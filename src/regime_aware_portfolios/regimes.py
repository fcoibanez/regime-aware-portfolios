"""The regime model fitted to the factor returns, which calibrates the simulation.

The model is fitted with rwls from many cold starts, and one fit is kept by the
rule in the ``[regimes]`` section of the configuration:

``"best"``
    the highest likelihood reached;
``"modal"``
    the model reached by the most starts;
``"nearest"``
    the optimum whose regime means are closest to the reference means given in
    the configuration, after matching regimes.

Regimes are then numbered by the volatility of one factor, calmest first, so
that the numbering does not depend on the starting values of EM.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rwls
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from regime_aware_portfolios.config import StudyConfig

__all__ = [
    "RegimeParameters",
    "fit_regimes",
    "load_regimes",
    "order_regimes",
    "reference_distance",
    "save_regimes",
]


def fit_regimes(
    factors: pd.DataFrame, cfg: StudyConfig, n_jobs: int = -1
) -> tuple[rwls.RegimeFit, dict]:
    """Fit the calibrating regime model.

    Parameters
    ----------
    factors : pandas.DataFrame
        Factor returns, one column per factor of the configuration.
    cfg : StudyConfig
        Configuration; its ``[regimes]`` and ``[seeds]`` sections are used.
    n_jobs : int
        Processes for the cold starts; ``-1`` uses all processors.

    Returns
    -------
    fit : rwls.RegimeFit
        The selected fit, with regimes numbered by ``regimes.order_by``.
    starts : dict
        How many starts were fitted, how many passed the health check, how many
        reached the selected model, and the selection rule.
    """
    factors = factors[cfg.factor_names]
    if cfg.regimes.selection == "nearest":
        fit, starts = _nearest(factors, cfg, n_jobs)
    else:
        fit = rwls.RegimeModel(cfg.regime_settings()).fit(factors, n_jobs=n_jobs)
        starts = dict(fit.starts)
    return order_regimes(fit, cfg.regimes.order_by), starts


def order_regimes(fit: rwls.RegimeFit, factor: str) -> rwls.RegimeFit:
    """The fit with regimes numbered by the variance of ``factor``, lowest first."""
    return fit.relabel(fit.variances[factor].sort_values(kind="stable").index)


def reference_distance(fit: rwls.RegimeFit, reference: pd.DataFrame) -> float:
    """Distance between a fit's regime means and reference means.

    Regimes are matched as in ``rwls.RegimeFit.alignment``: each factor is
    divided by the dispersion of the reference means across regimes, and the
    matching minimizes the total Euclidean distance, which is returned.
    """
    ref = reference[fit.factor_names].to_numpy()
    scale = ref.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    cost = cdist(fit.means.to_numpy() / scale, ref / scale)
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].sum())


def _nearest(factors: pd.DataFrame, cfg: StudyConfig, n_jobs: int):
    settings = cfg.regime_settings(n_starts=1)
    seeds = [cfg.seeds.regimes + j for j in range(cfg.regimes.n_starts)]
    fits = Parallel(n_jobs=n_jobs)(
        delayed(_single_start)(factors, settings.replace(seed=seed)) for seed in seeds
    )
    healthy = [f for f in fits if f is not None]
    if not healthy:
        raise RuntimeError("the regime model failed to converge from every start")

    # Reference means are given in percent, one value per regime and factor.
    # The nearest fit identifies the optimum; its best fit is kept.
    reference = pd.DataFrame(cfg.regimes.reference) / 100.0
    distances = [reference_distance(f, reference) for f in healthy]
    nearest = healthy[int(np.argmin(distances))]
    scale = factors.std(ddof=0).to_numpy()
    group = [f for f in healthy if _same_optimum(f, nearest, scale, settings)]
    selected = max(group, key=lambda f: f.loglike)
    starts = {
        "starts": len(fits),
        "healthy": len(healthy),
        "reached": len(group),
        "selection": "nearest",
        "distance": reference_distance(selected, reference),
        "seed": selected.settings.seed,
    }
    return selected, starts


def _single_start(factors: pd.DataFrame, settings: rwls.RegimeSettings):
    try:
        return rwls.RegimeModel(settings).fit(factors)
    except RuntimeError:
        return None


def _same_optimum(fit, other, scale, settings) -> bool:
    """Whether two fits reached the same optimum, by the rwls grouping tolerances."""
    aligned = fit.align_to(other)
    means = np.abs(aligned.means.to_numpy() - other.means.to_numpy()) / scale
    trans = np.abs(aligned.transmat.to_numpy() - other.transmat.to_numpy())
    return bool(
        means.max() < settings.modal_mean_tol
        and trans.max() < settings.modal_transition_tol
    )


# ---------------------------------------------------------------------- storage


@dataclass(frozen=True)
class RegimeParameters:
    """A fitted regime model as stored: parameters and posterior probabilities.

    Regimes are labelled 1 to M in every table. ``covariances[0]`` is regime 1.
    """

    factor_names: list[str]
    startprob: pd.Series
    transmat: pd.DataFrame
    means: pd.DataFrame
    covariances: np.ndarray
    posteriors: pd.DataFrame
    loglike: float
    record: dict


def save_regimes(fit: rwls.RegimeFit, starts: dict, folder: str | Path) -> None:
    """Store a fit as ``regimes.json`` and ``posteriors.parquet`` in ``folder``."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    record = {
        "factor_names": list(fit.factor_names),
        "first_period": f"{fit.index[0]:%Y-%m}",
        "last_period": f"{fit.index[-1]:%Y-%m}",
        "n_periods": int(fit.nobs),
        "loglike": fit.loglike,
        "n_iter": fit.n_iter,
        "converged": fit.converged,
        "startprob": fit.startprob.tolist(),
        "transmat": fit.transmat.to_numpy().tolist(),
        "means": fit.means.to_numpy().tolist(),
        "covariances": fit.covariances.tolist(),
        "starts": starts,
        "settings": dataclasses.asdict(fit.settings),
    }
    (folder / "regimes.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    posteriors = fit.posteriors.copy()
    posteriors.columns = [str(c) for c in posteriors.columns]
    posteriors.to_parquet(folder / "posteriors.parquet")


def load_regimes(folder: str | Path) -> RegimeParameters:
    """Read a fit stored by :func:`save_regimes`."""
    folder = Path(folder)
    record = json.loads((folder / "regimes.json").read_text(encoding="utf-8"))
    names = record["factor_names"]
    labels = pd.RangeIndex(1, len(record["startprob"]) + 1, name="regime")
    posteriors = pd.read_parquet(folder / "posteriors.parquet")
    posteriors.columns = labels
    return RegimeParameters(
        factor_names=names,
        startprob=pd.Series(record["startprob"], index=labels),
        transmat=pd.DataFrame(record["transmat"], index=labels, columns=labels),
        means=pd.DataFrame(record["means"], index=labels, columns=names),
        covariances=np.array(record["covariances"]),
        posteriors=posteriors,
        loglike=record["loglike"],
        record=record,
    )
