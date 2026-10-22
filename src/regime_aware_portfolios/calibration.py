"""Distributions of stock parameters that the simulation draws from.

Each stock of the calibration sample is regressed on the factors in every regime
by regime-weighted least squares (rwls), with the posterior probabilities of the
fitted regime model as weights. The estimates are then summarized across stocks:

* Loadings: the cross-sectional mean, and the cross-sectional covariance net of
  estimation error. The covariance of estimated loadings is that of the true
  loadings plus the average sampling covariance, and only the former belongs in
  the simulation. The loadings of one factor across regimes form a block; blocks
  of different factors are uncorrelated, and each is made positive definite.
* Residual variances: one per stock and regime. Their logarithms are regressed
  on the loadings, so that a simulated stock's idiosyncratic risk depends on its
  exposures. The log of a variance estimate is biased down by about one over its
  degrees of freedom and has a sampling variance of about two over them; both
  are removed. Stocks with few effective degrees of freedom in some regime are
  left out of this step only.

Every stock counts equally.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rwls

__all__ = [
    "Calibration",
    "StockEstimates",
    "calibrate",
    "estimate_stocks",
    "load_calibration",
    "save_calibration",
]


@dataclass(frozen=True)
class StockEstimates:
    """Regime-specific estimates of every stock of the calibration sample.

    Loadings are ordered by factor, then regime. ``sampling_cov[i]`` is the
    sandwich covariance of stock ``i``'s loadings in the same order.
    """

    loadings: pd.DataFrame
    sampling_cov: np.ndarray
    resid_var: pd.DataFrame
    dof: pd.DataFrame
    fallback: pd.DataFrame


def estimate_stocks(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    posteriors: pd.DataFrame,
    settings: rwls.RWLSSettings,
) -> StockEstimates:
    """RWLS estimates of every stock in ``returns``.

    Parameters
    ----------
    returns : pandas.DataFrame
        Excess returns, one column per stock; missing outside each stock's sample.
    factors : pandas.DataFrame
        Factor returns.
    posteriors : pandas.DataFrame
        Posterior regime probabilities, one column per regime.
    settings : rwls.RWLSSettings
        Estimator settings.
    """
    res = rwls.RWLS(returns, factors, posteriors, settings).fit()
    names = list(factors.columns)
    regimes = list(posteriors.columns)
    terms = res.terms
    columns = pd.MultiIndex.from_product([names, regimes], names=["factor", "regime"])

    # rwls orders coefficients by regime, then term; here by factor, then regime.
    k = len(terms)
    order = [s * k + terms.index(f) for f in names for s in range(len(regimes))]
    stocks = list(res.asset_names)
    loadings = res.params.to_numpy()[order].T
    sampling_cov = np.stack(
        [res.cov_params(a).to_numpy()[np.ix_(order, order)] for a in stocks]
    )

    # Effective degrees of freedom of each stock's residual variance in a regime:
    # (sum of weights)^2 / (sum of squared weights), less the coefficients.
    observed = returns[stocks].notna().to_numpy()
    P = posteriors.reindex(returns.index).to_numpy()
    sums = observed.T.astype(float) @ P
    squares = observed.T.astype(float) @ P**2
    dof = np.maximum(sums**2 / squares - k, 1.0)

    index = pd.Index(stocks, name=returns.columns.name)
    regime_index = pd.Index(regimes, name="regime")
    return StockEstimates(
        loadings=pd.DataFrame(loadings, index=index, columns=columns),
        sampling_cov=sampling_cov,
        resid_var=pd.DataFrame(
            res.resid_var.to_numpy().T, index=index, columns=regime_index
        ),
        dof=pd.DataFrame(dof, index=index, columns=regime_index),
        fallback=pd.DataFrame(
            res.fallback.to_numpy().T, index=index, columns=regime_index
        ),
    )


@dataclass(frozen=True)
class Calibration:
    """Parameters of the distributions the simulated stocks are drawn from.

    ``loadings_mean`` and ``loadings_cov`` describe the loadings, indexed by
    factor and regime. A stock's log residual variances are
    ``resid_intercept + resid_coef @ loadings`` plus a normal error with
    covariance ``resid_cov``, one value per regime.
    """

    loadings_mean: pd.Series
    loadings_cov: pd.DataFrame
    resid_intercept: pd.Series
    resid_coef: pd.DataFrame
    resid_cov: pd.DataFrame
    record: dict


def calibrate(
    estimates: StockEstimates,
    loading_eigen_floor: float,
    resid_eigen_floor: float,
    min_resid_dof: float,
) -> Calibration:
    """Distributions of loadings and residual variances across stocks.

    Parameters
    ----------
    estimates : StockEstimates
        Estimates of every stock.
    loading_eigen_floor : float
        Smallest eigenvalue allowed in each factor's block of the loading
        covariance.
    resid_eigen_floor : float
        Smallest eigenvalue allowed in the covariance of the log residual
        variances.
    min_resid_dof : float
        Effective degrees of freedom a stock needs in every regime to enter the
        residual-variance regression.
    """
    params = estimates.loadings.to_numpy()
    columns = estimates.loadings.columns
    factors = list(columns.get_level_values("factor").unique())
    regimes = estimates.resid_var.columns

    mean = params.mean(axis=0)
    observed = np.cov(params, rowvar=False)
    true = observed - estimates.sampling_cov.mean(axis=0)
    cov = np.zeros_like(true)
    for f in factors:
        ix = np.flatnonzero(columns.get_level_values("factor") == f)
        cov[np.ix_(ix, ix)] = _floor_eigenvalues(
            true[np.ix_(ix, ix)], loading_eigen_floor
        )

    s2 = estimates.resid_var.to_numpy()
    dof = estimates.dof.to_numpy()
    keep = (dof >= min_resid_dof).all(axis=1)
    z = np.log(s2[keep]) + 1.0 / dof[keep]
    design = np.column_stack([np.ones(keep.sum()), params[keep]])
    coef, *_ = np.linalg.lstsq(design, z, rcond=None)
    errors = z - design @ coef
    resid_cov = np.cov(errors, rowvar=False) - np.diag((2.0 / dof[keep]).mean(axis=0))
    resid_cov = _floor_eigenvalues(resid_cov, resid_eigen_floor)

    record = {
        "n_stocks": int(len(params)),
        "n_resid_stocks": int(keep.sum()),
        "fallback_estimates": int(estimates.fallback.to_numpy().sum()),
        "estimation_error_share": dict(
            zip(
                [f"{f} {r}" for f, r in columns],
                (1 - np.diag(cov) / np.diag(observed)).tolist(),
                strict=True,
            )
        ),
        "mean_resid_var": dict(
            zip(map(str, regimes), s2.mean(axis=0).tolist(), strict=True)
        ),
    }
    return Calibration(
        loadings_mean=pd.Series(mean, index=columns),
        loadings_cov=pd.DataFrame(cov, index=columns, columns=columns),
        resid_intercept=pd.Series(coef[0], index=regimes),
        resid_coef=pd.DataFrame(coef[1:].T, index=regimes, columns=columns),
        resid_cov=pd.DataFrame(resid_cov, index=regimes, columns=regimes),
        record=record,
    )


def _floor_eigenvalues(matrix: np.ndarray, floor: float) -> np.ndarray:
    """The symmetric part of ``matrix`` with its eigenvalues raised to ``floor``."""
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    return vectors @ np.diag(np.maximum(values, floor)) @ vectors.T


# ---------------------------------------------------------------------- storage


def save_calibration(calibration: Calibration, folder: str | Path) -> None:
    """Store a calibration as ``calibration.json`` in ``folder``."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    columns = calibration.loadings_mean.index
    record = {
        "factors": list(columns.get_level_values("factor").unique()),
        "regimes": [int(r) for r in calibration.resid_intercept.index],
        "loadings_mean": calibration.loadings_mean.tolist(),
        "loadings_cov": calibration.loadings_cov.to_numpy().tolist(),
        "resid_intercept": calibration.resid_intercept.tolist(),
        "resid_coef": calibration.resid_coef.to_numpy().tolist(),
        "resid_cov": calibration.resid_cov.to_numpy().tolist(),
        "record": calibration.record,
    }
    (folder / "calibration.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )


def load_calibration(folder: str | Path) -> Calibration:
    """Read a calibration stored by :func:`save_calibration`."""
    record = json.loads((Path(folder) / "calibration.json").read_text(encoding="utf-8"))
    regimes = pd.Index(record["regimes"], name="regime")
    columns = pd.MultiIndex.from_product(
        [record["factors"], record["regimes"]], names=["factor", "regime"]
    )
    return Calibration(
        loadings_mean=pd.Series(record["loadings_mean"], index=columns),
        loadings_cov=pd.DataFrame(
            record["loadings_cov"], index=columns, columns=columns
        ),
        resid_intercept=pd.Series(record["resid_intercept"], index=regimes),
        resid_coef=pd.DataFrame(record["resid_coef"], index=regimes, columns=columns),
        resid_cov=pd.DataFrame(record["resid_cov"], index=regimes, columns=regimes),
        record=record["record"],
    )
