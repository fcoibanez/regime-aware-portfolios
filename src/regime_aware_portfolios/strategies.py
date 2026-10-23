"""The portfolio strategies compared in the study.

Every strategy decides at the end of each out-of-sample month, using only the
data up to and including that month, and holds its portfolio through the next.
The first decision uses the n_in_sample months of the path, so a path of
n_in_sample + n_out_of_sample months has n_out_of_sample decisions, each earning
one month's return.

Strategies are combinations of a few choices, held in :class:`Strategy`:

* how return moments are estimated:

  - "regimes": a hidden Markov model refitted at every decision date on the
    factors (or a subset of them), and per-regime regressions of every stock on
    all the factors, weighted by the regime's posterior probabilities (RWLS) or
    by hard classification into the most recent ``window`` months of the regime;
  - "agnostic": one regression over the whole history, or the most recent
    ``window`` months, with the sample mean and covariance of the factors;
  - "minimum_variance": the sample covariance of the stocks' returns;
  - "equal_weight": none;

* how next month's regime probabilities are formed: the filtered probabilities
  propagated through the transition matrix ("filtered"), or the transition row of
  the regime with the highest filtered probability ("current_row", as in Costa
  and Kwon, 2020);
* whether each regime's factor covariance is diagonal or full;
* the objective: expected exponential utility over the mixture of regime
  distributions ("mixture"), or mean-variance on the mixture's mean and
  covariance ("mvo");
* whether the regime model is estimated or is that of the data-generating
  process ("oracle"). The oracle runs the same regressions; only its regime
  probabilities, transition matrix and factor moments are the true ones.

Expected returns include each stock's estimated intercept, so that they equal
the (weighted) sample mean of its returns.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
import rwls
from hmmlearn.hmm import GaussianHMM

from regime_aware_portfolios.config import StudyConfig
from regime_aware_portfolios.optimize import (
    MixtureMoments,
    Portfolio,
    mean_variance_portfolio,
    minimum_variance_portfolio,
    mixture_portfolio,
)
from regime_aware_portfolios.simulation import DataGeneratingProcess

__all__ = [
    "Decision",
    "Strategy",
    "decision_dates",
    "decisions",
    "portfolio",
    "study_strategies",
]


@dataclass(frozen=True)
class Strategy:
    """How a strategy estimates moments and chooses its portfolio."""

    name: str
    estimator: str
    objective: str = "mvo"
    n_regimes: int = 1
    regime_factors: tuple[str, ...] | None = None
    weighting: str = "posterior"
    window: int | None = None
    forecast: str = "filtered"
    factor_cov: str = "diag"
    oracle: bool = False

    def __post_init__(self) -> None:
        choices = {
            "estimator": ("regimes", "agnostic", "minimum_variance", "equal_weight"),
            "objective": ("mixture", "mvo"),
            "weighting": ("posterior", "hard"),
            "forecast": ("filtered", "current_row"),
            "factor_cov": ("diag", "full"),
        }
        for field, allowed in choices.items():
            if getattr(self, field) not in allowed:
                raise ValueError(f"{self.name}: {field} must be one of {allowed}")
        if self.weighting == "hard" and self.window is None:
            raise ValueError(f"{self.name}: hard classification needs a window")

    @property
    def depends_on_risk_aversion(self) -> bool:
        return self.estimator in ("regimes", "agnostic")


def study_strategies(cfg: StudyConfig) -> dict[str, Strategy]:
    """The strategies of a study, by name.

    The Costa and Kwon steps change one choice at a time, from their estimator
    to the framework's mean-variance variant: the number of regimes, the factors
    the regime model sees, the forecast rule, the factor covariance, and the
    observation weights. The framework then differs only in its objective.
    """
    st = cfg.strategies
    M = cfg.regimes.n_regimes
    ck = dict(
        estimator="regimes",
        objective="mvo",
        weighting="hard",
        window=st.costa_kwon_window,
        forecast="current_row",
        factor_cov="full",
    )
    out = [
        Strategy("framework", "regimes", "mixture", n_regimes=M),
        Strategy("framework_mvo", "regimes", "mvo", n_regimes=M),
        Strategy("oracle", "regimes", "mixture", n_regimes=M, oracle=True),
        Strategy("oracle_mvo", "regimes", "mvo", n_regimes=M, oracle=True),
        Strategy("agnostic", "agnostic"),
        *[
            Strategy(f"rolling_{w}", "agnostic", window=w)
            for w in st.rolling_window_sensitivity
        ],
        Strategy(
            "costa_kwon",
            n_regimes=st.costa_kwon_regimes,
            regime_factors=st.costa_kwon_factors,
            **ck,
        ),
        Strategy(
            "path_regimes", n_regimes=M, regime_factors=st.costa_kwon_factors, **ck
        ),
        Strategy("path_factors", n_regimes=M, **ck),
        Strategy("path_forecast", n_regimes=M, **{**ck, "forecast": "filtered"}),
        Strategy(
            "path_covariance",
            n_regimes=M,
            **{**ck, "forecast": "filtered", "factor_cov": "diag"},
        ),
        Strategy("minimum_variance", "minimum_variance"),
        Strategy("equal_weight", "equal_weight"),
    ]
    return {s.name: s for s in out}


@dataclass(frozen=True)
class Decision:
    """The inputs of one decision: the date and the moments or returns used.

    For strategies with a regime model, ``filtered`` holds the probability of
    each regime at the decision date given the data up to it, and
    ``regime_means`` the regime model's factor means (one row per regime), from
    which estimated regimes can be matched to the true ones.
    """

    date: int
    moments: MixtureMoments | None = None
    returns: np.ndarray | None = None
    n_assets: int = 0
    filtered: np.ndarray | None = None
    regime_means: np.ndarray | None = None


def decision_dates(cfg: StudyConfig) -> range:
    """Positions of the decision dates in a simulated path."""
    sim = cfg.simulation
    return range(sim.n_in_sample - 1, sim.n_periods - 1)


def decisions(
    strategy: Strategy,
    factors: pd.DataFrame,
    returns: pd.DataFrame,
    cfg: StudyConfig,
    dgp: DataGeneratingProcess | None = None,
) -> Iterator[Decision]:
    """The decisions of ``strategy`` on one path, in date order.

    ``factors`` and ``returns`` are the whole path, indexed by period; each
    decision uses the rows up to and including its date. The oracle needs the
    data-generating process ``dgp``.
    """
    dates = decision_dates(cfg)
    n = returns.shape[1]
    if strategy.estimator == "equal_weight":
        for d in dates:
            yield Decision(date=d, n_assets=n)
    elif strategy.estimator == "minimum_variance":
        for d in dates:
            yield Decision(date=d, returns=returns.iloc[: d + 1].to_numpy(), n_assets=n)
    elif strategy.estimator == "agnostic":
        yield from _agnostic(strategy, factors, returns, cfg, dates)
    elif strategy.oracle:
        if dgp is None:
            raise ValueError("the oracle needs the data-generating process")
        yield from _oracle(strategy, factors, returns, cfg, dates, dgp)
    else:
        yield from _estimated(strategy, factors, returns, cfg, dates)


def portfolio(
    strategy: Strategy, decision: Decision, phi: float, solver: str = "CLARABEL"
) -> Portfolio:
    """The portfolio of ``strategy`` at one decision, for risk aversion ``phi``."""
    if strategy.estimator == "equal_weight":
        n = decision.n_assets
        return Portfolio(weights=np.full(n, 1.0 / n), attempt=0)
    if strategy.estimator == "minimum_variance":
        return minimum_variance_portfolio(decision.returns, solver)
    if strategy.objective == "mixture":
        return mixture_portfolio(phi, decision.moments, solver)
    return mean_variance_portfolio(phi, decision.moments, solver)


# ------------------------------------------------------------------ estimators


def _rwls_settings(strategy: Strategy, cfg: StudyConfig) -> rwls.RWLSSettings:
    if strategy.weighting == "hard":
        return cfg.rwls_settings(weighting="hard", hard_window=strategy.window)
    return cfg.rwls_settings()


def _estimated(strategy, factors, returns, cfg, dates) -> Iterator[Decision]:
    settings = rwls.RegimeSettings(n_regimes=strategy.n_regimes, seed=cfg.seeds.regimes)
    rwls_settings = _rwls_settings(strategy, cfg)
    regime_factors = list(strategy.regime_factors) if strategy.regime_factors else None
    steps = rwls.expanding(
        returns,
        factors,
        start=dates[0] + 1,
        end=factors.index[dates[-1]],
        regime_settings=settings,
        rwls_settings=rwls_settings,
        regime_factors=regime_factors,
    )
    for d, step in zip(dates, steps, strict=True):
        if strategy.forecast == "filtered":
            probs = step.forecast.to_numpy()
        else:
            probs = _current_row(
                step.regimes.posteriors.to_numpy(),
                step.regimes.transmat.to_numpy(),
                settings.forecast_floor,
            )
        results = step.results
        if strategy.factor_cov == "diag":
            fm, fc = results.factor_means.to_numpy(), results.factor_covariances
        else:
            weights = regime_weights(
                step.regimes.posteriors.to_numpy(),
                rwls_settings,
                n_coefficients=factors.shape[1] + 1,
            )
            fm, fc = _factor_moments(
                factors.iloc[: d + 1].to_numpy(), weights, full=True
            )
        yield Decision(
            d,
            moments=_mixture(results, probs, fm, fc),
            n_assets=returns.shape[1],
            filtered=step.regimes.posteriors.to_numpy()[-1],
            regime_means=step.regimes.means.to_numpy(),
        )


def _oracle(strategy, factors, returns, cfg, dates, dgp) -> Iterator[Decision]:
    if list(factors.columns) != dgp.factor_names:
        raise ValueError("factors must be those of the data-generating process")
    hmm = GaussianHMM(n_components=dgp.n_regimes, covariance_type="diag")
    hmm.startprob_ = dgp.initial
    hmm.transmat_ = dgp.transmat
    hmm.means_ = dgp.factor_means
    hmm.covars_ = dgp.factor_vols**2
    labels = list(range(1, dgp.n_regimes + 1))
    floor = rwls.RegimeSettings().forecast_floor
    true_covs = np.stack([np.diag(v**2) for v in dgp.factor_vols])
    rwls_settings = _rwls_settings(strategy, cfg)
    for d in dates:
        X, Y = factors.iloc[: d + 1], returns.iloc[: d + 1]
        post = pd.DataFrame(
            hmm.predict_proba(X.to_numpy()), index=X.index, columns=labels
        )
        results = rwls.RWLS(Y, X, post, rwls_settings).fit()
        probs = _floored(post.to_numpy()[-1] @ dgp.transmat, floor)
        yield Decision(
            d,
            moments=_mixture(results, probs, dgp.factor_means, true_covs),
            n_assets=returns.shape[1],
            filtered=post.to_numpy()[-1],
            regime_means=dgp.factor_means,
        )


def _agnostic(strategy, factors, returns, cfg, dates) -> Iterator[Decision]:
    for d in dates:
        lo = 0 if strategy.window is None else max(0, d + 1 - strategy.window)
        X, Y = factors.iloc[lo : d + 1], returns.iloc[lo : d + 1]
        one = pd.DataFrame({1: np.ones(len(X))}, index=X.index)
        results = rwls.RWLS(Y, X, one, cfg.rwls_settings()).fit()
        fm = X.to_numpy().mean(axis=0)[None, :]
        fc = np.cov(X.to_numpy(), rowvar=False)[None, :, :]
        yield Decision(
            d,
            moments=_mixture(results, np.array([1.0]), fm, fc),
            n_assets=returns.shape[1],
        )


# --------------------------------------------------------------------- helpers


def regime_weights(
    posteriors: np.ndarray, settings: rwls.RWLSSettings, n_coefficients: int
) -> list[np.ndarray]:
    """Observation weights of every regime, as rwls forms them.

    Posterior probabilities, or hard classification into the most recent
    ``settings.hard_window`` months of each regime; a regime whose weight is at
    or below the sparse threshold takes the fallback weights.
    """
    T, M = posteriors.shape
    if settings.weighting == "posterior":
        base = [posteriors[:, s].copy() for s in range(M)]
    else:
        labels = posteriors.argmax(axis=1)
        base = []
        for s in range(M):
            w = np.zeros(T)
            w[np.flatnonzero(labels == s)[-settings.hard_window :]] = 1.0
            base.append(w)
    threshold = settings.resolved_min_mass(n_coefficients)
    out = []
    for w in base:
        if w.sum() > threshold:
            out.append(w)
            continue
        fallback = np.zeros(T)
        if settings.resolved_fallback() == "pooled":
            fallback[:] = 1.0
        else:
            fallback[-settings.hard_window :] = 1.0
        out.append(fallback)
    return out


def _factor_moments(X: np.ndarray, weights, full: bool):
    """Weighted factor means and covariances (divided by the total weight)."""
    means, covs = [], []
    for w in weights:
        m = w @ X / w.sum()
        dev = X - m
        c = (w[:, None] * dev).T @ dev / w.sum()
        means.append(m)
        covs.append((c + c.T) / 2 if full else np.diag(np.diag(c)))
    return np.array(means), np.array(covs)


def _current_row(posteriors: np.ndarray, transmat: np.ndarray, floor: float):
    """Transition row of the regime with the highest filtered probability."""
    return _floored(transmat[int(np.argmax(posteriors[-1]))], floor)


def _floored(probs: np.ndarray, floor: float) -> np.ndarray:
    p = np.clip(probs, floor, None)
    return p / p.sum()


def _mixture(
    results: rwls.RWLSResults, probs, factor_means, factor_covs
) -> MixtureMoments:
    m = results.moments(factor_means=factor_means, factor_covariances=factor_covs)
    return MixtureMoments.from_factor_model(
        probs, m.alphas, m.loadings, m.factor_means, m.factor_covariances, m.resid_var
    )
