"""Backtest of a strategy on one simulated path.

At every decision date the strategy chooses its portfolio, for every level of
risk aversion, using the data up to and including that date; the portfolio is
held through the next month and earns that month's return. Between decisions
the holdings drift with their returns, so the trade made at a decision is the
distance from the drifted holdings to the new portfolio. The first portfolio is
bought from cash.

For each date and level of risk aversion the result records the next month's
return, the turnover of the trade (two-sided: the sum of absolute changes in
weight, 1 for the opening purchase), the number of holdings, the effective
number of holdings, and which statement of the optimization problem the solver
solved. The full weights are kept only when asked for.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path

import numpy as np

from regime_aware_portfolios.config import StudyConfig, load_config
from regime_aware_portfolios.paths import RESULTS
from regime_aware_portfolios.simulation import (
    DataGeneratingProcess,
    SimulatedPath,
    simulate_path,
)
from regime_aware_portfolios.strategies import (
    Strategy,
    decision_dates,
    decisions,
    portfolio,
    study_strategies,
)

__all__ = [
    "PathResult",
    "backtest_path",
    "effective_holdings",
    "load_result",
    "result_file",
    "run_backtest",
    "save_result",
    "turnover",
]


@dataclass(frozen=True)
class PathResult:
    """A strategy's backtest on one path.

    Arrays have one row per level of risk aversion (in the order of ``phis``)
    and one column per decision date (in the order of ``dates``). ``weights``,
    when kept, has shape (risk aversions, dates, stocks).

    Strategies with a regime model also record, at every decision date, the
    filtered probabilities ``filtered`` and forecast probabilities ``forecast``
    of each regime (dates, regimes), and the regime model's factor means
    ``regime_means`` (dates, regimes, factors).
    """

    phis: np.ndarray
    dates: np.ndarray
    returns: np.ndarray
    turnover: np.ndarray
    holdings: np.ndarray
    effective_holdings: np.ndarray
    attempts: np.ndarray
    weights: np.ndarray | None = None
    filtered: np.ndarray | None = None
    forecast: np.ndarray | None = None
    regime_means: np.ndarray | None = None


def backtest_path(
    strategy: Strategy,
    path: SimulatedPath,
    cfg: StudyConfig,
    dgp: DataGeneratingProcess | None = None,
    keep_weights: bool = False,
) -> PathResult:
    """Run ``strategy`` on ``path`` at every decision date and risk aversion.

    Raises OptimizationError, naming the date and risk aversion, if a portfolio
    cannot be found.
    """
    phis = np.asarray(cfg.strategies.risk_aversion, dtype=float)
    dates = np.asarray(decision_dates(cfg))
    R = path.returns.to_numpy()
    n_stocks = R.shape[1]
    W = np.empty((len(phis), len(dates), n_stocks))
    attempts = np.zeros((len(phis), len(dates)), dtype=np.int8)
    solver = cfg.solver.name
    filtered, forecast, regime_means = [], [], []

    for j, dec in enumerate(decisions(strategy, path.factors, path.returns, cfg, dgp)):
        if dec.filtered is not None:
            filtered.append(dec.filtered)
            forecast.append(dec.moments.probs)
            regime_means.append(dec.regime_means)
        if strategy.depends_on_risk_aversion:
            for i, phi in enumerate(phis):
                port = _solve(strategy, dec, phi, solver)
                W[i, j], attempts[i, j] = port.weights, port.attempt
        else:
            port = _solve(strategy, dec, phis[0], solver)
            W[:, j], attempts[:, j] = port.weights, port.attempt

    held = R[dates + 1]  # the month after each decision
    returns = np.einsum("pdn,dn->pd", W, held)
    threshold = cfg.backtest.holdings_threshold
    return PathResult(
        phis=phis,
        dates=dates,
        returns=returns,
        turnover=np.stack([turnover(W[i], held) for i in range(len(phis))]),
        holdings=(W >= threshold).sum(axis=2),
        effective_holdings=effective_holdings(W, threshold),
        attempts=attempts,
        weights=W.astype(np.float32) if keep_weights else None,
        filtered=np.array(filtered) if filtered else None,
        forecast=np.array(forecast) if forecast else None,
        regime_means=np.array(regime_means) if regime_means else None,
    )


def turnover(weights: np.ndarray, held_returns: np.ndarray) -> np.ndarray:
    """Two-sided turnover of each trade, against the drifted holdings.

    ``weights`` has one row per decision and ``held_returns`` the returns of
    the month each portfolio is held. The first trade is the purchase from cash,
    whose turnover is the sum of the absolute weights.
    """
    grown = weights[:-1] * (1.0 + held_returns[:-1])
    drifted = grown / grown.sum(axis=1, keepdims=True)
    out = np.empty(len(weights))
    out[0] = np.abs(weights[0]).sum()
    out[1:] = np.abs(weights[1:] - drifted).sum(axis=1)
    return out


def effective_holdings(weights: np.ndarray, threshold: float) -> np.ndarray:
    """Exponential of the entropy of the holdings at or above ``threshold``.

    Equal to the number of holdings when they are equally weighted, and smaller
    when the portfolio is concentrated in a few of them. The last axis of
    ``weights`` is the stocks.
    """
    held = np.where(weights >= threshold, weights, 0.0)
    shares = held / held.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(shares > 0, shares * np.log(shares), 0.0)
    return np.exp(-terms.sum(axis=-1))


def _solve(strategy, decision, phi, solver):
    from regime_aware_portfolios.optimize import OptimizationError

    try:
        return portfolio(strategy, decision, phi, solver)
    except OptimizationError as e:
        raise OptimizationError(
            f"{strategy.name}, date {decision.date}, risk aversion {phi:g}: {e}"
        ) from e


# ---------------------------------------------------------------- study runs


def run_backtest(config: str | Path, name: str, path: int) -> Path:
    """Backtest strategy ``name`` on path number ``path`` of a study and store it.

    The path is generated from the study's seed. Returns the file written.
    """
    cfg, dgp = _study(str(config))
    sim = cfg.simulation
    simulated = simulate_path(
        dgp, sim.n_periods, sim.n_stocks, cfg.seeds.simulation, path
    )
    result = backtest_path(
        study_strategies(cfg)[name],
        simulated,
        cfg,
        dgp,
        keep_weights=path < cfg.backtest.weights_paths,
    )
    file = result_file(RESULTS / cfg.name / "backtest", name, path)
    save_result(result, file)
    return file


def result_file(folder: str | Path, name: str, path: int) -> Path:
    """Where the backtest of strategy ``name`` on path ``path`` is stored."""
    return Path(folder) / name / f"path_{path:04d}.npz"


@lru_cache(maxsize=4)
def _study(config: str):
    """A study's configuration and data-generating process, read once per process."""
    import logging
    import warnings

    from regime_aware_portfolios.calibration import load_calibration
    from regime_aware_portfolios.regimes import load_regimes

    logging.getLogger("hmmlearn").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    cfg = load_config(config)
    study = RESULTS / cfg.name
    dgp = DataGeneratingProcess.build(
        load_regimes(study / "regimes"), load_calibration(study / "calibration")
    )
    return cfg, dgp


# ---------------------------------------------------------------------- storage


def save_result(result: PathResult, file: str | Path) -> None:
    """Store a result as a compressed npz file, written atomically."""
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        f.name: getattr(result, f.name)
        for f in fields(PathResult)
        if getattr(result, f.name) is not None
    }
    partial = file.with_name(file.stem + ".partial.npz")
    np.savez_compressed(partial, **arrays)
    partial.replace(file)


def load_result(file: str | Path) -> PathResult:
    """Read a result stored by :func:`save_result`."""
    with np.load(file) as d:
        return PathResult(**{k: d[k] for k in d.files})
