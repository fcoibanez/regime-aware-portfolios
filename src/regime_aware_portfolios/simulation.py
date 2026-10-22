"""Simulated factor and stock returns.

A path is drawn from the data-generating process in five steps:

1. a regime path from the Markov chain of the regime model, started from the
   chain's stationary distribution;
2. factor returns, normal within each regime with that regime's means and
   variances;
3. each stock's loadings, from the calibrated loading distribution;
4. each stock's log residual variances in every regime, from the calibrated
   regression on its loadings;
5. each stock's returns: loadings of the current regime times the factor
   returns, plus a normal residual with that regime's variance.

Excess returns have no intercept. Every path has its own random streams, derived
from the study's seed and the path's number, so that any path can be generated
on its own. Within a path the regime path, the factor returns and every stock
draw from separate streams, so that a shorter path is the start of a longer one
and a smaller cross-section is part of a larger one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from regime_aware_portfolios.calibration import Calibration
from regime_aware_portfolios.regimes import RegimeParameters

__all__ = [
    "DataGeneratingProcess",
    "SimulatedPath",
    "path_seed",
    "save_process",
    "simulate_path",
    "stationary_distribution",
]


@dataclass(frozen=True)
class DataGeneratingProcess:
    """Parameters of the simulation, regimes labelled 1 to M.

    Factor means and volatilities have one row per regime. Loadings are ordered
    by factor, then regime, as in :class:`Calibration`.
    """

    factor_names: list[str]
    initial: np.ndarray
    transmat: np.ndarray
    factor_means: np.ndarray
    factor_vols: np.ndarray
    loadings_mean: np.ndarray
    loadings_factor: np.ndarray
    resid_intercept: np.ndarray
    resid_coef: np.ndarray
    resid_factor: np.ndarray

    @property
    def n_regimes(self) -> int:
        return len(self.initial)

    @classmethod
    def build(
        cls, regimes: RegimeParameters, calibration: Calibration
    ) -> DataGeneratingProcess:
        """Assemble the process from a fitted regime model and a calibration."""
        names = regimes.factor_names
        order = pd.MultiIndex.from_product(
            [names, list(regimes.means.index)], names=["factor", "regime"]
        )
        loadings_cov = calibration.loadings_cov.loc[order, order].to_numpy()
        transmat = regimes.transmat.to_numpy()
        return cls(
            factor_names=list(names),
            initial=stationary_distribution(transmat),
            transmat=transmat,
            factor_means=regimes.means.to_numpy(),
            factor_vols=np.sqrt(np.diagonal(regimes.covariances, axis1=1, axis2=2)),
            loadings_mean=calibration.loadings_mean.loc[order].to_numpy(),
            loadings_factor=np.linalg.cholesky(loadings_cov),
            resid_intercept=calibration.resid_intercept.to_numpy(),
            resid_coef=calibration.resid_coef.loc[:, order].to_numpy(),
            resid_factor=np.linalg.cholesky(calibration.resid_cov.to_numpy()),
        )

    def to_dict(self) -> dict:
        """The process as plain values, for storage."""
        return {
            "factor_names": self.factor_names,
            **{
                name: getattr(self, name).tolist()
                for name in (
                    "initial",
                    "transmat",
                    "factor_means",
                    "factor_vols",
                    "loadings_mean",
                    "loadings_factor",
                    "resid_intercept",
                    "resid_coef",
                    "resid_factor",
                )
            },
        }


@dataclass(frozen=True)
class SimulatedPath:
    """One simulated path.

    ``states`` holds the regime of every period, labelled 1 to M. ``loadings``
    has shape (stocks, factors, regimes) and ``resid_vol`` (stocks, regimes).
    """

    factors: pd.DataFrame
    returns: pd.DataFrame
    states: np.ndarray
    loadings: np.ndarray
    resid_vol: np.ndarray


def stationary_distribution(transmat: np.ndarray) -> np.ndarray:
    """The stationary distribution of a transition matrix."""
    values, vectors = np.linalg.eig(np.asarray(transmat).T)
    v = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
    return v / v.sum()


def path_seed(seed: int, path: int) -> np.random.SeedSequence:
    """The seed of one path: child ``path`` of the study's seed."""
    return np.random.SeedSequence(seed, spawn_key=(path,))


def simulate_path(
    dgp: DataGeneratingProcess, n_periods: int, n_stocks: int, seed: int, path: int
) -> SimulatedPath:
    """Draw path number ``path`` of the study whose seed is ``seed``."""
    # Sub-streams: 0 the regime path, 1 the factor returns, 2 + i stock i. Child
    # i of a seed does not depend on how many children are spawned with it.
    streams = path_seed(seed, path).spawn(2 + n_stocks)
    regime_rng = np.random.default_rng(streams[0])
    factor_rng = np.random.default_rng(streams[1])

    states = _markov_chain(dgp.initial, dgp.transmat, n_periods, regime_rng)
    shocks = factor_rng.standard_normal((n_periods, len(dgp.factor_names)))
    factors = dgp.factor_means[states] + dgp.factor_vols[states] * shocks

    K, M = len(dgp.factor_names), dgp.n_regimes
    loadings = np.empty((n_stocks, K, M))
    resid_vol = np.empty((n_stocks, M))
    returns = np.empty((n_periods, n_stocks))
    for i in range(n_stocks):
        rng = np.random.default_rng(streams[2 + i])
        b = dgp.loadings_mean + dgp.loadings_factor @ rng.standard_normal(K * M)
        log_var = (
            dgp.resid_intercept
            + dgp.resid_coef @ b
            + dgp.resid_factor @ rng.standard_normal(M)
        )
        loadings[i] = b.reshape(K, M)
        resid_vol[i] = np.sqrt(np.exp(log_var))
        systematic = np.einsum("tk,tk->t", factors, loadings[i].T[states])
        returns[:, i] = systematic + resid_vol[i, states] * rng.standard_normal(
            n_periods
        )

    index = pd.RangeIndex(n_periods, name="period")
    return SimulatedPath(
        factors=pd.DataFrame(factors, index=index, columns=dgp.factor_names),
        returns=pd.DataFrame(returns, index=index),
        states=states + 1,
        loadings=loadings,
        resid_vol=resid_vol,
    )


def _markov_chain(
    initial: np.ndarray, transmat: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Regimes 0 to M - 1 of an n-period path, by inverse transform sampling."""
    u = rng.random(n)
    cum_initial = np.cumsum(initial)
    cum_trans = np.cumsum(transmat, axis=1)
    states = np.empty(n, dtype=np.int64)
    states[0] = min(np.searchsorted(cum_initial, u[0], side="right"), len(initial) - 1)
    for t in range(1, n):
        row = cum_trans[states[t - 1]]
        states[t] = min(np.searchsorted(row, u[t], side="right"), len(initial) - 1)
    return states


def save_process(dgp: DataGeneratingProcess, folder: str | Path) -> None:
    """Store the data-generating process as ``process.json`` in ``folder``."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "process.json").write_text(
        json.dumps(dgp.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
