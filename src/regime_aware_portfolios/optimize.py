"""Portfolio optimization, long-only and fully invested.

Returns are a mixture of normal distributions, one component per regime: in
regime s with probability p_s, mean m_s and covariance V_s. Under exponential
utility with absolute risk aversion phi, expected utility is maximized by
minimizing

    log sum_s p_s exp(-phi m_s'w + phi^2/2 w'V_s w),

the logarithm of the mixture's moment-generating function at -phi w. With a
single component this is mean-variance optimization, phi m'w - phi^2/2 w'Vw.

Covariances are given in factor form, V_s = L_s L_s' + diag(d_s), where L_s has
one column per factor. Then w'V_s w = ||L_s'w||^2 + sum_i d_si w_i^2, which the
solver handles far more efficiently than a dense quadratic form in every stock.

The solver is not equally reliable on every equivalent statement of a problem.
Each problem is therefore tried in a fixed order of statements that have the
same solution (the objective multiplied by a positive constant, and with
Clarabel a shorter maximum step of its interior-point method) until one is
solved to the solver's default tolerances. The result records which statement
was used. A problem that none of them solves raises OptimizationError.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

__all__ = [
    "ATTEMPTS",
    "MixtureMoments",
    "OptimizationError",
    "Portfolio",
    "collapse",
    "mean_variance_portfolio",
    "minimum_variance_portfolio",
    "mixture_portfolio",
]

#: Statements of a problem tried in order, per solver and kind of problem: a
#: multiplier of the objective and solver settings. The orders were chosen on
#: the problems of the study's earlier code (10 paths, 60 dates, five levels of
#: risk aversion), on which they leave no problem unsolved. The last five
#: mixture statements were added, in turn, for two problems of the full studies
#: that the statements before them failed on (both at the lowest risk aversion);
#: each problem has the same solution in every statement that solves it. Being
#: last, they change no solution the earlier statements reach.
ATTEMPTS = {
    "CLARABEL": {
        "mixture": (
            (10.0, {"max_step_fraction": 0.9}),
            (100.0, {}),
            (1.0, {"max_step_fraction": 0.9}),
            (10.0, {}),
            (1.0, {}),
            (1000.0, {}),
            (10.0, {"max_step_fraction": 0.5}),
            (1.0, {"equilibrate_enable": False}),
            (1.0, {"max_step_fraction": 0.5}),
        ),
        "quadratic": (
            (1.0, {}),
            (1.0, {"max_step_fraction": 0.9}),
            (10.0, {}),
        ),
    },
}
_DEFAULT_ATTEMPTS = ((1.0, {}),)


class OptimizationError(RuntimeError):
    """No statement of the problem was solved to the solver's tolerances."""


@dataclass(frozen=True)
class Portfolio:
    """Optimal weights, and the position in ``ATTEMPTS`` of the statement solved."""

    weights: np.ndarray
    attempt: int


@dataclass(frozen=True)
class MixtureMoments:
    """Return moments of every regime, in factor form.

    ``probs`` has shape (M,), ``means`` (M, N), ``factor_roots`` (M, N, K) with
    V_s = factor_roots[s] @ factor_roots[s].T + diag(idio_var[s]), and
    ``idio_var`` (M, N).
    """

    probs: np.ndarray
    means: np.ndarray
    factor_roots: np.ndarray
    idio_var: np.ndarray

    @property
    def n_assets(self) -> int:
        return self.means.shape[1]

    @classmethod
    def from_factor_model(
        cls, probs, intercepts, loadings, factor_means, factor_covs, idio_var
    ) -> MixtureMoments:
        """Moments implied by a factor model in every regime.

        Parameters
        ----------
        probs : array, shape (M,)
            Regime probabilities.
        intercepts : array, shape (M, N)
            Intercepts of every asset in every regime.
        loadings : array, shape (M, N, K)
            Factor loadings.
        factor_means, factor_covs : arrays, shapes (M, K) and (M, K, K)
            Mean and covariance of the factors in every regime.
        idio_var : array, shape (M, N)
            Residual variances.
        """
        loadings = np.asarray(loadings, dtype=float)
        means = intercepts + np.einsum("snk,sk->sn", loadings, factor_means)
        roots = np.stack(
            [b @ _cov_root(c) for b, c in zip(loadings, factor_covs, strict=True)]
        )
        return cls(
            probs=np.asarray(probs, dtype=float),
            means=np.asarray(means, dtype=float),
            factor_roots=roots,
            idio_var=np.asarray(idio_var, dtype=float),
        )

    def covariance(self, s: int) -> np.ndarray:
        """Dense covariance matrix of regime ``s``."""
        L = self.factor_roots[s]
        return L @ L.T + np.diag(self.idio_var[s])


def collapse(moments: MixtureMoments) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of the mixture, as dense arrays."""
    p = moments.probs
    mean = p @ moments.means
    cov = sum(
        p[s] * (moments.covariance(s) + np.outer(moments.means[s], moments.means[s]))
        for s in range(len(p))
    )
    cov = cov - np.outer(mean, mean)
    return mean, (cov + cov.T) / 2


def mixture_portfolio(
    phi: float, moments: MixtureMoments, solver: str = "CLARABEL"
) -> Portfolio:
    """Weights maximizing expected exponential utility over the full mixture."""
    w = cp.Variable(moments.n_assets)
    terms = [
        np.log(moments.probs[s])
        - phi * moments.means[s] @ w
        + phi**2 / 2 * _variance(w, moments, s)
        for s in range(len(moments.probs))
    ]
    return _solve(cp.log_sum_exp(cp.hstack(terms)), w, solver, "mixture")


def mean_variance_portfolio(
    phi: float, moments: MixtureMoments, solver: str = "CLARABEL"
) -> Portfolio:
    """Weights maximizing phi m'w - phi^2/2 w'Vw for the mixture's mean and covariance.

    The mixture's variance of a portfolio is the probability-weighted variance
    within regimes plus the variance of its mean across regimes:
    w'Vw = sum_s p_s w'V_s w + sum_s p_s (m_s'w - m'w)^2.
    """
    w = cp.Variable(moments.n_assets)
    p = moments.probs
    mean = p @ moments.means
    within = sum(p[s] * _variance(w, moments, s) for s in range(len(p)))
    across = cp.sum_squares(cp.multiply(np.sqrt(p), (moments.means - mean) @ w))
    objective = -phi * mean @ w + phi**2 / 2 * (within + across)
    return _solve(objective, w, solver, "quadratic")


def minimum_variance_portfolio(
    returns: np.ndarray, solver: str = "CLARABEL"
) -> Portfolio:
    """Weights minimizing the sample variance of a portfolio of ``returns``.

    ``returns`` has one row per period. The sample covariance S (divisor
    T - 1) is written as R R', so that w'Sw = ||R'w||^2 has one term per asset
    rather than one per period.
    """
    returns = np.asarray(returns, dtype=float)
    root = _cov_root(np.cov(returns, rowvar=False))
    w = cp.Variable(returns.shape[1])
    return _solve(cp.sum_squares(root.T @ w), w, solver, "quadratic")


def _variance(w, moments: MixtureMoments, s: int):
    return cp.sum_squares(moments.factor_roots[s].T @ w) + cp.sum_squares(
        cp.multiply(np.sqrt(moments.idio_var[s]), w)
    )


def _solve(objective, w, solver: str, kind: str) -> Portfolio:
    attempts = ATTEMPTS.get(solver, {}).get(kind, _DEFAULT_ATTEMPTS)
    statuses = []
    for k, (scale, settings) in enumerate(attempts):
        problem = cp.Problem(cp.Minimize(scale * objective), [w >= 0, cp.sum(w) == 1])
        try:
            problem.solve(solver=solver, **settings)
        except cp.error.SolverError:
            statuses.append("solver error")
            continue
        if problem.status == cp.OPTIMAL and w.value is not None:
            return Portfolio(weights=np.asarray(w.value, dtype=float), attempt=k)
        statuses.append(problem.status)
    raise OptimizationError(f"no statement of the problem was solved: {statuses}")


def _cov_root(cov: np.ndarray) -> np.ndarray:
    """A matrix R with R R' = cov, for a positive semidefinite cov."""
    values, vectors = np.linalg.eigh((cov + cov.T) / 2)
    return vectors * np.sqrt(np.clip(values, 0.0, None))
