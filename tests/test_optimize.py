import cvxpy as cp
import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import logsumexp

from regime_aware_portfolios import optimize
from regime_aware_portfolios.optimize import (
    MixtureMoments,
    OptimizationError,
    collapse,
    mean_variance_portfolio,
    minimum_variance_portfolio,
    mixture_portfolio,
)


def _moments(n=12, regimes=2, factors=3, seed=0):
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(regimes))
    intercepts = rng.normal(0.002, 0.003, (regimes, n))
    loadings = rng.normal(1.0, 0.4, (regimes, n, factors))
    factor_means = rng.normal(0.005, 0.01, (regimes, factors))
    factor_covs = np.stack(
        [np.diag(rng.uniform(0.0005, 0.003, factors)) for _ in range(regimes)]
    )
    idio = rng.uniform(0.002, 0.01, (regimes, n))
    return MixtureMoments.from_factor_model(
        probs, intercepts, loadings, factor_means, factor_covs, idio
    ), (intercepts, loadings, factor_means, factor_covs, idio)


def _simplex_minimum(f, n):
    """Minimum of f over long-only, fully invested weights, by SLSQP."""
    res = minimize(
        f,
        np.full(n, 1 / n),
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    return res.x, res.fun


def test_factor_form_matches_the_dense_moments():
    mo, (a, b, m, F, d) = _moments()
    for s in range(2):
        np.testing.assert_allclose(mo.means[s], a[s] + b[s] @ m[s])
        np.testing.assert_allclose(
            mo.covariance(s), b[s] @ F[s] @ b[s].T + np.diag(d[s])
        )


def test_collapse_gives_the_mixture_mean_and_covariance():
    mo, _ = _moments()
    mean, cov = collapse(mo)
    p, m = mo.probs, mo.means
    np.testing.assert_allclose(mean, p @ m)
    within = sum(p[s] * mo.covariance(s) for s in range(2))
    across = sum(p[s] * np.outer(m[s] - mean, m[s] - mean) for s in range(2))
    np.testing.assert_allclose(cov, within + across, atol=1e-15)


@pytest.mark.parametrize("phi", [1, 10, 50])
def test_mixture_portfolio_minimizes_the_mixture_objective(phi):
    mo, _ = _moments()

    def objective(w):
        return logsumexp(
            [
                np.log(mo.probs[s])
                - phi * mo.means[s] @ w
                + phi**2 / 2 * w @ mo.covariance(s) @ w
                for s in range(2)
            ]
        )

    port = mixture_portfolio(phi, mo)
    _, best = _simplex_minimum(objective, mo.n_assets)
    assert objective(port.weights) <= best + 1e-8
    assert port.weights.min() >= -1e-9
    # The budget holds to the solver's feasibility tolerance.
    assert port.weights.sum() == pytest.approx(1.0, abs=1e-7)


@pytest.mark.parametrize("phi", [1, 10, 50])
def test_mean_variance_portfolio_uses_the_mixture_moments(phi):
    mo, _ = _moments()
    mean, cov = collapse(mo)

    def objective(w):
        return -phi * mean @ w + phi**2 / 2 * w @ cov @ w

    port = mean_variance_portfolio(phi, mo)
    _, best = _simplex_minimum(objective, mo.n_assets)
    assert objective(port.weights) <= best + 1e-8


def test_one_regime_mixture_is_mean_variance():
    mo, _ = _moments(regimes=1)
    np.testing.assert_allclose(
        mixture_portfolio(10, mo).weights,
        mean_variance_portfolio(10, mo).weights,
        atol=1e-6,
    )


def test_minimum_variance_has_the_closed_form_when_all_weights_are_positive():
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0, 0.05, (400, 8)) + rng.normal(0, 0.02, (400, 1))
    S = np.cov(returns, rowvar=False)
    inv = np.linalg.solve(S, np.ones(8))
    expected = inv / inv.sum()
    assert expected.min() > 0
    w = minimum_variance_portfolio(returns).weights
    # Near the optimum the variance is flat in the weights: compare variances.
    assert w @ S @ w == pytest.approx(expected @ S @ expected, rel=1e-6)
    np.testing.assert_allclose(w, expected, atol=1e-4)


def test_a_later_statement_is_used_when_the_first_is_not_solved(monkeypatch):
    mo, _ = _moments()
    original = cp.Problem.solve
    calls = []

    def first_fails(self, *args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise cp.error.SolverError("first statement fails")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", first_fails)
    port = mixture_portfolio(10, mo)
    assert port.attempt == 1
    assert calls[1].get("max_step_fraction") is None  # the second statement


def test_unsolved_problems_raise(monkeypatch):
    mo, _ = _moments()

    def always_fails(self, *args, **kwargs):
        raise cp.error.SolverError("fails")

    monkeypatch.setattr(cp.Problem, "solve", always_fails)
    with pytest.raises(OptimizationError, match="no statement"):
        mixture_portfolio(10, mo)
    assert len(optimize.ATTEMPTS["CLARABEL"]["mixture"]) == 4
