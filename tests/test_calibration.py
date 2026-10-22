import numpy as np
import pandas as pd
import pytest
import rwls

from regime_aware_portfolios.calibration import (
    StockEstimates,
    _floor_eigenvalues,
    calibrate,
    estimate_stocks,
    load_calibration,
    save_calibration,
)

NAMES = ["Mkt-RF", "SMB"]


@pytest.fixture(scope="module")
def panel():
    """Two factors, two regimes and five stocks with different histories."""
    rng = np.random.default_rng(0)
    T = 300
    dates = pd.date_range("1990-01-31", periods=T, freq=pd.offsets.MonthEnd())
    p = np.clip(0.5 + 0.45 * np.sin(np.arange(T) / 9), 0.01, 0.99)
    posteriors = pd.DataFrame({1: p, 2: 1 - p}, index=dates)
    factors = pd.DataFrame(rng.normal(0, 0.04, (T, 2)), index=dates, columns=NAMES)
    betas = rng.normal(1, 0.3, (5, 2, 2))  # stock, regime, factor
    returns = {}
    for i in range(5):
        r = sum(
            posteriors[s].to_numpy() * (factors.to_numpy() @ betas[i, s - 1])
            for s in (1, 2)
        )
        returns[100 + i] = r + rng.normal(0, 0.05, T)
    returns = pd.DataFrame(returns, index=dates)
    returns.iloc[: 40 * 1, 1] = np.nan  # later listing
    returns.iloc[-60:, 2] = np.nan  # earlier delisting
    returns.columns.name = "permno"
    return returns, factors, posteriors


def test_estimates_follow_rwls_in_factor_then_regime_order(panel):
    returns, factors, posteriors = panel
    est = estimate_stocks(returns, factors, posteriors, rwls.RWLSSettings())
    res = rwls.RWLS(returns, factors, posteriors).fit()
    assert list(est.loadings.columns) == [
        ("Mkt-RF", 1),
        ("Mkt-RF", 2),
        ("SMB", 1),
        ("SMB", 2),
    ]
    for f, s in est.loadings.columns:
        np.testing.assert_allclose(
            est.loadings[(f, s)].to_numpy(), res.params.loc[(s, f)].to_numpy()
        )
        j = list(est.loadings.columns).index((f, s))
        np.testing.assert_allclose(
            np.sqrt(est.sampling_cov[:, j, j]), res.bse.loc[(s, f)].to_numpy()
        )
    np.testing.assert_allclose(est.resid_var.to_numpy(), res.resid_var.to_numpy().T)
    assert not est.fallback.to_numpy().any()


def test_effective_degrees_of_freedom(panel):
    returns, factors, posteriors = panel
    est = estimate_stocks(returns, factors, posteriors, rwls.RWLSSettings())
    rows = returns[101].notna().to_numpy()
    w = posteriors[2].to_numpy()[rows]
    expected = w.sum() ** 2 / (w**2).sum() - 3  # constant and two loadings
    assert est.dof.loc[101, 2] == pytest.approx(expected)


def _estimates(params, sampling, resid_var, dof):
    columns = pd.MultiIndex.from_product([NAMES, [1, 2]], names=["factor", "regime"])
    regimes = pd.Index([1, 2], name="regime")
    index = pd.RangeIndex(len(params))
    return StockEstimates(
        loadings=pd.DataFrame(params, index=index, columns=columns),
        sampling_cov=sampling,
        resid_var=pd.DataFrame(resid_var, index=index, columns=regimes),
        dof=pd.DataFrame(dof, index=index, columns=regimes),
        fallback=pd.DataFrame(False, index=index, columns=regimes),
    )


def test_calibrate_removes_estimation_error_and_keeps_factor_blocks():
    rng = np.random.default_rng(1)
    n = 4000
    true = rng.multivariate_normal(np.ones(4), 0.04 * np.eye(4) + 0.02, n)
    noise = 0.01 * np.eye(4)
    params = true + rng.multivariate_normal(np.zeros(4), noise, n)
    # Log residual variances depend on the loadings exactly, with large dof.
    coef = np.array([[0.5, 0.0, 0.2, 0.0], [0.0, 0.3, 0.0, 0.1]])
    z = np.array([-6.0, -5.0]) + params @ coef.T
    dof = np.full((n, 2), 1e9)
    est = _estimates(params, np.tile(noise, (n, 1, 1)), np.exp(z), dof)
    cal = calibrate(est, 1e-4, 1e-3, 5)

    np.testing.assert_allclose(cal.loadings_mean.to_numpy(), params.mean(axis=0))
    cov = cal.loadings_cov.to_numpy()
    observed = np.cov(params, rowvar=False)
    # Within a factor's block: observed covariance less the sampling covariance.
    np.testing.assert_allclose(cov[:2, :2], observed[:2, :2] - noise[:2, :2])
    # Across factors: nothing.
    np.testing.assert_array_equal(cov[:2, 2:], 0.0)
    np.testing.assert_allclose(cal.resid_intercept.to_numpy(), [-6.0, -5.0], atol=1e-8)
    np.testing.assert_allclose(cal.resid_coef.to_numpy(), coef, atol=1e-8)
    assert np.linalg.eigvalsh(cal.resid_cov.to_numpy()).min() >= 1e-3 - 1e-12
    assert cal.record["n_stocks"] == cal.record["n_resid_stocks"] == n


def test_residual_model_leaves_out_stocks_with_few_degrees_of_freedom():
    rng = np.random.default_rng(2)
    n = 50
    params = rng.normal(1, 0.2, (n, 4))
    dof = np.full((n, 2), 100.0)
    dof[:7, 1] = 3.0
    est = _estimates(
        params, np.zeros((n, 4, 4)), np.exp(rng.normal(-5, 0.3, (n, 2))), dof
    )
    cal = calibrate(est, 1e-4, 1e-3, 5)
    assert cal.record["n_resid_stocks"] == n - 7
    assert cal.record["n_stocks"] == n


def test_floor_eigenvalues():
    m = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3 and -1
    out = _floor_eigenvalues(m, 0.5)
    np.testing.assert_allclose(np.linalg.eigvalsh(out), [0.5, 3.0])
    np.testing.assert_allclose(out, out.T)


def test_save_and_load(panel, tmp_path):
    returns, factors, posteriors = panel
    est = estimate_stocks(returns, factors, posteriors, rwls.RWLSSettings())
    cal = calibrate(est, 1e-4, 1e-3, 5)
    save_calibration(cal, tmp_path)
    back = load_calibration(tmp_path)
    pd.testing.assert_series_equal(back.loadings_mean, cal.loadings_mean)
    pd.testing.assert_frame_equal(back.loadings_cov, cal.loadings_cov)
    pd.testing.assert_series_equal(back.resid_intercept, cal.resid_intercept)
    pd.testing.assert_frame_equal(back.resid_coef, cal.resid_coef)
    pd.testing.assert_frame_equal(back.resid_cov, cal.resid_cov)
    assert back.record == cal.record
