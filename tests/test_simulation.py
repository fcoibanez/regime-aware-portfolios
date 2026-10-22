import numpy as np
import pandas as pd
import pytest

from regime_aware_portfolios.calibration import Calibration
from regime_aware_portfolios.regimes import RegimeParameters
from regime_aware_portfolios.simulation import (
    DataGeneratingProcess,
    simulate_path,
    stationary_distribution,
)

NAMES = ["Mkt-RF", "SMB"]
TRANS = np.array([[0.9, 0.1], [0.3, 0.7]])


@pytest.fixture(scope="module")
def dgp():
    labels = pd.RangeIndex(1, 3, name="regime")
    regimes = RegimeParameters(
        factor_names=NAMES,
        startprob=pd.Series([0.5, 0.5], index=labels),
        transmat=pd.DataFrame(TRANS, index=labels, columns=labels),
        means=pd.DataFrame(
            [[0.01, 0.002], [-0.02, 0.004]], index=labels, columns=NAMES
        ),
        covariances=np.array(
            [np.diag([0.04**2, 0.02**2]), np.diag([0.08**2, 0.03**2])]
        ),
        posteriors=pd.DataFrame(columns=labels),
        loglike=0.0,
        record={},
    )
    columns = pd.MultiIndex.from_product([NAMES, [1, 2]], names=["factor", "regime"])
    cov = np.diag([0.04, 0.09, 0.01, 0.04]) + 0.005
    calibration = Calibration(
        loadings_mean=pd.Series([1.0, 1.2, 0.3, 0.5], index=columns),
        loadings_cov=pd.DataFrame(cov, index=columns, columns=columns),
        resid_intercept=pd.Series([-5.5, -5.0], index=labels),
        resid_coef=pd.DataFrame(
            [[0.2, 0.0, 0.1, 0.0], [0.0, 0.3, 0.0, 0.2]], index=labels, columns=columns
        ),
        resid_cov=pd.DataFrame([[0.3, 0.1], [0.1, 0.4]], index=labels, columns=labels),
        record={},
    )
    return DataGeneratingProcess.build(regimes, calibration)


def test_stationary_distribution():
    pi = stationary_distribution(TRANS)
    np.testing.assert_allclose(pi, [0.75, 0.25])
    np.testing.assert_allclose(pi @ TRANS, pi)


def test_paths_are_reproducible_and_distinct(dgp):
    a = simulate_path(dgp, 120, 10, seed=7, path=3)
    b = simulate_path(dgp, 120, 10, seed=7, path=3)
    pd.testing.assert_frame_equal(a.returns, b.returns)
    c = simulate_path(dgp, 120, 10, seed=7, path=4)
    d = simulate_path(dgp, 120, 10, seed=8, path=3)
    assert not np.allclose(a.returns, c.returns)
    assert not np.allclose(a.returns, d.returns)


def test_shorter_and_smaller_paths_are_nested(dgp):
    big = simulate_path(dgp, 200, 30, seed=7, path=0)
    small = simulate_path(dgp, 150, 12, seed=7, path=0)
    np.testing.assert_array_equal(small.states, big.states[:150])
    np.testing.assert_array_equal(small.factors, big.factors.iloc[:150])
    np.testing.assert_array_equal(small.returns, big.returns.iloc[:150, :12])
    np.testing.assert_array_equal(small.loadings, big.loadings[:12])


def test_returns_are_loadings_times_factors_plus_residuals(dgp):
    path = simulate_path(dgp, 20000, 3, seed=1, path=0)
    s = path.states - 1
    systematic = np.stack(
        [
            np.einsum("tk,tk->t", path.factors.to_numpy(), path.loadings[i].T[s])
            for i in range(3)
        ],
        axis=1,
    )
    resid = path.returns.to_numpy() - systematic
    for m in range(2):
        np.testing.assert_allclose(
            resid[s == m].std(axis=0), path.resid_vol[:, m], rtol=0.05
        )
        assert np.abs(resid[s == m].mean(axis=0)).max() < 0.01


def test_regimes_and_factors_follow_the_process(dgp):
    path = simulate_path(dgp, 100_000, 1, seed=2, path=0)
    s = path.states - 1
    assert set(np.unique(path.states)) == {1, 2}
    counts = np.zeros((2, 2))
    np.add.at(counts, (s[:-1], s[1:]), 1)
    np.testing.assert_allclose(
        counts / counts.sum(axis=1, keepdims=True), TRANS, atol=0.01
    )
    np.testing.assert_allclose(np.bincount(s) / len(s), [0.75, 0.25], atol=0.01)
    X = path.factors.to_numpy()
    for m in range(2):
        np.testing.assert_allclose(
            X[s == m].mean(axis=0), dgp.factor_means[m], atol=0.002
        )
        np.testing.assert_allclose(X[s == m].std(axis=0), dgp.factor_vols[m], rtol=0.02)


def test_stock_parameters_follow_the_calibration(dgp):
    path = simulate_path(dgp, 2, 20000, seed=3, path=0)
    b = path.loadings.reshape(20000, -1)  # factor, then regime
    np.testing.assert_allclose(b.mean(axis=0), dgp.loadings_mean, atol=0.01)
    true_cov = dgp.loadings_factor @ dgp.loadings_factor.T
    np.testing.assert_allclose(np.cov(b, rowvar=False), true_cov, atol=0.005)
    log_var = 2 * np.log(path.resid_vol)
    fitted = dgp.resid_intercept + b @ dgp.resid_coef.T
    errors = log_var - fitted
    np.testing.assert_allclose(errors.mean(axis=0), 0.0, atol=0.02)
    resid_cov = dgp.resid_factor @ dgp.resid_factor.T
    np.testing.assert_allclose(np.cov(errors, rowvar=False), resid_cov, atol=0.02)
