import dataclasses

import numpy as np
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM

from regime_aware_portfolios import load_config, paths
from regime_aware_portfolios.regimes import (
    RegimeParameters,
    filtered_probabilities,
    fit_regimes,
    load_regimes,
    order_regimes,
    reference_distance,
    save_regimes,
)

NAMES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
# Two regimes: a calm one and a volatile one with a lower market mean.
MEANS = np.array(
    [
        [0.010, 0.002, 0.003, 0.002, 0.002, 0.008],
        [-0.020, 0.000, 0.010, 0.004, 0.006, -0.010],
    ]
)
VOLS = np.array(
    [
        [0.035, 0.020, 0.020, 0.015, 0.015, 0.025],
        [0.070, 0.040, 0.040, 0.030, 0.030, 0.060],
    ]
)


@pytest.fixture(scope="module")
def factors():
    gen = GaussianHMM(n_components=2, covariance_type="diag", random_state=1)
    gen.startprob_ = np.array([0.5, 0.5])
    gen.transmat_ = np.array([[0.95, 0.05], [0.15, 0.85]])
    gen.means_, gen.covars_ = MEANS, VOLS**2
    X, Z = gen.sample(480)
    dates = pd.date_range("1980-01-31", periods=480, freq=pd.offsets.MonthEnd())
    frame = pd.DataFrame(X, index=pd.DatetimeIndex(dates, name="date"), columns=NAMES)
    frame.attrs["state_means"] = np.array([X[Z == k].mean(axis=0) for k in (0, 1)])
    return frame


def _config(**regimes):
    cfg = load_config(paths.CONFIG / "main.toml")
    base = {"n_regimes": 2, "n_starts": 8, "reference": {}, "selection": "best"}
    return dataclasses.replace(
        cfg, regimes=dataclasses.replace(cfg.regimes, **{**base, **regimes})
    )


def test_best_fit_is_numbered_by_market_volatility(factors):
    fit, starts = fit_regimes(factors, _config(), n_jobs=1)
    assert starts["selection"] == "best" and starts["starts"] == 8
    vol = fit.variances["Mkt-RF"].to_numpy()
    assert vol[0] < vol[1]
    # Regime 1 is the calm one: its means are those of the calm months.
    expected = factors.attrs["state_means"]
    np.testing.assert_allclose(fit.means.to_numpy(), expected, atol=0.003)


def test_order_regimes(factors):
    fit, _ = fit_regimes(factors, _config(), n_jobs=1)
    flipped = order_regimes(fit.relabel([2, 1]), "Mkt-RF")
    np.testing.assert_allclose(flipped.means.to_numpy(), fit.means.to_numpy())
    by_smb = order_regimes(fit, "SMB")
    assert by_smb.variances["SMB"].is_monotonic_increasing


def test_reference_distance_matches_regimes_in_any_order(factors):
    fit, _ = fit_regimes(factors, _config(), n_jobs=1)
    own = fit.means.copy()
    assert reference_distance(fit, own) == pytest.approx(0.0, abs=1e-12)
    assert reference_distance(fit, own.iloc[::-1]) == pytest.approx(0.0, abs=1e-12)
    assert reference_distance(fit, own + 0.01) > 0


def test_nearest_selects_the_optimum_closest_to_the_reference(factors):
    reference = {name: list(100 * MEANS[:, j]) for j, name in enumerate(NAMES)}
    cfg = _config(selection="nearest", reference=reference)
    fit, starts = fit_regimes(factors, cfg, n_jobs=1)
    assert starts["selection"] == "nearest"
    assert 1 <= starts["reached"] <= starts["healthy"] <= starts["starts"] == 8
    assert starts["seed"] == fit.settings.seed
    np.testing.assert_allclose(
        fit.means.to_numpy(), factors.attrs["state_means"], atol=0.003
    )
    # The best fit of that optimum.
    best, _ = fit_regimes(factors, _config(), n_jobs=1)
    assert fit.loglike == pytest.approx(best.loglike, abs=1e-6)


def test_save_and_load(factors, tmp_path):
    fit, starts = fit_regimes(factors, _config(), n_jobs=1)
    save_regimes(fit, starts, tmp_path)
    stored = load_regimes(tmp_path)
    assert stored.factor_names == NAMES
    pd.testing.assert_frame_equal(stored.means, fit.means)
    pd.testing.assert_frame_equal(stored.transmat, fit.transmat)
    pd.testing.assert_series_equal(stored.startprob, fit.startprob)
    np.testing.assert_array_equal(stored.covariances, fit.covariances)
    pd.testing.assert_frame_equal(stored.posteriors, fit.posteriors, check_freq=False)
    assert stored.loglike == fit.loglike
    assert stored.record["starts"] == starts
    assert stored.record["first_period"] == "1980-01"


def test_filtered_probabilities_use_only_past_data():
    labels = pd.RangeIndex(1, 3, name="regime")
    means = pd.DataFrame([[0.01, 0.0], [-0.02, 0.01]], index=labels, columns=["a", "b"])
    covs = np.stack([np.diag([0.02, 0.01]) ** 2, np.diag([0.06, 0.03]) ** 2])
    trans = pd.DataFrame([[0.9, 0.1], [0.3, 0.7]], index=labels, columns=labels)
    start = pd.Series([0.6, 0.4], index=labels)
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(0, 0.04, (40, 2)), columns=["a", "b"])
    params = RegimeParameters(
        ["a", "b"], start, trans, means, covs, pd.DataFrame(), 0.0, {}
    )
    filtered, forecast = filtered_probabilities(params, x)

    hmm = GaussianHMM(n_components=2, covariance_type="full")
    hmm.startprob_, hmm.transmat_ = start.to_numpy(), trans.to_numpy()
    hmm.means_, hmm.covars_ = means.to_numpy(), covs
    for t in (0, 9, 39):
        expected = hmm.predict_proba(x.iloc[: t + 1].to_numpy())[-1]
        np.testing.assert_allclose(filtered.iloc[t], expected, atol=1e-10)
    np.testing.assert_allclose(forecast.iloc[0], start)
    np.testing.assert_allclose(forecast.iloc[10], filtered.iloc[9] @ trans.to_numpy())
