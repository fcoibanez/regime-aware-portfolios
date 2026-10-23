import dataclasses

import numpy as np
import pytest

from regime_aware_portfolios import backtest, load_config, paths
from regime_aware_portfolios.backtest import (
    backtest_path,
    effective_holdings,
    load_result,
    save_result,
    turnover,
)
from regime_aware_portfolios.optimize import OptimizationError
from regime_aware_portfolios.simulation import DataGeneratingProcess, simulate_path
from regime_aware_portfolios.strategies import study_strategies


@pytest.fixture(scope="module")
def cfg():
    base = load_config(paths.CONFIG / "main.toml")
    return dataclasses.replace(
        base,
        regimes=dataclasses.replace(base.regimes, n_regimes=2),
        simulation=dataclasses.replace(
            base.simulation, n_in_sample=150, n_out_of_sample=4, n_stocks=12
        ),
    )


@pytest.fixture(scope="module")
def path(cfg):
    dgp = DataGeneratingProcess(
        factor_names=["Mkt-RF", "SMB"],
        initial=np.array([2 / 3, 1 / 3]),
        transmat=np.array([[0.9, 0.1], [0.2, 0.8]]),
        factor_means=np.array([[0.01, 0.002], [-0.02, 0.004]]),
        factor_vols=np.array([[0.04, 0.02], [0.08, 0.03]]),
        loadings_mean=np.array([1.0, 1.2, 0.3, 0.5]),
        loadings_factor=0.2 * np.eye(4),
        resid_intercept=np.array([-5.5, -5.0]),
        resid_coef=np.zeros((2, 4)),
        resid_factor=0.3 * np.eye(2),
    )
    sim = cfg.simulation
    return simulate_path(dgp, sim.n_periods, sim.n_stocks, seed=3, path=0)


def test_turnover_is_measured_against_drifted_holdings():
    w = np.array([[0.5, 0.5], [0.5, 0.5], [0.0, 1.0]])
    held = np.array([[0.1, -0.1], [0.0, 0.0], [0.0, 0.0]])
    out = turnover(w, held)
    assert out[0] == 1.0  # bought from cash
    # After the first month the holdings drifted to 0.55 / 0.45.
    np.testing.assert_allclose(out[1], 0.1)
    np.testing.assert_allclose(out[2], 1.0)


def test_effective_holdings():
    w = np.array([[0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])
    np.testing.assert_allclose(effective_holdings(w, 1e-4), [4.0, 1.0, 2.0])
    tiny = np.array([[0.5, 0.5 - 1e-6, 1e-6, 0.0]])
    assert effective_holdings(tiny, 1e-4)[0] == pytest.approx(2.0, abs=1e-4)


@pytest.mark.parametrize("name", ["equal_weight", "agnostic", "minimum_variance"])
def test_backtest_returns_are_those_of_the_following_month(cfg, path, name):
    strategy = study_strategies(cfg)[name]
    res = backtest_path(strategy, path, cfg, keep_weights=True)
    assert res.returns.shape == (5, 4) and res.weights.shape == (5, 4, 12)
    np.testing.assert_array_equal(res.dates, [149, 150, 151, 152])
    R = path.returns.to_numpy()
    expected = np.einsum("pdn,dn->pd", res.weights.astype(float), R[res.dates + 1])
    np.testing.assert_allclose(res.returns, expected, atol=1e-6)
    np.testing.assert_allclose(res.weights.sum(axis=2), 1.0, atol=1e-6)
    np.testing.assert_array_equal(res.turnover[:, 0], pytest.approx(1.0, abs=1e-6))
    if not strategy.depends_on_risk_aversion:
        assert (res.weights == res.weights[0]).all()


def test_risk_aversion_changes_the_portfolio(cfg, path):
    res = backtest_path(study_strategies(cfg)["agnostic"], path, cfg, keep_weights=True)
    assert not np.allclose(res.weights[0], res.weights[-1])
    assert res.holdings.shape == (5, 4) and (res.holdings >= 1).all()


def test_save_and_load(cfg, path, tmp_path):
    res = backtest_path(study_strategies(cfg)["agnostic"], path, cfg)
    assert res.weights is None
    file = tmp_path / "agnostic" / "path_0000.npz"
    save_result(res, file)
    back = load_result(file)
    for field in dataclasses.fields(res):
        a, b = getattr(res, field.name), getattr(back, field.name)
        if a is None:
            assert b is None
        else:
            np.testing.assert_array_equal(a, b)
    assert not list(tmp_path.rglob("*.partial.npz"))


def test_failures_name_the_strategy_date_and_risk_aversion(cfg, path, monkeypatch):
    def fail(*args, **kwargs):
        raise OptimizationError("no statement of the problem was solved")

    monkeypatch.setattr(backtest, "portfolio", fail)
    with pytest.raises(OptimizationError, match="agnostic, date 149, risk aversion 1"):
        backtest_path(study_strategies(cfg)["agnostic"], path, cfg)


def test_regime_strategies_record_their_probabilities(cfg, path, tmp_path):
    s = study_strategies(cfg)
    res = backtest_path(s["framework"], path, cfg)
    assert res.filtered.shape == (4, 2) and res.forecast.shape == (4, 2)
    assert res.regime_means.shape == (4, 2, 2)
    np.testing.assert_allclose(res.filtered.sum(axis=1), 1.0)
    np.testing.assert_allclose(res.forecast.sum(axis=1), 1.0)
    save_result(res, tmp_path / "f.npz")
    np.testing.assert_array_equal(
        load_result(tmp_path / "f.npz").forecast, res.forecast
    )
    agnostic = backtest_path(s["agnostic"], path, cfg)
    assert agnostic.filtered is None and agnostic.regime_means is None
