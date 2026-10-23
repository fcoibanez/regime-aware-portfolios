import numpy as np
import pytest

from regime_aware_portfolios.metrics import (
    STATISTICS,
    certainty_equivalent,
    max_drawdown,
    net_returns,
    path_statistics,
    sharpe_ratio,
)


def test_sharpe_ratio():
    r = np.array([0.01, 0.03, -0.01, 0.02])
    assert sharpe_ratio(r) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(12))
    assert np.isnan(sharpe_ratio(np.full(5, 0.01)))


def test_certainty_equivalent():
    # A certain return is its own certainty equivalent.
    assert certainty_equivalent(np.full(12, 0.01), 10) == pytest.approx(0.12)
    # For normal returns it is close to the mean less phi/2 times the variance.
    r = np.random.default_rng(0).normal(0.01, 0.04, 200_000)
    expected = 12 * (0.01 - 5 / 2 * 0.04**2)
    assert certainty_equivalent(r, 5) == pytest.approx(expected, abs=0.002)


def test_max_drawdown_counts_a_loss_in_the_first_month():
    assert max_drawdown(np.array([-0.1, 0.2])) == pytest.approx(-0.1)
    assert max_drawdown(np.array([0.1, -0.5, 0.1])) == pytest.approx(-0.5)
    assert max_drawdown(np.array([0.1, 0.1])) == 0.0


def test_net_returns_do_not_charge_the_opening_purchase():
    r = np.array([0.01, 0.02, 0.03])
    u = np.array([1.0, 0.5, 0.2])
    np.testing.assert_allclose(net_returns(r, u, 0.01), [0.01, 0.015, 0.028])


def test_path_statistics():
    rng = np.random.default_rng(1)
    r = rng.normal(0.01, 0.05, 60)
    u = np.r_[1.0, rng.uniform(0, 1, 59)]
    out = path_statistics(r, u, np.full(60, 20), np.full(60, 12.5), phi=10, cost=0.001)
    assert tuple(out) == STATISTICS
    assert out["Portfolio Turnover"] == pytest.approx(u[1:].mean() * 12)
    assert out["Cost Drag"] == pytest.approx(0.001 * u[1:].sum() / 60 * 12)
    assert out["Sharpe Ratio (net)"] < out["Sharpe Ratio"]
    assert out["Value-at-Risk (95%)"] >= out["Expected Shortfall"]
    assert out["Avg. Num. Holdings"] == 20 and out["Effective Num. Holdings"] == 12.5
