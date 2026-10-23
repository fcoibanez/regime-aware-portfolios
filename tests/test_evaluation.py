import numpy as np
import pytest

from regime_aware_portfolios.evaluation import (
    align_probabilities,
    forecast_scores,
    match_regimes,
    regime_conditional,
)

TRUE = np.array([[0.01, 0.002], [-0.02, 0.004], [0.03, -0.01]])


def test_match_regimes_recovers_a_permutation():
    estimated = TRUE[[2, 0, 1]] + 0.0005
    np.testing.assert_array_equal(match_regimes(estimated, TRUE), [2, 0, 1])


def test_align_probabilities():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    means = np.stack([TRUE[[2, 0, 1]], TRUE])
    aligned = align_probabilities(probs, means, TRUE)
    np.testing.assert_allclose(aligned, [[0.2, 0.1, 0.7], [0.1, 0.1, 0.8]])


def test_forecast_scores():
    states = np.array([0, 0, 1, 2, 0, 1])
    perfect = np.eye(3)[states]
    out = forecast_scores(perfect, states)
    assert out["Hit rate"] == 1 and out["Brier score"] == 0 and out["Brier skill"] == 1
    freq = np.bincount(states) / len(states)
    constant = np.tile(freq, (len(states), 1))
    out = forecast_scores(constant, states)
    assert out["Brier skill"] == pytest.approx(0.0)
    assert out["Constant forecast"] == pytest.approx(0.5)


def test_regime_conditional():
    states = np.array([0, 0, 1, 1, 1, 0])
    a = np.array([0.02, 0.01, -0.01, 0.0, 0.01, 0.03])
    b = a - 0.01
    out = regime_conditional(a, b, states, 2)
    assert out[1]["Share of months"] == 0.5
    assert out[1]["Return advantage (ann.)"] == pytest.approx(0.12)
    assert out[2]["Volatility ratio"] == pytest.approx(1.0)
    assert out[2]["VaR (95%) improvement"] == pytest.approx(0.01)
