import numpy as np
import pandas as pd
import pytest

from regime_aware_portfolios.inference import (
    _sharpe_and_gradient,
    detection_by_horizon,
    hac_standard_error,
    ledoit_wolf,
    paired_comparison,
    rejection_summary,
)


def test_paired_comparison():
    rng = np.random.default_rng(0)
    control = pd.Series(rng.normal(0, 1, 200))
    treatment = control + 0.5 + rng.normal(0, 0.1, 200)
    out = paired_comparison(treatment, control)
    assert out["Delta"] == pytest.approx(0.5, abs=0.03)
    assert out["t-stat"] > 30 and out["Frac. > 0"] == 1.0
    assert out["N paths"] == 200


def test_the_gradient_is_that_of_the_sharpe_difference():
    rng = np.random.default_rng(1)
    r = rng.normal([0.01, 0.005], [0.04, 0.05], (60, 2))
    _, grad, mu, gamma = _sharpe_and_gradient(r)

    def diff(m):  # (mean_0, mean_1, second moment_0, second moment_1)
        return m[0] / np.sqrt(m[2] - m[0] ** 2) - m[1] / np.sqrt(m[3] - m[1] ** 2)

    point = np.array([mu[0], mu[1], gamma[0], gamma[1]])
    h = 1e-7
    numeric = [(diff(point + h * e) - diff(point - h * e)) / (2 * h) for e in np.eye(4)]
    np.testing.assert_allclose(grad, numeric, rtol=1e-5)


def test_hac_standard_error_is_close_to_the_iid_one_without_autocorrelation():
    rng = np.random.default_rng(2)
    r = rng.normal([0.01, 0.0], [0.04, 0.04], (20_000, 2))
    # Independent normal series: var of each Sharpe ratio is (1 + SR^2 / 2) / T.
    sr = r.mean(axis=0) / r.std(axis=0)
    iid = np.sqrt(((1 + sr**2 / 2) / len(r)).sum())
    assert hac_standard_error(r) == pytest.approx(iid, rel=0.05)


def test_ledoit_wolf_rejects_a_real_difference_and_is_reproducible():
    rng = np.random.default_rng(3)
    common = rng.normal(0, 0.04, 240)
    a = common + 0.01 + rng.normal(0, 0.01, 240)
    b = common + rng.normal(0, 0.01, 240)
    out = ledoit_wolf(a, b, draws=199, rng=np.random.default_rng(7))
    assert out["SR difference"] > 0 and out["p-value"] < 0.01
    again = ledoit_wolf(a, b, draws=199, rng=np.random.default_rng(7))
    assert again == out
    hac = ledoit_wolf(a, b, method="hac")
    assert hac["p-value"] < 0.01
    with pytest.raises(ValueError, match="random generator"):
        ledoit_wolf(a, b)


def test_ledoit_wolf_rarely_rejects_without_a_difference():
    rng = np.random.default_rng(4)
    p = []
    for k in range(40):
        common = rng.normal(0.005, 0.04, 60)
        a = common + rng.normal(0, 0.01, 60)
        b = common + rng.normal(0, 0.01, 60)
        p.append(ledoit_wolf(a, b, draws=99, rng=np.random.default_rng(k))["p-value"])
    assert np.mean(np.array(p) < 0.05) <= 0.15


def test_rejection_summary_and_detection():
    tests = pd.DataFrame(
        {"SR difference": [0.1, 0.2, -0.1, 0.05], "p-value": [0.01, 0.2, 0.02, 0.5]}
    )
    s = rejection_summary(tests, 0.05)
    assert s["Reject (favorable)"] == 0.25 and s["Reject (adverse)"] == 0.25
    assert s["Inconclusive"] == 0.5
    rng = np.random.default_rng(5)
    a = [rng.normal(0.01, 0.03, 12) for _ in range(8)]
    b = [x - 0.01 + rng.normal(0, 0.005, 12) for x in a]
    det = detection_by_horizon(a, b, [1, 2, 4], 0.05, 1, 49, seed=1)
    assert list(det.index) == [12, 24, 48]
    assert (det["Histories"] == [8, 4, 2]).all()
