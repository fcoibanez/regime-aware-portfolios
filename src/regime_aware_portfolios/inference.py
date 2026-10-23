"""Statistical inference on backtested performance.

Two kinds of uncertainty are kept apart. Across the simulated paths, every
strategy is evaluated on the same data, so strategies are compared path by path
(:func:`paired_comparison`). On a single path, an investor sees one history of
n_out_of_sample months, and the question is whether that history alone would
show a difference in Sharpe ratios: the test of Ledoit and Wolf (2008), applied
path by path and summarized by how often it rejects (:func:`ledoit_wolf`,
:func:`rejection_summary`, :func:`detection_by_horizon`).

The Ledoit and Wolf test is implemented from their paper. The Sharpe ratio of a
return series is its mean over its standard deviation (divisor T), written as a
function of the first two uncentered moments; its sampling variance follows from
the delta method. The HAC standard error uses the Parzen kernel with the
automatic bandwidth of Andrews (1991), each moment's autocorrelation taken from
an AR(1) fit. The bootstrap version resamples the pair of return series in
circular blocks, studentizes each resampled difference with a standard error
computed from the blocks, and compares it with the studentized difference of the
data; its p-value is (1 + #{d* >= d}) / (1 + draws). Both covariance estimates
carry the factor T / (T - 4) of the authors' code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "detection_by_horizon",
    "hac_standard_error",
    "ledoit_wolf",
    "paired_comparison",
    "rejection_summary",
]


# ------------------------------------------------------------ across paths


def paired_comparison(treatment: pd.Series, control: pd.Series) -> dict:
    """Paired tests of ``treatment`` against ``control``, both indexed by path.

    Reports the mean and median difference, the paired t-test, and the share of
    paths on which the treatment is ahead, with sign and Wilcoxon tests. Mean and
    median can disagree when a method helps on most paths and fails badly on a
    few, and both are reported.
    """
    treatment, control = treatment.align(control, join="inner")
    diff = (treatment - control).dropna()
    n = len(diff)
    t_stat, p_t = stats.ttest_rel(treatment.loc[diff.index], control.loc[diff.index])
    wins = int((diff > 0).sum())
    return {
        "Mean (treatment)": float(treatment.loc[diff.index].mean()),
        "Mean (control)": float(control.loc[diff.index].mean()),
        "Delta": float(diff.mean()),
        "Std. Err.": float(diff.std(ddof=1) / np.sqrt(n)),
        "t-stat": float(t_stat),
        "p (paired t)": float(p_t),
        "Median delta": float(diff.median()),
        "Frac. > 0": wins / n,
        "p (sign)": float(stats.binomtest(wins, n, 0.5).pvalue),
        "p (Wilcoxon)": float(stats.wilcoxon(diff)[1]) if (diff != 0).any() else np.nan,
        "N paths": n,
    }


# ------------------------------------------------------------ Ledoit and Wolf


def _sharpe_and_gradient(r: np.ndarray):
    """Sharpe ratios of the two columns, and the gradient of their difference.

    The difference is that of column 0 minus column 1, as a function of
    (mean_0, mean_1, second moment_0, second moment_1).
    """
    mu = r.mean(axis=0)
    gamma = (r**2).mean(axis=0)
    var = gamma - mu**2
    sr = mu / np.sqrt(var)
    grad = np.array(
        [
            gamma[0] / var[0] ** 1.5,
            -gamma[1] / var[1] ** 1.5,
            -0.5 * mu[0] / var[0] ** 1.5,
            0.5 * mu[1] / var[1] ** 1.5,
        ]
    )
    return sr, grad, mu, gamma


def _moment_deviations(r: np.ndarray, mu, gamma) -> np.ndarray:
    return np.column_stack([r - mu, r**2 - gamma])


def _parzen(x: float) -> float:
    x = abs(x)
    if x <= 0.5:
        return 1 - 6 * x**2 + 6 * x**3
    if x <= 1:
        return 2 * (1 - x) ** 3
    return 0.0


def _andrews_alpha(v: np.ndarray) -> float:
    """Andrews' (1991) plug-in for the Parzen bandwidth, from AR(1) fits."""
    num = den = 0.0
    for i in range(v.shape[1]):
        y, x = v[1:, i], v[:-1, i]
        design = np.column_stack([np.ones(len(x)), x])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        rho = coef[1]
        s2 = float(((y - design @ coef) ** 2).mean())
        num += 4 * rho**2 * s2**2 / (1 - rho) ** 8
        den += s2**2 / (1 - rho) ** 4
    return num / den


def _hac_covariance(v: np.ndarray) -> np.ndarray:
    T = len(v)
    bandwidth = min(2.6614 * (_andrews_alpha(v) * T) ** 0.2, T - 1)
    psi = v.T @ v / T
    j = 1
    while j < bandwidth:
        gamma = v[j:].T @ v[:-j] / T
        psi = psi + _parzen(j / bandwidth) * (gamma + gamma.T)
        j += 1
    return T / (T - 4) * psi


def hac_standard_error(r: np.ndarray) -> float:
    """HAC standard error of the difference in Sharpe ratios of the two columns."""
    sr, grad, mu, gamma = _sharpe_and_gradient(r)
    v = _moment_deviations(r, mu, gamma)
    return float(np.sqrt(grad @ _hac_covariance(v) @ grad / len(r)))


def ledoit_wolf(
    a: np.ndarray,
    b: np.ndarray,
    method: str = "bootstrap",
    block: int = 1,
    draws: int = 499,
    rng: np.random.Generator | None = None,
) -> dict:
    """Test of equal Sharpe ratios of two return series on the same periods.

    Parameters
    ----------
    a, b : arrays
        Returns of the two strategies; the difference reported is a minus b.
    method : {"bootstrap", "hac"}
        Studentized circular block bootstrap, or the asymptotic test with the
        HAC standard error.
    block, draws : int
        Block length and number of resamples of the bootstrap.
    rng : numpy.random.Generator
        Source of the resampling; required for the bootstrap.

    Returns
    -------
    dict
        Sharpe ratios (per period), their difference, the HAC standard error and
        the p-value.
    """
    r = np.column_stack([np.asarray(a, float), np.asarray(b, float)])
    r = r[np.isfinite(r).all(axis=1)]
    T = len(r)
    sr, grad, mu, gamma = _sharpe_and_gradient(r)
    diff = sr[0] - sr[1]
    se = hac_standard_error(r)
    if method == "hac":
        p = 2 * stats.norm.sf(abs(diff) / se)
    elif method == "bootstrap":
        if rng is None:
            raise ValueError("the bootstrap needs a random generator")
        d = abs(diff) / se
        n_blocks = T // block
        exceed = 0
        for _ in range(draws):
            starts = rng.integers(0, T, size=int(np.ceil(T / block)))
            index = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
            rs = r[index]
            sr_s, grad_s, mu_s, gamma_s = _sharpe_and_gradient(rs)
            v = _moment_deviations(rs, mu_s, gamma_s)
            blocks = v[: n_blocks * block].reshape(n_blocks, block, 4).mean(axis=1)
            zeta = np.sqrt(block) * blocks
            psi = T / (T - 4) * (zeta.T @ zeta) / n_blocks
            se_s = np.sqrt(grad_s @ psi @ grad_s / T)
            exceed += abs(sr_s[0] - sr_s[1] - diff) / se_s >= d
        p = (1 + exceed) / (1 + draws)
    else:
        raise ValueError("method must be 'bootstrap' or 'hac'")
    return {
        "SR (a)": float(sr[0]),
        "SR (b)": float(sr[1]),
        "SR difference": float(diff),
        "Std. Err.": se,
        "p-value": float(p),
    }


def rejection_summary(tests: pd.DataFrame, alpha: float) -> pd.Series:
    """How often a single path's history rejects equal Sharpe ratios.

    ``tests`` has one row per path with the columns of :func:`ledoit_wolf`.
    Rejections are split by the sign of the difference: in favor of the first
    strategy, or against it.
    """
    reject = tests["p-value"] < alpha
    better = tests["SR difference"] > 0
    scale = np.sqrt(12)
    return pd.Series(
        {
            "Mean SR difference (ann.)": tests["SR difference"].mean() * scale,
            "Median SR difference (ann.)": tests["SR difference"].median() * scale,
            "Frac. SR difference > 0": better.mean(),
            "Reject (favorable)": (reject & better).mean(),
            "Reject (adverse)": (reject & ~better).mean(),
            "Inconclusive": (~reject).mean(),
            "Median p-value": tests["p-value"].median(),
            "N paths": len(tests),
        }
    )


def detection_by_horizon(
    a: list[np.ndarray],
    b: list[np.ndarray],
    multiples,
    alpha: float,
    block: int,
    draws: int,
    seed: int,
    max_histories: int = 150,
) -> pd.DataFrame:
    """Rejection rate of the bootstrap test as histories grow longer.

    Histories of k paths are formed by joining k consecutive paths' returns, for
    each k in ``multiples``. The joined paths are independent draws from the
    same process, so the history has the right distribution of returns, though
    not the continuity of one regime sequence across the joins.
    """
    rows = {}
    for k in multiples:
        n = min(max_histories, len(a) // k)
        if n < 2:
            continue
        rejected = []
        for j in range(n):
            ha = np.concatenate(a[j * k : (j + 1) * k])
            hb = np.concatenate(b[j * k : (j + 1) * k])
            rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(k, j)))
            res = ledoit_wolf(ha, hb, block=block, draws=draws, rng=rng)
            rejected.append(res["p-value"] < alpha)
        months = k * len(a[0])
        rows[months] = {
            "Years": months / 12,
            "Detection rate": float(np.mean(rejected)),
            "Histories": n,
        }
    out = pd.DataFrame(rows).T
    out.index.name = "Months"
    return out
