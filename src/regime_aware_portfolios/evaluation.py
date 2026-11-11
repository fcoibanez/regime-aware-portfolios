"""Evaluation of the regime probabilities against the simulated regimes.

A regime model numbers its regimes arbitrarily, so before estimated
probabilities can be scored against the true regimes of a path, each estimated
regime is matched to a true one. The match is made at every decision date, by
the regimes' factor means: each factor's means are measured in units of its
typical within-regime volatility, and the matching minimizes the total distance.
Volatility is the scale on which estimated means are noisy; dividing instead by
the dispersion of the true means across regimes lets a factor whose true means
nearly coincide dominate the distance with estimation noise. This is a
reporting convention; the portfolios do not depend on the labels.

Probabilities are scored by their hit rate (the share of periods whose most
probable regime is the true one) and Brier score (the mean, over periods, of the
squared distance between the probabilities and the indicator of the true
regime). Both are compared with a constant forecast: always the most frequent
regime for the hit rate, and each regime's frequency over the scored periods for
the Brier skill score, 1 - Brier / Brier of the constant forecast. Those
frequencies are computed from the very periods being scored, which no
forecaster could have known.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

__all__ = [
    "align_probabilities",
    "forecast_scores",
    "match_regimes",
    "regime_conditional",
]


def match_regimes(
    estimated_means: np.ndarray, true_means: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """For each estimated regime, the true regime it matches (0-based).

    ``scale`` holds one positive value per factor, the unit in which its means
    are compared, normally its within-regime volatility averaged over regimes.
    """
    scale = np.asarray(scale, dtype=float)
    cost = cdist(estimated_means / scale, true_means / scale)
    rows, cols = linear_sum_assignment(cost)
    out = np.empty(len(rows), dtype=int)
    out[rows] = cols
    return out


def align_probabilities(
    probs: np.ndarray,
    estimated_means: np.ndarray,
    true_means: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Probabilities relabeled to the true regimes, date by date.

    ``probs`` has shape (dates, regimes) and ``estimated_means`` (dates,
    regimes, factors); ``scale`` is as in :func:`match_regimes`.
    """
    out = np.empty_like(probs)
    for d in range(len(probs)):
        out[d, match_regimes(estimated_means[d], true_means, scale)] = probs[d]
    return out


def forecast_scores(probs: np.ndarray, states: np.ndarray) -> dict[str, float]:
    """Hit rate and Brier score of ``probs`` (periods, regimes) for ``states``.

    ``states`` holds the true regime of each period, 0-based.
    """
    states = np.asarray(states, dtype=int)
    M = probs.shape[1]
    onehot = np.eye(M)[states]
    freq = onehot.mean(axis=0)
    brier = float(((probs - onehot) ** 2).sum(axis=1).mean())
    constant = float(((freq[None, :] - onehot) ** 2).sum(axis=1).mean())
    return {
        "Hit rate": float((probs.argmax(axis=1) == states).mean()),
        "Constant forecast": float(freq.max()),
        "Brier score": brier,
        "Brier skill": 1.0 - brier / constant,
        "Periods": len(states),
    }


def regime_conditional(
    a: np.ndarray, b: np.ndarray, states: np.ndarray, n_regimes: int
) -> dict[int, dict[str, float]]:
    """Strategy ``a`` against ``b`` in the months governed by each true regime.

    ``a`` and ``b`` are monthly returns pooled over paths and ``states`` the
    true regime of each of those months (0-based). Returns, per regime: the
    share of months, the difference in mean return (annualized), the ratio of
    volatilities, and the improvement in the 95% value-at-risk (a's fifth
    percentile minus b's, so positive when a loses less).
    """
    out = {}
    for s in range(n_regimes):
        m = states == s
        ra, rb = a[m], b[m]
        out[s + 1] = {
            "Share of months": float(m.mean()),
            "Return advantage (ann.)": float((ra.mean() - rb.mean()) * 12),
            "Volatility ratio": float(ra.std(ddof=1) / rb.std(ddof=1)),
            "VaR (95%) improvement": float(np.percentile(ra, 5) - np.percentile(rb, 5)),
        }
    return out
