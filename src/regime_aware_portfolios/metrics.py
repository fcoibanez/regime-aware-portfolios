"""Performance statistics of backtested strategies.

Statistics are computed path by path, so that each has a distribution across
the simulated paths and strategies can be compared path by path. Returns are
monthly excess returns; annualized figures multiply means by 12 and standard
deviations by the square root of 12.

Trading costs are proportional to turnover: a trade of two-sided turnover u at
a cost c per unit reduces the return of the month the new portfolio is held by
c u. The opening purchase is not charged, so that costs measure ongoing trading
and are comparable across strategies that start from the same cash position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp

__all__ = [
    "STATISTICS",
    "certainty_equivalent",
    "max_drawdown",
    "net_returns",
    "path_statistics",
    "sharpe_ratio",
]

PERIODS_PER_YEAR = 12

#: Statistics of :func:`path_statistics`, in the order of the tables.
STATISTICS = (
    "Ann. Excess Return",
    "Ann. Std. Deviation",
    "Skewness",
    "Excess Kurtosis",
    "Sharpe Ratio",
    "Certainty Equivalent",
    "Max. Drawdown",
    "Value-at-Risk (95%)",
    "Expected Shortfall",
    "Portfolio Turnover",
    "Avg. Num. Holdings",
    "Effective Num. Holdings",
    "Sharpe Ratio (net)",
    "Certainty Equivalent (net)",
    "Cost Drag",
)


def sharpe_ratio(returns: np.ndarray) -> float:
    """Annualized Sharpe ratio: mean over standard deviation (divisor T - 1)."""
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(PERIODS_PER_YEAR)) if sd > 0 else np.nan


def certainty_equivalent(returns: np.ndarray, phi: float) -> float:
    """Annualized certainty equivalent under exponential utility.

    -(1/phi) log mean exp(-phi r), the return that a certain monthly payoff
    must have to give the same expected utility as the realized returns.
    """
    r = np.asarray(returns, dtype=float)
    monthly = -(logsumexp(-phi * r) - np.log(len(r))) / phi
    return float(monthly * PERIODS_PER_YEAR)


def max_drawdown(returns: np.ndarray) -> float:
    """Largest fall of cumulative wealth from a previous peak, the start included."""
    wealth = np.concatenate([[1.0], np.cumprod(1.0 + np.asarray(returns, dtype=float))])
    return float((wealth / np.maximum.accumulate(wealth) - 1.0).min())


def net_returns(returns: np.ndarray, turnover: np.ndarray, cost: float) -> np.ndarray:
    """Returns after costs; the opening purchase (the first trade) is not charged."""
    charge = cost * np.asarray(turnover, dtype=float)
    charge[0] = 0.0
    return np.asarray(returns, dtype=float) - charge


def path_statistics(
    returns: np.ndarray,
    turnover: np.ndarray,
    holdings: np.ndarray,
    effective_holdings: np.ndarray,
    phi: float,
    cost: float,
) -> dict[str, float]:
    """Every statistic of :data:`STATISTICS` for one strategy, path and phi."""
    r = np.asarray(returns, dtype=float)
    var95 = float(np.percentile(r, 5))
    net = net_returns(r, turnover, cost)
    return {
        "Ann. Excess Return": float(r.mean() * PERIODS_PER_YEAR),
        "Ann. Std. Deviation": float(r.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)),
        "Skewness": float(stats.skew(r, bias=False)),
        "Excess Kurtosis": float(stats.kurtosis(r, fisher=True, bias=False)),
        "Sharpe Ratio": sharpe_ratio(r),
        "Certainty Equivalent": certainty_equivalent(r, phi),
        "Max. Drawdown": max_drawdown(r),
        "Value-at-Risk (95%)": var95,
        "Expected Shortfall": float(r[r <= var95].mean()),
        "Portfolio Turnover": float(np.mean(turnover[1:]) * PERIODS_PER_YEAR),
        "Avg. Num. Holdings": float(np.mean(holdings)),
        "Effective Num. Holdings": float(np.mean(effective_holdings)),
        "Sharpe Ratio (net)": sharpe_ratio(net),
        "Certainty Equivalent (net)": certainty_equivalent(net, phi),
        "Cost Drag": float((r - net).mean() * PERIODS_PER_YEAR),
    }


def statistics_frame(result, cost: float, strategy: str, path: int) -> pd.DataFrame:
    """Statistics of one backtest result, one row per risk aversion."""
    rows = []
    for i, phi in enumerate(result.phis):
        row = path_statistics(
            result.returns[i],
            result.turnover[i],
            result.holdings[i],
            result.effective_holdings[i],
            phi,
            cost,
        )
        rows.append({"strategy": strategy, "phi": float(phi), "path": path, **row})
    return pd.DataFrame(rows)
