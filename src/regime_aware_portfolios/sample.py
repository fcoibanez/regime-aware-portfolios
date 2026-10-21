"""The stocks whose returns calibrate the simulation.

Each month, only the largest stocks that together make up a given share of total
market value are kept, which leaves out micro-caps. A stock then enters with its
longest run of consecutive months in which it was kept and had a return, provided
the run is long enough to estimate its factor loadings in every regime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["calibration_sample", "longest_run", "market_coverage"]


def market_coverage(mktcap: pd.Series, coverage: float) -> pd.Series:
    """Flag, each month, the largest stocks making up ``coverage`` of market value.

    Stocks are ranked by market value, largest first, and kept while their
    cumulative market value is at most ``coverage`` times the month's total. The
    stock that crosses the threshold is left out. Stocks of equal market value
    are ranked in the order of the index, that is, by PERMNO.

    Parameters
    ----------
    mktcap : pandas.Series
        Market values indexed by date and PERMNO. Missing values are never kept.
    coverage : float
        Share of the month's total market value, between 0 and 1.

    Returns
    -------
    pandas.Series
        Booleans with the index of ``mktcap``.
    """
    values = mktcap.to_numpy(dtype=float)
    rows = np.flatnonzero(~np.isnan(values))
    dates = mktcap.index.get_level_values("date")[rows]
    flags = np.zeros(len(values), dtype=bool)
    for month in pd.Series(rows).groupby(dates.to_numpy()).groups.values():
        month = rows[month]
        order = month[np.argsort(-values[month], kind="stable")]
        cumulative = np.cumsum(values[order])
        flags[order[cumulative <= coverage * values[month].sum()]] = True
    return pd.Series(flags, index=mktcap.index)


def longest_run(available: np.ndarray) -> tuple[int, int]:
    """Start and length of the longest run of True values; the first if tied.

    Returns ``(0, 0)`` when no value is True.
    """
    available = np.asarray(available, dtype=bool)
    edges = np.diff(np.concatenate([[False], available, [False]]).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    if not len(starts):
        return 0, 0
    lengths = ends - starts
    i = int(np.argmax(lengths))
    return int(starts[i]), int(lengths[i])


def calibration_sample(
    stocks: pd.DataFrame, coverage: float, min_history: int
) -> pd.DataFrame:
    """Excess returns of the stocks that calibrate the simulation.

    Parameters
    ----------
    stocks : pandas.DataFrame
        Columns ``excess_ret`` and ``mktcap``, indexed by date and PERMNO, as
        written by the data stage.
    coverage : float
        Share of market value kept each month; see :func:`market_coverage`.
    min_history : int
        Months a stock's longest run of returns must have.

    Returns
    -------
    pandas.DataFrame
        One column per stock and one row per month. Each stock has values only
        over its longest run of months in which it was kept and had a return.
    """
    kept = market_coverage(stocks["mktcap"], coverage)
    returns = stocks.loc[kept & stocks["excess_ret"].notna(), "excess_ret"]
    wide = returns.unstack("permno")

    values = wide.to_numpy()
    columns, runs = [], []
    for j in range(values.shape[1]):
        start, length = longest_run(~np.isnan(values[:, j]))
        if length >= min_history:
            columns.append(j)
            runs.append((start, length))

    out = np.full((len(wide), len(columns)), np.nan)
    for k, (j, (start, length)) in enumerate(zip(columns, runs, strict=True)):
        out[start : start + length, k] = values[start : start + length, j]
    return pd.DataFrame(out, index=wide.index, columns=wide.columns[columns])
