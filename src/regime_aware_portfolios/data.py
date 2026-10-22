"""Inputs of the study: CRSP stock returns, factor returns and NBER recessions.

CRSP data come either from a stored extract (the default) or from WRDS through
:func:`fetch_crsp_monthly`, which runs the query in ``sql/crsp_monthly.sql`` and
applies the same processing that produced the extract. Both give a table with the
columns of :data:`CRSP_COLUMNS`, indexed by month-end date and PERMNO.
"""

from __future__ import annotations

import io
import urllib.request
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "CRSP_COLUMNS",
    "adjust_for_delisting",
    "crsp_query",
    "excess_returns",
    "fetch_crsp_monthly",
    "fetch_usrec",
    "load_crsp",
    "market_value",
    "prepare_crsp",
    "read_usrec",
    "save_crsp",
    "save_usrec",
]

#: Columns of the processed CRSP table and their types. ``ret_adj`` is the return
#: adjusted for delisting; the others are as in the CRSP monthly stock file.
CRSP_COLUMNS = {
    "ret": "float64",
    "ret_adj": "float64",
    "shrout": "float64",
    "vol": "float64",
    "prc": "float64",
    "altprc": "float64",
    "bid": "float64",
    "ask": "float64",
    "exchcd": "Int64",
    "siccd": "Int64",
}

#: Delisting codes for which a missing delisting return is set to -30%.
PERFORMANCE_DELISTINGS = frozenset([500, 520, 580, 584, *range(551, 575)])

USREC_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=USREC"


# ------------------------------------------------------------------------- CRSP


def crsp_query() -> str:
    """The SQL query of the CRSP monthly stock file."""
    return (
        resources.files("regime_aware_portfolios")
        .joinpath("sql/crsp_monthly.sql")
        .read_text(encoding="utf-8")
    )


def adjust_for_delisting(ret, dlret, dlstcd) -> np.ndarray:
    """Monthly returns adjusted for delisting.

    The rules, applied in this order, are those of Tidy Finance:

    1. no delisting in the month: the return;
    2. a delisting with a delisting return: the delisting return;
    3. a delisting for poor performance (codes 500, 520, 551 to 574, 580 and 584)
       without a delisting return: -30%;
    4. a delisting with code 100 (still active): the return;
    5. any other delisting without a delisting return: -100%.
    """
    ret = pd.to_numeric(pd.Series(ret), errors="coerce").to_numpy(dtype=float)
    dlret = pd.to_numeric(pd.Series(dlret), errors="coerce").to_numpy(dtype=float)
    code = pd.to_numeric(pd.Series(dlstcd), errors="coerce").to_numpy(dtype=float)
    listed = np.isnan(code)
    conditions = [
        listed,
        ~listed & ~np.isnan(dlret),
        np.isin(code, list(PERFORMANCE_DELISTINGS)),
        code == 100,
    ]
    choices = [ret, dlret, np.full_like(ret, -0.30), ret]
    return np.select(conditions, choices, default=-1.0)


def prepare_crsp(raw: pd.DataFrame) -> pd.DataFrame:
    """Process the result of :func:`crsp_query` into the study's CRSP table.

    Returns are adjusted for delisting, every observation is dated at the end of
    its month, and the table is indexed by date and PERMNO.
    """
    df = raw.copy()
    df["ret_adj"] = adjust_for_delisting(df["ret"], df["dlret"], df["dlstcd"])
    df["date"] = pd.DatetimeIndex(pd.to_datetime(df["month"])) + pd.offsets.MonthEnd(0)
    df["permno"] = df["permno"].astype("int64")
    df = df.set_index(["date", "permno"]).sort_index()
    return _with_schema(df)


def fetch_crsp_monthly(username: str, start: str, end: str) -> pd.DataFrame:
    """Download the CRSP monthly stock file from WRDS and process it.

    Needs the ``wrds`` package and a WRDS account with access to CRSP.

    Parameters
    ----------
    username : str
        WRDS user name. The password is asked for interactively, or read from
        the user's ``.pgpass`` file.
    start, end : str
        First and last month, for example ``"1960-01"`` and ``"2024-12"``.
    """
    import wrds

    first = pd.Period(start, freq="M").start_time.strftime("%m/%d/%Y")
    last = pd.Period(end, freq="M").end_time.strftime("%m/%d/%Y")
    conn = wrds.Connection(wrds_username=username)
    try:
        raw = conn.raw_sql(crsp_query(), params={"start": first, "end": last})
    finally:
        conn.close()
    return prepare_crsp(raw)


def save_crsp(crsp: pd.DataFrame, path: str | Path) -> None:
    """Write the CRSP table to a Parquet file."""
    _with_schema(crsp).to_parquet(path)


def load_crsp(path: str | Path) -> pd.DataFrame:
    """Read the CRSP table from a Parquet file and check its layout."""
    return _with_schema(pd.read_parquet(path))


def market_value(crsp: pd.DataFrame) -> pd.Series:
    """Market value in thousands of dollars: shares outstanding times price.

    CRSP reports shares in thousands and marks a price that is the bid-ask
    midpoint with a negative sign, hence the absolute value. A zero market value
    is treated as missing.
    """
    mv = (crsp["shrout"] * crsp["altprc"]).abs()
    return mv.where(mv > 0).rename("mktcap")


def excess_returns(crsp: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    """Monthly excess returns and market values of every stock in the months of ``rf``.

    The excess return is the delisting-adjusted return less the risk-free rate.
    Rows with a missing return are kept, so that each month's market values cover
    every stock listed in it.

    Returns
    -------
    pandas.DataFrame
        Columns ``excess_ret`` and ``mktcap``, indexed by date and PERMNO.
    """
    dates = crsp.index.get_level_values("date")
    rows = crsp[dates.isin(rf.index)]
    rf_by_row = rf.reindex(rows.index.get_level_values("date")).to_numpy()
    return pd.DataFrame(
        {
            "excess_ret": rows["ret_adj"].to_numpy() - rf_by_row,
            "mktcap": market_value(rows).to_numpy(),
        },
        index=rows.index,
    )


def _with_schema(df: pd.DataFrame) -> pd.DataFrame:
    if list(df.index.names) != ["date", "permno"]:
        raise ValueError("the CRSP table must be indexed by date and permno")
    missing = [c for c in CRSP_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"the CRSP table lacks column(s) {missing}")
    dates = df.index.get_level_values("date")
    if not isinstance(dates, pd.DatetimeIndex):
        raise ValueError("the CRSP dates must be datetimes")
    out = df[list(CRSP_COLUMNS)].astype(CRSP_COLUMNS)
    out.index = pd.MultiIndex.from_arrays(
        [dates, df.index.get_level_values("permno").astype("int64")],
        names=["date", "permno"],
    )
    if out.index.duplicated().any():
        raise ValueError("the CRSP table has duplicated date and permno pairs")
    return out


# ---------------------------------------------------------------------- recessions


def fetch_usrec(url: str = USREC_URL, timeout: float = 30.0) -> pd.Series:
    """NBER recession indicator (FRED series USREC), one value per month.

    1 in months of recession, from the month after a business-cycle peak through
    the trough, and 0 otherwise. Dated at the first day of the month, as on FRED.
    """
    return read_usrec(io.BytesIO(_download_usrec(url, timeout)))


def save_usrec(path: str | Path, url: str = USREC_URL, timeout: float = 30.0) -> None:
    """Download the USREC series and store FRED's CSV file unchanged."""
    Path(path).write_bytes(_download_usrec(url, timeout))


def _download_usrec(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "regime-aware-portfolios"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    # FRED answers with an HTML page, not an error, when the site is down.
    if not raw.lstrip().lower().startswith(b"observation_date"):
        raise RuntimeError(f"{url} did not return the USREC CSV file; try again later")
    return raw


def read_usrec(source) -> pd.Series:
    """Read the USREC series from a FRED CSV file or URL."""
    frame = pd.read_csv(source)
    date_col = frame.columns[0]
    series = frame.set_index(pd.to_datetime(frame[date_col]))["USREC"].astype(int)
    series.index.name = "date"
    return series
