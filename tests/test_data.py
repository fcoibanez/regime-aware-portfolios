import sys
import types

import numpy as np
import pandas as pd
import pytest

from regime_aware_portfolios import data

# The query of the original data script, with its date range written out.
ORIGINAL_QUERY = """
SELECT msf.permno, msf.date, date_trunc('month', msf.date)::date as month, msf.ret, msf.shrout, msf.vol, msf.prc, msf.altprc, msf.bid, msf.ask, msenames.exchcd, msenames.siccd, msedelist.dlret, msedelist.dlstcd
FROM crsp.msf AS msf
LEFT JOIN crsp.msenames as msenames ON msf.permno = msenames.permno AND msenames.namedt <= msf.date AND msf.date <= msenames.nameendt
LEFT JOIN crsp.msedelist as msedelist
ON msf.permno = msedelist.permno AND date_trunc('month', msf.date)::date = date_trunc('month', msedelist.dlstdt)::date
WHERE msf.date BETWEEN '01/01/1960' AND '12/31/2024' AND msenames.shrcd IN (10, 11)
"""  # noqa: E501


def _words(sql):
    lines = [line for line in sql.splitlines() if not line.strip().startswith("--")]
    return " ".join(" ".join(lines).split())


def test_query_is_the_original_with_the_dates_as_parameters():
    query = _words(data.crsp_query())
    original = _words(ORIGINAL_QUERY)
    assert (
        query.replace("%(start)s", "'01/01/1960'").replace("%(end)s", "'12/31/2024'")
        == original
    )


@pytest.mark.parametrize(
    "ret, dlret, dlstcd, expected",
    [
        (0.05, np.nan, np.nan, 0.05),  # not delisted
        (0.05, -0.20, 500, -0.20),  # delisting return available
        (0.05, -0.20, 100, -0.20),  # available, whatever the code
        (0.05, np.nan, 500, -0.30),  # performance delisting
        (0.05, np.nan, 551, -0.30),
        (0.05, np.nan, 574, -0.30),
        (0.05, np.nan, 584, -0.30),
        (0.05, np.nan, 100, 0.05),  # code 100: still active
        (0.05, np.nan, 575, -1.0),  # anything else
        (0.05, np.nan, 231, -1.0),
        (np.nan, np.nan, np.nan, np.nan),
    ],
)
def test_delisting_rules(ret, dlret, dlstcd, expected):
    out = data.adjust_for_delisting([ret], [dlret], [dlstcd])
    np.testing.assert_allclose(out, [expected])


def test_delisting_accepts_nullable_columns():
    ret = pd.Series([0.1, pd.NA], dtype="Float64")
    dlret = pd.Series([pd.NA, pd.NA], dtype="Float64")
    code = pd.Series([pd.NA, 520], dtype="Int64")
    np.testing.assert_allclose(data.adjust_for_delisting(ret, dlret, code), [0.1, -0.3])


def _raw():
    """A query result as WRDS returns it, with nullable column types."""
    return pd.DataFrame(
        {
            "permno": pd.Series([10002, 10001, 10001], dtype="Int64"),
            "date": pd.to_datetime(["2020-01-31", "2020-02-28", "2020-01-31"]),
            "month": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-01-01"]),
            "ret": pd.Series([0.01, 0.02, -0.03], dtype="Float64"),
            "shrout": pd.Series([100.0, 200.0, 200.0], dtype="Float64"),
            "vol": pd.Series([5.0, 6.0, 7.0], dtype="Float64"),
            "prc": pd.Series([-10.0, 20.0, 21.0], dtype="Float64"),
            "altprc": pd.Series([-10.0, 20.0, 21.0], dtype="Float64"),
            "bid": pd.Series([9.9, pd.NA, 20.9], dtype="Float64"),
            "ask": pd.Series([10.1, pd.NA, 21.1], dtype="Float64"),
            "exchcd": pd.Series([3, 1, 1], dtype="Int64"),
            "siccd": pd.Series([3740, 6020, 6020], dtype="Int64"),
            "dlret": pd.Series([pd.NA, -0.5, pd.NA], dtype="Float64"),
            "dlstcd": pd.Series([pd.NA, 520, pd.NA], dtype="Int64"),
        }
    )


def test_prepare_crsp():
    crsp = data.prepare_crsp(_raw())
    assert list(crsp.columns) == list(data.CRSP_COLUMNS)
    assert crsp.index.names == ["date", "permno"]
    assert list(crsp.index) == [
        (pd.Timestamp("2020-01-31"), 10001),
        (pd.Timestamp("2020-01-31"), 10002),
        (pd.Timestamp("2020-02-29"), 10001),
    ]
    assert crsp.loc[(pd.Timestamp("2020-02-29"), 10001), "ret_adj"] == -0.5
    assert crsp.loc[(pd.Timestamp("2020-01-31"), 10002), "ret_adj"] == pytest.approx(
        0.01
    )
    assert crsp["ret"].dtype == "float64" and crsp["exchcd"].dtype == "Int64"


def test_save_and_load_round_trip(tmp_path):
    crsp = data.prepare_crsp(_raw())
    path = tmp_path / "crsp.parquet"
    data.save_crsp(crsp, path)
    pd.testing.assert_frame_equal(data.load_crsp(path), crsp)


def test_load_rejects_malformed_tables(tmp_path):
    crsp = data.prepare_crsp(_raw())
    path = tmp_path / "bad.parquet"
    crsp.drop(columns="bid").to_parquet(path)
    with pytest.raises(ValueError, match="lacks"):
        data.load_crsp(path)
    crsp.reset_index().set_index("permno").to_parquet(path)
    with pytest.raises(ValueError, match="indexed"):
        data.load_crsp(path)
    pd.concat([crsp, crsp.iloc[:1]]).to_parquet(path)
    with pytest.raises(ValueError, match="duplicated"):
        data.load_crsp(path)


def test_load_gives_integer_permnos(tmp_path):
    crsp = data.prepare_crsp(_raw())
    crsp.index = crsp.index.set_levels(
        crsp.index.levels[1].astype("Int64"), level="permno"
    )
    path = tmp_path / "crsp.parquet"
    crsp.to_parquet(path)
    loaded = data.load_crsp(path)
    assert loaded.index.get_level_values("permno").dtype == "int64"
    pd.testing.assert_frame_equal(loaded, data.prepare_crsp(_raw()))


def test_market_value():
    crsp = data.prepare_crsp(_raw())
    mv = data.market_value(crsp)
    assert mv.loc[(pd.Timestamp("2020-01-31"), 10002)] == 1000.0  # negative price
    zero = crsp.copy()
    zero["shrout"] = 0.0
    assert data.market_value(zero).isna().all()


def test_excess_returns():
    crsp = data.prepare_crsp(_raw())
    crsp.loc[(pd.Timestamp("2020-01-31"), 10001), "ret_adj"] = np.nan
    rf = pd.Series([0.001], index=pd.DatetimeIndex(["2020-01-31"]))
    out = data.excess_returns(crsp, rf)
    assert list(out.columns) == ["excess_ret", "mktcap"]
    assert list(out.index) == [
        (pd.Timestamp("2020-01-31"), 10001),
        (pd.Timestamp("2020-01-31"), 10002),
    ]
    assert np.isnan(out["excess_ret"].iloc[0])  # kept, with its market value
    assert out["mktcap"].iloc[0] == 200.0 * 21.0
    assert out["excess_ret"].iloc[1] == pytest.approx(0.01 - 0.001)


def test_read_usrec(tmp_path):
    path = tmp_path / "usrec.csv"
    path.write_text("observation_date,USREC\n2020-01-01,0\n2020-02-01,1\n")
    usrec = data.read_usrec(path)
    assert list(usrec) == [0, 1]
    assert usrec.index[1] == pd.Timestamp("2020-02-01")
    assert usrec.index.name == "date"


def test_save_usrec_keeps_the_file_unchanged(tmp_path):
    source = tmp_path / "fred.csv"
    source.write_bytes(b"observation_date,USREC\r\n2020-01-01,0\r\n")
    target = tmp_path / "usrec.csv"
    data.save_usrec(target, url=source.as_uri())
    assert target.read_bytes() == source.read_bytes()
    assert list(data.fetch_usrec(url=source.as_uri())) == [0]


def test_usrec_download_rejects_a_page_that_is_not_the_csv(tmp_path):
    page = tmp_path / "maintenance.html"
    page.write_bytes(b"<!doctype html>\n<title>Site Maintenance</title>\n")
    target = tmp_path / "usrec.csv"
    with pytest.raises(RuntimeError, match="did not return"):
        data.save_usrec(target, url=page.as_uri())
    assert not target.exists()
    with pytest.raises(RuntimeError, match="did not return"):
        data.fetch_usrec(url=page.as_uri())


def test_fetch_runs_the_query_and_processes_it(monkeypatch):
    calls = {}

    class Connection:
        def __init__(self, wrds_username):
            calls["user"] = wrds_username

        def raw_sql(self, sql, params=None):
            calls["sql"], calls["params"] = sql, params
            return _raw()

        def close(self):
            calls["closed"] = True

    monkeypatch.setitem(
        sys.modules, "wrds", types.SimpleNamespace(Connection=Connection)
    )
    crsp = data.fetch_crsp_monthly("someone", "1960-01", "2024-12")
    assert calls["user"] == "someone"
    assert calls["params"] == {"start": "01/01/1960", "end": "12/31/2024"}
    assert calls["sql"] == data.crsp_query()
    assert calls["closed"]
    pd.testing.assert_frame_equal(crsp, data.prepare_crsp(_raw()))


@pytest.mark.network
def test_fetch_usrec_from_fred():
    usrec = data.fetch_usrec()
    assert set(usrec.unique()) <= {0, 1}
    assert usrec.loc["2008-12-01"] == 1
    assert usrec.loc["2019-06-01"] == 0
