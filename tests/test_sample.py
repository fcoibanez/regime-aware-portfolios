import numpy as np
import pandas as pd
import pytest

from regime_aware_portfolios import sample


def _panel(values, months):
    """A (date, permno) table from a dict of PERMNO -> list of monthly values."""
    dates = pd.date_range("2000-01-31", periods=months, freq=pd.offsets.MonthEnd())
    frame = pd.DataFrame(values, index=pd.DatetimeIndex(dates, name="date"))
    frame.columns.name = "permno"
    return frame.stack(future_stack=True)


def test_coverage_keeps_the_largest_stocks():
    mktcap = _panel({1: [50.0], 2: [30.0], 3: [15.0], 4: [5.0]}, 1)
    assert list(sample.market_coverage(mktcap, 0.95)) == [True, True, True, False]
    # 50 + 30 + 15 = 95 is at most 95% of 100; 80 crosses 79%
    assert list(sample.market_coverage(mktcap, 0.79)) == [True, False, False, False]


def test_coverage_is_computed_month_by_month():
    mktcap = _panel({1: [90.0, 10.0], 2: [10.0, 90.0]}, 2)
    flags = sample.market_coverage(mktcap, 0.9)
    assert flags.loc[(slice(None), 1)].tolist() == [True, False]
    assert flags.loc[(slice(None), 2)].tolist() == [False, True]


def test_coverage_never_keeps_missing_values():
    mktcap = _panel({1: [np.nan], 2: [60.0], 3: [40.0]}, 1)
    assert list(sample.market_coverage(mktcap, 0.6)) == [False, True, False]


def test_coverage_breaks_ties_by_permno():
    mktcap = _panel({1: [60.0], 7: [20.0], 3: [20.0]}, 1).sort_index()
    flags = sample.market_coverage(mktcap, 0.8)
    assert flags.loc[(slice(None), 3)].item()
    assert not flags.loc[(slice(None), 7)].item()


@pytest.mark.parametrize(
    "available, expected",
    [
        ([], (0, 0)),
        ([False, False], (0, 0)),
        ([True, True, True], (0, 3)),
        ([True, False, True, True, False], (2, 2)),
        ([True, True, False, True, True], (0, 2)),  # first of equal runs
        ([False, True, True, True], (1, 3)),
    ],
)
def test_longest_run(available, expected):
    assert sample.longest_run(np.array(available, dtype=bool)) == expected


def test_calibration_sample():
    nan = np.nan
    ret = {
        # a gap in month 3: the longer run before it is kept
        1: [0.1, 0.2, 0.3, nan, 0.5, 0.6],
        # one month not covered (tiny market value) breaks the run
        2: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        # too short
        3: [nan, nan, nan, nan, 0.5, 0.6],
    }
    mv = {
        1: [100.0] * 6,
        2: [100.0, 100.0, 100.0, 100.0, 1e-9, 100.0],
        3: [100.0] * 6,
    }
    stocks = pd.DataFrame({"excess_ret": _panel(ret, 6), "mktcap": _panel(mv, 6)})
    out = sample.calibration_sample(stocks, coverage=0.999, min_history=3)
    assert list(out.columns) == [1, 2]
    np.testing.assert_array_equal(out[1].to_numpy(), [0.1, 0.2, 0.3, nan, nan, nan])
    np.testing.assert_array_equal(out[2].to_numpy(), [0.1, 0.2, 0.3, 0.4, nan, nan])
    assert len(out) == 6
