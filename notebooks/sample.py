# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Section 4.1: the stocks that calibrate the simulation
#
# Each month only the largest stocks that together make up `[sample] coverage`
# of total market value are kept, and a stock then enters with its longest run
# of consecutive months in which it was kept and had a return, if the run lasts
# at least `[sample] min_history` months (`regime_aware_portfolios.sample`).
# The counts the text quotes, from the tables of stage 1.

# %%
import pandas as pd

from regime_aware_portfolios import load_config
from regime_aware_portfolios.paths import CONFIG, DERIVED
from regime_aware_portfolios.sample import calibration_sample, market_coverage

cfg = load_config(CONFIG / "main.toml")
stocks = pd.read_parquet(DERIVED / "stocks.parquet")
kept = market_coverage(stocks["mktcap"], cfg.sample.coverage)
sample = calibration_sample(stocks, cfg.sample.coverage, cfg.sample.min_history)


def n_stocks(rows):
    return rows.index.get_level_values("permno").nunique()


present = sample.notna()
value = stocks["mktcap"].unstack("permno")
share = value.reindex(columns=sample.columns).where(present).sum(axis=1) / value.sum(
    axis=1
)
last = present.apply(lambda c: c[c].index.max())
pd.Series(
    {
        "Stocks with a return": n_stocks(stocks[stocks["excess_ret"].notna()]),
        "Kept by the coverage filter in some month": n_stocks(
            stocks[kept & stocks["excess_ret"].notna()]
        ),
        "In the final sample": sample.shape[1],
        "Runs ending before the last month": int((last < sample.index[-1]).sum()),
        "Stocks per month, fewest": int(present.sum(axis=1).min()),
        "Stocks per month, most": int(present.sum(axis=1).max()),
        "Share of market value, mean": round(share.mean(), 3),
        "Share of market value, lowest": round(share.min(), 3),
        "Share of market value, highest": round(share.max(), 3),
    }
)
