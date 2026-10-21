# Data

`raw/` holds the inputs of the study as obtained from their sources. `derived/`
holds the tables `scripts/01_data.py` builds from them, which the later stages
read.

| File in `raw/` | Source | Obtained | In the repository |
|---|---|---|---|
| `crsp_monthly.parquet` | CRSP Monthly Stock File, through WRDS | 16 June 2026 | No (licensed) |
| `french/F-F_Research_Data_5_Factors_2x3_CSV.zip` | Kenneth R. French Data Library | 20 October 2026 | Yes |
| `french/F-F_Momentum_Factor_CSV.zip` | Kenneth R. French Data Library | 20 October 2026 | Yes |
| `usrec.csv` | FRED, series USREC | 20 October 2026 | Yes |

## CRSP stock returns

Monthly returns, prices, shares outstanding, trading volume, bid and ask prices,
exchange and industry codes of ordinary common shares (CRSP share codes 10 and
11), from January 1960 to December 2024. The query is in
[`src/regime_aware_portfolios/sql/crsp_monthly.sql`](../src/regime_aware_portfolios/sql/crsp_monthly.sql)
and follows
[Tidy Finance](https://www.tidy-finance.org/python/wrds-crsp-and-compustat.html).
Returns are adjusted for delisting as described in
`regime_aware_portfolios.data.adjust_for_delisting`.

The file used for every result was downloaded on 16 June 2026. It has 3,527,723
stock-months of 26,298 stocks. The CRSP license does not allow it to be shared,
so it is not in the repository. With a WRDS account that includes CRSP, the same
data are downloaded and processed by

```
python -m pip install wrds
python scripts/01_data.py --wrds NAME
```

which writes them to `raw/crsp_monthly.parquet`, replacing any file there, and
builds the derived tables from them. CRSP revises its historical data from time
to time, so a later download may differ slightly from the one used here.

## Factor returns

The Fama-French five factors (2x3 sorts), the momentum factor and the one-month
Treasury bill rate, from the
[Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html),
downloaded with [`rwls.load_factors`](https://pypi.org/project/rwls/). Both files
state that they were created using the 202608 CRSP database. French updates the
factors as CRSP data are revised, so the files are kept in the repository as
downloaded. If they are deleted, the script downloads the current ones.

## Recessions

The NBER-based recession indicator for the United States, series
[USREC](https://fred.stlouisfed.org/series/USREC) from FRED, Federal Reserve
Bank of St. Louis, as downloaded, with observations through September 2026. It
equals 1 from the month after a business-cycle peak through the trough, and 0
otherwise.

## Derived tables

`scripts/01_data.py` writes these to `derived/`, each covering July 1963 to
December 2024 (`start` and `end` in the `[data]` section of the configuration):

| File | Contents |
|---|---|
| `stocks.parquet` | Excess return (delisting-adjusted return less the Treasury bill rate) and market value in thousands of dollars, by month and PERMNO. Stock-months without a return are kept with their market value. |
| `factors.parquet` | The factors of the model named in the configuration, and the Treasury bill rate `RF`. |
| `usrec.parquet` | The recession indicator, dated at month ends. |
| `manifest.json` | The configuration, code commit, package versions and a summary of the run. |

## Terms of use

The code in this repository is under the MIT License; the data are not. CRSP
data are subject to the license of the Center for Research in Security Prices.
The factor files are made available by Kenneth R. French, and USREC by the
Federal Reserve Bank of St. Louis, under the terms of their sites.
