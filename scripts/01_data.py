"""Stage 1: the inputs of the study.

Gathers the CRSP monthly stock file, the factor returns and the NBER recession
indicator in data/raw, and writes the tables the later stages read to
data/derived:

    stocks.parquet    excess return and market value of every stock and month
    factors.parquet   factor returns and the risk-free rate
    usrec.parquet     NBER recession indicator
    costs.json        cost of a one-way trade, from the CRSP quoted spreads
    run.json          configuration and package versions of this run

The three tables cover the months from [data] start to end of the configuration.
The cost is half the [costs] quantile of the quoted spreads of every stock-month
from [costs] decimalization on. The market-value-weighted mean, a lower estimate,
is recorded with it for the sensitivity of the results to costs.

Usage:

    python scripts/01_data.py
    python scripts/01_data.py --wrds NAME

The first form uses the CRSP data in data/raw/crsp_monthly.parquet; see
data/README.md for how the file used for the paper was obtained. The second
downloads them again from WRDS, with the user name NAME, and replaces that file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import rwls

from regime_aware_portfolios import data, load_config
from regime_aware_portfolios.paths import CONFIG, CRSP, DERIVED, FRENCH, RAW, USREC
from regime_aware_portfolios.run_record import write_run_record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=CONFIG / "main.toml")
    parser.add_argument(
        "--wrds",
        metavar="USERNAME",
        help="download the CRSP data from WRDS with this user name",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    RAW.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)

    # Raw files
    if args.wrds:
        months = f"{cfg.data.crsp_start} to {cfg.data.crsp_end}"
        print(f"Downloading CRSP from WRDS, {months}")
        crsp = data.fetch_crsp_monthly(
            args.wrds, cfg.data.crsp_start, cfg.data.crsp_end
        )
        data.save_crsp(crsp, CRSP)
    elif not CRSP.exists():
        sys.exit(
            f"{CRSP} not found. CRSP data are licensed and not included in the "
            "repository; see data/README.md."
        )

    factors = rwls.load_factors(cfg.data.factor_model, cache_dir=FRENCH)

    if not USREC.exists():
        print("Downloading USREC from FRED")
        data.save_usrec(USREC)

    # Derived tables
    factors = factors.loc[_month(cfg.data.start) : _month(cfg.data.end)]
    months = pd.date_range(
        _month(cfg.data.start), _month(cfg.data.end), freq=pd.offsets.MonthEnd()
    )
    if not factors.index.equals(months):
        sys.exit("the factor returns do not cover every month from start to end")

    crsp = data.load_crsp(CRSP)
    stocks = data.excess_returns(crsp, factors["RF"])

    usrec = data.read_usrec(USREC)
    usrec.index = usrec.index + pd.offsets.MonthEnd(0)
    usrec = usrec.reindex(months)
    if usrec.isna().any():
        sys.exit("USREC does not cover every month from start to end")

    factors.to_parquet(DERIVED / "factors.parquet")
    stocks.to_parquet(DERIVED / "stocks.parquet")
    usrec.rename("usrec").to_frame().to_parquet(DERIVED / "usrec.parquet")

    spreads = data.quoted_spreads(crsp)
    after = spreads[
        spreads.index.get_level_values("date")
        >= pd.Period(cfg.costs.decimalization, freq="M").start_time
    ]
    costs = {
        "one_way": data.half_spread(crsp, cfg.costs.decimalization, cfg.costs.quantile),
        "one_way_cap_weighted": data.cap_weighted_half_spread(
            crsp, cfg.costs.decimalization
        ),
        "spread_quantile": cfg.costs.quantile,
        "from": cfg.costs.decimalization,
        "stock_months": int(len(after)),
    }
    (DERIVED / "costs.json").write_text(
        json.dumps(costs, indent=2) + "\n", encoding="utf-8"
    )

    permnos = stocks.index.get_level_values("permno")
    summary = {
        "months": len(months),
        "first_month": f"{months[0]:%Y-%m}",
        "last_month": f"{months[-1]:%Y-%m}",
        "stock_months": len(stocks),
        "stock_months_with_return": int(stocks["excess_ret"].notna().sum()),
        "stocks": int(permnos.nunique()),
        "factors": [c for c in factors.columns if c != "RF"],
        "recession_months": int(usrec.sum()),
        "one_way_cost": costs["one_way"],
    }
    write_run_record(
        DERIVED / "run.json",
        "01_data",
        cfg,
        crsp={"downloaded": bool(args.wrds), "rows": len(crsp)},
        factor_source={k: v for k, v in factors.attrs.items() if k != "retrieved"},
        summary=summary,
    )

    print(f"CRSP:    {len(crsp):,} stock-months")
    print(
        f"Factors: {cfg.data.factor_model}, CRSP vintage "
        f"{', '.join(factors.attrs['crsp_vintage'])}"
    )
    print(
        f"Sample:  {summary['first_month']} to {summary['last_month']}, "
        f"{summary['months']} months, {summary['stocks']:,} stocks, "
        f"{summary['stock_months_with_return']:,} stock-months with a return"
    )
    print(f"Written: {DERIVED}")


def _month(value: str) -> pd.Timestamp:
    return pd.Timestamp(value) + pd.offsets.MonthEnd(0)


if __name__ == "__main__":
    main()
