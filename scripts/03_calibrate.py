"""Stage 3: the distributions of stock parameters that the simulation draws from.

Selects the calibration sample from the stock returns of stage 1, regresses
every stock on the factors in each regime of the model of stage 2, and writes to
results/<study>/calibration:

    calibration.json         parameters of the loading and residual-variance
                             distributions, and a record of the estimation
    stock_estimates.parquet  each stock's loadings, residual variances and
                             effective degrees of freedom, by regime
    run.json                 configuration, versions and a summary of this run

Usage:

    python scripts/03_calibrate.py
    python scripts/03_calibrate.py --config config/alt_calibration.toml
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from regime_aware_portfolios import load_config
from regime_aware_portfolios.calibration import (
    calibrate,
    estimate_stocks,
    save_calibration,
)
from regime_aware_portfolios.paths import CONFIG, DERIVED, RESULTS
from regime_aware_portfolios.regimes import load_regimes
from regime_aware_portfolios.run_record import check_upstream, write_run_record
from regime_aware_portfolios.sample import calibration_sample


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=CONFIG / "main.toml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    regimes_folder = RESULTS / cfg.name / "regimes"
    check_upstream(regimes_folder / "run.json", cfg, ("data", "regimes", "seeds"))

    stocks = pd.read_parquet(DERIVED / "stocks.parquet")
    factors = pd.read_parquet(DERIVED / "factors.parquet")[cfg.factor_names]
    regimes = load_regimes(regimes_folder)

    returns = calibration_sample(stocks, cfg.sample.coverage, cfg.sample.min_history)
    # The sparse-regime safeguard of rwls is counted in the calibration record.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        estimates = estimate_stocks(
            returns, factors, regimes.posteriors, cfg.rwls_settings()
        )
    result = calibrate(
        estimates,
        loading_eigen_floor=cfg.calibration.loading_eigen_floor,
        resid_eigen_floor=cfg.calibration.resid_eigen_floor,
        min_resid_dof=cfg.calibration.min_resid_dof,
    )

    folder = RESULTS / cfg.name / "calibration"
    save_calibration(result, folder)
    parts = {
        "loading": estimates.loadings,
        "resid_var": estimates.resid_var,
        "dof": estimates.dof,
        "fallback": estimates.fallback,
    }
    per_stock = pd.concat(
        [frame.set_axis(_labels(name, frame), axis=1) for name, frame in parts.items()],
        axis=1,
    )
    per_stock.to_parquet(folder / "stock_estimates.parquet")
    write_run_record(folder / "run.json", "03_calibrate", cfg, summary=result.record)

    rec = result.record
    vols = np.sqrt(12 * np.array(list(rec["mean_resid_var"].values())))
    print(
        f"Sample:   {rec['n_stocks']:,} stocks, {returns.notna().sum().sum():,} returns"
    )
    print(f"Fallback: {rec['fallback_estimates']} regime estimates used the safeguard")
    print(f"Residual model: {rec['n_resid_stocks']:,} stocks")
    print("Mean residual volatility by regime (annual):", np.round(vols, 3).tolist())
    print("Loading means by regime:")
    print(result.loadings_mean.unstack("regime").round(3).to_string())
    print(f"Written:  {folder}")


def _labels(name: str, frame: pd.DataFrame) -> list[str]:
    """Flat column names: the quantity, then the factor if any, then the regime."""
    flat = frame.columns.to_flat_index()
    return [
        " ".join([name, *map(str, c if isinstance(c, tuple) else (c,))]) for c in flat
    ]


if __name__ == "__main__":
    main()
