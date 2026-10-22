"""Stage 2: the regime model that calibrates the simulation.

Fits the hidden Markov model to the factor returns written by stage 1, from the
cold starts and by the selection rule of the configuration, and writes to
results/<study>/regimes:

    regimes.json        parameters of the selected model and a record of the starts
    posteriors.parquet  posterior regime probabilities, one row per month
    summary.txt         the rwls summary of the fit
    run.json            configuration, versions and a summary of this run

Usage:

    python scripts/02_regimes.py
    python scripts/02_regimes.py --config config/alt_calibration.toml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from regime_aware_portfolios import load_config
from regime_aware_portfolios.paths import CONFIG, DERIVED, RESULTS
from regime_aware_portfolios.regimes import fit_regimes, save_regimes
from regime_aware_portfolios.run_record import check_upstream, write_run_record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=CONFIG / "main.toml")
    parser.add_argument(
        "--jobs", type=int, default=-1, help="processes for the cold starts"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    check_upstream(DERIVED / "run.json", cfg, ("data",))
    factors = pd.read_parquet(DERIVED / "factors.parquet")

    fit, starts = fit_regimes(factors, cfg, n_jobs=args.jobs)

    folder = RESULTS / cfg.name / "regimes"
    save_regimes(fit, starts, folder)
    text = (
        fit.summary(periods_per_year=12).as_text()
        + f"\n\nSelection: {describe_selection(starts, cfg)}"
    )
    (folder / "summary.txt").write_text(text + "\n", encoding="utf-8")
    write_run_record(
        folder / "run.json",
        "02_regimes",
        cfg,
        starts=starts,
        summary={
            "loglike": fit.loglike,
            "converged": fit.converged,
            "n_iter": fit.n_iter,
            "persistence": np.diag(fit.transmat).tolist(),
        },
    )

    print(text)
    print(f"Written: {folder}")


def describe_selection(starts: dict, cfg) -> str:
    """One sentence on how the fit was chosen among the cold starts."""
    n, reached = starts["starts"], starts["reached"]
    rule = starts["selection"]
    if rule == "best":
        what = "the highest likelihood reached"
    elif rule == "modal":
        what = "the model reached by the most starts"
    else:
        what = (
            "the optimum whose regime means are closest to the reference means of "
            f"the configuration, at its highest likelihood (seed {starts['seed']})"
        )
    return (
        f"{what} from {n:,} cold starts, reached by {reached:,}. Regimes are "
        f"numbered by the volatility of {cfg.regimes.order_by}, lowest first."
    )


if __name__ == "__main__":
    main()
