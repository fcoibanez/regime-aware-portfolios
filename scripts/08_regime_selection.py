"""Stage 8: information criteria for the number of regimes (Figure 1).

Fits regime models with each number of regimes in [regime_selection] n_regimes
to the factor returns of stage 1: on the whole sample, and on each of
[regime_selection] n_subperiods consecutive sub-periods of equal length. Each is
fitted from [regime_selection] n_starts cold starts and the fit with the highest
likelihood is kept. Writes to results/<study>/regime_selection:

    criteria.parquet   log-likelihood, number of parameters, AIC, BIC, HQIC and
                       ICL, by sample and number of regimes
    run.json           configuration, versions and a summary of this run

With --costa-kwon, the models are fitted to the factors of the Costa and Kwon
regime model ([strategies] costa_kwon_factors) instead, and the files are
criteria_costa_kwon.parquet and run_costa_kwon.json.

Usage:

    python scripts/08_regime_selection.py
    python scripts/08_regime_selection.py --costa-kwon
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rwls

from regime_aware_portfolios import load_config
from regime_aware_portfolios.paths import CONFIG, DERIVED, RESULTS
from regime_aware_portfolios.run_record import check_upstream, write_run_record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=CONFIG / "main.toml")
    parser.add_argument("--jobs", type=int, default=-2)
    parser.add_argument(
        "--costa-kwon",
        action="store_true",
        help="fit to the factors of the Costa and Kwon regime model",
    )
    args = parser.parse_args(argv)
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)

    cfg = load_config(args.config)
    check_upstream(DERIVED / "run.json", cfg, ("data",))
    names = (
        list(cfg.strategies.costa_kwon_factors) if args.costa_kwon else cfg.factor_names
    )
    factors = pd.read_parquet(DERIVED / "factors.parquet")[names]
    sel = cfg.regime_selection
    samples = {"Full sample": factors}
    for part in np.array_split(np.arange(len(factors)), sel.n_subperiods):
        sub = factors.iloc[part]
        samples[f"{sub.index[0]:%Y-%m} to {sub.index[-1]:%Y-%m}"] = sub

    settings = cfg.regime_settings(selection="best")
    frames, selected = [], {}
    for label, data in samples.items():
        result = rwls.select_regimes(
            data,
            n_regimes=sel.n_regimes,
            settings=settings,
            n_starts=sel.n_starts,
            n_jobs=args.jobs,
        )
        table = result.table.reset_index()
        table.insert(0, "sample", label)
        frames.append(table)
        selected[label] = result.selected
        print(f"{label}: selected {result.selected}", flush=True)

    out = RESULTS / cfg.name / "regime_selection"
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_costa_kwon" if args.costa_kwon else ""
    criteria = pd.concat(frames, ignore_index=True)
    criteria.to_parquet(out / f"criteria{suffix}.parquet")
    write_run_record(
        out / f"run{suffix}.json",
        "08_regime_selection",
        cfg,
        summary={"factors": names, "selected": selected},
    )
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
