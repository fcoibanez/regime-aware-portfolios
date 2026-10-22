"""Stage 4: the simulation.

Assembles the data-generating process from the regime model of stage 2 and the
calibration of stage 3, draws every path once to check it against that process,
and writes to results/<study>/simulation:

    process.json        the data-generating process
    diagnostics.parquet one row per path: regime occupancy, factor and stock moments
    run.json            configuration, versions and a summary of this run

Paths are not stored. Each is generated from the study's seed and its number by
regime_aware_portfolios.simulation.simulate_path, in a few milliseconds, by the
stages that use it.

Usage:

    python scripts/04_simulate.py
    python scripts/04_simulate.py --config config/alt_calibration.toml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from regime_aware_portfolios import load_config
from regime_aware_portfolios.calibration import load_calibration
from regime_aware_portfolios.paths import CONFIG, RESULTS
from regime_aware_portfolios.regimes import load_regimes
from regime_aware_portfolios.run_record import check_upstream, write_run_record
from regime_aware_portfolios.simulation import (
    DataGeneratingProcess,
    save_process,
    simulate_path,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=CONFIG / "main.toml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    study = RESULTS / cfg.name
    check_upstream(
        study / "calibration" / "run.json",
        cfg,
        ("data", "sample", "calibration", "regimes", "seeds"),
    )
    dgp = DataGeneratingProcess.build(
        load_regimes(study / "regimes"), load_calibration(study / "calibration")
    )
    sim = cfg.simulation
    folder = study / "simulation"
    save_process(dgp, folder)

    M, K = dgp.n_regimes, len(dgp.factor_names)
    rows, transitions = [], np.zeros((M, M))
    loading_sum = np.zeros(K * M)
    loading_sq = np.zeros((K * M, K * M))
    sums = {"x": np.zeros((M, K)), "x2": np.zeros((M, K)), "n": np.zeros(M)}
    n_drawn = 0
    for p in range(sim.n_paths):
        path = simulate_path(dgp, sim.n_periods, sim.n_stocks, cfg.seeds.simulation, p)
        s = path.states - 1
        np.add.at(transitions, (s[:-1], s[1:]), 1)
        X = path.factors.to_numpy()
        for m in range(M):
            sums["x"][m] += X[s == m].sum(axis=0)
            sums["x2"][m] += (X[s == m] ** 2).sum(axis=0)
            sums["n"][m] += (s == m).sum()
        b = path.loadings.reshape(sim.n_stocks, K * M)
        loading_sum += b.sum(axis=0)
        loading_sq += b.T @ b
        n_drawn += sim.n_stocks
        vols = path.returns.std().to_numpy() * np.sqrt(12)
        rows.append(
            {
                "path": p,
                **{f"occupancy {m + 1}": float(np.mean(s == m)) for m in range(M)},
                "stock vol median": float(np.median(vols)),
                "resid vol median": float(np.median(path.resid_vol) * np.sqrt(12)),
                "market beta mean": float(path.loadings[:, 0, :].mean()),
            }
        )
    diagnostics = pd.DataFrame(rows).set_index("path")
    diagnostics.to_parquet(folder / "diagnostics.parquet")

    # Simulated against true moments, pooled over all paths.
    trans = transitions / transitions.sum(axis=1, keepdims=True)
    means = sums["x"] / sums["n"][:, None]
    vols = np.sqrt(sums["x2"] / sums["n"][:, None] - means**2)
    b_mean = loading_sum / n_drawn
    b_cov = loading_sq / n_drawn - np.outer(b_mean, b_mean)
    true_cov = dgp.loadings_factor @ dgp.loadings_factor.T
    checks = {
        "occupancy": np.abs(
            diagnostics.filter(like="occupancy").mean().to_numpy() - dgp.initial
        ).max(),
        "transitions": np.abs(trans - dgp.transmat).max(),
        "factor means": np.abs(means - dgp.factor_means).max(),
        "factor vols": np.abs(vols - dgp.factor_vols).max(),
        "loading means": np.abs(b_mean - dgp.loadings_mean).max(),
        "loading covariance": np.abs(b_cov - true_cov).max(),
    }
    summary = {
        "paths": sim.n_paths,
        "periods": sim.n_periods,
        "stocks": sim.n_stocks,
        "max abs difference from the process": {k: float(v) for k, v in checks.items()},
        "median stock volatility (annual)": float(
            diagnostics["stock vol median"].median()
        ),
    }
    write_run_record(folder / "run.json", "04_simulate", cfg, summary=summary)

    print(f"{sim.n_paths:,} paths of {sim.n_periods} months and {sim.n_stocks} stocks")
    print("Largest difference between pooled simulated and true values:")
    for k, v in checks.items():
        print(f"  {k:20s} {v:.2e}")
    print(
        "Median annual stock volatility, median across paths: "
        f"{summary['median stock volatility (annual)']:.3f}"
    )
    print(f"Written: {folder}")


if __name__ == "__main__":
    main()
