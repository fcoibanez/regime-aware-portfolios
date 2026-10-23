"""Stage 6: performance statistics of every backtest.

Reads the backtests of stage 5 and the cost estimate of stage 1, and writes to
results/<study>/metrics:

    path_statistics.parquet  every statistic of regime_aware_portfolios.metrics,
                             one row per strategy, risk aversion and path
    returns.parquet          monthly returns gross and net of costs, and the
                             turnover of each month, one row per strategy, risk
                             aversion, path and kind
    run.json                 configuration, versions and a summary of this run

Usage:

    python scripts/06_metrics.py
    python scripts/06_metrics.py --config config/alt_calibration.toml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from regime_aware_portfolios import load_config
from regime_aware_portfolios.backtest import load_result, result_file
from regime_aware_portfolios.metrics import net_returns, statistics_frame
from regime_aware_portfolios.paths import CONFIG, DERIVED, RESULTS
from regime_aware_portfolios.run_record import check_upstream, write_run_record
from regime_aware_portfolios.strategies import study_strategies


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=CONFIG / "main.toml")
    parser.add_argument("--jobs", type=int, default=-2)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    study = RESULTS / cfg.name
    check_upstream(
        study / "backtest" / "run.json",
        cfg,
        (
            "data",
            "sample",
            "calibration",
            "regimes",
            "seeds",
            "simulation",
            "strategies",
        ),
    )
    cost = json.loads((DERIVED / "costs.json").read_text(encoding="utf-8"))["one_way"]

    folder = study / "backtest"
    tasks = [
        (name, p)
        for name in study_strategies(cfg)
        for p in range(cfg.simulation.n_paths)
        if result_file(folder, name, p).exists()
    ]
    missing = len(study_strategies(cfg)) * cfg.simulation.n_paths - len(tasks)
    if missing:
        raise SystemExit(f"{missing} backtests are missing; finish stage 5 first")

    parts = Parallel(n_jobs=args.jobs, batch_size=64)(
        delayed(_one)(str(result_file(folder, n, p)), n, p, cost) for n, p in tasks
    )
    stats = pd.concat([s for s, _ in parts], ignore_index=True)
    returns = pd.concat([r for _, r in parts], ignore_index=True)

    out = study / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    stats.to_parquet(out / "path_statistics.parquet")
    returns.to_parquet(out / "returns.parquet")
    means = stats.groupby(["strategy", "phi"])[
        ["Sharpe Ratio", "Certainty Equivalent"]
    ].mean()
    write_run_record(
        out / "run.json",
        "06_metrics",
        cfg,
        summary={
            "one_way_cost": cost,
            "backtests": len(tasks),
            "mean_sharpe": {
                f"{s} {phi:g}": v for (s, phi), v in means["Sharpe Ratio"].items()
            },
        },
    )
    pd.set_option("display.width", 160)
    print(means["Sharpe Ratio"].unstack("phi").round(3).to_string())
    print(f"Written: {out}")


def _one(file: str, name: str, p: int, cost: float):
    result = load_result(file)
    stats = statistics_frame(result, cost, name, p)
    rows = []
    for i, phi in enumerate(result.phis):
        net = net_returns(result.returns[i], result.turnover[i], cost)
        for kind, values in (
            ("gross", result.returns[i]),
            ("net", net),
            ("turnover", result.turnover[i]),
        ):
            rows.append(
                {
                    "strategy": name,
                    "phi": float(phi),
                    "path": p,
                    "kind": kind,
                    **{f"m{j:02d}": v for j, v in enumerate(np.asarray(values))},
                }
            )
    return stats, pd.DataFrame(rows)


if __name__ == "__main__":
    main()
