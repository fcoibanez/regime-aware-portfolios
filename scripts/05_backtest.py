"""Stage 5: backtests of every strategy on every simulated path.

Each path is generated from the study's seed and its number, and each strategy
is run on it at every decision date and level of risk aversion. One file is
written per strategy and path, to results/<study>/backtest/<strategy>/, so an
interrupted run resumes where it stopped. The run stops at the first portfolio
the solver cannot find, naming the strategy, path, date and risk aversion.

    run.json   configuration, versions and a summary of the backtests

Usage:

    python scripts/05_backtest.py
    python scripts/05_backtest.py --config config/alt_calibration.toml
    python scripts/05_backtest.py --strategies framework agnostic --paths 0:100
"""

from __future__ import annotations

import os

# One thread per process: the paths are run in parallel instead.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from joblib import Parallel, delayed  # noqa: E402

from regime_aware_portfolios import load_config  # noqa: E402
from regime_aware_portfolios.backtest import (  # noqa: E402
    load_result,
    result_file,
    run_backtest,
)
from regime_aware_portfolios.paths import CONFIG, RESULTS  # noqa: E402
from regime_aware_portfolios.run_record import (  # noqa: E402
    check_upstream,
    write_run_record,
)
from regime_aware_portfolios.strategies import study_strategies  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=CONFIG / "main.toml")
    parser.add_argument("--strategies", nargs="*", help="default: every strategy")
    parser.add_argument("--paths", help="a range such as 0:100; default: every path")
    parser.add_argument(
        "--jobs", type=int, default=-2, help="processes (default: all but one)"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    study = RESULTS / cfg.name
    check_upstream(
        study / "simulation" / "run.json",
        cfg,
        ("data", "sample", "calibration", "regimes", "seeds", "simulation"),
    )
    strategies = study_strategies(cfg)
    names = args.strategies or list(strategies)
    unknown = set(names) - set(strategies)
    if unknown:
        parser.error(f"unknown strategies {sorted(unknown)}; known: {list(strategies)}")
    paths = _path_range(args.paths, cfg.simulation.n_paths)

    folder = study / "backtest"
    tasks = [
        (name, p)
        for p in paths
        for name in names
        if not result_file(folder, name, p).exists()
    ]
    print(f"{len(tasks):,} of {len(paths) * len(names):,} backtests to run")
    start, done = time.time(), 0
    config = str(args.config.resolve())
    jobs = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
        delayed(run_backtest)(config, name, p) for name, p in tasks
    )
    for _ in jobs:
        done += 1
        if done % 500 == 0 or done == len(tasks):
            print(f"  {done:,} done, {time.time() - start:,.0f}s", flush=True)

    summary = _summary(folder, names, paths)
    write_run_record(folder / "run.json", "05_backtest", cfg, summary=summary)
    for name, s in summary["strategies"].items():
        print(
            f"{name:18s} {s['paths']:5d} paths; later statements used: {s['fallbacks']}"
        )
    print(f"Written: {folder}")


def _path_range(spec: str | None, n_paths: int) -> range:
    if spec is None:
        return range(n_paths)
    lo, hi = (int(x) for x in spec.split(":"))
    if not 0 <= lo < hi <= n_paths:
        raise SystemExit(f"--paths must lie within 0:{n_paths}")
    return range(lo, hi)


def _summary(folder: Path, names, paths) -> dict:
    out = {}
    for name in names:
        files = [result_file(folder, name, p) for p in paths]
        files = [f for f in files if f.exists()]
        fallbacks = sum(int((load_result(f).attempts > 0).sum()) for f in files)
        out[name] = {"paths": len(files), "fallbacks": fallbacks}
    return {"paths": [paths.start, paths.stop], "strategies": out}


if __name__ == "__main__":
    main()
