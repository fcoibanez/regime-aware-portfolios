"""Stage 9: recovery of known loadings by RWLS (Figure 3).

On the first n_in_sample months of every simulated path, fits the regime model
to the factor returns and RWLS to the stock returns, using only those data, and
compares the estimated loadings with the true ones. Estimated regimes are
matched to the true regimes by their factor means. Writes to
results/<study>/recovery:

    loadings.npz      true and estimated loadings, each of shape
                      (paths, stocks, factors, regimes)
    summary.parquet   root-mean-squared error and correlation of estimated and
                      true loadings, by factor and regime
    run.json          configuration, versions and a summary of this run

Usage:

    python scripts/09_recovery.py
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from regime_aware_portfolios import load_config  # noqa: E402
from regime_aware_portfolios.paths import CONFIG, RESULTS  # noqa: E402
from regime_aware_portfolios.recovery import recover_path  # noqa: E402
from regime_aware_portfolios.run_record import (  # noqa: E402
    check_upstream,
    write_run_record,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=CONFIG / "main.toml")
    parser.add_argument("--jobs", type=int, default=-2)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    study = RESULTS / cfg.name
    check_upstream(
        study / "simulation" / "run.json",
        cfg,
        ("data", "sample", "calibration", "regimes", "seeds", "simulation"),
    )
    config = str(args.config.resolve())
    parts = Parallel(n_jobs=args.jobs)(
        delayed(recover_path)(config, p) for p in range(cfg.simulation.n_paths)
    )
    true = np.stack([t for t, _ in parts])
    est = np.stack([e for _, e in parts])

    names = cfg.factor_names
    rows = []
    for k, factor in enumerate(names):
        for s in range(true.shape[3]):
            t, e = true[:, :, k, s].ravel(), est[:, :, k, s].ravel()
            rows.append(
                {
                    "factor": factor,
                    "regime": s + 1,
                    "RMSE": float(np.sqrt(np.mean((e - t) ** 2))),
                    "Correlation": float(np.corrcoef(e, t)[0, 1]),
                    "Mean error": float(np.mean(e - t)),
                }
            )
    summary = pd.DataFrame(rows)
    out = study / "recovery"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "loadings.npz",
        true=true.astype(np.float32),
        estimated=est.astype(np.float32),
        factors=np.array(names),
    )
    summary.to_parquet(out / "summary.parquet")
    write_run_record(
        out / "run.json",
        "09_recovery",
        cfg,
        summary={
            "min_correlation": float(summary["Correlation"].min()),
            "max_rmse": float(summary["RMSE"].max()),
        },
    )
    print(
        summary.pivot(index="factor", columns="regime", values="Correlation").round(3)
    )
    print(summary.pivot(index="factor", columns="regime", values="RMSE").round(3))
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
