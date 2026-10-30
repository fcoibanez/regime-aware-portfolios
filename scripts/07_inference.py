"""Stage 7: inference and the evaluation of the regime probabilities.

Reads the returns of stage 6 and the backtests of stage 5, regenerates the true
regimes of every path, and writes to results/<study>/inference:

    significance.parquet     the framework against each benchmark, net of costs,
                             at every reported risk aversion: paired comparison of
                             Sharpe ratios across paths, and how often the Ledoit
                             and Wolf test rejects on a single path
    ledoit_wolf.parquet      the test on every path
    detection.parquet        rejection rate of the test as histories lengthen,
                             framework against the regime-agnostic benchmark and
                             against Costa and Kwon, at the first reported risk
                             aversion
    forecast_quality.parquet hit rates and Brier scores of the regime
                             probabilities, estimated and with known parameters
    regime_conditional.parquet the framework against the regime-agnostic
                             benchmark, by the true regime of each month
    run.json                 configuration, versions and a summary of this run

Usage:

    python scripts/07_inference.py
    python scripts/07_inference.py --config config/alt_calibration.toml
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
from regime_aware_portfolios.backtest import load_result, result_file  # noqa: E402
from regime_aware_portfolios.calibration import load_calibration  # noqa: E402
from regime_aware_portfolios.evaluation import (  # noqa: E402
    align_probabilities,
    forecast_scores,
    regime_conditional,
)
from regime_aware_portfolios.inference import (  # noqa: E402
    detection_by_horizon,
    ledoit_wolf,
    paired_comparison,
    rejection_summary,
)
from regime_aware_portfolios.metrics import sharpe_ratio  # noqa: E402
from regime_aware_portfolios.paths import CONFIG, RESULTS  # noqa: E402
from regime_aware_portfolios.regimes import load_regimes  # noqa: E402
from regime_aware_portfolios.run_record import (  # noqa: E402
    check_upstream,
    write_run_record,
)
from regime_aware_portfolios.simulation import (  # noqa: E402
    DataGeneratingProcess,
    simulate_path,
)

#: Benchmarks of Table 3, in the order of its columns.
BENCHMARKS = (
    "agnostic",
    "rolling_36",
    "equal_weight",
    "minimum_variance",
    "costa_kwon",
)
FRAMEWORK = "framework"
#: Benchmarks of the detection analysis: the closest and an existing regime model.
DETECTION = ("agnostic", "costa_kwon")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=CONFIG / "main.toml")
    parser.add_argument("--jobs", type=int, default=-2)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    study = RESULTS / cfg.name
    check_upstream(
        study / "metrics" / "run.json",
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
    out = study / "inference"
    out.mkdir(parents=True, exist_ok=True)
    inf = cfg.inference
    reported = cfg.strategies.reported_risk_aversion
    rolling = f"rolling_{cfg.strategies.rolling_window}"
    benchmarks = [b if b != "rolling_36" else rolling for b in BENCHMARKS]

    returns = pd.read_parquet(study / "metrics" / "returns.parquet")
    months = [c for c in returns.columns if c.startswith("m")]
    net = returns[returns["kind"] == "net"].set_index(["strategy", "phi", "path"])[
        months
    ]

    # ---------------------------------------------------- Table 3: significance
    tasks = [(b, phi) for phi in reported for b in benchmarks]
    paths = range(cfg.simulation.n_paths)
    tests = Parallel(n_jobs=args.jobs, batch_size=8)(
        delayed(_lw)(
            net.loc[(FRAMEWORK, float(phi), p)].to_numpy(),
            net.loc[(b, float(phi), p)].to_numpy(),
            inf.bootstrap_block,
            inf.bootstrap_draws,
            cfg.seeds.bootstrap,
            p,
        )
        for b, phi in tasks
        for p in paths
    )
    lw = pd.DataFrame(tests)
    lw.insert(0, "path", [p for _ in tasks for p in paths])
    lw.insert(0, "phi", [float(phi) for b, phi in tasks for _ in paths])
    lw.insert(0, "benchmark", [b for b, phi in tasks for _ in paths])
    lw.to_parquet(out / "ledoit_wolf.parquet")

    rows = []
    for b, phi in tasks:
        fw = net.loc[FRAMEWORK, float(phi)].apply(sharpe_ratio, axis=1)
        bm = net.loc[b, float(phi)].apply(sharpe_ratio, axis=1)
        sel = lw[(lw["benchmark"] == b) & (lw["phi"] == float(phi))]
        rows.append(
            {
                "benchmark": b,
                "phi": float(phi),
                **paired_comparison(fw, bm),
                **rejection_summary(sel, inf.alpha).to_dict(),
            }
        )
    significance = pd.DataFrame(rows)
    significance.to_parquet(out / "significance.parquet")

    # --------------------------------------------- detection by history length
    phi0 = float(reported[0])
    a = [net.loc[(FRAMEWORK, phi0, p)].to_numpy() for p in paths]
    parts = []
    for benchmark in DETECTION:
        b = [net.loc[(benchmark, phi0, p)].to_numpy() for p in paths]
        part = detection_by_horizon(
            a,
            b,
            inf.detection_multiples,
            inf.alpha,
            inf.bootstrap_block,
            inf.bootstrap_draws,
            cfg.seeds.bootstrap,
            inf.detection_histories,
        ).reset_index()
        part.insert(0, "benchmark", benchmark)
        parts.append(part)
    detection = pd.concat(parts, ignore_index=True)
    detection.to_parquet(out / "detection.parquet")

    # ------------------------------------------- Table 5: regime probabilities
    dgp = DataGeneratingProcess.build(
        load_regimes(study / "regimes"), load_calibration(study / "calibration")
    )
    sim = cfg.simulation
    folder = study / "backtest"
    pooled = {
        k: [] for k in ("now_est", "now_true", "fc_est", "fc_true", "s_now", "s_next")
    }
    by_phi = {phi: {"fw": [], "ag": [], "s": []} for phi in reported}
    for p in paths:
        states = (
            simulate_path(dgp, sim.n_periods, 1, cfg.seeds.simulation, p).states - 1
        )
        est = load_result(result_file(folder, FRAMEWORK, p))
        orc = load_result(result_file(folder, "oracle", p))
        ag = load_result(result_file(folder, "agnostic", p))
        dates = est.dates
        pooled["now_est"].append(
            align_probabilities(est.filtered, est.regime_means, dgp.factor_means)
        )
        pooled["fc_est"].append(
            align_probabilities(est.forecast, est.regime_means, dgp.factor_means)
        )
        pooled["now_true"].append(orc.filtered)
        pooled["fc_true"].append(orc.forecast)
        pooled["s_now"].append(states[dates])
        pooled["s_next"].append(states[dates + 1])
        for phi in reported:
            i = list(est.phis).index(float(phi))
            by_phi[phi]["fw"].append(est.returns[i])
            by_phi[phi]["ag"].append(ag.returns[i])
            by_phi[phi]["s"].append(states[dates + 1])
    cat = {k: np.concatenate(v) for k, v in pooled.items()}
    quality = pd.DataFrame(
        {
            "Nowcast, known parameters": forecast_scores(cat["now_true"], cat["s_now"]),
            "Nowcast, estimated": forecast_scores(cat["now_est"], cat["s_now"]),
            "One-step forecast, known parameters": forecast_scores(
                cat["fc_true"], cat["s_next"]
            ),
            "One-step forecast, estimated": forecast_scores(
                cat["fc_est"], cat["s_next"]
            ),
        }
    ).T
    quality.to_parquet(out / "forecast_quality.parquet")

    rows = []
    for phi in reported:
        res = regime_conditional(
            np.concatenate(by_phi[phi]["fw"]),
            np.concatenate(by_phi[phi]["ag"]),
            np.concatenate(by_phi[phi]["s"]),
            dgp.n_regimes,
        )
        for regime, values in res.items():
            rows.append({"phi": float(phi), "regime": regime, **values})
    conditional = pd.DataFrame(rows)
    conditional.to_parquet(out / "regime_conditional.parquet")

    write_run_record(
        out / "run.json",
        "07_inference",
        cfg,
        summary={
            "lw_reject_favorable": {
                f"{r.benchmark} {r.phi:g}": r["Reject (favorable)"]
                for _, r in significance.iterrows()
            },
            "forecast_hit_rate": quality["Hit rate"].to_dict(),
        },
    )
    pd.set_option("display.width", 200)
    print(
        significance[
            [
                "benchmark",
                "phi",
                "Delta",
                "t-stat",
                "Reject (favorable)",
                "Reject (adverse)",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print(quality.round(3).to_string())
    print(detection.round(3).to_string(index=False))
    print(f"Written: {out}")


def _lw(a, b, block, draws, seed, path):
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(path,)))
    return ledoit_wolf(a, b, block=block, draws=draws, rng=rng)


if __name__ == "__main__":
    main()
