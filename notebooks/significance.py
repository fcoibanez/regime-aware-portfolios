# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Table 3 and the Online Appendix tests
#
# Sharpe ratios gross and net of trading costs, the paired test of the net
# Sharpe ratio across paths, and how often the Ledoit and Wolf (2008) test
# rejects on a single path (stage 7). Also: how that rejection rate grows with
# the length of the history, and paired tests of every measure of Table 2
# against the regime-agnostic and the equally weighted portfolios.

# %%
import json

import numpy as np
import pandas as pd

from regime_aware_portfolios import exhibits, load_config
from regime_aware_portfolios.paths import CONFIG, DERIVED, OUTPUT, RESULTS

TABLES = OUTPUT / "tables"


def sharpe(r):
    """Annualized Sharpe ratio of each row."""
    return r.mean(axis=1) / r.std(axis=1, ddof=1) * np.sqrt(12)


COST = json.loads((DERIVED / "costs.json").read_text(encoding="utf-8"))["one_way"]


def load(study):
    cfg = load_config(CONFIG / f"{study}.toml")
    folder = RESULTS / cfg.name
    if not (folder / "inference" / "significance.parquet").exists():
        return cfg, None, None, None
    return (
        cfg,
        pd.read_parquet(folder / "metrics" / "path_statistics.parquet"),
        pd.read_parquet(folder / "inference" / "significance.parquet"),
        pd.read_parquet(folder / "inference" / "detection.parquet"),
    )


# %% [markdown]
# ## Main study

# %%
cfg, stats, significance, detection = load("main")
columns = exhibits.table_columns(cfg.strategies.rolling_window)
phis = cfg.strategies.reported_risk_aversion
text = exhibits.significance_table(
    stats, significance, "framework", columns, phis, COST
)
exhibits.write(text, TABLES / "significance.tex")
print(text)

# %% [markdown]
# Rejections by direction: in the framework's favor, and in the benchmark's.

# %%
significance.set_index(["benchmark", "phi"])[
    ["Delta", "t-stat", "Reject (favorable)", "Reject (adverse)", "Median p-value"]
].round(3)

# %%
text = exhibits.detection_table(
    detection, {"agnostic": "Agnostic", "costa_kwon": r"C\&K"}
)
exhibits.write(text, TABLES / "detection_horizon.tex")
print(text)

# %% [markdown]
# Why a single history cannot reveal the difference from the regime-agnostic
# benchmark: the correlation of the two strategies' monthly net returns, and the
# mean and the spread across paths of the difference in their net Sharpe ratios.

# %%
returns = pd.read_parquet(RESULTS / cfg.name / "metrics" / "returns.parquet")
months = [c for c in returns.columns if c.startswith("m")]
net = returns[returns["kind"] == "net"].set_index(["strategy", "phi", "path"])[months]
rows = {}
for benchmark in ("agnostic", "costa_kwon"):
    for phi in phis:
        a = net.loc[("framework", float(phi))].to_numpy()
        b = net.loc[(benchmark, float(phi))].to_numpy()
        corr = np.mean([np.corrcoef(x, y)[0, 1] for x, y in zip(a, b, strict=True)])
        gap = sharpe(a) - sharpe(b)
        rows[(benchmark, phi)] = {
            "Return correlation": corr,
            "Mean Sharpe gap": gap.mean(),
            "Spread of the gap": gap.std(ddof=1),
        }
pd.DataFrame(rows).T.round(3)

# %% [markdown]
# ## Every measure, paired (Online Appendix)

# %%
for benchmark, name in (("agnostic", "agnostic"), ("equal_weight", "ew")):
    text = exhibits.paired_measures_table(stats, "framework", benchmark, phis)
    exhibits.write(text, TABLES / "appendix" / f"significance_{name}.tex")
    print(text)

# %% [markdown]
# ## Online Appendix studies

# %%
for study, prefix in exhibits.APPENDIX_PREFIX.items():
    cfg, stats, significance, _ = load(study)
    if stats is None:
        print(f"{study}: no results yet")
        continue
    columns = exhibits.table_columns(cfg.strategies.rolling_window)
    text = exhibits.significance_table(
        stats,
        significance,
        "framework",
        columns,
        cfg.strategies.reported_risk_aversion,
        COST,
    )
    exhibits.write(text, TABLES / "appendix" / f"{prefix}_significance.tex")
    print(study)
    print(text)
