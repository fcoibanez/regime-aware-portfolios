# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Table 2 and Figure 4: out-of-sample performance
#
# Means across simulated paths of the statistics computed in stage 6, for the
# framework and the five benchmarks, at the reported levels of risk aversion;
# and the distribution of the Sharpe ratio across paths. The same table is
# written for each appendix study whose results exist.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from regime_aware_portfolios import exhibits, figures, load_config
from regime_aware_portfolios.paths import CONFIG, OUTPUT, RESULTS

figures.style()
TABLES, FIGURES = OUTPUT / "tables", OUTPUT / "figures"


def load(study):
    cfg = load_config(CONFIG / f"{study}.toml")
    path = RESULTS / cfg.name / "metrics" / "path_statistics.parquet"
    return cfg, (pd.read_parquet(path) if path.exists() else None)


# %% [markdown]
# ## Main study

# %%
cfg, stats = load("main")
columns = exhibits.table_columns(cfg.strategies.rolling_window)
phis = cfg.strategies.reported_risk_aversion
text = exhibits.performance_table(stats, columns, phis)
exhibits.write(text, TABLES / "performance.tex")
print(text)

# %%
fig = figures.ecdf_figure(stats, list(columns), phis)
fig.savefig(FIGURES / "benchmark_ecdf.pdf")
plt.show()

# %% [markdown]
# ## Online Appendix studies

# %%
for study, prefix in exhibits.APPENDIX_PREFIX.items():
    cfg, stats = load(study)
    if stats is None:
        print(f"{study}: no results yet")
        continue
    columns = exhibits.table_columns(cfg.strategies.rolling_window)
    text = exhibits.performance_table(
        stats, columns, cfg.strategies.reported_risk_aversion
    )
    exhibits.write(text, TABLES / "appendix" / f"{prefix}_performance.tex")
    print(study)
    print(text)
