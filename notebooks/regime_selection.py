# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Figure 1: the number of regimes
#
# Information criteria of models with one to eight regimes, on the full sample
# and on three consecutive sub-periods of equal length (stage 8). Each model is
# the best of 1,000 cold starts. The table lists the number of regimes each
# criterion selects, for the six factors and, where stage 8 was also run with
# `--costa-kwon`, for the market factor alone.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from regime_aware_portfolios import exhibits, figures, load_config
from regime_aware_portfolios.paths import CONFIG, OUTPUT, RESULTS

figures.style()
cfg = load_config(CONFIG / "main.toml")
folder = RESULTS / cfg.name / "regime_selection"
criteria = pd.read_parquet(folder / "criteria.parquet")

# %%
fig = figures.selection_figure(criteria)
fig.savefig(OUTPUT / "figures" / "state_calibration.pdf")
plt.show()


# %%
def selected(table):
    return table.groupby("sample", sort=False).apply(
        lambda g: g.set_index("n_regimes")[["BIC", "ICL", "HQIC", "AIC"]].idxmin()
    )


text = exhibits.state_selection_table(selected(criteria))
exhibits.write(text, OUTPUT / "tables" / "state_selection.tex")
print(text)

# %% [markdown]
# Differences in BIC between neighboring numbers of regimes; a negative entry
# favors the larger model.

# %%
bic = criteria.pivot(index="sample", columns="n_regimes", values="BIC")
bic.diff(axis=1).round(1)

# %% [markdown]
# ## The market factor alone

# %%
market = folder / "criteria_costa_kwon.parquet"
if market.exists():
    table = pd.read_parquet(market)
    print(selected(table))
    print(
        table.pivot(index="sample", columns="n_regimes", values="BIC")
        .diff(axis=1)
        .round(1)
    )
