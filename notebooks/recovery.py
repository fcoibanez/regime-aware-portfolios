# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Figure 3: recovery of the loadings
#
# On the in-sample months of every simulated path, the regime model is fitted to
# the factor returns and RWLS to the stock returns, with nothing known about the
# data-generating process, and the estimated loadings are compared with the true
# ones (stage 9). One panel per factor; regimes are matched to the true regimes
# by their factor means.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regime_aware_portfolios import figures, load_config
from regime_aware_portfolios.paths import CONFIG, OUTPUT, RESULTS

figures.style()
cfg = load_config(CONFIG / "main.toml")
folder = RESULTS / cfg.name / "recovery"
loadings = np.load(folder / "loadings.npz")
summary = pd.read_parquet(folder / "summary.parquet")

# %% [markdown]
# Each panel plots 1,000 stocks: the 200 of each of the first five paths.
# The summary statistics below use every path.

# %%
SHOWN_PATHS = 5
true, estimated = loadings["true"], loadings["estimated"]
for k, factor in enumerate(loadings["factors"]):
    t = true[:SHOWN_PATHS, :, k, :].reshape(-1, true.shape[3])
    e = estimated[:SHOWN_PATHS, :, k, :].reshape(-1, true.shape[3])
    fig = figures.recovery_figure(t, e)
    name = {"Mkt-RF": "mktrf", "Mom": "umd"}.get(str(factor), str(factor).lower())
    figures.save(
        fig, OUTPUT / "figures" / f"recovery_{name}.pdf", dpi=50, transparent=False
    )
    plt.show()

# %% [markdown]
# Correlation and root-mean-squared error of estimated and true loadings.

# %%
summary.pivot(index="factor", columns="regime", values="Correlation").round(3)

# %%
summary.pivot(index="factor", columns="regime", values="RMSE").round(3)

# %% [markdown]
# Mean error of the estimated loadings: close to zero against the RMSE, so the
# estimates are centered on the truth.

# %%
summary.pivot(index="factor", columns="regime", values="Mean error").round(3)
