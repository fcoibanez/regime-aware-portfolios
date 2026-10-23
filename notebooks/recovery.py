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

# %%
true, estimated = loadings["true"], loadings["estimated"]
for k, factor in enumerate(loadings["factors"]):
    t = true[:, :, k, :].reshape(-1, true.shape[3])
    e = estimated[:, :, k, :].reshape(-1, true.shape[3])
    fig = figures.recovery_figure(t, e, cfg.regime_labels)
    name = {"Mkt-RF": "mktrf", "Mom": "umd"}.get(str(factor), str(factor).lower())
    fig.savefig(OUTPUT / "figures" / f"recovery_{name}.pdf")
    plt.show()

# %% [markdown]
# Correlation and root-mean-squared error of estimated and true loadings.

# %%
summary.pivot(index="factor", columns="regime", values="Correlation").round(3)

# %%
summary.pivot(index="factor", columns="regime", values="RMSE").round(3)
