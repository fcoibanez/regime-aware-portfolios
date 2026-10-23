# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Online Appendix: length of the rolling window
#
# The paired difference in Sharpe ratio between the framework and the rolling
# OLS benchmark, for each window in `[strategies] rolling_window_sensitivity`,
# with 95% confidence intervals. Returns are before trading costs.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from regime_aware_portfolios import figures, load_config
from regime_aware_portfolios.inference import paired_comparison
from regime_aware_portfolios.paths import CONFIG, OUTPUT, RESULTS

figures.style()
cfg = load_config(CONFIG / "main.toml")
stats = pd.read_parquet(RESULTS / cfg.name / "metrics" / "path_statistics.parquet")
sharpe = stats.set_index(["strategy", "phi", "path"])["Sharpe Ratio"]

rows = []
for phi in cfg.strategies.reported_risk_aversion:
    for window in cfg.strategies.rolling_window_sensitivity:
        r = paired_comparison(
            sharpe.loc[("framework", float(phi))],
            sharpe.loc[(f"rolling_{window}", float(phi))],
        )
        rows.append(
            {
                "phi": phi,
                "window": window,
                **r,
                "Rolling Sharpe": r["Mean (control)"],
            }
        )
differences = pd.DataFrame(rows)

# %%
fig = figures.rolling_window_figure(differences)
fig.savefig(OUTPUT / "figures" / "rolling_window_sensitivity.pdf")
plt.show()

# %%
differences.set_index(["phi", "window"])[
    ["Rolling Sharpe", "Delta", "Std. Err.", "t-stat", "Frac. > 0"]
].round(3)
