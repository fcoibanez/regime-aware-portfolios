# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Online Appendix: performance across trading costs
#
# Mean net Sharpe ratio across paths as a function of the one-way cost charged
# per unit of two-sided turnover, from zero to 3%. The vertical line is the
# cost used in the paper, half the median quoted spread since decimalization;
# the band reaches down to half the market-value-weighted mean spread (stage 1).
# Below the figure, each benchmark's break-even cost against the framework.

# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regime_aware_portfolios import exhibits, figures, load_config
from regime_aware_portfolios.paths import CONFIG, DERIVED, OUTPUT, RESULTS

figures.style()
cfg = load_config(CONFIG / "main.toml")
costs = json.loads((DERIVED / "costs.json").read_text(encoding="utf-8"))
returns = pd.read_parquet(RESULTS / cfg.name / "metrics" / "returns.parquet")
months = [c for c in returns.columns if c.startswith("m")]
by_kind = {
    kind: returns[returns["kind"] == kind].set_index(["strategy", "phi", "path"])[
        months
    ]
    for kind in ("gross", "turnover")
}
strategies = list(exhibits.table_columns(cfg.strategies.rolling_window))
phis = cfg.strategies.reported_risk_aversion
grid = np.linspace(0, 0.03, 61)

# %%
fig = figures.cost_figure(
    by_kind["gross"],
    by_kind["turnover"],
    strategies,
    phis,
    grid,
    costs["one_way"],
    costs["one_way_cap_weighted"],
)
fig.savefig(OUTPUT / "figures" / "cost_sensitivity.pdf")
plt.show()


# %% [markdown]
# ## Break-even costs
#
# The cost at which the mean net Sharpe ratio of a benchmark reaches the
# framework's (the average path), and the median across paths of each path's own
# break-even cost. Paths on which the framework trails before costs, or on which
# it trades less than the benchmark, have no break-even and are left out of the
# median; the share of paths with one is reported.


# %%
def net_sharpe(r, u, c):
    u = u.copy()
    u[:, 0] = 0.0
    net = r - c * u
    return net.mean(axis=1) / net.std(axis=1, ddof=1) * np.sqrt(12)


fine = np.linspace(0, 0.10, 2001)
rows = []
for phi in phis:
    r0 = by_kind["gross"].loc[("framework", float(phi))].to_numpy()
    u0 = by_kind["turnover"].loc[("framework", float(phi))].to_numpy()
    for s in strategies[1:]:
        r1 = by_kind["gross"].loc[(s, float(phi))].to_numpy()
        u1 = by_kind["turnover"].loc[(s, float(phi))].to_numpy()
        gaps = np.array([net_sharpe(r0, u0, c) - net_sharpe(r1, u1, c) for c in fine])
        mean_gap = gaps.mean(axis=1)
        crossing = fine[np.argmax(mean_gap <= 0)] if (mean_gap <= 0).any() else np.nan
        ahead = gaps[0] > 0
        crosses = ahead & (gaps <= 0).any(axis=0)
        path_cost = fine[np.argmax(gaps <= 0, axis=0)]
        rows.append(
            {
                "phi": phi,
                "benchmark": s,
                "Average path": crossing,
                "Median of paths": np.median(path_cost[crosses])
                if crosses.any()
                else np.nan,
                "Share of paths with one": crosses.mean(),
            }
        )
pd.DataFrame(rows).set_index(["phi", "benchmark"]).round(4)
