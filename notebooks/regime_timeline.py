# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Figure 2: the turbulent regime and NBER recessions
#
# The smoothed probability of the most volatile regime of the main model, with
# NBER recessions shaded (FRED series USREC: 1 from the month after each
# business-cycle peak through the trough). The probability is drawn as a
# six-month centered moving average; every statistic below uses the unsmoothed
# series. The sample is split across two rows so that episodes can be resolved.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regime_aware_portfolios import figures, load_config
from regime_aware_portfolios.paths import CONFIG, DERIVED, OUTPUT, RESULTS
from regime_aware_portfolios.regimes import filtered_probabilities, load_regimes

figures.style()
SMOOTH_MONTHS = 6
SPLIT = "1994-06-30"

cfg = load_config(CONFIG / "main.toml")
regimes = load_regimes(RESULTS / cfg.name / "regimes")
# Regimes are numbered by market volatility, so the last is the most volatile.
regime = cfg.regimes.n_regimes
label = cfg.regime_labels[regime - 1]
probability = regimes.posteriors[regime]
recession = pd.read_parquet(DERIVED / "usrec.parquet")["usrec"].reindex(
    probability.index
)

# %%
fig = figures.timeline_figure(probability, recession, label, SPLIT, SMOOTH_MONTHS)
figures.save(fig, OUTPUT / "figures" / "regime_timeline.pdf")
plt.show()

# %% [markdown]
# ## Statistics quoted in the text


# %%
def summary(p):
    inside = p[recession == 1].mean()
    outside = p[recession == 0].mean()
    return {
        "Peak": p.max(),
        "Mean, recessions": inside,
        "Mean, expansions": outside,
        "Ratio": inside / outside,
        "Episodes above one half": len(figures.recession_spans(p > 0.5)),
        "Recession months above one half": int(((p > 0.5) & (recession == 1)).sum()),
    }


smoothed = probability.rolling(SMOOTH_MONTHS, center=True).mean()
pd.DataFrame(
    {"Unsmoothed": summary(probability), "Smoothed": summary(smoothed.dropna())}
).round(3)

# %%
print(f"Recession months: {int(recession.sum())} of {len(recession)}")
spans = figures.recession_spans(recession.astype(bool))
for start, end in spans:
    p = probability[start:end]
    print(f"{start:%Y-%m} to {end:%Y-%m}: mean {p.mean():.2f}, max {p.max():.2f}")
print(np.round(probability.groupby(recession).describe(), 3))

# %% [markdown]
# ## Without hindsight, and as the most likely regime
#
# The posteriors above are smoothed: each uses the whole sample. The filtered
# probability uses only the data up to each month (with the parameters of the
# whole sample). The modal regime is the one with the highest posterior.

# %%
factors = pd.read_parquet(DERIVED / "factors.parquet").loc[probability.index]
filtered, forecast = filtered_probabilities(regimes, factors)
modal = regimes.posteriors.idxmax(axis=1) == regime
pd.Series(
    {
        "Filtered, recessions": filtered[regime][recession == 1].mean(),
        "Filtered, expansions": filtered[regime][recession == 0].mean(),
        "Modal in recession months": modal[recession == 1].mean(),
        "Modal in all months": modal.mean(),
    }
).round(3)

# %% [markdown]
# Correlation of the posterior with the recession indicator k months ahead
# (negative k: behind).

# %%
pd.Series(
    {k: probability.corr(recession.shift(-k)) for k in range(-6, 7)},
    name="correlation",
).round(3)

# %% [markdown]
# The longest episodes above one half that overlap no recession.

# %%
episodes = [
    (start, end, (end.to_period("M") - start.to_period("M")).n + 1)
    for start, end in figures.recession_spans(probability > 0.5)
    if not recession[start:end].any()
]
for start, end, months in sorted(episodes, key=lambda e: -e[2])[:6]:
    print(f"{start:%Y-%m} to {end:%Y-%m}: {months} months")
