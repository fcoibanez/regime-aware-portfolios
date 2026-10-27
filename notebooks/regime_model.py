# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Table 1, the appendix regime models, and the momentum crashes
#
# The regime model of each study (stage 2): monthly factor means and
# volatilities by regime, and the transition probabilities. Regimes are numbered
# by the volatility of the market factor, calmest first, and named as in
# `[exhibits] regime_labels` of each study's configuration. Then the worst months
# for momentum, with the smoothed probability of each regime of the main model.

# %%
import numpy as np
import pandas as pd

from regime_aware_portfolios import exhibits, load_config
from regime_aware_portfolios.paths import CONFIG, DERIVED, OUTPUT, RESULTS
from regime_aware_portfolios.regimes import filtered_probabilities, load_regimes

TABLES = OUTPUT / "tables"


def parameters(study):
    cfg = load_config(CONFIG / f"{study}.toml")
    r = load_regimes(RESULTS / cfg.name / "regimes")
    vols = pd.DataFrame(
        np.sqrt(np.diagonal(r.covariances, axis1=1, axis2=2)),
        index=r.means.index,
        columns=r.means.columns,
    )
    return cfg, r, vols


# %% [markdown]
# ## Table 1

# %%
cfg, regimes, vols = parameters("main")
text = exhibits.regime_parameter_table(
    regimes.means, vols, regimes.transmat, cfg.regime_labels
)
exhibits.write(text, TABLES / "regime_model.tex")
print(text)
print(f"Log-likelihood {regimes.loglike:.2f}")
print("Expected duration (months):")
print((1 / (1 - np.diag(regimes.transmat.to_numpy()))).round(2))

# %% [markdown]
# ## Online Appendix models

# %%
for study, prefix in exhibits.APPENDIX_PREFIX.items():
    c, r, v = parameters(study)
    panels = exhibits.regime_panels(r.means, v, r.transmat, c.regime_labels)
    for name, text in zip(("means", "vols", "transitions"), panels, strict=True):
        exhibits.write(text, TABLES / "appendix" / f"{prefix}_hmm_{name}.tex")
    print(study, f"log-likelihood {r.loglike:.2f}")
    print("\n".join(panels))

# %% [markdown]
# ## The worst months for momentum
#
# Months in the first percentile of the momentum factor, with the smoothed
# regime probabilities of the main model.

# %%
factors = pd.read_parquet(DERIVED / "factors.parquet").loc[
    regimes.posteriors.index, cfg.factor_names
]
text = exhibits.momentum_table(factors, regimes.posteriors, cfg.regime_labels)
exhibits.write(text, TABLES / "appendix" / "momentum_crashes.tex")
print(text)

# %% [markdown]
# The assignment of these months to the turbulent regime is close to
# definitional: the regime takes most of the extreme months of every factor.
# Share of each factor's months in its lowest and highest percentile whose most
# likely regime is the most volatile one:

# %%
turbulent = cfg.regimes.n_regimes
modal = regimes.posteriors.idxmax(axis=1)
tails = {}
for f in cfg.factor_names:
    lo, hi = factors[f].quantile(0.01), factors[f].quantile(0.99)
    extreme = (factors[f] <= lo) | (factors[f] >= hi)
    tails[exhibits.FACTOR_LABELS.get(f, f)] = (modal[extreme] == turbulent).mean()
pd.Series(tails).round(2)

# %% [markdown]
# What is not definitional: the forecast probability of the turbulent regime
# for each crash month, from the data up to the month before, and the market's
# cumulative excess return over the preceding year.

# %%
_, forecast = filtered_probabilities(regimes, factors)
mom = factors["Mom"]
worst = mom.index[mom <= mom.quantile(0.01)]
crashes = pd.DataFrame(
    {
        "Forecast probability": forecast.loc[worst, turbulent].to_numpy(),
        "Market, prior 12 months": [
            factors["Mkt-RF"].iloc[i - 12 : i].sum()
            for i in factors.index.get_indexer(worst)
        ],
    },
    index=worst.strftime("%Y-%m"),
)
print(crashes.round(3))
print(
    f"Mean forecast {crashes['Forecast probability'].mean():.3f}, "
    f"unconditional {forecast[turbulent].mean():.3f}"
)

# %% [markdown]
# ## The external checks for the four-regime model (Online Appendix B)
#
# Mean posterior probability of each regime inside and outside NBER recessions,
# and the regime probabilities in the worst months for momentum.

# %%
four = load_regimes(RESULTS / "four_regimes" / "regimes")
usrec = pd.read_parquet(DERIVED / "usrec.parquet")["usrec"].reindex(
    four.posteriors.index
)
pd.DataFrame(
    {
        "Recessions": four.posteriors[usrec == 1].mean(),
        "Expansions": four.posteriors[usrec == 0].mean(),
        "Occupancy": four.posteriors.mean(),
    }
).round(3)

# %%
four.posteriors.loc[worst].round(3)
