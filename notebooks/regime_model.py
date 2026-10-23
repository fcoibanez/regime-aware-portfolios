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
from regime_aware_portfolios.regimes import load_regimes

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
