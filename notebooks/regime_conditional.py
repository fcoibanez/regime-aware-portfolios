# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Table 5: regime-conditional performance and the regime probabilities
#
# Panel A compares the framework with the regime-agnostic benchmark in the months
# governed by each true regime, pooled across paths. Panel B scores the regime
# probabilities against the true regimes: the filtered probabilities (nowcast)
# and the one-step forecast, with the parameters of the data-generating process
# and with the parameters re-estimated at each date. Both are computed in stage 7.

# %%
import pandas as pd

from regime_aware_portfolios import exhibits, load_config
from regime_aware_portfolios.paths import CONFIG, OUTPUT, RESULTS

TABLES = OUTPUT / "tables"
cfg = load_config(CONFIG / "main.toml")
folder = RESULTS / cfg.name / "inference"
conditional = pd.read_parquet(folder / "regime_conditional.parquet")
quality = pd.read_parquet(folder / "forecast_quality.parquet")

# %%
text = exhibits.regime_conditional_table(
    conditional, cfg.regime_labels, cfg.strategies.reported_risk_aversion
)
exhibits.write(text, TABLES / "regime_conditional.tex")
print(text)

# %%
text = exhibits.forecast_quality_table(quality)
exhibits.write(text, TABLES / "forecast_quality.tex")
print(text)
