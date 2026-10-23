# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Numbers quoted in the text
#
# Every number the text of the paper quotes from the results, as a LaTeX command
# in `output/numbers.tex`. The paper includes that file and writes, for example,
# `\SharpeFrameworkTen` where it quotes the framework's Sharpe ratio at
# $\varphi = 10$, so that the text cannot fall out of step with the tables.
# Names use words for numbers, since LaTeX command names have letters only.

# %%
import json

import numpy as np
import pandas as pd

from regime_aware_portfolios import exhibits, load_config
from regime_aware_portfolios.paths import CONFIG, DERIVED, OUTPUT, RESULTS

cfg = load_config(CONFIG / "main.toml")
study = RESULTS / cfg.name
phis = cfg.strategies.reported_risk_aversion
WORDS = {1: "One", 5: "Five", 10: "Ten", 25: "TwentyFive", 50: "Fifty"}
NAMES = {
    "framework": "Framework",
    "agnostic": "Agnostic",
    f"rolling_{cfg.strategies.rolling_window}": "Rolling",
    "equal_weight": "EqualWeight",
    "minimum_variance": "MinimumVariance",
    "costa_kwon": "CostaKwon",
}
values = {}


def pct(x, digits=2):
    return f"{100 * x:.{digits}f}\\%"


# %% [markdown]
# ## Data and costs

# %%
data = json.loads((DERIVED / "run.json").read_text(encoding="utf-8"))["summary"]
costs = json.loads((DERIVED / "costs.json").read_text(encoding="utf-8"))
values |= {
    "SampleMonths": f"{data['months']:,}".replace(",", "{,}"),
    "SampleStocks": f"{data['stocks']:,}".replace(",", "{,}"),
    "RecessionMonths": str(data["recession_months"]),
    "OneWayCost": pct(costs["one_way"], 4),
    "OneWayCostCapWeighted": pct(costs["one_way_cap_weighted"], 4),
}

# %% [markdown]
# ## The regime model

# %%
record = json.loads((study / "regimes" / "regimes.json").read_text(encoding="utf-8"))
transmat = np.array(record["transmat"])
values |= {
    "LogLikelihood": f"{record['loglike']:,.2f}".replace(",", "{,}"),
    "StartsReached": str(record["starts"]["reached"]),
    "StartsDistinct": str(record["starts"]["distinct"]),
}
for s, label in enumerate(cfg.regime_labels):
    values[f"Persistence{label}"] = f"{transmat[s, s]:.2f}"
    values[f"Duration{label}"] = f"{1 / (1 - transmat[s, s]):.2f}"

posteriors = pd.read_parquet(study / "regimes" / "posteriors.parquet")
recession = pd.read_parquet(DERIVED / "usrec.parquet")["usrec"].reindex(
    posteriors.index
)
turbulent = posteriors.iloc[:, -1]
inside, outside = turbulent[recession == 1].mean(), turbulent[recession == 0].mean()
values |= {
    "RecessionProbInside": f"{inside:.2f}",
    "RecessionProbOutside": f"{outside:.2f}",
    "RecessionProbRatio": f"{inside / outside:.2f}",
}

# %% [markdown]
# ## Performance

# %%
stats = pd.read_parquet(study / "metrics" / "path_statistics.parquet")
means = stats.groupby(["strategy", "phi"]).mean(numeric_only=True)
significance = pd.read_parquet(study / "inference" / "significance.parquet")
sig = significance.set_index(["benchmark", "phi"])
for phi in phis:
    w = WORDS[int(phi)]
    for s, name in NAMES.items():
        row = means.loc[(s, float(phi))]
        values[f"Sharpe{name}{w}"] = f"{row['Sharpe Ratio']:.2f}"
        values[f"SharpeNet{name}{w}"] = f"{row['Sharpe Ratio (net)']:.2f}"
        values[f"CE{name}{w}"] = pct(row["Certainty Equivalent"])
        values[f"Volatility{name}{w}"] = pct(row["Ann. Std. Deviation"])
        values[f"Turnover{name}{w}"] = f"{row['Portfolio Turnover']:.2f}"
        if s == "framework":
            continue
        r = sig.loc[(s, float(phi))]
        values[f"DeltaSharpe{name}{w}"] = f"{r['Delta']:.3f}"
        values[f"TStat{name}{w}"] = f"{r['t-stat']:.2f}"
        values[f"LWReject{name}{w}"] = pct(
            r["Reject (favorable)"] + r["Reject (adverse)"], 1
        )
        values[f"LWRejectAdverse{name}{w}"] = pct(r["Reject (adverse)"], 1)

detection = pd.read_parquet(study / "inference" / "detection.parquet")
values["DetectionFiveYears"] = pct(detection["Detection rate"].iloc[0], 1)
values["DetectionLongest"] = pct(detection["Detection rate"].iloc[-1], 1)
values["DetectionLongestYears"] = f"{detection['Years'].iloc[-1]:.0f}"

# %% [markdown]
# ## Recovery and the regime probabilities

# %%
recovery = pd.read_parquet(study / "recovery" / "summary.parquet")
quality = pd.read_parquet(study / "inference" / "forecast_quality.parquet")
nowcast = quality.loc["Nowcast, estimated"]
forecast = quality.loc["One-step forecast, estimated"]
values |= {
    "RecoveryCorrelationMin": f"{recovery['Correlation'].min():.2f}",
    "RecoveryCorrelationMax": f"{recovery['Correlation'].max():.2f}",
    "HitRateNowcast": f"{nowcast['Hit rate']:.3f}",
    "HitRateForecast": f"{forecast['Hit rate']:.3f}",
    "HitRateConstant": f"{forecast['Constant forecast']:.3f}",
}

# %%
text = exhibits.macros(values)
exhibits.write(text, OUTPUT / "numbers.tex")
print(text)
