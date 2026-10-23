# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Table 4: contribution of each design choice
#
# Panel A walks from the estimator of Costa and Kwon (2020) to the RWLS
# framework one change at a time, under the mean--variance objective; each row is
# the paired change in Sharpe ratio across paths, so the rows add up to the total.
# Panel B compares the mean--variance objective with the expected utility of the
# full mixture, with the regime parameters estimated and known. The steps are
# `exhibits.DECOMPOSITION_STEPS` and the strategies are defined in
# `regime_aware_portfolios.strategies`.

# %%
import pandas as pd

from regime_aware_portfolios import exhibits, load_config
from regime_aware_portfolios.paths import CONFIG, OUTPUT, RESULTS

TABLES = OUTPUT / "tables"


def write_tables(study, folder):
    cfg = load_config(CONFIG / f"{study}.toml")
    path = RESULTS / cfg.name / "metrics" / "path_statistics.parquet"
    if not path.exists():
        print(f"{study}: no results yet")
        return
    stats = pd.read_parquet(path)
    phis = cfg.strategies.reported_risk_aversion
    prefix = "" if study == "main" else exhibits.APPENDIX_PREFIX[study] + "_"
    for name, text in (
        (
            "decomposition",
            exhibits.decomposition_table(stats, exhibits.DECOMPOSITION_STEPS, phis),
        ),
        ("objective", exhibits.objective_table(stats, exhibits.OBJECTIVE_PAIRS, phis)),
    ):
        exhibits.write(text, folder / f"{prefix}{name}.tex")
        print(text)


# %%
write_tables("main", TABLES)

# %% [markdown]
# ## Four regimes (Online Appendix)

# %%
write_tables("four_regimes", TABLES / "appendix")
