# regime-aware-portfolios

Replication code for the study of regime-aware portfolio construction with
regime-weighted least squares. The estimator is the
[rwls](https://pypi.org/project/rwls/) package; this repository holds everything
else: data preparation, calibration of the simulation, the simulation itself, the
strategies and their backtests, the evaluation, and one notebook per table and
figure of the paper.

## Layout

```
config/          every parameter of the study, one file per study
  main.toml        the main text
  alt_calibration.toml, four_regimes.toml   the Online Appendix studies
data/            inputs (raw) and rebuilt intermediate files
src/regime_aware_portfolios/   the study's code, imported by scripts and notebooks
scripts/         the computing stages, run in order
notebooks/       one notebook per table and figure; they only read results
output/          the figures, tables and numbers the paper includes
tests/
```

No module defines its own constants: every value comes from a file in `config/`,
read and checked by `regime_aware_portfolios.config`. Each run records the
configuration and package versions that produced it.

## Environment

```
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
```

`requirements.lock` pins the exact version of every package used for the
results in the paper. The code runs on Python 3.11 or later; the results were
produced with Python 3.13.

## Running

The stages run in order from the repository's root folder. Each writes a
`run.json` next to its results.

| Stage | Command | Writes |
|---|---|---|
| 1. Inputs | `python scripts/01_data.py` | `data/derived/` |
| 2. Regime model | `python scripts/02_regimes.py --config config/<study>.toml` | `results/<study>/regimes/` |
| 3. Calibration | `python scripts/03_calibrate.py --config config/<study>.toml` | `results/<study>/calibration/` |
| 4. Simulation | `python scripts/04_simulate.py --config config/<study>.toml` | `results/<study>/simulation/` |
| 5. Backtests | `python scripts/05_backtest.py --config config/<study>.toml` | `results/<study>/backtest/` |
| 6. Performance statistics | `python scripts/06_metrics.py --config config/<study>.toml` | `results/<study>/metrics/` |
| 7. Inference | `python scripts/07_inference.py --config config/<study>.toml` | `results/<study>/inference/` |
| 8. Number of regimes | `python scripts/08_regime_selection.py` (and `--costa-kwon`) | `results/main/regime_selection/` |
| 9. Recovery of the loadings | `python scripts/09_recovery.py` | `results/main/recovery/` |

Stage 1 needs the CRSP extract, which is licensed and not included; see
[data/README.md](data/README.md) for how to download it from WRDS. Stage 2 and
later stages run once per study, `main` by default. Simulated paths are not
stored: each is generated from the study's seed and its number, in a few
milliseconds, by `regime_aware_portfolios.simulation.simulate_path`.

The studies are `main`, `alt_calibration` and `four_regimes`. Stages 2 to 7
run for each; stages 8 and 9 for `main` only. Stage 5 is the long one, about an
hour per study on a 32-thread processor, and can be resumed: it skips the
backtests already on disk.

## Tables and figures

Each notebook in `notebooks/` reads the results of the stages and writes
exhibits to `output/tables` and `output/figures`, under the file names the paper
uses. The notebooks are Python files in the percent format of
[jupytext](https://jupytext.readthedocs.io); they run as scripts, or open as
notebooks once jupytext is installed (`pip install -e .[notebooks]`).

| Exhibit | Notebook | Output | Stages |
|---|---|---|---|
| Section 4.1, the calibration sample | `sample.py` | (printed counts) | 1 |
| Table 1, regime model | `regime_model.py` | `tables/regime_model.tex` | 2 |
| Figure 1, number of regimes | `regime_selection.py` | `figures/state_calibration.pdf`, `tables/state_selection.tex` | 8 |
| Figure 2, regimes and recessions | `regime_timeline.py` | `figures/regime_timeline.pdf` | 1, 2 |
| Figure 3, recovery of the loadings | `recovery.py` | `figures/recovery_<factor>.pdf` | 9 |
| Figure 4, Sharpe ratios across paths | `performance.py` | `figures/benchmark_ecdf.pdf` | 6 |
| Table 2, performance | `performance.py` | `tables/performance.tex` | 6 |
| Table 3, significance and costs | `significance.py` | `tables/significance.tex`, `tables/detection_horizon.tex` | 7 |
| Table 4, design choices | `decomposition.py` | `tables/decomposition.tex`, `tables/objective.tex` | 6 |
| Table 5, regime-conditional performance | `regime_conditional.py` | `tables/regime_conditional.tex`, `tables/forecast_quality.tex` | 7 |
| Online Appendix A, alternative calibration | `regime_model.py`, `performance.py`, `significance.py` | `tables/appendix/alt_*.tex` | 2 to 7 |
| Online Appendix B, four regimes | `regime_model.py`, `performance.py`, `significance.py`, `decomposition.py` | `tables/appendix/m4_*.tex` | 2 to 7 |
| Online Appendix, momentum crashes | `regime_model.py` | `tables/appendix/momentum_crashes.tex` | 2 |
| Online Appendix, trading costs | `costs.py` | `figures/cost_sensitivity.pdf` | 1, 6 |
| Online Appendix, paired tests of every measure | `significance.py` | `tables/appendix/significance_{agnostic,ew}.tex` | 6 |
| Online Appendix, rolling window | `rolling_window.py` | `figures/rolling_window_sensitivity.pdf` | 6 |
| Numbers quoted in the text | `quoted_numbers.py` | `numbers.tex` | all |

## License

The code is released under the MIT License. The data are not covered by it; see
[data/README.md](data/README.md).
