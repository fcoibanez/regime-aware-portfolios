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
`manifest.json` next to its results.

| Stage | Command | Writes |
|---|---|---|
| 1. Inputs | `python scripts/01_data.py` | `data/derived/` |

Stage 1 needs the CRSP extract, which is licensed and not included; see
[data/README.md](data/README.md) for how to download it from WRDS.

## Status

Under construction. The stages, notebooks and the map from each exhibit to the
code that produces it will be added here as they are completed.

## License

The code is released under the MIT License. The data are not covered by it; see
[data/README.md](data/README.md).
