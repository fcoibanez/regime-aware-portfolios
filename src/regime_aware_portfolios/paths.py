"""Locations of the inputs and outputs of the study within the repository.

The package is installed in editable mode (see the README), so the repository is
found from the location of this file.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "CONFIG",
    "CRSP",
    "DERIVED",
    "FRENCH",
    "OUTPUT",
    "RAW",
    "RESULTS",
    "ROOT",
    "USREC",
]

ROOT = Path(__file__).resolve().parents[2]

CONFIG = ROOT / "config"
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"
RESULTS = ROOT / "results"
OUTPUT = ROOT / "output"

#: The processed CRSP monthly stock file.
CRSP = RAW / "crsp_monthly.parquet"
#: Zipped CSV files of the Kenneth R. French Data Library.
FRENCH = RAW / "french"
#: FRED's CSV file of the NBER recession indicator.
USREC = RAW / "usrec.csv"
