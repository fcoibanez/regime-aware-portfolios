"""Replication code for the regime-aware portfolio study.

The estimator itself comes from the rwls package; this package holds the data
preparation, calibration, simulation, strategies, backtests and evaluation of the
study, configured from the files in ``config/``.
"""

from regime_aware_portfolios.config import StudyConfig, load_config

__all__ = ["StudyConfig", "load_config"]
