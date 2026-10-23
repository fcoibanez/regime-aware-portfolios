"""Study configuration.

Every value the study uses is read from a TOML file in ``config/`` and held in the
frozen dataclasses below, which check it when it is loaded. Modules receive these
objects and define no constants of their own.

A file can extend another with ``extends = "main.toml"``: its sections are merged
into those of the file it extends, key by key.
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rwls

__all__ = ["StudyConfig", "load_config"]


class ConfigError(ValueError):
    """A configuration file has a missing, unknown or invalid value."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


@dataclass(frozen=True)
class DataConfig:
    start: str
    end: str
    factor_model: str
    crsp_start: str
    crsp_end: str

    def __post_init__(self) -> None:
        _require(self.start < self.end, "data.start must precede data.end")
        _require(
            self.crsp_start <= self.start and self.end <= self.crsp_end,
            "the CRSP months must cover data.start to data.end",
        )
        _require(
            self.factor_model in rwls.factors.MODELS,
            f"data.factor_model must be one of {sorted(rwls.factors.MODELS)}",
        )


@dataclass(frozen=True)
class SampleConfig:
    coverage: float
    min_history: int

    def __post_init__(self) -> None:
        _require(0 < self.coverage <= 1, "sample.coverage must be in (0, 1]")
        _require(self.min_history >= 1, "sample.min_history must be positive")


@dataclass(frozen=True)
class CalibrationConfig:
    loading_eigen_floor: float
    resid_eigen_floor: float
    min_resid_dof: float

    def __post_init__(self) -> None:
        _require(
            self.loading_eigen_floor > 0 and self.resid_eigen_floor > 0,
            "calibration eigenvalue floors must be positive",
        )
        _require(self.min_resid_dof > 0, "calibration.min_resid_dof must be positive")


@dataclass(frozen=True)
class RegimeConfig:
    n_regimes: int
    n_starts: int
    selection: str
    order_by: str
    reference: dict[str, tuple[float, ...]] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(self.n_regimes >= 1, "regimes.n_regimes must be positive")
        _require(self.n_starts >= 1, "regimes.n_starts must be positive")
        _require(
            self.selection in ("best", "modal", "nearest"),
            "regimes.selection must be 'best', 'modal' or 'nearest'",
        )
        reference = {k: tuple(v) for k, v in self.reference.items()}
        object.__setattr__(self, "reference", reference)
        _require(
            (self.selection == "nearest") == bool(reference),
            "regimes.reference is required with selection 'nearest', and only then",
        )
        _require(
            all(len(v) == self.n_regimes for v in reference.values()),
            "regimes.reference must give one mean per regime for each factor",
        )


@dataclass(frozen=True)
class RegimeSelectionConfig:
    n_regimes: tuple[int, ...]
    n_starts: int
    n_subperiods: int

    def __post_init__(self) -> None:
        _require(
            len(self.n_regimes) > 0 and min(self.n_regimes) >= 1,
            "regime_selection.n_regimes must list positive integers",
        )
        _require(self.n_starts >= 1, "regime_selection.n_starts must be positive")
        _require(
            self.n_subperiods >= 1, "regime_selection.n_subperiods must be positive"
        )


@dataclass(frozen=True)
class SimulationConfig:
    n_paths: int
    n_stocks: int
    n_in_sample: int
    n_out_of_sample: int

    def __post_init__(self) -> None:
        for name in ("n_paths", "n_stocks", "n_in_sample", "n_out_of_sample"):
            _require(getattr(self, name) >= 1, f"simulation.{name} must be positive")

    @property
    def n_periods(self) -> int:
        return self.n_in_sample + self.n_out_of_sample


@dataclass(frozen=True)
class StrategyConfig:
    risk_aversion: tuple[float, ...]
    reported_risk_aversion: tuple[float, ...]
    rolling_window: int
    rolling_window_sensitivity: tuple[int, ...]
    costa_kwon_window: int
    costa_kwon_regimes: int
    costa_kwon_factors: tuple[str, ...]

    def __post_init__(self) -> None:
        _require(
            len(self.risk_aversion) > 0 and min(self.risk_aversion) > 0,
            "strategies.risk_aversion must list positive values",
        )
        _require(
            set(self.reported_risk_aversion) <= set(self.risk_aversion),
            "strategies.reported_risk_aversion must be a subset of risk_aversion",
        )
        _require(
            self.rolling_window >= 2, "strategies.rolling_window must be at least 2"
        )
        _require(
            self.rolling_window in self.rolling_window_sensitivity,
            "strategies.rolling_window_sensitivity must include rolling_window",
        )
        _require(
            self.costa_kwon_window >= 1, "strategies.costa_kwon_window must be positive"
        )
        _require(
            self.costa_kwon_regimes >= 1,
            "strategies.costa_kwon_regimes must be positive",
        )
        _require(
            len(self.costa_kwon_factors) > 0, "strategies.costa_kwon_factors is empty"
        )


@dataclass(frozen=True)
class CostConfig:
    decimalization: str
    quantile: float

    def __post_init__(self) -> None:
        _require(0 < self.quantile < 1, "costs.quantile must be in (0, 1)")


@dataclass(frozen=True)
class InferenceConfig:
    alpha: float
    bootstrap_draws: int
    bootstrap_block: int
    detection_multiples: tuple[int, ...]

    def __post_init__(self) -> None:
        _require(0 < self.alpha < 1, "inference.alpha must be in (0, 1)")
        _require(
            self.bootstrap_draws >= 1, "inference.bootstrap_draws must be positive"
        )
        _require(
            self.bootstrap_block >= 1, "inference.bootstrap_block must be positive"
        )
        _require(
            len(self.detection_multiples) > 0 and min(self.detection_multiples) >= 1,
            "inference.detection_multiples must list positive integers",
        )


@dataclass(frozen=True)
class SeedConfig:
    regimes: int
    simulation: int
    bootstrap: int


@dataclass(frozen=True)
class SolverConfig:
    name: str


@dataclass(frozen=True)
class BacktestConfig:
    holdings_threshold: float
    weights_paths: int

    def __post_init__(self) -> None:
        _require(
            0 <= self.holdings_threshold < 1,
            "backtest.holdings_threshold must be in [0, 1)",
        )
        _require(self.weights_paths >= 0, "backtest.weights_paths must be non-negative")


@dataclass(frozen=True)
class StudyConfig:
    """All parameters of one study (main text or an appendix)."""

    name: str
    data: DataConfig
    sample: SampleConfig
    calibration: CalibrationConfig
    regimes: RegimeConfig
    regime_selection: RegimeSelectionConfig
    simulation: SimulationConfig
    strategies: StrategyConfig
    costs: CostConfig
    inference: InferenceConfig
    seeds: SeedConfig
    solver: SolverConfig
    backtest: BacktestConfig
    source: Path = dataclasses.field(compare=False, default=Path())

    def __post_init__(self) -> None:
        factors = self.factor_names
        _require(
            self.regimes.order_by in factors,
            f"regimes.order_by must be one of the factors {factors}",
        )
        _require(
            not self.regimes.reference or set(self.regimes.reference) == set(factors),
            f"regimes.reference must give the means of the factors {factors}",
        )

    @property
    def factor_names(self) -> list[str]:
        """The factors of ``data.factor_model``, as named in French's files."""
        return [
            c for _, cols in rwls.factors.MODELS[self.data.factor_model] for c in cols
        ]

    def regime_settings(self, **overrides) -> rwls.RegimeSettings:
        """Settings of the regime model: rwls defaults with the study's choices.

        With ``selection = "nearest"`` the starts are fitted one by one and the
        choice is made by :func:`regime_aware_portfolios.regimes.fit_regimes`, so
        the rwls rule is left at its default.
        """
        selection = self.regimes.selection
        base = rwls.RegimeSettings(
            n_regimes=self.regimes.n_regimes,
            n_starts=self.regimes.n_starts,
            selection="best" if selection == "nearest" else selection,
            seed=self.seeds.regimes,
        )
        return base.replace(**overrides) if overrides else base

    def rwls_settings(self, **overrides) -> rwls.RWLSSettings:
        """Settings of the RWLS estimator: the rwls defaults."""
        base = rwls.RWLSSettings()
        return base.replace(**overrides) if overrides else base

    def as_dict(self) -> dict[str, Any]:
        """The configuration as plain values, for run records."""
        out = dataclasses.asdict(self)
        out["source"] = str(self.source)
        return out


_SECTIONS = {
    "data": DataConfig,
    "sample": SampleConfig,
    "calibration": CalibrationConfig,
    "regimes": RegimeConfig,
    "regime_selection": RegimeSelectionConfig,
    "simulation": SimulationConfig,
    "strategies": StrategyConfig,
    "costs": CostConfig,
    "inference": InferenceConfig,
    "seeds": SeedConfig,
    "solver": SolverConfig,
    "backtest": BacktestConfig,
}


def load_config(path: str | Path) -> StudyConfig:
    """Read, merge and validate a study configuration file."""
    path = Path(path)
    raw = _read(path)
    unknown = set(raw) - set(_SECTIONS) - {"name"}
    _require(not unknown, f"{path.name}: unknown section(s) {sorted(unknown)}")
    _require("name" in raw, f"{path.name}: missing 'name'")
    sections = {}
    for key, cls in _SECTIONS.items():
        _require(key in raw, f"{path.name}: missing section [{key}]")
        sections[key] = _build(cls, raw[key], f"{path.name} [{key}]")
    return StudyConfig(name=raw["name"], source=path, **sections)


def _read(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    _require(path not in seen, f"circular 'extends' at {path.name}")
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    base = _read(path.parent / parent, seen + (path,))
    for key, value in raw.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return base


def _build(cls, values: dict[str, Any], where: str):
    fields = {f.name: f for f in dataclasses.fields(cls)}
    required = {
        name
        for name, f in fields.items()
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    }
    unknown = set(values) - set(fields)
    missing = required - set(values)
    _require(not unknown, f"{where}: unknown key(s) {sorted(unknown)}")
    _require(not missing, f"{where}: missing key(s) {sorted(missing)}")
    converted = {k: tuple(v) if isinstance(v, list) else v for k, v in values.items()}
    return cls(**converted)
