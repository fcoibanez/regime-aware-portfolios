from pathlib import Path

import pytest
import rwls

from regime_aware_portfolios.config import ConfigError, load_config

CONFIG = Path(__file__).resolve().parents[1] / "config"


def test_main_configuration_loads():
    cfg = load_config(CONFIG / "main.toml")
    assert cfg.name == "main"
    assert cfg.sample.coverage == 0.99
    assert cfg.sample.min_history == 120
    assert cfg.simulation.n_periods == 660
    assert cfg.strategies.risk_aversion == (1, 5, 10, 25, 50)
    assert cfg.regime_selection.n_regimes == tuple(range(1, 9))
    assert cfg.solver.name == "CLARABEL"


def test_every_configuration_loads():
    names = {load_config(p).name for p in CONFIG.glob("*.toml")}
    assert names == {"main", "alt_calibration", "four_regimes"}


def test_appendix_files_override_only_what_they_name():
    main = load_config(CONFIG / "main.toml")
    alt = load_config(CONFIG / "alt_calibration.toml")
    four = load_config(CONFIG / "four_regimes.toml")
    assert alt.regimes.selection == "best"
    assert alt.regimes.n_regimes == main.regimes.n_regimes
    assert alt.regimes.n_starts == main.regimes.n_starts
    assert four.regimes.n_regimes == 4
    assert four.simulation == main.simulation
    assert four.seeds == main.seeds


def test_rwls_settings_follow_the_configuration():
    cfg = load_config(CONFIG / "main.toml")
    s = cfg.regime_settings()
    assert isinstance(s, rwls.RegimeSettings)
    assert (s.n_regimes, s.n_starts, s.selection, s.seed) == (3, 1000, "modal", 13)
    # Everything else keeps the rwls defaults.
    assert s.tol == rwls.RegimeSettings().tol
    assert cfg.regime_settings(n_regimes=2).n_regimes == 2
    assert cfg.rwls_settings() == rwls.RWLSSettings()


def test_manifest_dictionary():
    d = load_config(CONFIG / "main.toml").as_dict()
    assert d["seeds"] == {"regimes": 13, "simulation": 20261020, "bootstrap": 20261211}
    assert d["source"].endswith("main.toml")


def _write(tmp_path, text, name="study.toml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _main_text():
    return (CONFIG / "main.toml").read_text(encoding="utf-8")


def test_unknown_and_missing_values_are_rejected(tmp_path):
    text = _main_text()
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(
            _write(
                tmp_path, text.replace("coverage = 0.99", "coverage = 0.99\nextra = 1")
            )
        )
    with pytest.raises(ConfigError, match="missing key"):
        load_config(_write(tmp_path, text.replace("min_history = 120", "")))
    with pytest.raises(ConfigError, match="unknown section"):
        load_config(_write(tmp_path, text + "\n[other]\nx = 1\n"))


@pytest.mark.parametrize(
    "old, new",
    [
        ("coverage = 0.99", "coverage = 1.5"),
        ('selection = "modal"', 'selection = "median"'),
        ('factor_model = "ff5+mom"', 'factor_model = "ff7"'),
        ("reported_risk_aversion = [10, 25, 50]", "reported_risk_aversion = [10, 30]"),
        (
            "rolling_window_sensitivity = [24, 36, 60]",
            "rolling_window_sensitivity = [24, 60]",
        ),
        ("alpha = 0.05", "alpha = 1.5"),
    ],
)
def test_invalid_values_are_rejected(tmp_path, old, new):
    text = _main_text()
    assert old in text
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, text.replace(old, new)))


def test_circular_extends_is_rejected(tmp_path):
    _write(tmp_path, 'extends = "b.toml"\nname = "a"\n', "a.toml")
    _write(tmp_path, 'extends = "a.toml"\nname = "b"\n', "b.toml")
    with pytest.raises(ConfigError, match="circular"):
        load_config(tmp_path / "a.toml")
