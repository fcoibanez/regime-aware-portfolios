import dataclasses
import json

import pytest
import rwls

from regime_aware_portfolios import load_config, paths, run_record


def test_paths_point_into_the_repository():
    assert (paths.ROOT / "pyproject.toml").exists()
    assert (paths.CONFIG / "main.toml").exists()
    assert paths.CRSP.parent == paths.RAW


def test_environment_records_the_packages():
    env = run_record.environment()
    assert set(env["packages"]) == set(run_record.PACKAGES)
    assert env["packages"]["rwls"] == rwls.__version__
    assert env["python"].count(".") == 2


def test_write_run_record(tmp_path):
    cfg = load_config(paths.CONFIG / "main.toml")
    path = tmp_path / "run.json"
    record = run_record.write_run_record(
        path, "01_data", cfg, inputs={"a.csv": "0" * 64}
    )
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == json.loads(json.dumps(record, default=str))
    assert stored["stage"] == "01_data"
    assert stored["config"]["name"] == "main"
    assert stored["config"]["data"]["factor_model"] == "ff5+mom"
    assert stored["inputs"] == {"a.csv": "0" * 64}
    assert set(stored) >= {"created", "code", "environment"}


def test_check_upstream(tmp_path):
    # The appendix configuration holds tuples, which the record stores as lists.
    cfg = load_config(paths.CONFIG / "alt_calibration.toml")
    path = tmp_path / "run.json"
    run_record.write_run_record(path, "02_regimes", cfg)
    assert (
        run_record.check_upstream(path, cfg, ("data", "regimes"))["stage"]
        == "02_regimes"
    )
    changed = dataclasses.replace(
        cfg, regimes=dataclasses.replace(cfg.regimes, n_starts=10)
    )
    assert run_record.check_upstream(path, changed, ("data",))
    with pytest.raises(SystemExit, match="regimes"):
        run_record.check_upstream(path, changed, ("data", "regimes"))
    with pytest.raises(SystemExit, match="not found"):
        run_record.check_upstream(tmp_path / "missing.json", cfg, ("data",))
