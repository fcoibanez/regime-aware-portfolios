import json

import rwls

from regime_aware_portfolios import load_config, manifest, paths


def test_paths_point_into_the_repository():
    assert (paths.ROOT / "pyproject.toml").exists()
    assert (paths.CONFIG / "main.toml").exists()
    assert paths.CRSP.parent == paths.RAW


def test_environment_records_the_packages():
    env = manifest.environment()
    assert set(env["packages"]) == set(manifest.PACKAGES)
    assert env["packages"]["rwls"] == rwls.__version__
    assert env["python"].count(".") == 2


def test_write_manifest(tmp_path):
    cfg = load_config(paths.CONFIG / "main.toml")
    path = tmp_path / "manifest.json"
    record = manifest.write_manifest(path, "01_data", cfg, inputs={"a.csv": "0" * 64})
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == json.loads(json.dumps(record, default=str))
    assert stored["stage"] == "01_data"
    assert stored["config"]["name"] == "main"
    assert stored["config"]["data"]["factor_model"] == "ff5+mom"
    assert stored["inputs"] == {"a.csv": "0" * 64}
    assert set(stored) >= {"created", "code", "environment"}
