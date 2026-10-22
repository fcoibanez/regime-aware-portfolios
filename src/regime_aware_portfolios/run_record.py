"""Records of how each stage's results were produced.

Every stage writes a ``run.json`` next to its results with the configuration,
the commit of the code, the versions of Python and the packages that compute, and
anything else the stage reports.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from regime_aware_portfolios.config import StudyConfig
from regime_aware_portfolios.paths import ROOT

__all__ = [
    "PACKAGES",
    "check_upstream",
    "environment",
    "git_commit",
    "write_run_record",
]

#: Packages whose versions are recorded.
PACKAGES = (
    "regime-aware-portfolios",
    "rwls",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "hmmlearn",
    "cvxpy",
    "clarabel",
    "pyarrow",
)


def git_commit() -> dict[str, Any] | None:
    """The commit checked out in the repository and whether files were modified.

    None when git or the repository is not available.
    """
    try:
        commit = _git("rev-parse", "HEAD")
        status = _git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"commit": commit, "modified": bool(status)}


def environment() -> dict[str, Any]:
    """Versions of Python and of :data:`PACKAGES`, and the platform."""
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


def write_run_record(
    path: str | Path, stage: str, config: StudyConfig, **sections: Any
) -> dict[str, Any]:
    """Write the record of a stage's run and return it.

    Parameters
    ----------
    path : path-like
        File to write, usually ``run.json`` in the stage's output folder.
    stage : str
        Name of the stage, such as ``"01_data"``.
    config : StudyConfig
        Configuration of the run.
    **sections
        Further entries, such as a summary of the results. Values must be
        serializable to JSON.
    """
    record = {
        "stage": stage,
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "code": git_commit(),
        "environment": environment(),
        "config": config.as_dict(),
        **sections,
    }
    Path(path).write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return record


def check_upstream(
    path: str | Path, config: StudyConfig, sections: tuple[str, ...]
) -> dict[str, Any]:
    """Stop unless an earlier stage ran with the same settings in ``sections``.

    Reads the record at ``path``, written by an earlier stage, and compares
    the named sections of its configuration with those of ``config``. Returns the
    record.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} not found; run the earlier stage first")
    record = json.loads(path.read_text(encoding="utf-8"))
    current = json.loads(json.dumps(config.as_dict(), default=str))
    for section in sections:
        if record["config"][section] != current[section]:
            raise SystemExit(
                f"{record['stage']} ran with different [{section}] settings; rerun it"
            )
    return record


def _git(*args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()
