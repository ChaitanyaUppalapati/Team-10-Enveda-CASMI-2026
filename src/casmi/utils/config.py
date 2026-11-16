"""Config and path loading. Owner: Member 1."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PATHS_FILE = REPO_ROOT / "configs" / "paths.yaml"

_ENV_OVERRIDES = {
    "data_dir": "CASMI_DATA_DIR",
    "output_dir": "CASMI_OUTPUT_DIR",
    "candidates_dir": "CASMI_CANDIDATES_DIR",
}


def detect_env() -> str:
    """Return ``"kaggle"`` inside a Kaggle kernel (or if CASMI_ENV=kaggle), else ``"local"``."""
    if os.environ.get("CASMI_ENV"):
        return os.environ["CASMI_ENV"]
    return "kaggle" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else "local"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict (empty file -> empty dict)."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_paths(env: str | None = None, paths_file: str | Path | None = None) -> dict[str, Path]:
    """Resolve the path roots for ``env`` from configs/paths.yaml plus env-var overrides.

    When the package is installed from a wheel (Kaggle), point ``CASMI_PATHS_FILE`` at the
    copy of configs/paths.yaml shipped in the code dataset.
    """
    env = env or detect_env()
    paths_file = paths_file or os.environ.get("CASMI_PATHS_FILE") or PATHS_FILE
    profiles = load_config(paths_file)
    if env not in profiles:
        raise KeyError(f"unknown env {env!r}; expected one of {sorted(profiles)}")
    out: dict[str, Path] = {}
    for key, value in profiles[env].items():
        override = os.environ.get(_ENV_OVERRIDES[key]) if key in _ENV_OVERRIDES else None
        p = Path(override or value)
        out[key] = p if p.is_absolute() else REPO_ROOT / p
    return out
