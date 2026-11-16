"""Guards the repo structure: every model folder follows the template and every area
has a CODEOWNERS entry. Owner: Member 1."""

from fnmatch import fnmatch
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = ("README.md", "config.yaml", "train.py", "predict.py")


def _arch_dirs():
    dirs = sorted(p for p in (ROOT / "models").glob("*/arch*") if p.is_dir())
    dirs += sorted(p for p in (ROOT / "baselines").glob("b*") if p.is_dir())
    return dirs


def _codeowners_patterns():
    lines = (ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8").splitlines()
    return [ln.split()[0] for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def _owned(rel: str, patterns) -> bool:
    for pat in patterns:
        if pat == "*":
            continue
        p = pat.lstrip("/")
        if p.endswith("/") and (rel + "/").startswith(p):
            return True
        if fnmatch(rel, p):
            return True
    return False


def test_model_dirs_exist():
    assert len(_arch_dirs()) >= 20  # 16 architectures + 4 baselines


@pytest.mark.parametrize("d", _arch_dirs(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_model_dir_template(d):
    for name in MODEL_FILES:
        assert (d / name).is_file(), f"{d.relative_to(ROOT)} is missing {name}"
    cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["model_id"] == f"{d.parent.name}.{d.name}"
    assert "Owner:" in (d / "README.md").read_text(encoding="utf-8")


AREAS = [
    "src/casmi/data", "src/casmi/preprocess", "src/casmi/fingerprints", "src/casmi/retrieval",
    "src/casmi/candidates", "src/casmi/evaluation", "src/casmi/submission", "src/casmi/utils",
    "data/splits", "kaggle", "kaggle/datasets/coconut", "notebooks", "experiments/entries",
    "experiments/leaderboard", "report", "slides", "docs/error_analysis", "docs/ablations",
]


@pytest.mark.parametrize("rel", AREAS + [p.relative_to(ROOT).as_posix() for p in _arch_dirs()])
def test_every_area_has_an_owner(rel):
    assert _owned(rel, _codeowners_patterns()), f"no CODEOWNERS entry covers {rel}"
