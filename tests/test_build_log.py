"""Tests for the experiment-log builder. Owner: Member 1."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_log", ROOT / "experiments" / "build_log.py")
build_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_log)


def test_template_is_a_valid_entry(tmp_path):
    (tmp_path / "e.yaml").write_text((ROOT / "experiments" / "TEMPLATE.yaml").read_text())
    df = build_log.load_entries(tmp_path)
    assert len(df) == 1
    assert "fingerprint_mlp.arch1_mlp" in build_log.to_markdown(df)


def test_missing_keys_rejected(tmp_path):
    (tmp_path / "bad.yaml").write_text("date: 2026-11-17\nmember: 1\n")
    with pytest.raises(ValueError, match="missing keys"):
        build_log.load_entries(tmp_path)


def test_committed_entries_are_valid():
    build_log.load_entries(ROOT / "experiments" / "entries")
