"""Helpers shared by the training scripts of both architectures. Owner: Member 4."""

from __future__ import annotations

from pathlib import Path

from casmi.candidates.pool import load_pool
from casmi.utils import load_paths

from . import team


def pick_device(name: str = "auto"):
    """``auto``: CUDA, then Apple MPS, then CPU."""
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def checkpoint_path(cfg: dict) -> Path:
    return load_paths()["output_dir"] / "models" / cfg["model_id"] / "model.pt"


def build_protocol(cfg: dict) -> team.Protocol:
    """Team protocol with the external pools named in ``data.external_pools`` (tables under candidates_dir)."""
    external = [load_pool(f"{name}.parquet") for name in cfg["data"].get("external_pools", [])]
    return team.load_protocol(cfg["data"], external)
