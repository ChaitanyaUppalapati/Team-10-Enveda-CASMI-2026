"""Expose models/spectrum_formula/common as the importable package ``sf_common`` for tests."""

import importlib.util
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[2] / "models" / "spectrum_formula" / "common"
if "sf_common" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "sf_common", _PKG / "__init__.py", submodule_search_locations=[str(_PKG)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["sf_common"] = module
    spec.loader.exec_module(module)
