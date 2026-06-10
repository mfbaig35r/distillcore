"""Guard against pyproject.toml and distillcore.__version__ drifting.

We were bitten by exactly this between 0.7.0 → 0.7.1: pyproject got bumped,
__init__.py didn't, and the divergence shipped to PyPI silently.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import distillcore


def _pyproject_version() -> str:
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


def test_version_matches_pyproject() -> None:
    assert distillcore.__version__ == _pyproject_version(), (
        f"distillcore.__version__={distillcore.__version__!r} does not match "
        f"pyproject.toml version={_pyproject_version()!r}. Bump both."
    )
