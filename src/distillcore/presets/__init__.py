"""Preset registry for domain-specific configurations."""

from __future__ import annotations

from ..config import DomainConfig

_PRESETS: dict[str, DomainConfig] = {}


def register_preset(name: str, config: DomainConfig) -> None:
    """Register a domain preset."""
    _PRESETS[name] = config


def load_preset(name: str) -> DomainConfig:
    """Load a registered domain preset by name."""
    if name not in _PRESETS:
        available = ", ".join(sorted(_PRESETS.keys())) or "(none)"
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return _PRESETS[name]


def list_presets() -> list[str]:
    """Return names of all registered presets."""
    return sorted(_PRESETS.keys())


# Auto-register built-in presets
from .generic import GENERIC_PRESET  # noqa: E402
from .legal import LEGAL_PRESET  # noqa: E402

register_preset("generic", GENERIC_PRESET)
register_preset("legal", LEGAL_PRESET)
