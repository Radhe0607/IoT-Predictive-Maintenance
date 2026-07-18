"""
src/configs/__init__.py
=======================
Public interface for the IoT Predictive Maintenance configuration layer.

Import these helpers at any point in the pipeline to access the
centralised, validated project configuration without re-reading disk:

    from src.configs import get_config, get_absolute_path, setup_logging

Exported:
    get_config         — Singleton accessor returning the active ProjectConfig.
    load_config        — Force-load a ProjectConfig from a YAML file path.
    get_absolute_path  — Resolve a config-relative path to an absolute Path.
    setup_logging      — Configure the root Python logger from config settings.
    display_config     — Pretty-print the active configuration to stdout.
    ProjectConfig      — Root frozen dataclass (for type annotations).
"""

from .config import (
    ProjectConfig,
    display_config,
    get_absolute_path,
    get_config,
    load_config,
    setup_logging,
)

__all__ = [
    "ProjectConfig",
    "get_config",
    "load_config",
    "get_absolute_path",
    "setup_logging",
    "display_config",
]
