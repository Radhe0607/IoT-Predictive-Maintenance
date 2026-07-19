"""
config.py
=========
Centralised configuration loader for the IoT Predictive Maintenance project.

This module is the single source of truth for all project settings. Every
other module in the pipeline obtains its defaults by calling helpers exposed
here rather than embedding literal values in source code.

Design:
    - The canonical settings live in ``config.yaml`` (same directory as this file).
    - ``ProjectConfig`` is a frozen, type-annotated dataclass that holds every
      setting after loading and validation. Immutability prevents accidental
      mutation during a pipeline run.
    - ``load_config()`` reads the YAML file, deep-merges any caller-supplied
      overrides, and returns a validated ``ProjectConfig`` instance.
    - ``get_config()`` is a singleton accessor — it loads the config exactly
      once per process and returns the same object on subsequent calls.
    - ``get_absolute_path()`` resolves config-relative paths to absolute
      ``pathlib.Path`` objects, anchored at the project root.
    - ``setup_logging()`` configures the root logger from the config's
      ``logging`` section.

Usage::

    # Anywhere in the project:
    from src.configs.config import get_config, get_absolute_path

    cfg = get_config()

    # Access settings as attributes:
    model_dir   = get_absolute_path(cfg.paths.models_dir)
    target_col  = cfg.model.target_col
    n_estimators = cfg.model.n_estimators

    # Bootstrap logging at program entry point:
    from src.configs.config import setup_logging
    setup_logging()

Part of the Infotact Solutions Data Science & Machine Learning Internship project.
"""

import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# ---------------------------------------------------------------------------
# Module-level logger (pre-logging-setup; uses basicConfig fallback)
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Location of the YAML config file
# ---------------------------------------------------------------------------
# This resolves to the directory that *this* Python file lives in,
# meaning config.yaml must sit alongside config.py.
_CONFIG_DIR:  Path = Path(__file__).resolve().parent
_CONFIG_YAML: Path = _CONFIG_DIR / "config.yaml"

# Cached singleton — populated on first call to get_config()
_CONFIG_SINGLETON: Optional["ProjectConfig"] = None


# ---------------------------------------------------------------------------
# Frozen sub-config dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectMeta:
    """Top-level project identity fields."""
    name:        str = "IoT Predictive Maintenance"
    version:     str = "1.0.0"
    description: str = ""
    random_seed: int = 42


@dataclass(frozen=True)
class PathsConfig:
    """All input / output directory and file paths (relative to project root)."""
    raw_data_dir:          str = "data/raw"
    processed_data_dir:    str = "data/processed"
    raw_data_file:         str = "data/raw/sensor_data.csv"
    engineered_data_file:  str = "data/processed/features.csv"
    models_dir:            str = "outputs/models"
    model_filename:        str = "random_forest_baseline.joblib"
    plots_dir:             str = "outputs/plots"
    reports_dir:           str = "outputs/reports"
    predictions_output:    str = "outputs/reports/predictions.csv"


@dataclass(frozen=True)
class DataConfig:
    """DataLoader options."""
    encoding:  str = "utf-8"
    separator: str = ","


@dataclass(frozen=True)
class PreprocessingConfig:
    """DataPreprocessor defaults."""
    numerical_cols:        Optional[List[str]] = None
    categorical_cols:      Optional[List[str]] = None
    num_fill_strategy:     str   = "median"
    cat_fill_strategy:     str   = "mode"
    cat_fill_constant:     str   = "UNKNOWN"
    cat_encoding:          str   = "label"
    drop_missing_thresh:   float = 0.0
    normalize:             bool  = True


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """FeatureEngineer defaults."""
    sensor_cols:            Optional[List[str]] = None
    timestamp_col:          Optional[str]       = None
    rolling_windows:        List[int]  = field(default_factory=lambda: [3, 5, 10])
    lag_steps:              List[int]  = field(default_factory=lambda: [1, 3, 5])
    max_interaction_pairs:  int        = 10
    variance_threshold:     float      = 0.01
    correlation_threshold:  float      = 0.95
    enable_rolling:         bool       = True
    enable_lags:            bool       = True
    enable_delta:           bool       = True
    enable_interactions:    bool       = True
    enable_time:            bool       = True
    enable_selection:       bool       = True
    save_format:            str        = "csv"


@dataclass(frozen=True)
class ModelConfig:
    """BaselineModel / RandomForestClassifier hyperparameters."""
    target_col:        str            = "failure"
    test_size:         float          = 0.20
    n_estimators:      int            = 200
    max_depth:         Optional[int]  = None
    min_samples_split: int            = 5
    min_samples_leaf:  int            = 2
    class_weight:      Optional[str]  = "balanced"


@dataclass(frozen=True)
class EvaluationConfig:
    """ModelEvaluator, EDA, and PredictionPipeline options."""
    cv_folds:                    int   = 5
    cv_scoring:                  str   = "f1_weighted"
    confidence_threshold:        float = 0.50
    top_n_features:              int   = 20
    eda_max_cols_per_figure:     int   = 20
    eda_hist_bins:               int   = 30
    eda_missing_heatmap_sample:  int   = 300


@dataclass(frozen=True)
class LoggingConfig:
    """Python logging configuration."""
    level:   str = "INFO"
    format:  str = "%(asctime)s  [%(levelname)s]  %(name)s — %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class PlotConfig:
    """Global matplotlib / seaborn aesthetics."""
    dpi:            int   = 150
    style:          str   = "whitegrid"
    palette:        str   = "muted"
    font_scale:     float = 0.95
    title_fontsize: int   = 13
    label_fontsize: int   = 10
    tick_fontsize:  int   = 8
    grid_alpha:     float = 0.35
    spine_colour:   str   = "#CCCCCC"


@dataclass(frozen=True)
class ProjectConfig:
    """
    Root configuration object for the IoT Predictive Maintenance project.

    Holds all sub-configs as frozen nested dataclasses. Because ``frozen=True``
    is set, fields cannot be mutated after construction, preventing accidental
    config drift during a pipeline run.

    Attributes:
        project  (ProjectMeta):             Project identity.
        paths    (PathsConfig):             All I/O paths.
        data     (DataConfig):              Data-loading options.
        preprocessing (PreprocessingConfig): Preprocessor defaults.
        feature_engineering (FeatureEngineeringConfig): Feature-engineer defaults.
        model    (ModelConfig):             Model hyperparameters.
        evaluation (EvaluationConfig):      Evaluator / EDA options.
        logging  (LoggingConfig):           Logging settings.
        plot     (PlotConfig):              Plot aesthetics.
    """
    project:             ProjectMeta             = field(default_factory=ProjectMeta)
    paths:               PathsConfig             = field(default_factory=PathsConfig)
    data:                DataConfig              = field(default_factory=DataConfig)
    preprocessing:       PreprocessingConfig     = field(default_factory=PreprocessingConfig)
    feature_engineering: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    model:               ModelConfig             = field(default_factory=ModelConfig)
    evaluation:          EvaluationConfig        = field(default_factory=EvaluationConfig)
    logging:             LoggingConfig           = field(default_factory=LoggingConfig)
    plot:                PlotConfig              = field(default_factory=PlotConfig)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce_field(target_cls: type, key: str, value: Any) -> Any:
    """
    Coerce a raw YAML value to the annotated type of *key* in *target_cls*.

    Handles:
        - ``None`` / null  → kept as ``None``
        - ``list``         → kept as ``list``
        - ``int`` / ``float`` / ``bool`` / ``str`` → cast using annotation

    Args:
        target_cls: The frozen dataclass whose annotations define the target type.
        key (str):  Field name.
        value:      Raw value from YAML.

    Returns:
        The coerced value.
    """
    if value is None:
        return None

    hints = {f.name: f.type for f in fields(target_cls)}
    hint  = hints.get(key)

    if hint is None or isinstance(value, list):
        return value

    try:
        # Strip Optional / Union wrappers to get the inner type
        origin = getattr(hint, "__origin__", None)
        if origin is Union:
            inner_types = [t for t in hint.__args__ if t is not type(None)]
            if inner_types:
                return inner_types[0](value)
        if isinstance(hint, str):
            return value   # string annotation — no coercion needed
        return value       # already correct type from YAML parsing
    except (TypeError, ValueError):
        return value


def _build_sub_config(cls: type, raw: Dict[str, Any]) -> Any:
    """
    Construct a frozen dataclass instance of *cls* from a raw YAML dict.

    Unknown keys in *raw* are silently ignored so that the YAML can contain
    comments or future keys without breaking older code.

    Args:
        cls:       A frozen dataclass class.
        raw (dict): Flat dict of field_name → value from YAML.

    Returns:
        An instance of *cls*.
    """
    valid_keys  = {f.name for f in fields(cls)}
    filtered    = {k: v for k, v in raw.items() if k in valid_keys}
    coerced     = {k: _coerce_field(cls, k, v) for k, v in filtered.items()}
    return cls(**coerced)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides:   Optional[Dict[str, Any]]   = None,
) -> "ProjectConfig":
    """
    Load, validate, and return a ``ProjectConfig`` from a YAML file.

    This function does **not** use the singleton cache — it always reads the
    file from disk. Use :func:`get_config` for the cached singleton.

    Args:
        config_path (str | Path, optional):
            Path to the YAML config file.
            Defaults to ``config.yaml`` in the same directory as this module.
        overrides (dict, optional):
            A nested dict of section → {key: value} pairs that will be
            deep-merged on top of the YAML values *before* building the
            dataclass.  Useful for programmatic overrides in tests or notebooks::

                cfg = load_config(overrides={"model": {"n_estimators": 100}})

    Returns:
        ProjectConfig: A fully populated, frozen configuration object.

    Raises:
        FileNotFoundError: If the config YAML file is not found.
        yaml.YAMLError:    If the YAML is malformed.
    """
    path = Path(config_path).resolve() if config_path else _CONFIG_YAML

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: '{path}'.\n"
            "Ensure config.yaml is present alongside config.py."
        )

    with path.open("r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    logger.debug("Config loaded from '%s'.", path)

    # Deep-merge caller-supplied overrides
    if overrides:
        for section, values in overrides.items():
            if isinstance(values, dict):
                raw.setdefault(section, {}).update(values)
            else:
                raw[section] = values

    # Build sub-configs, fall back to defaults if section is missing
    return ProjectConfig(
        project             = _build_sub_config(ProjectMeta,              raw.get("project",             {})),
        paths               = _build_sub_config(PathsConfig,              raw.get("paths",               {})),
        data                = _build_sub_config(DataConfig,               raw.get("data",                {})),
        preprocessing       = _build_sub_config(PreprocessingConfig,      raw.get("preprocessing",       {})),
        feature_engineering = _build_sub_config(FeatureEngineeringConfig, raw.get("feature_engineering", {})),
        model               = _build_sub_config(ModelConfig,              raw.get("model",               {})),
        evaluation          = _build_sub_config(EvaluationConfig,         raw.get("evaluation",          {})),
        logging             = _build_sub_config(LoggingConfig,            raw.get("logging",             {})),
        plot                = _build_sub_config(PlotConfig,               raw.get("plot",                {})),
    )


def get_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides:   Optional[Dict[str, Any]]   = None,
    reload:      bool = False,
) -> "ProjectConfig":
    """
    Return the singleton ``ProjectConfig``, loading it on first call.

    Subsequent calls return the cached object without re-reading disk,
    making this safe and cheap to call at module import time.

    Args:
        config_path (str | Path, optional):
            Path to the YAML file. Only used on the first call (or when
            *reload* is ``True``).
        overrides (dict, optional):
            Section-level overrides applied on top of the YAML values.
            Only used on the first call (or when *reload* is ``True``).
        reload (bool):
            Force a fresh load from disk, discarding the cached config.
            Useful in test suites or when the YAML has been edited at runtime.
            Defaults to ``False``.

    Returns:
        ProjectConfig: The singleton configuration object.
    """
    global _CONFIG_SINGLETON

    if _CONFIG_SINGLETON is None or reload:
        _CONFIG_SINGLETON = load_config(config_path=config_path, overrides=overrides)
        logger.info(
            "ProjectConfig loaded — project='%s' v%s, seed=%d.",
            _CONFIG_SINGLETON.project.name,
            _CONFIG_SINGLETON.project.version,
            _CONFIG_SINGLETON.project.random_seed,
        )

    return _CONFIG_SINGLETON


def get_absolute_path(relative_path: str, root: Optional[Path] = None) -> Path:
    """
    Resolve a config-relative path string to an absolute ``pathlib.Path``.

    The project root is determined as the grandparent of the ``configs/``
    directory (i.e., ``<project_root>/data/notebook/``).  All relative paths
    in ``config.yaml`` are anchored there.

    Args:
        relative_path (str): A path string from a ``PathsConfig`` field,
                             e.g. ``"outputs/models"``.
        root (Path, optional): Override the project root. Defaults to the
                               directory two levels above this file.

    Returns:
        Path: Resolved absolute path (directory may not yet exist).

    Example::

        from src.configs.config import get_config, get_absolute_path

        cfg      = get_config()
        plot_dir = get_absolute_path(cfg.paths.plots_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
    """
    if root is None:
        # config.py lives at  <root>/src/configs/config.py
        # Project root is     <root>/  (three levels up from this file)
        root = _CONFIG_DIR.parent.parent

    return (root / relative_path).resolve()


def setup_logging(cfg: Optional["ProjectConfig"] = None) -> None:
    """
    Configure the root Python logger from the ``logging`` section of the config.

    This should be called **once** at the application entry point (e.g., the
    top of a notebook or ``main.py``) before any other module starts logging.

    Args:
        cfg (ProjectConfig, optional):
            Config object to read logging settings from.
            Defaults to the singleton returned by :func:`get_config`.

    Example::

        from src.configs.config import setup_logging
        setup_logging()
    """
    if cfg is None:
        cfg = get_config()

    log_cfg = cfg.logging

    numeric_level = getattr(logging, log_cfg.level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format=log_cfg.format,
        datefmt=log_cfg.datefmt,
        force=True,   # override any existing basicConfig from other modules
    )
    logger.debug("Logging configured — level=%s.", log_cfg.level)


def display_config(cfg: Optional["ProjectConfig"] = None) -> None:
    """
    Pretty-print the active configuration to stdout.

    Useful for debugging and notebook header cells to confirm the config
    that is in effect for a given run.

    Args:
        cfg (ProjectConfig, optional):
            Config object to display. Defaults to the singleton.
    """
    if cfg is None:
        cfg = get_config()

    sep  = "─" * 68
    sep2 = "═" * 68

    print(f"\n{sep2}")
    print(f"  PROJECT CONFIG — {cfg.project.name} v{cfg.project.version}")
    print(sep2)

    section_map = {
        "Project":             cfg.project,
        "Paths":               cfg.paths,
        "Data":                cfg.data,
        "Preprocessing":       cfg.preprocessing,
        "Feature Engineering": cfg.feature_engineering,
        "Model":               cfg.model,
        "Evaluation":          cfg.evaluation,
        "Logging":             cfg.logging,
        "Plot":                cfg.plot,
    }

    for section_name, section_obj in section_map.items():
        print(f"\n  [{section_name}]")
        print(f"  {sep[:50]}")
        for f in fields(section_obj):
            val = getattr(section_obj, f.name)
            print(f"    {f.name:<30} {val}")

    print(f"\n{sep2}\n")

def validate_config(config):

    required = [
        "DATA_PATH",
        "MODEL_PATH",
        "OUTPUT_PATH"
    ]

    for item in required:

        if item not in config:
            raise ValueError(f"{item} is missing!")

    print("Configuration is valid.")