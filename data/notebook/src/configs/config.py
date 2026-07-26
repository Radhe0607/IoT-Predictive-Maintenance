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
    - ``validate_config()`` validates parameter bounds, numeric thresholds,
      and required path specifications.
    - Convenience path and parameter getters (``get_model_path``,
      ``get_raw_data_path``, ``get_engineered_data_path``, ``get_models_dir``,
      ``get_reports_dir``, ``get_plots_dir``, ``get_training_params``) provide
      type-safe access to common settings across modules.

Usage::

    # Anywhere in the project:
    from src.configs.config import get_config, get_absolute_path, get_model_path

    cfg = get_config()

    # Access settings as attributes:
    model_path = get_model_path(cfg)
    target_col = cfg.model.target_col
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
_CONFIG_DIR: Path = Path(__file__).resolve().parent
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
    description: str = "Infotact Solutions — DS & ML Internship baseline pipeline"
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
    hint = hints.get(key)

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
    valid_keys = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in raw.items() if k in valid_keys}
    coerced = {k: _coerce_field(cls, k, v) for k, v in filtered.items()}
    return cls(**coerced)


def _apply_env_overrides(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check environment variables and apply overrides to raw config dict.

    Supported env variables:
        - IOT_RAW_DATA_FILE
        - IOT_ENGINEERED_DATA_FILE
        - IOT_MODELS_DIR
        - IOT_MODEL_FILENAME
        - IOT_REPORTS_DIR
        - IOT_PLOTS_DIR
        - IOT_TEST_SIZE
        - IOT_N_ESTIMATORS
        - IOT_RANDOM_SEED
    """
    env_map = {
        "IOT_RAW_DATA_FILE": ("paths", "raw_data_file"),
        "IOT_ENGINEERED_DATA_FILE": ("paths", "engineered_data_file"),
        "IOT_MODELS_DIR": ("paths", "models_dir"),
        "IOT_MODEL_FILENAME": ("paths", "model_filename"),
        "IOT_REPORTS_DIR": ("paths", "reports_dir"),
        "IOT_PLOTS_DIR": ("paths", "plots_dir"),
        "IOT_TEST_SIZE": ("model", "test_size"),
        "IOT_N_ESTIMATORS": ("model", "n_estimators"),
        "IOT_RANDOM_SEED": ("project", "random_seed"),
    }

    for env_var, (section, key) in env_map.items():
        val = os.environ.get(env_var)
        if val is not None:
            raw.setdefault(section, {})[key] = val
            logger.info("Config override from env %s -> %s.%s = %s", env_var, section, key, val)

    return raw


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
            dataclass. Useful for programmatic overrides in tests or notebooks::

                cfg = load_config(overrides={"model": {"n_estimators": 100}})

    Returns:
        ProjectConfig: A fully populated, frozen configuration object.

    Raises:
        FileNotFoundError: If the config YAML file is not found.
        yaml.YAMLError:    If the YAML is malformed.
    """
    env_config_path = os.environ.get("IOT_CONFIG_PATH")
    if env_config_path and not config_path:
        config_path = env_config_path

    path = Path(config_path).resolve() if config_path else _CONFIG_YAML

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: '{path}'.\n"
            "Ensure config.yaml is present alongside config.py."
        )

    try:
        with path.open("r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}
    except Exception as exc:
        logger.error("Failed to parse YAML config file '%s': %s", path, exc)
        raise yaml.YAMLError(f"Malformed YAML in config file '{path}': {exc}") from exc

    logger.debug("Config loaded from '%s'.", path)

    # Apply environment variable overrides
    raw = _apply_env_overrides(raw)

    # Deep-merge caller-supplied overrides
    if overrides:
        for section, values in overrides.items():
            if isinstance(values, dict):
                raw.setdefault(section, {}).update(values)
            else:
                raw[section] = values

    # Build sub-configs, fall back to defaults if section is missing
    cfg = ProjectConfig(
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

    # Automatically validate configuration
    validate_config(cfg)
    return cfg


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
    directory (i.e., ``<project_root>/data/notebook/``). All relative paths
    in ``config.yaml`` are anchored there.

    Args:
        relative_path (str): A path string from a ``PathsConfig`` field,
                             e.g. ``"outputs/models"``.
        root (Path, optional): Override the project root. Defaults to the
                               directory two levels above this file.

    Returns:
        Path: Resolved absolute path (directory may not yet exist).
    """
    if root is None:
        root = _CONFIG_DIR.parent.parent

    return (root / relative_path).resolve()


# ---------------------------------------------------------------------------
# Convenience Path & Parameter Getters
# ---------------------------------------------------------------------------


def get_model_path(cfg: Optional["ProjectConfig"] = None) -> Path:
    """Return absolute Path to the trained model artifact."""
    c = cfg or get_config()
    return get_absolute_path(f"{c.paths.models_dir}/{c.paths.model_filename}")


def get_raw_data_path(cfg: Optional["ProjectConfig"] = None) -> Path:
    """Return absolute Path to the raw sensor dataset."""
    c = cfg or get_config()
    return get_absolute_path(c.paths.raw_data_file)


def get_engineered_data_path(cfg: Optional["ProjectConfig"] = None) -> Path:
    """Return absolute Path to the feature-engineered dataset."""
    c = cfg or get_config()
    return get_absolute_path(c.paths.engineered_data_file)


def get_models_dir(cfg: Optional["ProjectConfig"] = None) -> Path:
    """Return absolute Path to the models output directory."""
    c = cfg or get_config()
    return get_absolute_path(c.paths.models_dir)


def get_reports_dir(cfg: Optional["ProjectConfig"] = None) -> Path:
    """Return absolute Path to the reports output directory."""
    c = cfg or get_config()
    return get_absolute_path(c.paths.reports_dir)


def get_plots_dir(cfg: Optional["ProjectConfig"] = None) -> Path:
    """Return absolute Path to the plots output directory."""
    c = cfg or get_config()
    return get_absolute_path(c.paths.plots_dir)


def get_training_params(cfg: Optional["ProjectConfig"] = None) -> Dict[str, Any]:
    """
    Return a dictionary of model training hyper-parameters.

    Keys include: ``target_col``, ``test_size``, ``n_estimators``, ``max_depth``,
    ``min_samples_split``, ``min_samples_leaf``, ``class_weight``, ``random_state``.
    """
    c = cfg or get_config()
    return {
        "target_col":        c.model.target_col,
        "test_size":         c.model.test_size,
        "n_estimators":      c.model.n_estimators,
        "max_depth":         c.model.max_depth,
        "min_samples_split": c.model.min_samples_split,
        "min_samples_leaf":  c.model.min_samples_leaf,
        "class_weight":      c.model.class_weight,
        "random_state":      c.project.random_seed,
    }


def validate_config(cfg: Optional[Union["ProjectConfig", Dict[str, Any]]] = None) -> bool:
    """
    Validate parameter bounds, numeric thresholds, and path specifications.

    Args:
        cfg (ProjectConfig | dict, optional):
            Config object or dictionary to validate.
            Defaults to the singleton returned by :func:`get_config`.

    Returns:
        bool: True if configuration is valid.

    Raises:
        ValueError: If any configuration value is invalid or out of bounds.
        TypeError:  If cfg is of an unsupported type.
    """
    if cfg is None:
        cfg = get_config()

    if isinstance(cfg, dict):
        # Dictionary validation path
        required_keys = ["DATA_PATH", "MODEL_PATH", "OUTPUT_PATH"]
        missing = [k for k in required_keys if k not in cfg]
        if missing:
            raise ValueError(f"Missing required dictionary configuration keys: {missing}")
        logger.debug("Dictionary configuration validated successfully.")
        return True

    if not isinstance(cfg, ProjectConfig):
        raise TypeError(f"Expected ProjectConfig or dict, got {type(cfg).__name__}.")

    # --- 1. Path validations -----------------------------------------------
    paths = cfg.paths
    for path_attr in ["raw_data_file", "engineered_data_file", "models_dir", "model_filename", "reports_dir", "plots_dir"]:
        val = getattr(paths, path_attr, None)
        if not val or not isinstance(val, str) or not val.strip():
            raise ValueError(f"paths.{path_attr} must be a non-empty string.")

    # --- 2. Model hyperparameter validations ------------------------------
    model = cfg.model
    if not (0.0 < model.test_size < 1.0):
        raise ValueError(f"model.test_size must be in (0.0, 1.0), got {model.test_size}")
    if model.n_estimators <= 0:
        raise ValueError(f"model.n_estimators must be > 0, got {model.n_estimators}")
    if model.min_samples_split < 2:
        raise ValueError(f"model.min_samples_split must be >= 2, got {model.min_samples_split}")
    if model.min_samples_leaf < 1:
        raise ValueError(f"model.min_samples_leaf must be >= 1, got {model.min_samples_leaf}")

    # --- 3. Preprocessing validations -------------------------------------
    prep = cfg.preprocessing
    valid_num_fill = ["median", "mean", "zero", "drop"]
    if prep.num_fill_strategy not in valid_num_fill:
        raise ValueError(f"preprocessing.num_fill_strategy must be one of {valid_num_fill}, got '{prep.num_fill_strategy}'")

    valid_cat_fill = ["mode", "constant", "drop"]
    if prep.cat_fill_strategy not in valid_cat_fill:
        raise ValueError(f"preprocessing.cat_fill_strategy must be one of {valid_cat_fill}, got '{prep.cat_fill_strategy}'")

    valid_cat_enc = ["label", "onehot", "none"]
    if prep.cat_encoding not in valid_cat_enc:
        raise ValueError(f"preprocessing.cat_encoding must be one of {valid_cat_enc}, got '{prep.cat_encoding}'")

    # --- 4. Evaluation validations ----------------------------------------
    ev = cfg.evaluation
    if ev.cv_folds < 2:
        raise ValueError(f"evaluation.cv_folds must be >= 2, got {ev.cv_folds}")
    if not (0.0 <= ev.confidence_threshold <= 1.0):
        raise ValueError(f"evaluation.confidence_threshold must be in [0.0, 1.0], got {ev.confidence_threshold}")

    logger.debug("ProjectConfig validated successfully.")
    return True


def setup_logging(cfg: Optional["ProjectConfig"] = None) -> None:
    """
    Configure the root Python logger from the ``logging`` section of the config.

    Args:
        cfg (ProjectConfig, optional):
            Config object to read logging settings from.
            Defaults to the singleton returned by :func:`get_config`.
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

    Args:
        cfg (ProjectConfig, optional):
            Config object to display. Defaults to the singleton.
    """
    if cfg is None:
        cfg = get_config()

    sep = "─" * 68
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