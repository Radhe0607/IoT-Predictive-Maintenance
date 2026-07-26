"""
src/configs/__init__.py
=======================
Public interface for the IoT Predictive Maintenance configuration layer.

Import these helpers at any point in the pipeline to access the
centralised, validated project configuration without re-reading disk:

    from src.configs import (
        get_config,
        load_config,
        get_absolute_path,
        setup_logging,
        validate_config,
        get_model_path,
        get_raw_data_path,
        get_engineered_data_path,
        get_models_dir,
        get_reports_dir,
        get_plots_dir,
        get_training_params,
    )

Exported:
    ProjectConfig             — Root frozen dataclass for configuration.
    ProjectMeta               — Top-level project identity metadata.
    PathsConfig               — All I/O directory and file path settings.
    DataConfig                — Data loading options.
    PreprocessingConfig       — Preprocessor settings.
    FeatureEngineeringConfig   — Feature engineering settings.
    ModelConfig               — Model hyperparameters.
    EvaluationConfig          — Evaluation & EDA settings.
    LoggingConfig             — Logger settings.
    PlotConfig                — Aesthetics & plot settings.

    get_config                — Singleton accessor returning the active ProjectConfig.
    load_config               — Force-load a ProjectConfig from a YAML file path.
    get_absolute_path         — Resolve a config-relative path to an absolute Path.
    setup_logging             — Configure the root Python logger from config settings.
    display_config            — Pretty-print the active configuration to stdout.
    validate_config           — Validate parameter bounds, ranges, and required paths.

    get_model_path            — Absolute Path to the trained model artifact.
    get_raw_data_path         — Absolute Path to the raw dataset file.
    get_engineered_data_path  — Absolute Path to the engineered features dataset file.
    get_models_dir            — Absolute Path to the models directory.
    get_reports_dir           — Absolute Path to the reports directory.
    get_plots_dir             — Absolute Path to the plots directory.
    get_training_params       — Dict of model training hyper-parameters.
"""

from .config import (
    DataConfig,
    EvaluationConfig,
    FeatureEngineeringConfig,
    LoggingConfig,
    ModelConfig,
    PathsConfig,
    PlotConfig,
    PreprocessingConfig,
    ProjectConfig,
    ProjectMeta,
    display_config,
    get_absolute_path,
    get_config,
    get_engineered_data_path,
    get_model_path,
    get_models_dir,
    get_plots_dir,
    get_raw_data_path,
    get_reports_dir,
    get_training_params,
    load_config,
    setup_logging,
    validate_config,
)

__all__ = [
    "ProjectConfig",
    "ProjectMeta",
    "PathsConfig",
    "DataConfig",
    "PreprocessingConfig",
    "FeatureEngineeringConfig",
    "ModelConfig",
    "EvaluationConfig",
    "LoggingConfig",
    "PlotConfig",
    "get_config",
    "load_config",
    "get_absolute_path",
    "setup_logging",
    "display_config",
    "validate_config",
    "get_model_path",
    "get_raw_data_path",
    "get_engineered_data_path",
    "get_models_dir",
    "get_reports_dir",
    "get_plots_dir",
    "get_training_params",
]
