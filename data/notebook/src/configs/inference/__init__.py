"""
src/configs/inference/__init__.py
==================================
Public interface for the IoT Predictive Maintenance inference sub-package.

This package is the **canonical inference layer** of the project.
It provides a reusable, model-agnostic pipeline that loads a trained model
artifact, accepts raw or pre-processed sensor data in any convenient format,
applies optional preprocessing and feature-engineering stages, and returns
richly-structured prediction results.

Usage::

    # One-liner convenience function (recommended for notebooks / scripts)
    from src.configs.inference import run_inference

    result = run_inference("data/raw/new_sensor_batch.csv")
    result.display()
    df = result.to_dataframe()

    # Fine-grained control
    from src.configs.inference import InferencePipeline, InferenceResult

    pipeline = InferencePipeline(
        model_path="outputs/models/random_forest_baseline.joblib",
        preprocessor=fitted_preprocessor,   # optional
        feature_engineer=fitted_engineer,   # optional
        target_col="failure",
    )
    result = pipeline.run(sensor_df)        # DataFrame, dict, ndarray, or CSV path
    pipeline.display_results(result)
    pipeline.save_results(result, "outputs/reports/predictions.json", fmt="json")

Exported symbols:
    InferencePipeline  — end-to-end inference orchestrator:
                         loads a trained model, accepts raw sensor data
                         (dict, DataFrame, ndarray, or CSV path), applies
                         optional preprocessing / feature engineering,
                         generates class predictions and confidence scores,
                         displays colour-coded maintenance recommendations,
                         and exports results to CSV or JSON.

    InferenceResult    — structured result container returned by every
                         InferencePipeline.run() / predict*() call;
                         exposes .predictions, .probabilities, .confidence,
                         .urgency_summary, .to_dataframe(), .to_dict(),
                         and .display().

    run_inference      — module-level one-liner convenience function that
                         creates an InferencePipeline, runs inference,
                         optionally displays and saves the results, and
                         returns the InferenceResult.  Mirrors the
                         run_evaluation() pattern from the evaluation package.
"""

from .inference_pipeline import InferencePipeline, InferenceResult, run_inference

__all__ = [
    "InferencePipeline",
    "InferenceResult",
    "run_inference",
]
