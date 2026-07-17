"""
src/pipeline/__init__.py
========================
Exposes the public interface of the prediction pipeline sub-package.

Exports:
    PredictionPipeline — end-to-end inference orchestrator:
                         loads a trained model, accepts raw sensor data
                         (CSV, DataFrame, or single dict), applies optional
                         preprocessing / feature engineering, generates
                         class predictions and confidence scores, displays
                         colour-coded maintenance recommendations, and
                         exports results to CSV or JSON.

    PredictionResult   — lightweight result container returned by every
                         PredictionPipeline.predict* call; exposes
                         .predictions, .probabilities, .confidence,
                         and .to_dataframe().
"""

from .predict import PredictionPipeline, PredictionResult

__all__ = ["PredictionPipeline", "PredictionResult"]
