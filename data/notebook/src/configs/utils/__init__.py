"""
src/configs/utils/__init__.py
==============================
Exposes the public interface of the project utility sub-package.

.. note::
    **Canonical inference location has moved.**

    The dedicated, first-class inference module for this project is now
    ``src.configs.inference``.  For new code, prefer importing from there::

        from src.configs.inference import InferencePipeline, InferenceResult, run_inference

    The symbols below (``PredictionPipeline``, ``PredictionResult``) are
    retained in this module for **backward-compatibility** and will continue
    to work without any changes to existing call sites.

Exports:
    PredictionPipeline — end-to-end inference orchestrator (back-compat):
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

    save_model         — persist any joblib-serialisable model to disk;
                         wraps joblib.dump with directory auto-creation,
                         atomic temp-file writes, and structured logging.

    load_model         — deserialise a saved model artifact from disk;
                         raises an informative FileNotFoundError when the
                         file is absent (instead of an opaque IOError).
"""

from .predict import PredictionPipeline, PredictionResult
from .model_manager import save_model, load_model

__all__ = [
    "PredictionPipeline",
    "PredictionResult",
    "save_model",
    "load_model",
]
