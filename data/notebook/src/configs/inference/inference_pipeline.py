"""
inference_pipeline.py
=====================
Reusable inference and prediction pipeline for the IoT Predictive Maintenance project.

This module is the **canonical inference layer** of the project.  It loads a
pre-trained model artifact from disk, accepts new sensor data in any of several
convenient formats, applies optional preprocessing and feature-engineering
stages, and returns a richly-structured ``InferenceResult`` that includes
predictions, confidence scores, and human-readable maintenance recommendations.

Pipeline:
    raw input  →  [optional: preprocess]  →  [optional: feature-engineer]
    →  feature alignment  →  model.predict  →  model.predict_proba
    →  InferenceResult

Design goals:
    - **Reusability**: ``InferencePipeline`` is model-agnostic; it works with
      any joblib-serialised sklearn-compatible estimator.
    - **Flexible input**: The :meth:`InferencePipeline.run` entry point auto-
      detects whether the caller supplied a dict, DataFrame, ndarray, or a
      path to a CSV file, so no boilerplate is needed at call sites.
    - **Model persistence integration**: Uses :func:`load_model` from the
      project's ``model_manager`` utility (the single source of truth for
      model serialisation / deserialisation) rather than calling
      ``joblib.load`` directly.
    - **Structured output**: Every predict call returns an ``InferenceResult``
      with predictions, probabilities, per-sample confidence, urgency levels,
      recommended actions, a timestamp, and pipeline metadata.  The result
      can be converted to a tidy ``pd.DataFrame`` or a JSON-serialisable
      ``dict`` with a single method call.
    - **Traceability**: All significant operations are logged through the
      standard Python ``logging`` framework.
    - **Safety**: Informative exceptions are raised for every error condition
      (missing file, empty input, wrong types, shape mismatches …) so that
      callers receive actionable messages rather than cryptic tracebacks.

Typical usage::

    # --- One-liner convenience function -----------------------------------
    from src.configs.inference.inference_pipeline import run_inference

    result = run_inference(sensor_data)          # dict, DataFrame, or CSV path
    result.display()
    df = result.to_dataframe()

    # --- Fine-grained control ---------------------------------------------
    from src.configs.inference import InferencePipeline

    pipeline = InferencePipeline(
        model_path="outputs/models/random_forest_baseline.joblib",
        preprocessor=fitted_preprocessor,          # optional
        feature_engineer=fitted_engineer,          # optional
        target_col="failure",
    )
    result = pipeline.run(sensor_df)
    pipeline.display_results(result)
    pipeline.save_results(result, "outputs/reports/predictions.csv")

References:
    - src/configs/utils/model_manager.py   — load_model / save_model
    - src/configs/config.py                — get_config / get_absolute_path
    - src/configs/evaluation/evaluation_pipeline.py — architectural reference

Part of the Infotact Solutions Data Science & Machine Learning Internship project.
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Suppress non-critical sklearn warnings
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Centralised configuration defaults
# ---------------------------------------------------------------------------
# Falls back to hard-coded sensible defaults when the config layer is not
# importable (e.g. during isolated unit tests or standalone script runs).
try:
    from src.configs.config import get_config as _get_cfg, get_absolute_path
    _cfg = _get_cfg()

    _DEFAULT_MODEL_PATH  = str(
        get_absolute_path(
            f"{_cfg.paths.models_dir}/{_cfg.paths.model_filename}"
        )
    )
    _DEFAULT_CONF_THRESH  = _cfg.evaluation.confidence_threshold
    _DEFAULT_REPORTS_DIR  = str(get_absolute_path(_cfg.paths.reports_dir))
    _DEFAULT_TARGET_COL   = _cfg.model.target_col
    _DEFAULT_PRED_OUTPUT  = str(get_absolute_path(_cfg.paths.predictions_output))

except Exception:
    _DEFAULT_MODEL_PATH  = "outputs/models/random_forest_baseline.joblib"
    _DEFAULT_CONF_THRESH  = 0.50
    _DEFAULT_REPORTS_DIR  = "outputs/reports"
    _DEFAULT_TARGET_COL   = "failure"
    _DEFAULT_PRED_OUTPUT  = "outputs/reports/predictions.csv"

# ---------------------------------------------------------------------------
# ANSI colour codes (terminal + Jupyter compatible)
# ---------------------------------------------------------------------------
_ANSI: Dict[str, str] = {
    "red":    "\033[91m",
    "yellow": "\033[93m",
    "green":  "\033[92m",
    "cyan":   "\033[96m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}

# ---------------------------------------------------------------------------
# Default maintenance-action label map
# ---------------------------------------------------------------------------
# Maps predicted class integer label → human-readable urgency + action text.
# Users can override this by passing a custom ``label_map`` to __init__.
_DEFAULT_LABEL_MAP: Dict[Any, Dict[str, str]] = {
    0: {
        "urgency": "NORMAL",
        "action":  "No maintenance required. Continue normal monitoring.",
        "colour":  "green",
    },
    1: {
        "urgency": "MAINTENANCE REQUIRED",
        "action":  (
            "Schedule preventive maintenance. Inspect components, "
            "lubricate moving parts, and verify sensor calibration."
        ),
        "colour":  "yellow",
    },
    2: {
        "urgency": "CRITICAL — IMMEDIATE ACTION",
        "action":  (
            "STOP MACHINE IMMEDIATELY. Critical failure risk detected. "
            "Full inspection and component replacement required."
        ),
        "colour":  "red",
    },
}


# ===========================================================================
# InferenceResult — structured result container
# ===========================================================================


class InferenceResult:
    """
    Structured container for the output of a single inference call.

    Every ``predict*`` / ``run`` call on :class:`InferencePipeline` returns
    one ``InferenceResult``.  The object is intentionally lightweight — it
    stores arrays and a metadata dict and exposes two conversion helpers
    (:meth:`to_dataframe` and :meth:`to_dict`) that produce analysis-ready
    representations.

    Attributes:
        predictions    (np.ndarray): Predicted class labels  (shape: n_samples).
        probabilities  (np.ndarray | None):
            Class probability matrix (n_samples, n_classes), or ``None``
            when the model does not expose ``predict_proba``.
        confidence     (np.ndarray):
            Maximum probability per sample (n_samples,).
            Filled with ``np.nan`` when probabilities are unavailable.
        feature_matrix (pd.DataFrame):
            Aligned feature matrix that was sent to the model.
        label_map      (dict):  Class label → urgency / action mapping.
        class_labels   (list | None): Class labels from the fitted model.
        n_samples      (int):   Number of samples predicted.
        timestamp      (str):   ISO-8601 timestamp of the inference call.
        metadata       (dict):  Pipeline metadata snapshot (model name,
                                feature count, pipeline stages, etc.).

    Example::

        result = pipeline.run({"temperature": 85.3, "vibration": 0.42})
        print(result)                   # concise repr
        df = result.to_dataframe()      # tidy DataFrame
        d  = result.to_dict()           # JSON-serialisable dict
        result.display()                # coloured terminal report
    """

    def __init__(
        self,
        predictions:    np.ndarray,
        probabilities:  Optional[np.ndarray],
        feature_matrix: pd.DataFrame,
        label_map:      Dict[Any, Dict[str, str]],
        class_labels:   Optional[List[Any]] = None,
        metadata:       Optional[Dict[str, Any]] = None,
    ) -> None:
        self.predictions:    np.ndarray           = predictions
        self.probabilities:  Optional[np.ndarray] = probabilities
        self.feature_matrix: pd.DataFrame         = feature_matrix
        self.label_map:      Dict                 = label_map
        self.class_labels:   Optional[List[Any]]  = class_labels
        self.n_samples:      int                  = len(predictions)
        self.timestamp:      str                  = datetime.now().isoformat(timespec="seconds")
        self.metadata:       Dict[str, Any]       = metadata or {}

        # ── Confidence = max(predict_proba) per row ───────────────────────
        if probabilities is not None:
            proba_2d = (
                probabilities if probabilities.ndim == 2
                else probabilities.reshape(-1, 1)
            )
            self.confidence: np.ndarray = proba_2d.max(axis=1)
        else:
            self.confidence = np.full(self.n_samples, np.nan)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def urgency_summary(self) -> Dict[str, int]:
        """
        Return a dict mapping urgency label → count of samples at that level.

        Example::

            >>> result.urgency_summary
            {"NORMAL": 80, "MAINTENANCE REQUIRED": 15, "CRITICAL — IMMEDIATE ACTION": 5}
        """
        summary: Dict[str, int] = {}
        for pred in self.predictions:
            info = self.label_map.get(pred, {"urgency": str(pred)})
            urg = info["urgency"]
            summary[urg] = summary.get(urg, 0) + 1
        return summary

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the inference results to a tidy ``pd.DataFrame``.

        Columns produced:
            - ``sample_id``   — 0-based row index
            - ``prediction``  — predicted class label
            - ``confidence``  — maximum class probability (``None`` if N/A)
            - ``urgency``     — human-readable urgency level
            - ``action``      — recommended maintenance action
            - ``prob_{cls}``  — per-class probability (one column per class,
                                 only when ``probabilities`` is available)

        Returns:
            pd.DataFrame: One row per sample.

        Example::

            df = result.to_dataframe()
            critical = df[df["urgency"] == "CRITICAL — IMMEDIATE ACTION"]
        """
        records = []
        for i in range(self.n_samples):
            pred = self.predictions[i]
            conf = self.confidence[i]
            info = self.label_map.get(pred, {
                "urgency": str(pred),
                "action":  "Unknown class — check label_map configuration.",
                "colour":  "cyan",
            })

            row: Dict[str, Any] = {
                "sample_id":  i,
                "prediction": pred,
                "confidence": round(float(conf), 4) if not np.isnan(conf) else None,
                "urgency":    info["urgency"],
                "action":     info["action"],
            }

            # Per-class probability columns
            if self.probabilities is not None and self.probabilities.ndim == 2:
                for j, cls in enumerate(
                    self.class_labels or range(self.probabilities.shape[1])
                ):
                    row[f"prob_{cls}"] = round(float(self.probabilities[i, j]), 4)

            records.append(row)

        return pd.DataFrame(records)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the inference results to a JSON-serialisable ``dict``.

        Structure::

            {
                "timestamp":  "2026-07-23T18:00:00",
                "n_samples":  100,
                "metadata":   { ... pipeline info ... },
                "urgency_summary": { "NORMAL": 80, ... },
                "predictions": [
                    {
                        "sample_id":  0,
                        "prediction": 0,
                        "confidence": 0.94,
                        "urgency":    "NORMAL",
                        "action":     "No maintenance required ...",
                        "prob_0":     0.94,
                        "prob_1":     0.06,
                    },
                    ...
                ]
            }

        Returns:
            dict: Fully JSON-serialisable representation of the results.
        """
        df = self.to_dataframe()
        records = df.where(pd.notna(df), other=None).to_dict(orient="records")
        return {
            "timestamp":       self.timestamp,
            "n_samples":       self.n_samples,
            "metadata":        {k: str(v) for k, v in self.metadata.items()},
            "urgency_summary": self.urgency_summary,
            "predictions":     records,
        }

    def display(self, max_rows: int = 50, colour: bool = True) -> None:
        """
        Print a formatted prediction report to stdout.

        Delegates to :meth:`InferencePipeline.display_results` using the
        metadata stored on this result object.

        Args:
            max_rows (int): Maximum sample rows to print. Defaults to ``50``.
            colour (bool):  ANSI colour output. Defaults to ``True``.
        """
        conf_thresh = self.metadata.get("confidence_threshold", _DEFAULT_CONF_THRESH)
        _display_results(self, conf_thresh=conf_thresh, max_rows=max_rows, colour=colour)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"InferenceResult("
            f"n_samples={self.n_samples}, "
            f"timestamp='{self.timestamp}', "
            f"has_probabilities={self.probabilities is not None})"
        )

    def __len__(self) -> int:
        return self.n_samples


# ===========================================================================
# Module-level display helper (used by both InferenceResult.display and
# InferencePipeline.display_results to avoid duplicating rendering logic)
# ===========================================================================


def _display_results(
    result: "InferenceResult",
    conf_thresh: float = _DEFAULT_CONF_THRESH,
    max_rows: int = 50,
    colour: bool = True,
) -> None:
    """Render a formatted inference report to stdout."""
    sep  = "─" * 72
    sep2 = "═" * 72
    c    = _ANSI if colour else {k: "" for k in _ANSI}

    print(f"\n{c['bold']}{sep2}{c['reset']}")
    print(f"{c['bold']}  PREDICTIVE MAINTENANCE — INFERENCE RESULTS{c['reset']}")
    print(f"{c['bold']}{sep2}{c['reset']}")
    print(f"  Timestamp        : {result.timestamp}")
    print(f"  Samples          : {result.n_samples:,}")
    print(
        f"  Confidence scores: "
        f"{'Available' if result.probabilities is not None else 'Not available'}"
    )
    print(f"  Confidence thresh: {conf_thresh:.0%}")
    if result.metadata:
        print(f"  Model            : {result.metadata.get('model_name', 'N/A')}")
        print(f"  Model type       : {result.metadata.get('model_type', 'N/A')}")
    print(sep)

    # ── Per-sample table ──────────────────────────────────────────────
    display_df   = result.to_dataframe()
    rows_to_show = min(max_rows, len(display_df))

    print(
        f"\n  {'#':<6} {'Prediction':<14} {'Confidence':>11}  "
        f"{'Urgency Level':<28} {'Action (truncated)'}"
    )
    print(f"  {'─'*6} {'─'*14} {'─'*11}  {'─'*28} {'─'*30}")

    for _, row in display_df.iloc[:rows_to_show].iterrows():
        pred   = row["prediction"]
        conf   = row["confidence"]
        urg    = row["urgency"]
        action = row["action"][:55] + "…" if len(row["action"]) > 55 else row["action"]

        conf_str = f"{conf:.1%}" if conf is not None else "  N/A  "
        flag     = ""
        if conf is not None and conf < conf_thresh:
            flag = f" {c['yellow']}⚠ low conf{c['reset']}"

        info    = result.label_map.get(pred, {"colour": "cyan"})
        urg_col = c.get(info.get("colour", "cyan"), "")

        print(
            f"  {int(row['sample_id']):<6} {str(pred):<14} {conf_str:>11}  "
            f"{urg_col}{urg:<28}{c['reset']} {action}{flag}"
        )

    if rows_to_show < result.n_samples:
        print(
            f"\n  … {result.n_samples - rows_to_show:,} more rows not shown "
            f"(max_rows={max_rows})."
        )

    # ── Summary section ───────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  PREDICTION SUMMARY")
    print(sep)

    class_counts = pd.Series(result.predictions).value_counts().sort_index()
    for cls, cnt in class_counts.items():
        pct     = cnt / result.n_samples * 100
        info    = result.label_map.get(cls, {"urgency": str(cls), "colour": "cyan"})
        urg_col = c.get(info.get("colour", "cyan"), "")
        bar     = "█" * int(pct / 5)
        print(
            f"  Class {str(cls):<6} → {cnt:>5,} sample(s)  "
            f"({pct:>5.1f}%)  {urg_col}{bar}{c['reset']}"
        )

    valid_conf = result.confidence[~np.isnan(result.confidence)]
    if len(valid_conf) > 0:
        print(sep)
        print(
            f"  Confidence  min={valid_conf.min():.1%} | "
            f"mean={valid_conf.mean():.1%} | "
            f"max={valid_conf.max():.1%}"
        )
        low_conf_n = int((valid_conf < conf_thresh).sum())
        if low_conf_n > 0:
            print(
                f"  {c['yellow']}⚠  {low_conf_n:,} sample(s) below confidence "
                f"threshold ({conf_thresh:.0%}) — review manually.{c['reset']}"
            )
        else:
            print(f"  {c['green']}✓  All samples above confidence threshold.{c['reset']}")

    # ── Urgency summary ───────────────────────────────────────────────
    print(sep)
    print("  URGENCY BREAKDOWN")
    print(sep)
    for urg_label, count in result.urgency_summary.items():
        print(f"  {urg_label:<38} : {count:>5,} sample(s)")

    print(f"{sep2}\n")


# ===========================================================================
# InferencePipeline — main orchestrator
# ===========================================================================


class InferencePipeline:
    """
    End-to-end reusable inference pipeline for IoT predictive maintenance.

    Loads a pre-trained joblib model artifact and orchestrates the full
    inference workflow from raw sensor input to a structured
    :class:`InferenceResult`::

        raw input  →  [optional: preprocess]  →  [optional: feature-engineer]
        →  feature alignment  →  model.predict  →  model.predict_proba
        →  InferenceResult

    The pipeline is designed to be run independently of the training
    workflow.  Preprocessing and feature-engineering stages are optional —
    when the caller provides already-processed feature DataFrames, those
    stages are skipped.

    Attributes:
        model_path           (Path):   Resolved path to the ``.joblib`` file.
        model                (Any):    Loaded sklearn estimator.
        feature_names        (list | None): Expected feature column names.
        preprocessor         (Any | None): Fitted preprocessor instance.
        feature_engineer     (Any | None): Fitted feature engineer instance.
        target_col           (str | None): Target column to drop before inference.
        label_map            (dict):   Class label → urgency / action mapping.
        confidence_threshold (float):  Min confidence to flag as high-confidence.
        _results_history     (list):   All InferenceResult objects from this session.

    Example — minimal (pre-engineered features)::

        pipeline = InferencePipeline("outputs/models/random_forest_baseline.joblib")
        result   = pipeline.run(feature_df)
        result.display()

    Example — full pipeline (raw sensor dict)::

        pipeline = InferencePipeline(
            model_path="outputs/models/random_forest_baseline.joblib",
            preprocessor=fitted_preprocessor,
            feature_engineer=fitted_engineer,
            target_col="failure",
        )
        result = pipeline.run({"temperature": 85.3, "vibration": 0.42})
        pipeline.save_results(result, "outputs/reports/predictions.json", fmt="json")

    Example — CSV batch inference::

        result = pipeline.run("data/raw/new_sensor_batch.csv")
    """

    def __init__(
        self,
        model_path:           Union[str, Path]    = _DEFAULT_MODEL_PATH,
        feature_names:        Optional[List[str]] = None,
        preprocessor:         Optional[Any]       = None,
        feature_engineer:     Optional[Any]       = None,
        target_col:           Optional[str]       = _DEFAULT_TARGET_COL,
        label_map:            Optional[Dict]      = None,
        confidence_threshold: float               = _DEFAULT_CONF_THRESH,
    ) -> None:
        """
        Initialise the InferencePipeline and load the model from disk.

        Args:
            model_path (str | Path):
                Absolute or relative path to the saved ``.joblib`` model
                artifact.  Defaults to the path configured in ``config.yaml``.
            feature_names (List[str], optional):
                Ordered list of feature column names the model was trained on.
                When ``None``, the pipeline attempts to read
                ``feature_names_in_`` from the loaded model (sklearn ≥ 1.0).
                If neither source is available, no column reordering is applied.
            preprocessor (DataPreprocessor, optional):
                A **fitted** ``DataPreprocessor`` instance.  When provided,
                raw input DataFrames are passed through
                ``preprocessor.transform(df)`` before feature engineering.
            feature_engineer (FeatureEngineer, optional):
                A **fitted** ``FeatureEngineer`` instance.  When provided,
                the preprocessed DataFrame is passed through
                ``feature_engineer.transform(df)`` before prediction.
            target_col (str, optional):
                Name of the target / label column.  Dropped automatically
                from the input DataFrame if present.  Defaults to
                ``config.yaml → model.target_col`` (``"failure"``).
            label_map (dict, optional):
                Mapping from predicted class label → dict with keys
                ``"urgency"``, ``"action"``, and ``"colour"``.
                Defaults to :data:`_DEFAULT_LABEL_MAP`.
            confidence_threshold (float):
                Minimum confidence score to consider a prediction
                "high-confidence".  Used in display formatting only.
                Must be in ``[0.0, 1.0]``.  Defaults to ``0.50``.

        Raises:
            FileNotFoundError: If the model file does not exist on disk.
            ValueError:        If ``confidence_threshold`` is outside ``[0, 1]``.
        """
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {confidence_threshold!r}."
            )

        self.model_path:           Path             = Path(model_path).resolve()
        self.preprocessor:         Optional[Any]    = preprocessor
        self.feature_engineer:     Optional[Any]    = feature_engineer
        self.target_col:           Optional[str]    = target_col
        self.label_map:            Dict             = label_map or _DEFAULT_LABEL_MAP
        self.confidence_threshold: float            = confidence_threshold
        self._results_history:     List[InferenceResult] = []

        # ── Load model artifact ───────────────────────────────────────────
        self.model = self._load_model(self.model_path)

        # ── Resolve training feature names ────────────────────────────────
        self.feature_names: Optional[List[str]] = (
            feature_names
            or self._infer_feature_names(self.model)
        )

        # ── Determine class labels ────────────────────────────────────────
        self._class_labels: Optional[List[Any]] = (
            list(self.model.classes_)
            if hasattr(self.model, "classes_")
            else None
        )

        logger.info(
            "InferencePipeline initialised — model='%s', "
            "n_features=%s, n_classes=%s, "
            "preprocessor=%s, feature_engineer=%s.",
            self.model_path.name,
            len(self.feature_names) if self.feature_names else "unknown",
            len(self._class_labels) if self._class_labels else "unknown",
            type(preprocessor).__name__ if preprocessor else "None",
            type(feature_engineer).__name__ if feature_engineer else "None",
        )

    # ==================================================================
    # Primary public entry point
    # ==================================================================

    def run(
        self,
        data: Union[pd.DataFrame, np.ndarray, Dict[str, float], str, Path],
        apply_preprocessing:      bool = True,
        apply_feature_engineering: bool = True,
        csv_kwargs:               Optional[Dict] = None,
    ) -> "InferenceResult":
        """
        Run the full inference pipeline on any supported input format.

        This is the **recommended entry point** for all inference calls.
        It auto-detects the input type and dispatches to the appropriate
        internal method.

        Supported input types:
            - ``dict``        — single sensor-reading sample
                                e.g. ``{"temperature": 85.3, "vibration": 0.42}``
            - ``pd.DataFrame`` — feature matrix (single or batch)
            - ``np.ndarray``  — numeric array (columns inferred from model)
            - ``str`` / ``Path`` — path to a CSV file

        Args:
            data:
                Input sensor data.  See supported types above.
            apply_preprocessing (bool):
                Apply the configured :attr:`preprocessor` (if set).
                Defaults to ``True``.
            apply_feature_engineering (bool):
                Apply the configured :attr:`feature_engineer` (if set).
                Defaults to ``True``.
            csv_kwargs (dict, optional):
                Extra keyword arguments forwarded to ``pd.read_csv`` when
                *data* is a file path.

        Returns:
            InferenceResult: Structured prediction result.

        Raises:
            TypeError:        If *data* is an unsupported type.
            FileNotFoundError: If *data* is a path that does not exist.
            ValueError:       If the processed input is empty.

        Example::

            # Single reading
            result = pipeline.run({"temperature": 85.3, "vibration": 0.42})

            # Batch DataFrame
            result = pipeline.run(feature_df)

            # CSV file
            result = pipeline.run("data/raw/new_readings.csv")
        """
        logger.info("InferencePipeline.run() — input type: %s.", type(data).__name__)

        # ── Dispatch by input type ────────────────────────────────────────
        if isinstance(data, dict):
            return self.predict_single(data)

        if isinstance(data, (str, Path)):
            return self.predict_from_csv(
                data,
                apply_preprocessing=apply_preprocessing,
                apply_feature_engineering=apply_feature_engineering,
                csv_kwargs=csv_kwargs,
            )

        if isinstance(data, (pd.DataFrame, np.ndarray)):
            return self.predict_from_dataframe(
                data if isinstance(data, pd.DataFrame)
                else self._ndarray_to_dataframe(data),
                apply_preprocessing=apply_preprocessing,
                apply_feature_engineering=apply_feature_engineering,
            )

        raise TypeError(
            f"Unsupported input type '{type(data).__name__}'. "
            "Expected one of: dict, pd.DataFrame, np.ndarray, str (CSV path), or Path."
        )

    # ==================================================================
    # Specialised prediction entry points
    # ==================================================================

    def predict(
        self,
        data: Union[pd.DataFrame, np.ndarray],
    ) -> "InferenceResult":
        """
        Generate predictions from a pre-engineered feature matrix.

        Use this method when *data* already contains the exact feature
        columns the model expects (i.e., preprocessing and feature
        engineering have been applied upstream).

        Args:
            data (pd.DataFrame | np.ndarray):
                Feature matrix.  If a DataFrame is passed, any column
                named :attr:`target_col` is silently dropped, and columns
                are reordered / padded to match :attr:`feature_names`.

        Returns:
            InferenceResult: Structured prediction result.

        Raises:
            TypeError:  If *data* is not a DataFrame or ndarray.
            ValueError: If no usable feature columns remain after alignment.
        """
        if isinstance(data, np.ndarray):
            data = self._ndarray_to_dataframe(data)
        elif not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"data must be pd.DataFrame or np.ndarray, got {type(data).__name__}."
            )

        # Drop target column if accidentally present
        if self.target_col and self.target_col in data.columns:
            data = data.drop(columns=[self.target_col])
            logger.debug("Dropped target column '%s' from input.", self.target_col)

        # Align feature columns to training layout
        X = self._align_features(data)

        if X.shape[1] == 0:
            raise ValueError(
                "No feature columns remain after alignment. "
                "Ensure the input DataFrame contains the expected sensor columns."
            )

        logger.info(
            "predict() — %d sample(s), %d feature(s).",
            X.shape[0], X.shape[1],
        )

        # ── Core inference ────────────────────────────────────────────────
        predictions   = self.model.predict(X)
        probabilities = self._get_probabilities(X)

        result = InferenceResult(
            predictions=predictions,
            probabilities=probabilities,
            feature_matrix=X,
            label_map=self.label_map,
            class_labels=self._class_labels,
            metadata=self._build_metadata(),
        )
        self._results_history.append(result)

        logger.info(
            "Inference complete — %d sample(s) | classes predicted: %s.",
            result.n_samples, np.unique(predictions).tolist(),
        )
        return result

    def predict_single(
        self,
        sensor_readings: Dict[str, float],
    ) -> "InferenceResult":
        """
        Generate a prediction for a single sensor reading dictionary.

        Convenience method for real-time single-sample inference without
        needing to construct a DataFrame manually.

        Args:
            sensor_readings (Dict[str, float]):
                Mapping of feature name → current sensor value.
                Example: ``{"temperature": 85.3, "vibration": 0.42}``

        Returns:
            InferenceResult: Single-sample prediction result.

        Raises:
            ValueError: If *sensor_readings* is empty.

        Example::

            result = pipeline.predict_single(
                {"temperature": 85.3, "vibration": 0.42, "pressure": 2.1}
            )
        """
        if not sensor_readings:
            raise ValueError(
                "sensor_readings must not be empty. "
                "Provide at least one feature name → value mapping."
            )

        df = pd.DataFrame([sensor_readings])
        logger.info(
            "predict_single() — %d feature(s): %s.",
            len(sensor_readings), list(sensor_readings.keys()),
        )
        return self.predict(df)

    def predict_from_csv(
        self,
        csv_path:                  Union[str, Path],
        apply_preprocessing:       bool = True,
        apply_feature_engineering: bool = True,
        csv_kwargs:                Optional[Dict] = None,
    ) -> "InferenceResult":
        """
        Load sensor data from a CSV file and run the full prediction pipeline.

        Args:
            csv_path (str | Path):
                Path to the input CSV file.
            apply_preprocessing (bool):
                Apply the configured preprocessor (if set). Defaults to ``True``.
            apply_feature_engineering (bool):
                Apply the configured feature engineer (if set). Defaults to ``True``.
            csv_kwargs (dict, optional):
                Extra keyword arguments forwarded to ``pd.read_csv``
                (e.g. ``{"sep": ";", "encoding": "latin-1"}``).

        Returns:
            InferenceResult: Prediction results for all rows in the CSV.

        Raises:
            FileNotFoundError: If *csv_path* does not exist.
            ValueError:        If the loaded file is empty.
        """
        path = Path(csv_path).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"CSV file not found: '{path}'. "
                "Check the path and ensure the file exists before running inference."
            )

        logger.info("Loading sensor data from '%s' …", path)
        try:
            df = pd.read_csv(path, **(csv_kwargs or {}))
        except Exception as exc:
            logger.error("Failed to read CSV '%s': %s", path, exc)
            raise

        if df.empty:
            raise ValueError(
                f"The CSV file '{path}' is empty. "
                "Provide a file with at least one data row."
            )

        logger.info("CSV loaded — %d rows × %d cols.", df.shape[0], df.shape[1])
        return self.predict_from_dataframe(
            df=df,
            apply_preprocessing=apply_preprocessing,
            apply_feature_engineering=apply_feature_engineering,
        )

    def predict_from_dataframe(
        self,
        df:                        pd.DataFrame,
        apply_preprocessing:       bool = True,
        apply_feature_engineering: bool = True,
    ) -> "InferenceResult":
        """
        Run the full pipeline on a raw sensor DataFrame.

        Applies preprocessing → feature engineering → prediction in sequence.
        Stages are skipped when the corresponding component is ``None`` or
        the corresponding ``apply_*`` flag is ``False``.

        Args:
            df (pd.DataFrame): Raw or semi-processed sensor DataFrame.
            apply_preprocessing (bool):
                Run the preprocessor (if configured). Defaults to ``True``.
            apply_feature_engineering (bool):
                Run the feature engineer (if configured). Defaults to ``True``.

        Returns:
            InferenceResult: Prediction results for all rows in *df*.

        Raises:
            TypeError: If *df* is not a ``pd.DataFrame``.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected pd.DataFrame, got {type(df).__name__}. "
                "Use predict_single() for dict inputs or run() for auto-dispatch."
            )

        result_df = df.copy(deep=True)

        # ── Stage 1: Preprocessing ────────────────────────────────────────
        if apply_preprocessing and self.preprocessor is not None:
            logger.info("Stage 1/3 — Applying preprocessing …")
            try:
                result_df = self.preprocessor.transform(result_df)
                logger.info(
                    "Preprocessing complete — shape after: %s.", result_df.shape
                )
            except Exception as exc:
                logger.error("Preprocessing failed: %s", exc)
                raise

        # ── Stage 2: Feature engineering ──────────────────────────────────
        if apply_feature_engineering and self.feature_engineer is not None:
            logger.info("Stage 2/3 — Applying feature engineering …")
            try:
                result_df = self.feature_engineer.transform(result_df)
                logger.info(
                    "Feature engineering complete — shape after: %s.", result_df.shape
                )
            except Exception as exc:
                logger.error("Feature engineering failed: %s", exc)
                raise

        # ── Stage 3: Predict ──────────────────────────────────────────────
        logger.info("Stage 3/3 — Running model inference …")
        return self.predict(result_df)

    # ==================================================================
    # Display and persistence
    # ==================================================================

    def display_results(
        self,
        result: "InferenceResult",
        max_rows: int = 50,
        colour:   bool = True,
    ) -> None:
        """
        Print a formatted inference report to stdout.

        For each sample the report shows the predicted class, confidence
        score, colour-coded urgency level, and recommended maintenance
        action.  A summary section displays class distribution, confidence
        statistics, and urgency breakdown.

        Args:
            result (InferenceResult): Output from any ``predict*`` / ``run`` call.
            max_rows (int): Max sample rows to display. Defaults to ``50``.
            colour (bool):  ANSI colour codes. Defaults to ``True``.
        """
        _display_results(
            result,
            conf_thresh=self.confidence_threshold,
            max_rows=max_rows,
            colour=colour,
        )
        logger.info(
            "Results displayed — %d sample(s), urgency: %s.",
            result.n_samples, result.urgency_summary,
        )

    def save_results(
        self,
        result:           "InferenceResult",
        output_path:      Union[str, Path] = _DEFAULT_PRED_OUTPUT,
        fmt:              str = "csv",
        include_features: bool = False,
    ) -> Path:
        """
        Persist inference results to a CSV or JSON file.

        Args:
            result (InferenceResult):
                Output from any ``predict*`` / ``run`` call.
            output_path (str | Path):
                Destination file path.  Parent directories are created
                automatically.  Defaults to
                ``config.yaml → paths.predictions_output``.
            fmt (str):
                Output format — ``"csv"`` (default) or ``"json"``.
            include_features (bool):
                When ``True``, appends the aligned feature matrix columns
                alongside the prediction columns.  Defaults to ``False``.

        Returns:
            Path: Resolved absolute path of the saved file.

        Raises:
            ValueError: If *fmt* is not ``"csv"`` or ``"json"``.
            OSError:    If the file cannot be written.

        Example::

            path = pipeline.save_results(result, "outputs/reports/run1.json", fmt="json")
            print(f"Saved to: {path}")
        """
        if fmt not in ("csv", "json"):
            raise ValueError(
                f"fmt must be 'csv' or 'json', got '{fmt}'. "
                "Choose one of the supported formats."
            )

        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_df = result.to_dataframe()

        if include_features:
            feat_df    = result.feature_matrix.reset_index(drop=True)
            results_df = pd.concat([results_df, feat_df], axis=1)

        try:
            if fmt == "csv":
                results_df.to_csv(output_path, index=False)
            else:
                payload = result.to_dict()
                if include_features:
                    # Replace flat prediction records with feature-augmented ones
                    feat_records = feat_df.where(pd.notna(feat_df), other=None).to_dict(
                        orient="records"
                    )
                    for rec, feat in zip(payload["predictions"], feat_records):
                        rec.update(feat)
                with output_path.open("w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2, default=str)
        except Exception as exc:
            logger.error("Failed to save results to '%s': %s", output_path, exc)
            raise

        size_kb = output_path.stat().st_size / 1024
        logger.info(
            "Inference results saved → '%s'  [%.1f KB, fmt=%s].",
            output_path, size_kb, fmt,
        )
        print(f"\n  ✓ Results saved → {output_path}  ({size_kb:.1f} KB)")
        return output_path

    # ==================================================================
    # Utility / introspection
    # ==================================================================

    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Return a dictionary of model metadata and pipeline configuration.

        Useful for logging the pipeline state or including as metadata in
        output reports.

        Returns:
            dict: Keys include ``model_type``, ``model_path``,
                  ``n_features``, ``class_labels``, ``has_preprocessor``,
                  ``has_engineer``, ``confidence_threshold``, and
                  ``total_predictions_this_session``.

        Example::

            info = pipeline.get_pipeline_info()
            print(info["model_type"])       # → "RandomForestClassifier"
            print(info["n_features"])       # → 42
        """
        return {
            "model_name":   self.model_path.name,
            "model_type":   type(self.model).__name__,
            "model_path":   str(self.model_path),
            "n_features":   (
                len(self.feature_names) if self.feature_names else "unknown"
            ),
            "feature_names":    self.feature_names,
            "class_labels":     self._class_labels,
            "has_preprocessor": self.preprocessor is not None,
            "has_engineer":     self.feature_engineer is not None,
            "target_col":       self.target_col,
            "confidence_threshold": self.confidence_threshold,
            "total_predictions_this_session": sum(
                r.n_samples for r in self._results_history
            ),
        }

    def display_pipeline_info(self) -> None:
        """Print a formatted summary of the loaded model and pipeline configuration."""
        info = self.get_pipeline_info()
        sep  = "─" * 65

        print(f"\n{sep}")
        print("  INFERENCE PIPELINE CONFIGURATION")
        print(sep)
        for key, val in info.items():
            if key == "feature_names":
                n = len(val) if val else 0
                print(f"  {'feature_names':<30} {n} column(s)")
                if val:
                    for i, feat in enumerate(val[:8], 1):
                        print(f"    {i:>2}. {feat}")
                    if len(val) > 8:
                        print(f"    … {len(val) - 8} more")
            else:
                print(f"  {str(key):<30} {val}")
        print(f"{sep}\n")

    # ==================================================================
    # Private helpers
    # ==================================================================

    @staticmethod
    def _load_model(path: Path) -> Any:
        """
        Load a joblib model artifact from *path* using the project's
        model persistence utility (:func:`load_model` from ``model_manager``).

        Args:
            path (Path): Absolute path to the ``.joblib`` file.

        Returns:
            Any: The deserialised sklearn estimator.

        Raises:
            FileNotFoundError: If the model file does not exist.
            RuntimeError:      If model loading fails for any other reason.
        """
        try:
            from src.configs.utils.model_manager import load_model
            return load_model(path)
        except ImportError:
            # Fallback: load directly with joblib if model_manager is unavailable
            import joblib
            if not path.exists():
                raise FileNotFoundError(
                    f"Model file not found: '{path}'. "
                    "Run the training pipeline first to generate the artifact."
                )
            model = joblib.load(path)
            size_kb = path.stat().st_size / 1024
            logger.info(
                "Model loaded from '%s'  [%.1f KB, type=%s].",
                path, size_kb, type(model).__name__,
            )
            print(f"  ✓ Model loaded — {path.name}  ({size_kb:.1f} KB)")
            return model
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model from '{path}': {exc}. "
                "The file may be corrupt. Try retraining and saving the model again."
            ) from exc

    @staticmethod
    def _infer_feature_names(model: Any) -> Optional[List[str]]:
        """
        Attempt to read feature names from the model's ``feature_names_in_``
        attribute (available on sklearn ≥ 1.0 estimators fitted on DataFrames).

        Args:
            model: Any fitted sklearn estimator.

        Returns:
            List[str] | None: Feature names, or ``None`` if unavailable.
        """
        if hasattr(model, "feature_names_in_"):
            names = list(model.feature_names_in_)
            logger.info(
                "Feature names inferred from model.feature_names_in_ (%d cols).",
                len(names),
            )
            return names
        logger.info(
            "model.feature_names_in_ not available — "
            "column reordering will not be applied."
        )
        return None

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder and zero-pad the input DataFrame to match the training feature layout.

        Algorithm:
            1. Select only numeric columns (sklearn does not accept object / datetime).
            2. If :attr:`feature_names` is known:
               a. Add any missing columns with value ``0.0`` (zero imputation).
               b. Reorder to the exact training column order.
               c. Drop any extra columns not seen during training.
            3. Otherwise return the numeric-only DataFrame as-is.

        Args:
            df (pd.DataFrame): Input feature DataFrame.

        Returns:
            pd.DataFrame: Aligned feature matrix ready for ``model.predict()``.
        """
        numeric_df = df.select_dtypes(include=[np.number])

        if not self.feature_names:
            logger.debug(
                "No training feature names known — passing %d numeric column(s) to model.",
                numeric_df.shape[1],
            )
            return numeric_df.reset_index(drop=True)

        # Add missing columns (zero-imputed)
        missing_cols = set(self.feature_names) - set(numeric_df.columns)
        if missing_cols:
            logger.warning(
                "%d training feature(s) missing from input — zero-imputing: %s.",
                len(missing_cols), sorted(missing_cols),
            )
            for col in missing_cols:
                numeric_df[col] = 0.0

        # Drop extra columns not seen during training
        extra_cols = set(numeric_df.columns) - set(self.feature_names)
        if extra_cols:
            logger.debug(
                "Dropping %d extra column(s) not in training feature set: %s.",
                len(extra_cols), sorted(extra_cols),
            )
            numeric_df = numeric_df.drop(columns=list(extra_cols))

        # Reorder to match the training column order exactly
        numeric_df = numeric_df[self.feature_names]

        logger.debug(
            "Feature alignment complete — %d column(s) in training order.",
            numeric_df.shape[1],
        )
        return numeric_df.reset_index(drop=True)

    def _get_probabilities(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Call ``predict_proba`` on the model if available.

        Args:
            X (pd.DataFrame): Aligned feature matrix.

        Returns:
            np.ndarray | None:
                Full probability matrix (n_samples, n_classes), or ``None``
                if the model does not expose ``predict_proba`` or if the
                call raises an exception.
        """
        if not hasattr(self.model, "predict_proba"):
            logger.info(
                "Model does not expose predict_proba — confidence scores unavailable."
            )
            return None
        try:
            proba = self.model.predict_proba(X)
            logger.debug("predict_proba returned shape %s.", proba.shape)
            return proba
        except Exception as exc:
            logger.warning("predict_proba call failed: %s. Confidence unavailable.", exc)
            return None

    def _ndarray_to_dataframe(self, arr: np.ndarray) -> pd.DataFrame:
        """
        Convert a raw numpy array to a DataFrame, using training feature names
        as column labels when available.

        Args:
            arr (np.ndarray): Input array of shape (n_samples, n_features).

        Returns:
            pd.DataFrame: DataFrame with appropriate column names.
        """
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        cols = (
            self.feature_names[:arr.shape[1]]
            if self.feature_names
            else [f"feature_{i}" for i in range(arr.shape[1])]
        )
        return pd.DataFrame(arr, columns=cols)

    def _build_metadata(self) -> Dict[str, Any]:
        """Build the metadata dict attached to every InferenceResult."""
        return {
            "model_name":           self.model_path.name,
            "model_type":           type(self.model).__name__,
            "model_path":           str(self.model_path),
            "n_features":           (
                len(self.feature_names) if self.feature_names else "unknown"
            ),
            "has_preprocessor":     self.preprocessor is not None,
            "has_engineer":         self.feature_engineer is not None,
            "confidence_threshold": self.confidence_threshold,
        }

    # ==================================================================
    # Dunder helpers
    # ==================================================================

    def __repr__(self) -> str:
        return (
            f"InferencePipeline("
            f"model='{self.model_path.name}', "
            f"type={type(self.model).__name__}, "
            f"n_features={len(self.feature_names) if self.feature_names else 'unknown'}, "
            f"sessions={len(self._results_history)})"
        )

    def __str__(self) -> str:
        total = sum(r.n_samples for r in self._results_history)
        return (
            f"InferencePipeline [{self.model_path.name}] — "
            f"{total:,} prediction(s) made this session"
        )


# ===========================================================================
# Module-level convenience function
# ===========================================================================


def run_inference(
    data:                      Union[pd.DataFrame, np.ndarray, Dict[str, float], str, Path],
    model_path:                Union[str, Path]    = _DEFAULT_MODEL_PATH,
    preprocessor:              Optional[Any]       = None,
    feature_engineer:          Optional[Any]       = None,
    target_col:                Optional[str]       = _DEFAULT_TARGET_COL,
    label_map:                 Optional[Dict]      = None,
    confidence_threshold:      float               = _DEFAULT_CONF_THRESH,
    display:                   bool               = True,
    save_output:               bool               = False,
    output_path:               Union[str, Path]   = _DEFAULT_PRED_OUTPUT,
    output_fmt:                str                = "csv",
    apply_preprocessing:       bool               = True,
    apply_feature_engineering: bool               = True,
    csv_kwargs:                Optional[Dict]     = None,
) -> "InferenceResult":
    """
    One-liner convenience function for end-to-end inference.

    Creates an :class:`InferencePipeline`, runs the full inference workflow
    on *data*, optionally displays the results to stdout, optionally saves
    the results to disk, and returns the :class:`InferenceResult`.

    This mirrors the ``run_evaluation()`` pattern from the evaluation sub-package
    and is the recommended way to run inference from notebooks or scripts.

    Args:
        data:
            Input sensor data.  Accepts any of:
            ``dict``, ``pd.DataFrame``, ``np.ndarray``, ``str`` (CSV path),
            or ``pathlib.Path`` (CSV path).
        model_path (str | Path):
            Path to the trained ``.joblib`` model artifact.
            Defaults to ``config.yaml → paths.models_dir/model_filename``.
        preprocessor (DataPreprocessor, optional):
            Fitted preprocessor.  ``None`` = skip preprocessing.
        feature_engineer (FeatureEngineer, optional):
            Fitted feature engineer.  ``None`` = skip feature engineering.
        target_col (str, optional):
            Target / label column name to drop from input if present.
            Defaults to ``config.yaml → model.target_col`` (``"failure"``).
        label_map (dict, optional):
            Custom class label → urgency / action mapping.
            ``None`` uses :data:`_DEFAULT_LABEL_MAP`.
        confidence_threshold (float):
            Minimum confidence score for "high-confidence" classification.
            Must be in ``[0.0, 1.0]``.  Defaults to ``0.50``.
        display (bool):
            Whether to print the formatted results report.  Defaults to ``True``.
        save_output (bool):
            Whether to save the results to *output_path*.  Defaults to ``False``.
        output_path (str | Path):
            Destination file path when *save_output* is ``True``.
            Defaults to ``config.yaml → paths.predictions_output``.
        output_fmt (str):
            Output format — ``"csv"`` or ``"json"``.  Defaults to ``"csv"``.
        apply_preprocessing (bool):
            Apply the preprocessor stage (if configured).  Defaults to ``True``.
        apply_feature_engineering (bool):
            Apply the feature-engineering stage (if configured).  Defaults to ``True``.
        csv_kwargs (dict, optional):
            Extra keyword arguments for ``pd.read_csv`` when *data* is a file path.

    Returns:
        InferenceResult: Structured inference result.

    Example — notebook one-liner::

        from src.configs.inference import run_inference

        result = run_inference("data/raw/new_sensor_batch.csv")
        df = result.to_dataframe()

    Example — with preprocessing::

        result = run_inference(
            data={"temperature": 85.3, "vibration": 0.42, "pressure": 2.1},
            display=True,
            save_output=True,
            output_path="outputs/reports/live_prediction.json",
            output_fmt="json",
        )
    """
    logger.info("run_inference() — starting end-to-end inference pipeline.")

    pipeline = InferencePipeline(
        model_path=model_path,
        preprocessor=preprocessor,
        feature_engineer=feature_engineer,
        target_col=target_col,
        label_map=label_map,
        confidence_threshold=confidence_threshold,
    )

    result = pipeline.run(
        data=data,
        apply_preprocessing=apply_preprocessing,
        apply_feature_engineering=apply_feature_engineering,
        csv_kwargs=csv_kwargs,
    )

    if display:
        pipeline.display_results(result)

    if save_output:
        pipeline.save_results(result, output_path=output_path, fmt=output_fmt)

    logger.info(
        "run_inference() — complete. %d sample(s) predicted.", result.n_samples
    )
    return result
