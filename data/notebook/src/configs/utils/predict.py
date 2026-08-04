"""
predict.py
==========
End-to-end prediction pipeline for the IoT Predictive Maintenance project.

This module provides a PredictionPipeline class that orchestrates the complete
inference workflow from raw sensor data to actionable maintenance predictions:

    Load model  →  Ingest sensor data  →  Preprocess  →  Engineer features
    →  Align feature columns  →  Predict class  →  Score confidence
    →  Display / export results

The pipeline is designed to be run independently of the training workflow —
it loads a pre-trained joblib model artifact from disk and applies the same
transformation stages that were used during training, so predictions are
consistent with what the model has learned.

Key design principles:
    - Model-agnostic: accepts any joblib-serialised sklearn estimator.
    - Stateless inputs: raw CSV files or pandas DataFrames both accepted.
    - Feature-alignment: automatically reorders / pads columns to match the
      training feature set, preventing shape-mismatch errors at inference time.
    - Confidence scores: uses predict_proba when available; falls back to
      hard predictions gracefully.
    - Actionable output: maps raw predictions to human-readable maintenance
      recommendations with colour-coded urgency levels.

Part of the Infotact Solutions Data Science & Machine Learning Internship project.

Note:
    The preprocessing and feature-engineering stages applied here must mirror
    the exact configuration used during training. If the training preprocessor
    or feature engineer was fitted with specific settings, pass the same fitted
    objects to PredictionPipeline to avoid data leakage and transformation drift.
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Suppress non-critical warnings
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")



# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Centralised configuration defaults
# ---------------------------------------------------------------------------
try:
    from src.configs.config import get_config as _get_cfg
    _cfg = _get_cfg()
    _DEFAULT_MODEL_PATH  = f"{_cfg.paths.models_dir}/{_cfg.paths.model_filename}"
    _DEFAULT_CONF_THRESH = _cfg.evaluation.confidence_threshold
    _DEFAULT_REPORTS_DIR = _cfg.paths.reports_dir
except (ImportError, AttributeError, FileNotFoundError) as exc:   # fallback when running module in isolation
    logger.debug("Config singleton unavailable, using defaults: %s", exc)
    _DEFAULT_MODEL_PATH  = "outputs/models/random_forest_baseline.joblib"
    _DEFAULT_CONF_THRESH = 0.50
    _DEFAULT_REPORTS_DIR = "outputs/reports"

# ---------------------------------------------------------------------------
# Urgency level colour codes (ANSI — works in terminals & Jupyter)
# ---------------------------------------------------------------------------
_ANSI = {
    "red":    "\033[91m",
    "yellow": "\033[93m",
    "green":  "\033[92m",
    "cyan":   "\033[96m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}

# ---------------------------------------------------------------------------
# Maintenance action mapping
# ---------------------------------------------------------------------------
# Maps predicted class label → (urgency, recommended_action)
# Users can override this mapping by passing `label_map` to __init__.
_DEFAULT_LABEL_MAP: Dict[Any, Dict[str, str]] = {
    0: {
        "urgency": "NORMAL",
        "action":  "No maintenance required. Continue monitoring.",
        "colour":  "green",
    },
    1: {
        "urgency": "MAINTENANCE REQUIRED",
        "action":  (
            "Schedule preventive maintenance. Inspect components, "
            "lubricate moving parts, and check sensor calibration."
        ),
        "colour":  "yellow",
    },
    # Additional classes (e.g., fault types) can be added here
    2: {
        "urgency": "CRITICAL — IMMEDIATE ACTION",
        "action":  (
            "STOP MACHINE. Immediate inspection required. "
            "Risk of catastrophic failure detected."
        ),
        "colour":  "red",
    },
}


# ---------------------------------------------------------------------------
# PredictionResult dataclass-style container
# ---------------------------------------------------------------------------


class PredictionResult:
    """
    Lightweight container for a single-sample or batch prediction result.

    Attributes:
        predictions    (np.ndarray):   Predicted class labels (shape: n_samples).
        probabilities  (np.ndarray | None): Class probabilities (n_samples, n_classes)
                                            or (n_samples,) for binary positive-class.
        confidence     (np.ndarray):   Maximum probability per sample (n_samples,).
        feature_matrix (pd.DataFrame): Aligned feature matrix sent to the model.
        n_samples      (int):          Number of samples predicted.
        timestamp      (str):          ISO timestamp of when predictions were made.
        label_map      (dict):         Mapping from class label → action info.

    Notes:
        When probabilities are unavailable (model does not expose
        ``predict_proba``), :attr:`confidence` is filled with ``np.nan``.
    """

    def __init__(
        self,
        predictions:    np.ndarray,
        probabilities:  Optional[np.ndarray],
        feature_matrix: pd.DataFrame,
        label_map:      Dict[Any, Dict[str, str]],
        class_labels:   Optional[List[Any]] = None,
    ) -> None:
        self.predictions:    np.ndarray           = predictions
        self.probabilities:  Optional[np.ndarray] = probabilities
        self.feature_matrix: pd.DataFrame         = feature_matrix
        self.label_map:      Dict                 = label_map
        self.class_labels:   Optional[List[Any]]  = class_labels
        self.n_samples:      int                  = len(predictions)
        self.timestamp:      str                  = datetime.now().isoformat(timespec="seconds")

        # Confidence = max probability per row (or NaN if no proba)
        if probabilities is not None:
            proba_2d = (
                probabilities if probabilities.ndim == 2
                else probabilities.reshape(-1, 1)
            )
            self.confidence = proba_2d.max(axis=1)
        else:
            self.confidence = np.full(self.n_samples, np.nan)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return predictions and confidence scores as a tidy DataFrame.

        Columns:
            - ``sample_id``   — 0-based row index
            - ``prediction``  — predicted class label
            - ``confidence``  — maximum class probability (NaN if unavailable)
            - ``urgency``     — human-readable urgency level
            - ``action``      — recommended maintenance action
            - ``prob_{cls}``  — per-class probability (one column per class,
                                 only when probabilities are available)

        Returns:
            pd.DataFrame: One row per sample.
        """
        records = []
        for i in range(self.n_samples):
            pred  = self.predictions[i]
            conf  = self.confidence[i]
            info  = self.label_map.get(pred, {
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
                for j, cls in enumerate(self.class_labels or range(self.probabilities.shape[1])):
                    row[f"prob_{cls}"] = round(float(self.probabilities[i, j]), 4)

            records.append(row)

        return pd.DataFrame(records)

    def __repr__(self) -> str:
        return (
            f"PredictionResult(n_samples={self.n_samples}, "
            f"timestamp='{self.timestamp}', "
            f"has_probabilities={self.probabilities is not None})"
        )

    def __len__(self) -> int:
        return self.n_samples


# ---------------------------------------------------------------------------
# PredictionPipeline class
# ---------------------------------------------------------------------------


class PredictionPipeline:
    """
    End-to-end inference pipeline for IoT predictive maintenance models.

    Loads a pre-trained joblib model artifact and orchestrates the full
    inference workflow:

        raw input  →  [optional: preprocess]  →  [optional: feature-engineer]
        →  feature alignment  →  model.predict  →  PredictionResult

    Preprocessing and feature-engineering are optional — when the caller
    provides pre-processed / pre-engineered feature DataFrames directly,
    those stages are bypassed.

    Attributes:
        model_path       (Path):           Path of the loaded .joblib file.
        model            (BaseEstimator):  Loaded sklearn estimator.
        feature_names    (List[str] | None): Expected feature columns
                                              (auto-detected from the model
                                               if available).
        preprocessor     (DataPreprocessor | None): Fitted preprocessor.
        feature_engineer (FeatureEngineer | None):  Fitted feature engineer.
        target_col       (str | None):     Target column to drop before
                                            inference (if present in input).
        label_map        (dict):           Class label → urgency/action map.
        _results_history (List[PredictionResult]): All results from this session.

    Example — minimal usage (pre-engineered features)::

        pipeline = PredictionPipeline("outputs/models/random_forest_baseline.joblib")
        result   = pipeline.predict(feature_df)
        pipeline.display_results(result)

    Example — full pipeline (raw sensor CSV)::

        pipeline = PredictionPipeline(
            model_path="outputs/models/random_forest_baseline.joblib",
            preprocessor=fitted_preprocessor,
            feature_engineer=fitted_engineer,
            target_col="failure",
        )
        result = pipeline.predict_from_csv("data/raw/new_sensor_batch.csv")
        pipeline.display_results(result)
        pipeline.save_results(result, "outputs/reports/predictions.csv")
    """

    def __init__(
        self,
        model_path:       Union[str, Path]     = _DEFAULT_MODEL_PATH,
        feature_names:    Optional[List[str]]  = None,
        preprocessor:     Optional[Any]        = None,
        feature_engineer: Optional[Any]        = None,
        target_col:       Optional[str]        = None,
        label_map:        Optional[Dict]       = None,
        confidence_threshold: float            = _DEFAULT_CONF_THRESH,
    ) -> None:
        """
        Initialise the PredictionPipeline and load the model from disk.

        Args:
            model_path (str | Path):
                Absolute or relative path to the saved ``.joblib`` model file.
            feature_names (List[str], optional):
                Ordered list of feature column names the model was trained on.
                When ``None``, the pipeline attempts to read
                ``feature_names_in_`` from the model object (sklearn ≥ 1.0).
                If neither source is available, no column reordering is applied.
            preprocessor (DataPreprocessor, optional):
                A **fitted** ``DataPreprocessor`` instance.
                When provided, raw input DataFrames are passed through
                ``preprocessor.transform(df)`` before feature engineering.
                Defaults to ``None`` (preprocessing skipped).
            feature_engineer (FeatureEngineer, optional):
                A **fitted/configured** ``FeatureEngineer`` instance.
                When provided, the preprocessed DataFrame is passed through
                ``feature_engineer.transform(df)`` before prediction.
                Defaults to ``None`` (feature engineering skipped).
            target_col (str, optional):
                Name of the target/label column. If present in the input
                DataFrame it is automatically dropped before inference.
                Defaults to ``None``.
            label_map (dict, optional):
                Mapping from predicted class label → dict with keys
                ``"urgency"``, ``"action"``, ``"colour"``.
                Defaults to :data:`_DEFAULT_LABEL_MAP`.
            confidence_threshold (float):
                Minimum confidence score to flag a prediction as
                "high-confidence". Used in display formatting only.
                Defaults to ``0.50``.

        Raises:
            FileNotFoundError: If the model file does not exist.
            ValueError:        If ``confidence_threshold`` is outside [0, 1].
        """
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {confidence_threshold}."
            )

        self.model_path:            Path                = Path(model_path).resolve()
        self.preprocessor:          Optional[Any]       = preprocessor
        self.feature_engineer:      Optional[Any]       = feature_engineer
        self.target_col:            Optional[str]       = target_col
        self.label_map:             Dict                = label_map or _DEFAULT_LABEL_MAP
        self.confidence_threshold:  float               = confidence_threshold
        self._results_history:      List[PredictionResult] = []

        # ── Load model ────────────────────────────────────────────────────
        self.model = self._load_model(self.model_path)

        # ── Resolve training feature names ────────────────────────────────
        self.feature_names: Optional[List[str]] = (
            feature_names
            or self._infer_feature_names(self.model)
        )

        # ── Determine class labels ─────────────────────────────────────────
        self._class_labels: Optional[List[Any]] = (
            list(self.model.classes_)
            if hasattr(self.model, "classes_")
            else None
        )

        logger.info(
            "PredictionPipeline ready — model='%s', "
            "n_features=%s, n_classes=%s, "
            "preprocessor=%s, feature_engineer=%s.",
            self.model_path.name,
            len(self.feature_names) if self.feature_names else "unknown",
            len(self._class_labels) if self._class_labels else "unknown",
            type(preprocessor).__name__ if preprocessor else "None",
            type(feature_engineer).__name__ if feature_engineer else "None",
        )

    # ------------------------------------------------------------------
    # Primary prediction entry points
    # ------------------------------------------------------------------

    def predict(
        self,
        data: Union[pd.DataFrame, np.ndarray],
    ) -> "PredictionResult":
        """
        Generate maintenance predictions from a pre-engineered feature matrix.

        Use this method when the input *data* already contains the exact
        feature columns the model expects (i.e., preprocessing and feature
        engineering have already been applied upstream).

        Args:
            data (DataFrame | ndarray):
                Feature matrix for inference. If a DataFrame is passed, any
                column named :attr:`target_col` is silently dropped. Columns
                are reordered / padded to match :attr:`feature_names`.

        Returns:
            PredictionResult: Container with predictions, probabilities,
                              confidence scores, and the aligned feature matrix.

        Raises:
            TypeError:  If *data* is not a DataFrame or ndarray.
            ValueError: If no usable feature columns remain after processing.
        """
        # Coerce to DataFrame for uniform handling
        if isinstance(data, np.ndarray):
            cols = (
                self.feature_names[:data.shape[1]]
                if self.feature_names
                else [f"feature_{i}" for i in range(data.shape[1])]
            )
            data = pd.DataFrame(data, columns=cols)
        elif not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"data must be a pd.DataFrame or np.ndarray, got {type(data).__name__}."
            )

        # Drop target column if accidentally present
        if self.target_col and self.target_col in data.columns:
            data = data.drop(columns=[self.target_col])
            logger.debug("Dropped target column '%s' from input.", self.target_col)

        # Align feature columns to the training layout
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

        # ── Inference ─────────────────────────────────────────────────────
        predictions = self.model.predict(X)
        probabilities = self._get_probabilities(X)

        result = PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            feature_matrix=X,
            label_map=self.label_map,
            class_labels=self._class_labels,
        )
        self._results_history.append(result)

        logger.info(
            "Predictions complete — %d sample(s) | classes predicted: %s.",
            result.n_samples, np.unique(predictions).tolist(),
        )
        return result

    def predict_from_csv(
        self,
        csv_path:   Union[str, Path],
        csv_kwargs: Optional[Dict] = None,
        apply_preprocessing:     bool = True,
        apply_feature_engineering: bool = True,
    ) -> "PredictionResult":
        """
        Load sensor data from a CSV file and run the full prediction pipeline.

        This method applies the optional preprocessing and feature-engineering
        stages configured at construction time before calling :meth:`predict`.

        Args:
            csv_path (str | Path):
                Path to the input CSV file.
            csv_kwargs (dict, optional):
                Extra keyword arguments forwarded to ``pd.read_csv``.
                Useful for custom separators, encodings, etc.
            apply_preprocessing (bool):
                Whether to apply :attr:`preprocessor` (if set).
                Defaults to ``True``.
            apply_feature_engineering (bool):
                Whether to apply :attr:`feature_engineer` (if set).
                Defaults to ``True``.

        Returns:
            PredictionResult: Prediction results for the loaded data.

        Raises:
            FileNotFoundError: If *csv_path* does not exist.
            ValueError:        If the loaded file is empty.
        """
        path = Path(csv_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: '{path}'.")

        logger.info("Loading sensor data from '%s' …", path)
        df = pd.read_csv(path, **(csv_kwargs or {}))

        if df.empty:
            raise ValueError(f"The CSV file '{path}' is empty.")

        logger.info("CSV loaded — %d rows × %d cols.", df.shape[0], df.shape[1])
        return self.predict_from_dataframe(
            df=df,
            apply_preprocessing=apply_preprocessing,
            apply_feature_engineering=apply_feature_engineering,
        )

    def predict_from_dataframe(
        self,
        df:                       pd.DataFrame,
        apply_preprocessing:      bool = True,
        apply_feature_engineering: bool = True,
    ) -> "PredictionResult":
        """
        Run the full pipeline on a raw sensor DataFrame.

        Applies preprocessing → feature engineering → prediction in sequence.
        Stages are skipped if the corresponding component is ``None`` or
        if the ``apply_*`` flag is ``False``.

        Args:
            df (pd.DataFrame): Raw sensor DataFrame.
            apply_preprocessing (bool):
                Run the preprocessor (if configured). Defaults to ``True``.
            apply_feature_engineering (bool):
                Run the feature engineer (if configured). Defaults to ``True``.

        Returns:
            PredictionResult: Prediction results for the input data.

        Raises:
            TypeError: If *df* is not a ``pd.DataFrame``.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected a pd.DataFrame, got {type(df).__name__}."
            )

        result_df = df.copy(deep=True)

        # ── Stage 1: Preprocessing ────────────────────────────────────────
        if apply_preprocessing and self.preprocessor is not None:
            logger.info("Applying preprocessing stage …")
            result_df = self.preprocessor.transform(result_df)
            logger.info(
                "Preprocessing complete — shape after: %s.", result_df.shape
            )

        # ── Stage 2: Feature engineering ──────────────────────────────────
        if apply_feature_engineering and self.feature_engineer is not None:
            logger.info("Applying feature engineering stage …")
            result_df = self.feature_engineer.transform(result_df)
            logger.info(
                "Feature engineering complete — shape after: %s.", result_df.shape
            )

        # ── Stage 3: Predict ──────────────────────────────────────────────
        return self.predict(result_df)

    def predict_single(
        self,
        sensor_readings: Dict[str, float],
    ) -> "PredictionResult":
        """
        Generate a prediction for a single sensor reading dictionary.

        Convenience method for real-time single-sample inference without
        needing to construct a DataFrame manually.

        Args:
            sensor_readings (Dict[str, float]):
                Mapping of feature name → current sensor value.
                Example: ``{"temperature": 85.3, "vibration": 0.42, ...}``

        Returns:
            PredictionResult: Single-sample prediction result.

        Raises:
            ValueError: If *sensor_readings* is empty.
        """
        if not sensor_readings:
            raise ValueError("sensor_readings dict must not be empty.")

        df = pd.DataFrame([sensor_readings])
        logger.info(
            "predict_single() — %d feature(s): %s",
            len(sensor_readings), list(sensor_readings.keys()),
        )
        return self.predict(df)

    # ------------------------------------------------------------------
    # Display methods
    # ------------------------------------------------------------------

    def display_results(
        self,
        result: "PredictionResult",
        max_rows: int = 50,
        colour:   bool = True,
    ) -> None:
        """
        Print a formatted prediction report to stdout.

        For each sample the report shows:
            - Sample index
            - Predicted class label
            - Confidence score (% or N/A)
            - Urgency level (colour-coded in terminal environments)
            - Recommended maintenance action

        A summary section at the end shows the class distribution and
        any low-confidence warnings.

        Args:
            result (PredictionResult): Output from any ``predict*`` method.
            max_rows (int):
                Maximum number of individual sample rows to display before
                truncating. Defaults to ``50``.
            colour (bool):
                Whether to use ANSI colour codes in terminal output.
                Set to ``False`` for plain-text logging. Defaults to ``True``.
        """
        sep  = "─" * 72
        sep2 = "═" * 72
        c    = _ANSI if colour else {k: "" for k in _ANSI}

        print(f"\n{c['bold']}{sep2}{c['reset']}")
        print(f"{c['bold']}  PREDICTIVE MAINTENANCE — PREDICTION RESULTS{c['reset']}")
        print(f"{c['bold']}{sep2}{c['reset']}")
        print(f"  Timestamp        : {result.timestamp}")
        print(f"  Samples          : {result.n_samples:,}")
        print(
            f"  Confidence scores: "
            f"{'Available' if result.probabilities is not None else 'Not available'}"
        )
        print(f"  Confidence thresh: {self.confidence_threshold:.0%}")
        print(sep)

        # ── Per-sample rows ───────────────────────────────────────────────
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

            # Confidence string
            conf_str = f"{conf:.1%}" if conf is not None else "  N/A  "

            # Low-confidence flag
            flag = ""
            if conf is not None and conf < self.confidence_threshold:
                flag = f" {c['yellow']}⚠ low conf{c['reset']}"

            # Urgency colour
            info    = self.label_map.get(pred, {"colour": "cyan"})
            urg_col = c.get(info["colour"], "")

            print(
                f"  {int(row['sample_id']):<6} {str(pred):<14} {conf_str:>11}  "
                f"{urg_col}{urg:<28}{c['reset']} {action}{flag}"
            )

        if rows_to_show < result.n_samples:
            print(f"\n  … {result.n_samples - rows_to_show:,} more rows not shown "
                  f"(max_rows={max_rows}).")

        # ── Summary ───────────────────────────────────────────────────────
        print(f"\n{sep}")
        print(f"  PREDICTION SUMMARY")
        print(sep)

        class_counts = pd.Series(result.predictions).value_counts().sort_index()
        for cls, cnt in class_counts.items():
            pct  = cnt / result.n_samples * 100
            info = self.label_map.get(cls, {"urgency": str(cls), "colour": "cyan"})
            urg_col = c.get(info["colour"], "")
            bar  = "█" * int(pct / 5)
            print(
                f"  Class {str(cls):<6} → {cnt:>5,} sample(s)  "
                f"({pct:>5.1f}%)  {urg_col}{bar}{c['reset']}"
            )

        # Confidence stats
        valid_conf = result.confidence[~np.isnan(result.confidence)]
        if len(valid_conf) > 0:
            print(sep)
            print(
                f"  Confidence  min={valid_conf.min():.1%} | "
                f"mean={valid_conf.mean():.1%} | "
                f"max={valid_conf.max():.1%}"
            )
            low_conf_n = int((valid_conf < self.confidence_threshold).sum())
            if low_conf_n > 0:
                print(
                    f"  {c['yellow']}⚠  {low_conf_n:,} sample(s) below confidence "
                    f"threshold ({self.confidence_threshold:.0%}) — "
                    f"review manually.{c['reset']}"
                )
            else:
                print(
                    f"  {c['green']}✓  All samples above confidence threshold.{c['reset']}"
                )

        print(f"{sep2}\n")
        logger.info(
            "Results displayed — %d samples, class dist: %s.",
            result.n_samples, class_counts.to_dict(),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(
        self,
        result:       "PredictionResult",
        output_path:  Union[str, Path],
        fmt:          str = "csv",
        include_features: bool = False,
    ) -> Path:
        """
        Save prediction results to a CSV or JSON file.

        Args:
            result (PredictionResult):
                Output from any ``predict*`` method.
            output_path (str | Path):
                Destination file path. Parent directories are created
                automatically if absent.
            fmt (str):
                Output format — ``"csv"`` (default) or ``"json"``.
            include_features (bool):
                When ``True``, append the aligned feature matrix columns
                to the output alongside the prediction columns.
                Defaults to ``False``.

        Returns:
            Path: Resolved absolute path of the saved file.

        Raises:
            ValueError: If *fmt* is not ``"csv"`` or ``"json"``.
        """
        if fmt not in ("csv", "json"):
            raise ValueError(f"fmt must be 'csv' or 'json', got '{fmt}'.")

        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_df = result.to_dataframe()

        if include_features:
            # Align feature matrix index before concatenation
            feat_df    = result.feature_matrix.reset_index(drop=True)
            results_df = pd.concat([results_df, feat_df], axis=1)

        if fmt == "csv":
            results_df.to_csv(output_path, index=False)
        else:
            # JSON — convert NaN → None for serialisability
            records = results_df.where(pd.notna(results_df), other=None).to_dict(
                orient="records"
            )
            payload = {
                "model":     self.model_path.name,
                "timestamp": result.timestamp,
                "n_samples": result.n_samples,
                "predictions": records,
            }
            with output_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=str)

        size_kb = output_path.stat().st_size / 1024
        logger.info("Prediction results saved → %s  [%.1f KB].", output_path, size_kb)
        return output_path

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def get_model_info(self) -> Dict[str, Any]:
        """
        Return a dictionary of model metadata and pipeline configuration.

        Returns:
            dict: Keys include ``model_type``, ``model_path``, ``n_features``,
                  ``class_labels``, ``has_preprocessor``, ``has_engineer``.
        """
        return {
            "model_type":       type(self.model).__name__,
            "model_path":       str(self.model_path),
            "n_features":       len(self.feature_names) if self.feature_names else "unknown",
            "feature_names":    self.feature_names,
            "class_labels":     self._class_labels,
            "has_preprocessor": self.preprocessor is not None,
            "has_engineer":     self.feature_engineer is not None,
            "target_col":       self.target_col,
            "confidence_threshold": self.confidence_threshold,
            "n_predictions_this_session": sum(
                r.n_samples for r in self._results_history
            ),
        }

    def display_model_info(self) -> None:
        """Print a formatted summary of the loaded model and pipeline config."""
        info = self.get_model_info()
        sep  = "─" * 65

        print(f"\n{sep}")
        print("  PIPELINE CONFIGURATION")
        print(sep)
        for key, val in info.items():
            if key == "feature_names":
                n = len(val) if val else 0
                print(f"  {'feature_names':<30} {n} column(s)")
                if val:
                    for i, f in enumerate(val[:8], 1):
                        print(f"    {i:>2}. {f}")
                    if len(val) > 8:
                        print(f"    … {len(val) - 8} more")
            else:
                print(f"  {str(key):<30} {val}")
        print(f"{sep}\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model(path: Path) -> Any:
        """
        Load a joblib model artifact from *path*.

        Args:
            path (Path): Absolute path to the ``.joblib`` file.

        Returns:
            Any: The deserialised sklearn estimator.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: '{path}'. "
                f"Run the training pipeline first to generate the artifact."
            )
        model = joblib.load(path)
        size_kb = path.stat().st_size / 1024
        logger.info(
            "Model loaded from '%s'  [%.1f KB, type=%s].",
            path, size_kb, type(model).__name__,
        )
        return model

    @staticmethod
    def _infer_feature_names(model: Any) -> Optional[List[str]]:
        """
        Attempt to read feature names from the model object itself.

        sklearn ≥ 1.0 estimators that were fitted on a DataFrame store
        the training feature names in ``feature_names_in_``.

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
            "no automatic feature alignment will be applied."
        )
        return None

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder and pad the input DataFrame to match the training feature layout.

        Algorithm:
            1. Select only numeric columns (sklearn does not accept object/datetime).
            2. If :attr:`feature_names` is known:
               a. Add any missing columns with value ``0.0`` (zero-imputation).
               b. Reorder to the exact training column order.
               c. Drop any extra columns not seen during training.
            3. Otherwise, return the numeric-only DataFrame as-is.

        Args:
            df (pd.DataFrame): Input feature DataFrame (post-preprocessing,
                                post-feature-engineering).

        Returns:
            pd.DataFrame: Aligned feature matrix ready for model.predict().
        """
        # Keep only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        if not self.feature_names:
            logger.debug(
                "No training feature names known — passing %d numeric "
                "columns directly to model.", numeric_df.shape[1],
            )
            return numeric_df.reset_index(drop=True)

        # ── Add missing columns with zero fill ────────────────────────────
        missing_cols = set(self.feature_names) - set(numeric_df.columns)
        if missing_cols:
            logger.warning(
                "%d training feature(s) missing from input — "
                "zero-imputing: %s",
                len(missing_cols), sorted(missing_cols),
            )
            for col in missing_cols:
                numeric_df[col] = 0.0

        # ── Drop extra columns not seen during training ────────────────────
        extra_cols = set(numeric_df.columns) - set(self.feature_names)
        if extra_cols:
            logger.debug(
                "Dropping %d column(s) not in training feature set: %s",
                len(extra_cols), sorted(extra_cols),
            )
            numeric_df = numeric_df.drop(columns=list(extra_cols))

        # ── Reorder to match training column order ─────────────────────────
        numeric_df = numeric_df[self.feature_names]

        logger.debug(
            "Feature alignment complete — %d columns in correct order.",
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
                if the model does not expose ``predict_proba``.
        """
        if not hasattr(self.model, "predict_proba"):
            logger.info(
                "Model does not expose predict_proba — confidence unavailable."
            )
            return None
        try:
            proba = self.model.predict_proba(X)
            logger.debug("predict_proba returned shape %s.", proba.shape)
            return proba
        except Exception as exc:
            logger.warning("predict_proba failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PredictionPipeline("
            f"model='{self.model_path.name}', "
            f"type={type(self.model).__name__}, "
            f"n_features={len(self.feature_names) if self.feature_names else 'unknown'}, "
            f"sessions={len(self._results_history)})"
        )

    def __str__(self) -> str:
        total = sum(r.n_samples for r in self._results_history)
        return (
            f"PredictionPipeline [{self.model_path.name}] — "
            f"{total:,} prediction(s) made this session"
        )
