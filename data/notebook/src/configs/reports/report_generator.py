"""
report_generator.py
===================
Predictive Maintenance Reporting Module for the IoT Predictive Maintenance project.

This module provides a reusable, highly modular reporting engine that generates
structured prediction and evaluation reports after model inference.

Pipeline:
    InferenceResult / Predictions (+ optional y_true / metrics)
    →  PredictiveMaintenanceReport
    →  Export to JSON & TXT in outputs/reports/

Design goals:
    - **Reusability**: ``ReportGenerator`` accepts an ``InferenceResult`` (from
      :class:`~src.configs.inference.InferencePipeline`), a pandas DataFrame, or
      dict/array structures, along with optional ground-truth labels or
      pre-computed metrics.
    - **Dual Export**: Generates both machine-readable JSON reports and
      human-readable ASCII TXT reports.
    - **Comprehensive Summaries**:
          1. Model & Pipeline Metadata (name, type, timestamp, feature count)
          2. Prediction Summary (class distribution, percentages, urgency levels)
          3. Confidence Statistics (mean, min, max, std, low/high confidence counts)
          4. Actionable Alerts (maintenance recommendations, critical flags)
          5. Evaluation Metrics (Accuracy, Precision, Recall, F1, Confusion Matrix,
             ROC-AUC when ground truth or metrics are available)
    - **Automatic Persistence**: Automatically saves reports to ``outputs/reports/``
      with ISO-8601 timestamped filenames.
    - **Traceability & Safety**: Full logging via Python's ``logging`` framework
      and graceful exception handling throughout.

Typical usage::

    # --- Convenience Function ---------------------------------------------
    from src.configs.reports import generate_report

    # Generate and save report after inference
    report = generate_report(inference_result, save=True)

    # --- Object-Oriented Interface ---------------------------------------
    from src.configs.reports import ReportGenerator

    generator = ReportGenerator()
    report = generator.generate_report(
        inference_result=result,
        y_true=y_test,  # optional ground truth for evaluation metrics
    )
    report.save_both()  # exports JSON + TXT to outputs/reports/

Part of the Infotact Solutions Data Science & Machine Learning Internship project.
"""

import json
import logging
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

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
try:
    from src.configs.config import get_config as _get_cfg, get_absolute_path
    _cfg = _get_cfg()

    _DEFAULT_REPORTS_DIR = str(get_absolute_path(_cfg.paths.reports_dir))
    _DEFAULT_CONF_THRESH = _cfg.evaluation.confidence_threshold

except (ImportError, AttributeError, FileNotFoundError) as exc:  # Fallback when running module in isolation
    logger.debug("Config singleton unavailable, using defaults: %s", exc)
    _DEFAULT_REPORTS_DIR = "outputs/reports"
    _DEFAULT_CONF_THRESH = 0.50

# ---------------------------------------------------------------------------
# Maintenance action label map (Fallback if not provided in input)
# ---------------------------------------------------------------------------
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
        "urgency": "CRITICAL - IMMEDIATE ACTION",
        "action":  (
            "STOP MACHINE IMMEDIATELY. Critical failure risk detected. "
            "Full inspection and component replacement required."
        ),
        "colour":  "red",
    },
}


# ===========================================================================
# PredictiveMaintenanceReport Container
# ===========================================================================


class PredictiveMaintenanceReport:
    """
    Structured report container holding all prediction summaries, evaluation
    metrics, and metadata for a predictive maintenance run.

    Attributes:
        report_id (str): Unique report ID.
        timestamp (str): ISO-8601 generation timestamp.
        title (str): Human-readable report title.
        metadata (dict): Model, pipeline, and execution environment metadata.
        prediction_summary (dict): Statistical breakdown of predictions and confidence.
        urgency_summary (dict): Count and percentage per urgency level.
        evaluation_metrics (dict, optional): Accuracy, Precision, Recall, F1, etc.
        sample_details (list, optional): Per-sample prediction records.
        reports_dir (Path): Output directory for saved reports.
    """

    def __init__(
        self,
        title: str,
        metadata: Dict[str, Any],
        prediction_summary: Dict[str, Any],
        urgency_summary: Dict[str, Any],
        evaluation_metrics: Optional[Dict[str, Any]] = None,
        sample_details: Optional[List[Dict[str, Any]]] = None,
        reports_dir: Union[str, Path] = _DEFAULT_REPORTS_DIR,
    ) -> None:
        self.report_id: str = f"REP-{uuid.uuid4().hex[:8].upper()}"
        self.timestamp: str = datetime.now().isoformat(timespec="seconds")
        self.title: str = title
        self.metadata: Dict[str, Any] = metadata
        self.prediction_summary: Dict[str, Any] = prediction_summary
        self.urgency_summary: Dict[str, Any] = urgency_summary
        self.evaluation_metrics: Optional[Dict[str, Any]] = evaluation_metrics
        self.sample_details: Optional[List[Dict[str, Any]]] = sample_details
        self.reports_dir: Path = Path(reports_dir).resolve()

        # Ensure reports output directory exists
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Serialisation & Export
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert report to a clean, JSON-serialisable dictionary.
        """
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "title": self.title,
            "metadata": self.metadata,
            "prediction_summary": self.prediction_summary,
            "urgency_summary": self.urgency_summary,
            "evaluation_metrics": self.evaluation_metrics,
            "sample_details": self.sample_details,
        }

    def to_txt(self) -> str:
        """
        Format report as a clean, human-readable ASCII text report.
        """
        lines: List[str] = []
        double_line = "=" * 78
        single_line = "-" * 78

        lines.append(double_line)
        lines.append(f"  {self.title.upper()}")
        lines.append(double_line)
        lines.append(f"  Report ID   : {self.report_id}")
        lines.append(f"  Timestamp   : {self.timestamp}")
        if "model_name" in self.metadata:
            lines.append(f"  Model Name  : {self.metadata['model_name']}")
        if "model_type" in self.metadata:
            lines.append(f"  Model Type  : {self.metadata['model_type']}")
        lines.append(single_line)

        # ── 1. Prediction Summary ──────────────────────────────────────────
        ps = self.prediction_summary
        lines.append("\n1. PREDICTION SUMMARY")
        lines.append(single_line)
        lines.append(f"  Total Samples Predicted : {ps.get('total_samples', 0):,}")
        lines.append(f"  Confidence Threshold   : {ps.get('confidence_threshold', 0.5):.0%}")
        lines.append(f"  High Confidence Samples : {ps.get('high_confidence_count', 0):,} ({ps.get('high_confidence_pct', 0.0):.1f}%)")
        lines.append(f"  Low Confidence Samples  : {ps.get('low_confidence_count', 0):,} ({ps.get('low_confidence_pct', 0.0):.1f}%)")

        cs = ps.get("confidence_stats", {})
        if cs:
            lines.append(
                f"  Confidence Stats        : Min={cs.get('min', 0.0):.1%} | "
                f"Mean={cs.get('mean', 0.0):.1%} | Max={cs.get('max', 0.0):.1%}"
            )

        # Class breakdown
        class_dist = ps.get("class_distribution", {})
        if class_dist:
            lines.append("\n  Class Distribution:")
            for cls_key, cls_info in class_dist.items():
                cnt = cls_info.get("count", 0)
                pct = cls_info.get("pct", 0.0)
                urg = cls_info.get("urgency", f"Class {cls_key}")
                lines.append(f"    - Class {cls_key:<4} [{urg:<25}] : {cnt:>6,} samples ({pct:>5.1f}%)")

        # ── 2. Urgency Breakdown ───────────────────────────────────────────
        lines.append("\n2. MAINTENANCE URGENCY BREAKDOWN")
        lines.append(single_line)
        for urg_label, info in self.urgency_summary.items():
            cnt = info.get("count", 0) if isinstance(info, dict) else info
            pct = info.get("pct", 0.0) if isinstance(info, dict) else (cnt / ps.get('total_samples', 1) * 100)
            act = info.get("action", "") if isinstance(info, dict) else ""
            lines.append(f"  * {urg_label:<32} : {cnt:>6,} samples ({pct:>5.1f}%)")
            if act:
                lines.append(f"    Action: {act}")

        # ── 3. Evaluation Metrics (If available) ─────────────────────────
        if self.evaluation_metrics:
            lines.append("\n3. MODEL EVALUATION METRICS")
            lines.append(single_line)
            em = self.evaluation_metrics
            for metric_key, metric_val in em.items():
                if metric_key == "confusion_matrix":
                    lines.append("\n  Confusion Matrix:")
                    cm = metric_val
                    if isinstance(cm, list):
                        for row in cm:
                            lines.append(f"    {row}")
                    else:
                        lines.append(f"    {cm}")
                elif metric_key == "classification_report":
                    lines.append("\n  Classification Report:")
                    if isinstance(metric_val, str):
                        for line in metric_val.split("\n"):
                            lines.append(f"    {line}")
                elif isinstance(metric_val, (float, int)):
                    if isinstance(metric_val, float):
                        lines.append(f"  - {metric_key:<25} : {metric_val:.4f} ({metric_val:.2%})")
                    else:
                        lines.append(f"  - {metric_key:<25} : {metric_val}")

        # ── 4. Critical / High-Risk Alerts Summary ───────────────────────
        alerts = ps.get("critical_alerts", [])
        if alerts:
            lines.append("\n4. CRITICAL RISK ALERTS & LOW CONFIDENCE WARNINGS")
            lines.append(single_line)
            for alert in alerts[:25]:  # Top 25 alerts
                lines.append(
                    f"  [Sample #{alert.get('sample_id', 'N/A')}] "
                    f"Pred={alert.get('prediction', 'N/A')} | "
                    f"Conf={alert.get('confidence', 0.0):.1%} | "
                    f"Urgency={alert.get('urgency', 'N/A')}"
                )
            if len(alerts) > 25:
                lines.append(f"  ... and {len(alerts) - 25} more critical/low-confidence alerts.")

        lines.append("\n" + double_line)
        lines.append("  END OF REPORT")
        lines.append(double_line + "\n")

        return "\n".join(lines)

    def save_json(self, file_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Save the report as a JSON file.
        """
        if file_path is None:
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.reports_dir / f"predictive_maintenance_report_{ts_str}.json"
        else:
            file_path = Path(file_path).resolve()

        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with file_path.open("w", encoding="utf-8") as fh:
                json.dump(self.to_dict(), fh, indent=2, default=str)
            size_kb = file_path.stat().st_size / 1024
            logger.info("Saved JSON report -> '%s' [%.1f KB]", file_path, size_kb)
            return file_path
        except Exception as exc:
            logger.error("Failed to save JSON report to '%s': %s", file_path, exc)
            raise OSError(f"Failed to write JSON report file: {exc}") from exc

    def save_txt(self, file_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Save the report as a human-readable TXT file.
        """
        if file_path is None:
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.reports_dir / f"predictive_maintenance_report_{ts_str}.txt"
        else:
            file_path = Path(file_path).resolve()

        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            txt_content = self.to_txt()
            with file_path.open("w", encoding="utf-8") as fh:
                fh.write(txt_content)
            size_kb = file_path.stat().st_size / 1024
            logger.info("Saved TXT report -> '%s' [%.1f KB]", file_path, size_kb)
            return file_path
        except Exception as exc:
            logger.error("Failed to save TXT report to '%s': %s", file_path, exc)
            raise OSError(f"Failed to write TXT report file: {exc}") from exc

    def save_both(
        self, base_name: Optional[str] = None
    ) -> Tuple[Path, Path]:
        """
        Save the report in both JSON and TXT formats.

        Args:
            base_name (str, optional): Base filename (without extension).

        Returns:
            Tuple[Path, Path]: (json_path, txt_path)
        """
        if base_name is None:
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"predictive_maintenance_report_{ts_str}"

        json_path = self.reports_dir / f"{base_name}.json"
        txt_path = self.reports_dir / f"{base_name}.txt"

        saved_json = self.save_json(json_path)
        saved_txt = self.save_txt(txt_path)
        return saved_json, saved_txt

    def __repr__(self) -> str:
        return (
            f"PredictiveMaintenanceReport(id='{self.report_id}', "
            f"timestamp='{self.timestamp}', "
            f"samples={self.prediction_summary.get('total_samples', 0)})"
        )


# ===========================================================================
# ReportGenerator Class
# ===========================================================================


class ReportGenerator:
    """
    Reusable reporting generator for generating structured JSON and TXT reports
    after model inference and evaluation.

    Attributes:
        reports_dir (Path): Directory where output reports are persisted.
        confidence_threshold (float): Minimum confidence threshold for classification.
    """

    def __init__(
        self,
        reports_dir: Union[str, Path] = _DEFAULT_REPORTS_DIR,
        confidence_threshold: float = _DEFAULT_CONF_THRESH,
    ) -> None:
        """
        Initialise ReportGenerator.

        Args:
            reports_dir (str | Path): Path to reports output directory.
            confidence_threshold (float): Confidence threshold for flagging low confidence.
        """
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, got {confidence_threshold}"
            )

        self.reports_dir: Path = Path(reports_dir).resolve()
        self.confidence_threshold: float = confidence_threshold
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "ReportGenerator initialised — reports_dir='%s', confidence_threshold=%.2f",
            self.reports_dir,
            self.confidence_threshold,
        )

    # ------------------------------------------------------------------
    # Main Report Generation Entry Points
    # ------------------------------------------------------------------

    def generate_report(
        self,
        inference_result: Any,
        y_true: Optional[Union[np.ndarray, pd.Series, List[Any]]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        report_title: str = "Predictive Maintenance Inference Report",
        include_samples: bool = True,
        label_map: Optional[Dict[Any, Dict[str, str]]] = None,
    ) -> PredictiveMaintenanceReport:
        """
        Generate a structured PredictiveMaintenanceReport from inference results.

        Args:
            inference_result: An ``InferenceResult`` instance, ``PredictionResult`` instance,
                             or a DataFrame/dict containing predictions.
            y_true (array-like, optional): Ground-truth target labels for calculating metrics.
            metrics (dict, optional): Pre-computed metrics dictionary.
            report_title (str): Title of the generated report.
            include_samples (bool): Include per-sample prediction records.
            label_map (dict, optional): Class label mapping override.

        Returns:
            PredictiveMaintenanceReport: Constructed report object ready for export.
        """
        logger.info("Generating predictive maintenance report...")

        # ── 1. Normalise Input Data ────────────────────────────────────────
        preds, probs, confs, feat_df, meta, l_map = self._parse_inference_input(
            inference_result, label_map
        )

        n_samples = len(preds)
        if n_samples == 0:
            raise ValueError("Cannot generate report for empty inference predictions.")

        # ── 2. Build Prediction Summary ───────────────────────────────────
        prediction_summary = self._build_prediction_summary(
            preds=preds,
            confs=confs,
            label_map=l_map,
        )

        # ── 3. Build Urgency Breakdown ────────────────────────────────────
        urgency_summary = self._build_urgency_summary(
            preds=preds,
            label_map=l_map,
            n_samples=n_samples,
        )

        # ── 4. Compute / Package Evaluation Metrics ───────────────────────
        eval_metrics = self._resolve_evaluation_metrics(
            y_true=y_true,
            y_pred=preds,
            y_prob=probs,
            provided_metrics=metrics,
        )

        # ── 5. Prepare Per-Sample Records ─────────────────────────────────
        sample_records = None
        if include_samples:
            sample_records = self._build_sample_records(
                preds=preds,
                confs=confs,
                probs=probs,
                label_map=l_map,
            )

        # ── 6. Assemble Report Object ─────────────────────────────────────
        report = PredictiveMaintenanceReport(
            title=report_title,
            metadata=meta,
            prediction_summary=prediction_summary,
            urgency_summary=urgency_summary,
            evaluation_metrics=eval_metrics,
            sample_details=sample_records,
            reports_dir=self.reports_dir,
        )

        logger.info(
            "Report successfully generated [ID: %s, Samples: %d]",
            report.report_id,
            n_samples,
        )
        return report

    def generate_and_save(
        self,
        inference_result: Any,
        y_true: Optional[Union[np.ndarray, pd.Series, List[Any]]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        report_title: str = "Predictive Maintenance Report",
        base_filename: Optional[str] = None,
        formats: Union[str, List[str]] = "both",
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Generate and automatically save reports in JSON, TXT, or both formats.

        Args:
            inference_result: Inference output object or DataFrame.
            y_true: Ground truth labels (optional).
            metrics: Pre-computed metrics (optional).
            report_title: Title string.
            base_filename: Custom output filename prefix.
            formats: Format(s) to export — ``"json"``, ``"txt"``, or ``"both"``.

        Returns:
            Tuple[Optional[Path], Optional[Path]]: (json_path, txt_path)
        """
        report = self.generate_report(
            inference_result=inference_result,
            y_true=y_true,
            metrics=metrics,
            report_title=report_title,
        )

        if isinstance(formats, str):
            formats = [formats.lower()]
        else:
            formats = [f.lower() for f in formats]

        json_path: Optional[Path] = None
        txt_path: Optional[Path] = None

        if "both" in formats or ("json" in formats and "txt" in formats):
            json_path, txt_path = report.save_both(base_name=base_filename)
        elif "json" in formats:
            fname = f"{base_filename}.json" if base_filename else None
            json_path = report.save_json(fname)
        elif "txt" in formats:
            fname = f"{base_filename}.txt" if base_filename else None
            txt_path = report.save_txt(fname)
        else:
            raise ValueError(f"Invalid export format '{formats}'. Must be 'json', 'txt', or 'both'.")

        return json_path, txt_path

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def _parse_inference_input(
        self,
        inp: Any,
        custom_label_map: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
        """
        Parse input objects into standardised arrays and metadata dicts.
        """
        preds: np.ndarray
        probs: Optional[np.ndarray] = None
        confs: np.ndarray
        feat_df: pd.DataFrame = pd.DataFrame()
        meta: Dict[str, Any] = {}
        label_map: Dict[str, Any] = custom_label_map or _DEFAULT_LABEL_MAP

        if hasattr(inp, "predictions"):
            preds = np.asarray(inp.predictions)
            probs = getattr(inp, "probabilities", None)
            feat_df = getattr(inp, "feature_matrix", pd.DataFrame())
            meta = getattr(inp, "metadata", {}) or {}
            label_map = custom_label_map or getattr(inp, "label_map", _DEFAULT_LABEL_MAP)

            if hasattr(inp, "confidence") and inp.confidence is not None:
                confs = np.asarray(inp.confidence)
            elif probs is not None:
                p_2d = probs if probs.ndim == 2 else probs.reshape(-1, 1)
                confs = p_2d.max(axis=1)
            else:
                confs = np.full(len(preds), np.nan)

        elif isinstance(inp, pd.DataFrame):
            if "prediction" in inp.columns:
                preds = inp["prediction"].to_numpy()
            elif "predictions" in inp.columns:
                preds = inp["predictions"].to_numpy()
            else:
                preds = inp.iloc[:, 0].to_numpy()

            if "confidence" in inp.columns:
                confs = inp["confidence"].to_numpy()
            else:
                confs = np.full(len(preds), np.nan)

            feat_df = inp
        elif isinstance(inp, (list, np.ndarray)):
            preds = np.asarray(inp)
            confs = np.full(len(preds), np.nan)
        else:
            raise TypeError(
                f"Unsupported inference result type: {type(inp)}. "
                "Expected InferenceResult, PredictionResult, DataFrame, or array."
            )

        if not meta:
            meta = {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "sample_count": len(preds),
            }

        return preds, probs, confs, feat_df, meta, label_map

    def _build_prediction_summary(
        self,
        preds: np.ndarray,
        confs: np.ndarray,
        label_map: Dict[Any, Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Build statistical prediction summary.
        """
        n_samples = len(preds)
        valid_confs = confs[~np.isnan(confs)]

        high_conf_cnt = int((valid_confs >= self.confidence_threshold).sum()) if len(valid_confs) > 0 else 0
        low_conf_cnt = int((valid_confs < self.confidence_threshold).sum()) if len(valid_confs) > 0 else 0

        conf_stats = {}
        if len(valid_confs) > 0:
            conf_stats = {
                "min": round(float(valid_confs.min()), 4),
                "max": round(float(valid_confs.max()), 4),
                "mean": round(float(valid_confs.mean()), 4),
                "std": round(float(valid_confs.std()), 4),
            }

        # Class distribution
        class_counts = pd.Series(preds).value_counts().sort_index()
        class_dist = {}
        for cls_val, count in class_counts.items():
            info = label_map.get(cls_val, {"urgency": str(cls_val)})
            class_dist[str(cls_val)] = {
                "count": int(count),
                "pct": round(float(count / n_samples * 100), 2),
                "urgency": info.get("urgency", str(cls_val)),
                "action": info.get("action", ""),
            }

        # Critical / Low confidence risk alerts
        alerts = []
        for idx in range(n_samples):
            p = preds[idx]
            c = confs[idx]
            info = label_map.get(p, {"urgency": str(p)})
            urg = info.get("urgency", str(p))

            # Flag if urgency is critical or confidence is below threshold
            is_critical = "CRITICAL" in urg.upper() or "MAINTENANCE" in urg.upper()
            is_low_conf = not np.isnan(c) and c < self.confidence_threshold

            if is_critical or is_low_conf:
                alerts.append({
                    "sample_id": idx,
                    "prediction": int(p) if isinstance(p, (np.integer, int)) else str(p),
                    "confidence": round(float(c), 4) if not np.isnan(c) else None,
                    "urgency": urg,
                    "reason": "CRITICAL_RISK" if is_critical else "LOW_CONFIDENCE",
                })

        return {
            "total_samples": n_samples,
            "confidence_threshold": self.confidence_threshold,
            "high_confidence_count": high_conf_cnt,
            "high_confidence_pct": round(float(high_conf_cnt / n_samples * 100), 2) if n_samples > 0 else 0.0,
            "low_confidence_count": low_conf_cnt,
            "low_confidence_pct": round(float(low_conf_cnt / n_samples * 100), 2) if n_samples > 0 else 0.0,
            "confidence_stats": conf_stats,
            "class_distribution": class_dist,
            "critical_alerts": alerts,
        }

    def _build_urgency_summary(
        self,
        preds: np.ndarray,
        label_map: Dict[Any, Dict[str, str]],
        n_samples: int,
    ) -> Dict[str, Any]:
        """
        Build urgency breakdown dictionary.
        """
        summary: Dict[str, Dict[str, Any]] = {}
        for p in preds:
            info = label_map.get(p, {"urgency": f"Class {p}", "action": ""})
            urg = info.get("urgency", f"Class {p}")

            if urg not in summary:
                summary[urg] = {
                    "count": 0,
                    "pct": 0.0,
                    "action": info.get("action", ""),
                }
            summary[urg]["count"] += 1

        for urg in summary:
            summary[urg]["pct"] = round(float(summary[urg]["count"] / n_samples * 100), 2)

        return summary

    def _resolve_evaluation_metrics(
        self,
        y_true: Optional[Union[np.ndarray, pd.Series, List[Any]]],
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        provided_metrics: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate or format evaluation metrics.
        """
        if provided_metrics is not None:
            return provided_metrics

        if y_true is None:
            return None

        y_true_arr = np.asarray(y_true)
        if len(y_true_arr) != len(y_pred):
            logger.warning(
                "Length mismatch between y_true (%d) and y_pred (%d). Skipping metric computation.",
                len(y_true_arr),
                len(y_pred),
            )
            return None

        try:
            acc = float(accuracy_score(y_true_arr, y_pred))
            prec_macro = float(precision_score(y_true_arr, y_pred, average="macro", zero_division=0))
            prec_weighted = float(precision_score(y_true_arr, y_pred, average="weighted", zero_division=0))
            rec_macro = float(recall_score(y_true_arr, y_pred, average="macro", zero_division=0))
            rec_weighted = float(recall_score(y_true_arr, y_pred, average="weighted", zero_division=0))
            f1_macro = float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0))
            f1_weighted = float(f1_score(y_true_arr, y_pred, average="weighted", zero_division=0))

            cm = confusion_matrix(y_true_arr, y_pred).tolist()
            clf_rep = classification_report(y_true_arr, y_pred, zero_division=0)

            roc_auc = None
            if y_prob is not None:
                try:
                    if y_prob.ndim == 1 or (y_prob.ndim == 2 and y_prob.shape[1] == 2):
                        p = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                        roc_auc = float(roc_auc_score(y_true_arr, p))
                    else:
                        roc_auc = float(roc_auc_score(y_true_arr, y_prob, multi_class="ovr"))
                except Exception as auc_err:
                    logger.debug("ROC-AUC calculation skipped: %s", auc_err)

            metrics_dict = {
                "accuracy": round(acc, 4),
                "precision_macro": round(prec_macro, 4),
                "precision_weighted": round(prec_weighted, 4),
                "recall_macro": round(rec_macro, 4),
                "recall_weighted": round(rec_weighted, 4),
                "f1_macro": round(f1_macro, 4),
                "f1_weighted": round(f1_weighted, 4),
                "confusion_matrix": cm,
                "classification_report": clf_rep,
            }

            if roc_auc is not None:
                metrics_dict["roc_auc"] = round(roc_auc, 4)

            return metrics_dict

        except Exception as exc:
            logger.error("Error computing evaluation metrics: %s", exc)
            return None

    def _build_sample_records(
        self,
        preds: np.ndarray,
        confs: np.ndarray,
        probs: Optional[np.ndarray],
        label_map: Dict[Any, Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """
        Build per-sample dictionary records.
        """
        records = []
        for i in range(len(preds)):
            pred = preds[i]
            conf = confs[i]
            info = label_map.get(pred, {"urgency": str(pred), "action": ""})

            rec: Dict[str, Any] = {
                "sample_id": i,
                "prediction": int(pred) if isinstance(pred, (np.integer, int)) else str(pred),
                "confidence": round(float(conf), 4) if not np.isnan(conf) else None,
                "urgency": info.get("urgency", str(pred)),
                "action": info.get("action", ""),
            }

            if probs is not None and probs.ndim == 2:
                for j in range(probs.shape[1]):
                    rec[f"prob_class_{j}"] = round(float(probs[i, j]), 4)

            records.append(rec)

        return records


# ===========================================================================
# Module-level convenience function
# ===========================================================================


def generate_report(
    inference_result: Any,
    y_true: Optional[Union[np.ndarray, pd.Series, List[Any]]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    report_title: str = "Predictive Maintenance Inference Report",
    output_dir: Union[str, Path] = _DEFAULT_REPORTS_DIR,
    confidence_threshold: float = _DEFAULT_CONF_THRESH,
    save: bool = True,
    base_filename: Optional[str] = None,
    formats: Union[str, List[str]] = "both",
) -> PredictiveMaintenanceReport:
    """
    Convenience function for quick generation and optional export of predictive maintenance reports.

    Args:
        inference_result: Output from InferencePipeline, PredictionResult, DataFrame, etc.
        y_true: Ground truth target values (optional).
        metrics: Pre-computed metrics dictionary (optional).
        report_title: Title string for report.
        output_dir: Directory to save generated reports.
        confidence_threshold: Threshold for flagging low confidence predictions.
        save: Save report to disk automatically if True.
        base_filename: Custom filename prefix.
        formats: Export formats (``"json"``, ``"txt"``, or ``"both"``).

    Returns:
        PredictiveMaintenanceReport: The generated report object.

    Example::

        from src.configs.reports import generate_report

        report = generate_report(result, save=True)
    """
    generator = ReportGenerator(
        reports_dir=output_dir,
        confidence_threshold=confidence_threshold,
    )

    report = generator.generate_report(
        inference_result=inference_result,
        y_true=y_true,
        metrics=metrics,
        report_title=report_title,
    )

    if save:
        generator.generate_and_save(
            inference_result=inference_result,
            y_true=y_true,
            metrics=metrics,
            report_title=report_title,
            base_filename=base_filename,
            formats=formats,
        )

    return report
