"""
model_evaluation.py
===================
Standalone model evaluation module for the IoT Predictive Maintenance project.

This module provides a ModelEvaluator class that works with **any** fitted
sklearn-compatible classifier — it is deliberately decoupled from BaselineModel
so it can be reused as future models are introduced (SVM, XGBoost, LSTM, etc.).

Evaluation suite:
    1. Core metrics      — Accuracy, Precision, Recall, F1-score
                           (both macro and weighted averages)
    2. ROC-AUC           — Binary targets only; multi-class OvR also supported
    3. Confusion matrix  — Raw-count and row-normalised variants
    4. Classification report — Full per-class breakdown (sklearn format)
    5. Precision-Recall curve — Binary targets
    6. ROC curve         — Binary and multi-class OvR
    7. Cross-validation  — Stratified k-fold CV score summary
    8. Report export     — Saves JSON metrics + TXT human-readable report
                           to outputs/reports/

All plots are saved under outputs/plots/ and returned as Figure objects
for interactive use in Jupyter notebooks.

Part of the Infotact Solutions Data Science & Machine Learning Internship project.

Note:
    This module is *evaluation only* — it never re-trains or modifies any model.
    All modelling logic remains in src/models/baseline_model.py.
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import label_binarize

# ---------------------------------------------------------------------------
# Suppress non-critical sklearn / matplotlib warnings
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global plot aesthetics (consistent across the pipeline)
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=0.95)
plt.rcParams.update({
    "figure.dpi":        150,
    "axes.titlesize":    13,
    "axes.labelsize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#FAFAFA",
})

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
MetricsDict  = Dict[str, float]
ArrayLike    = Union[np.ndarray, pd.Series, List]

# ---------------------------------------------------------------------------
# Centralised configuration defaults
# ---------------------------------------------------------------------------
try:
    from src.configs.config import get_config as _get_cfg
    _cfg = _get_cfg()
    _D_REPORTS_DIR    = _cfg.paths.reports_dir
    _D_PLOTS_DIR      = _cfg.paths.plots_dir
    _D_CV_FOLDS       = _cfg.evaluation.cv_folds
    _D_CV_SCORING     = _cfg.evaluation.cv_scoring
    _D_RANDOM_STATE   = _cfg.project.random_seed
except Exception:   # fallback when running module in isolation
    _D_REPORTS_DIR    = "outputs/reports"
    _D_PLOTS_DIR      = "outputs/plots"
    _D_CV_FOLDS       = 5
    _D_CV_SCORING     = "f1_weighted"
    _D_RANDOM_STATE   = 42


# ---------------------------------------------------------------------------
# ModelEvaluator class
# ---------------------------------------------------------------------------


class ModelEvaluator:
    """
    A model-agnostic evaluation harness for sklearn-compatible classifiers.

    Accepts any fitted estimator alongside ground-truth labels and predictions
    (or raw test data from which predictions are inferred automatically).
    Produces a comprehensive evaluation suite and persists results to disk.

    The class is deliberately decoupled from ``BaselineModel`` — it can evaluate
    any current or future classifier without modification.

    Attributes:
        model_name   (str):       Human-readable name for report headers.
        reports_dir  (Path):      Directory where text / JSON reports are saved.
        plots_dir    (Path):      Directory where evaluation plots are saved.
        _y_true      (np.ndarray): Ground-truth labels.
        _y_pred      (np.ndarray): Predicted class labels.
        _y_prob      (np.ndarray|None): Predicted probabilities (if available).
        _classes     (np.ndarray): Sorted unique class labels.
        _is_binary   (bool):      Whether the problem is binary classification.
        metrics      (MetricsDict): Computed scalar metrics after evaluate().
        _saved_plots (List[Path]): Paths of figures saved in this session.
        _saved_reports(List[Path]): Paths of report files saved this session.

    Example — from a fitted BaselineModel::

        from src.models.baseline_model import BaselineModel
        from src.evaluation.model_evaluation import ModelEvaluator

        bm = BaselineModel(target_col="failure")
        bm.train(enriched_df).evaluate()

        evaluator = ModelEvaluator(model_name="RandomForest_v1")
        evaluator.from_predictions(
            y_true=bm.y_test,
            y_pred=bm._y_pred,
            y_prob=bm._y_prob,
        )
        evaluator.evaluate()
        evaluator.run_full_evaluation()

    Example — from a raw estimator + test data::

        evaluator = ModelEvaluator(model_name="RandomForest_v1")
        evaluator.from_estimator(
            estimator=bm.model,
            X_test=bm.X_test,
            y_test=bm.y_test,
        )
        evaluator.evaluate()
        evaluator.run_full_evaluation()
    """

    def __init__(
        self,
        model_name:  str            = "Model",
        reports_dir: Union[str, Path] = _D_REPORTS_DIR,
        plots_dir:   Union[str, Path] = _D_PLOTS_DIR,
    ) -> None:
        """
        Initialise the ModelEvaluator.

        Args:
            model_name (str):
                A short identifier used in report headers and file names.
                Defaults to ``"Model"``.
            reports_dir (str | Path):
                Directory for JSON and TXT report files.
                Created automatically if absent. Defaults to ``"outputs/reports"``.
            plots_dir (str | Path):
                Directory for PNG plot files.
                Created automatically if absent. Defaults to ``"outputs/plots"``.
        """
        self.model_name:  str  = model_name
        self.reports_dir: Path = Path(reports_dir).resolve()
        self.plots_dir:   Path = Path(plots_dir).resolve()

        # Ensure output directories exist
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # State populated by from_predictions() / from_estimator()
        self._y_true:   Optional[np.ndarray] = None
        self._y_pred:   Optional[np.ndarray] = None
        self._y_prob:   Optional[np.ndarray] = None   # shape (n,) binary | (n, k) multi
        self._classes:  Optional[np.ndarray] = None
        self._is_binary: bool                = False
        self._estimator: Optional[BaseEstimator] = None  # kept for cross-val

        self.metrics:         MetricsDict = {}
        self._saved_plots:    List[Path]  = []
        self._saved_reports:  List[Path]  = []

        logger.info(
            "ModelEvaluator initialised — model_name='%s', "
            "reports_dir=%s, plots_dir=%s.",
            model_name, self.reports_dir, self.plots_dir,
        )

    # ------------------------------------------------------------------
    # Data-ingestion methods (two entry points)
    # ------------------------------------------------------------------

    def from_predictions(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        y_prob: Optional[ArrayLike] = None,
    ) -> "ModelEvaluator":
        """
        Provide ground-truth labels and pre-computed predictions directly.

        Use this path when predictions have already been generated externally
        (e.g., from BaselineModel.evaluate()).

        Args:
            y_true (array-like): Ground-truth class labels.
            y_pred (array-like): Predicted class labels (hard predictions).
            y_prob (array-like, optional):
                Predicted probability scores.
                - Binary: 1-D array of positive-class probabilities.
                - Multi-class: 2-D array of shape ``(n_samples, n_classes)``.
                Required for ROC-AUC and ROC/PR curve plots.

        Returns:
            ModelEvaluator: ``self`` — enables method chaining.

        Raises:
            ValueError: If *y_true* and *y_pred* have different lengths.
        """
        self._y_true = np.asarray(y_true)
        self._y_pred = np.asarray(y_pred)

        if len(self._y_true) != len(self._y_pred):
            raise ValueError(
                f"y_true and y_pred must have the same length, "
                f"got {len(self._y_true)} and {len(self._y_pred)}."
            )

        self._y_prob   = np.asarray(y_prob) if y_prob is not None else None
        self._classes  = np.unique(self._y_true)
        self._is_binary = (len(self._classes) == 2)

        logger.info(
            "from_predictions() — %d samples, %d classes, binary=%s, "
            "probabilities=%s.",
            len(self._y_true), len(self._classes),
            self._is_binary, self._y_prob is not None,
        )
        return self

    def from_estimator(
        self,
        estimator: BaseEstimator,
        X_test:    Union[pd.DataFrame, np.ndarray],
        y_test:    ArrayLike,
    ) -> "ModelEvaluator":
        """
        Derive predictions from a fitted sklearn estimator and test data.

        This is the preferred entry point when working directly with a fitted
        model object. Probability scores are extracted automatically if the
        estimator exposes ``predict_proba``.

        Args:
            estimator (BaseEstimator): Any fitted sklearn-compatible classifier.
            X_test (DataFrame | ndarray): Feature matrix for the test set.
            y_test (array-like): Ground-truth labels for the test set.

        Returns:
            ModelEvaluator: ``self`` — enables method chaining.

        Raises:
            TypeError: If *estimator* does not expose a ``predict`` method.
        """
        if not hasattr(estimator, "predict"):
            raise TypeError(
                f"estimator must implement a 'predict' method, "
                f"got {type(estimator).__name__}."
            )

        self._estimator = estimator
        y_pred = estimator.predict(X_test)
        y_prob = None

        if hasattr(estimator, "predict_proba"):
            proba  = estimator.predict_proba(X_test)
            y_true_arr = np.asarray(y_test)
            n_classes  = len(np.unique(y_true_arr))
            # Binary → use positive-class column; multi-class → full matrix
            y_prob = proba[:, 1] if n_classes == 2 else proba

        return self.from_predictions(y_true=y_test, y_pred=y_pred, y_prob=y_prob)

    # ------------------------------------------------------------------
    # Core metric computation
    # ------------------------------------------------------------------

    def evaluate(self) -> MetricsDict:
        """
        Compute all scalar evaluation metrics and cache them in :attr:`metrics`.

        Metrics computed:
            - ``accuracy``                — overall correct-prediction rate
            - ``precision_macro``         — macro-averaged precision
            - ``precision_weighted``      — sample-weighted precision
            - ``recall_macro``            — macro-averaged recall
            - ``recall_weighted``         — sample-weighted recall
            - ``f1_macro``                — macro-averaged F1-score
            - ``f1_weighted``             — sample-weighted F1-score
            - ``roc_auc``                 — AUC-ROC (binary: standard;
                                            multi-class: OvR macro)
            - ``avg_precision``           — average precision score
                                            (binary only, from PR curve)

        Returns:
            MetricsDict: Mapping of metric name → float value.

        Raises:
            RuntimeError: If neither :meth:`from_predictions` nor
                          :meth:`from_estimator` has been called first.
        """
        self._require_data()

        y_true = self._y_true
        y_pred = self._y_pred

        # ── Scalar classification metrics ─────────────────────────────────
        self.metrics = {
            "accuracy":           float(accuracy_score(y_true, y_pred)),
            "precision_macro":    float(precision_score(
                y_true, y_pred, average="macro",    zero_division=0)),
            "precision_weighted": float(precision_score(
                y_true, y_pred, average="weighted", zero_division=0)),
            "recall_macro":       float(recall_score(
                y_true, y_pred, average="macro",    zero_division=0)),
            "recall_weighted":    float(recall_score(
                y_true, y_pred, average="weighted", zero_division=0)),
            "f1_macro":           float(f1_score(
                y_true, y_pred, average="macro",    zero_division=0)),
            "f1_weighted":        float(f1_score(
                y_true, y_pred, average="weighted", zero_division=0)),
        }

        # ── ROC-AUC ───────────────────────────────────────────────────────
        if self._y_prob is not None:
            try:
                if self._is_binary:
                    self.metrics["roc_auc"] = float(
                        roc_auc_score(y_true, self._y_prob)
                    )
                    self.metrics["avg_precision"] = float(
                        average_precision_score(y_true, self._y_prob)
                    )
                else:
                    # Multi-class: one-vs-rest macro AUC
                    self.metrics["roc_auc_ovr_macro"] = float(
                        roc_auc_score(
                            y_true, self._y_prob,
                            multi_class="ovr", average="macro",
                        )
                    )
            except ValueError as exc:
                logger.warning("Could not compute AUC metric: %s", exc)

        logger.info(
            "evaluate() — accuracy=%.4f, f1_weighted=%.4f, "
            "precision_w=%.4f, recall_w=%.4f.",
            self.metrics["accuracy"],
            self.metrics["f1_weighted"],
            self.metrics["precision_weighted"],
            self.metrics["recall_weighted"],
        )
        return self.metrics

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def display_report(self) -> None:
        """
        Print a fully formatted evaluation report to stdout.

        Sections shown:
            1. Header (model name, sample counts, class info)
            2. Scalar metrics table with inline ASCII score bars
            3. Full sklearn per-class classification report
            4. ASCII confusion matrix

        Raises:
            RuntimeError: If :meth:`evaluate` has not been called first.
        """
        self._require_evaluated()

        sep = "─" * 70

        # ── 1. Header ─────────────────────────────────────────────────────
        print(f"\n{sep}")
        print(f"  MODEL EVALUATION REPORT — {self.model_name.upper()}")
        print(sep)
        print(f"  Samples evaluated  : {len(self._y_true):,}")
        print(f"  Number of classes  : {len(self._classes)}")
        print(f"  Classes            : {list(self._classes)}")
        print(f"  Problem type       : {'Binary' if self._is_binary else 'Multi-class'}")
        print(f"  Probabilities      : {'Available' if self._y_prob is not None else 'Not provided'}")
        print(sep)

        # ── 2. Scalar metrics table ───────────────────────────────────────
        metric_labels = {
            "accuracy":             "Accuracy",
            "precision_macro":      "Precision   (macro)",
            "precision_weighted":   "Precision   (weighted)",
            "recall_macro":         "Recall      (macro)",
            "recall_weighted":      "Recall      (weighted)",
            "f1_macro":             "F1-Score    (macro)",
            "f1_weighted":          "F1-Score    (weighted)",
            "roc_auc":              "ROC-AUC     (binary)",
            "roc_auc_ovr_macro":    "ROC-AUC OvR (macro)",
            "avg_precision":        "Avg Precision (PR-AUC)",
        }

        print(f"\n  {'Metric':<30} {'Score':>8}  {'Visual bar'}")
        print(f"  {'──────':<30} {'─────':>8}  {'──────────'}")
        for key, label in metric_labels.items():
            if key in self.metrics:
                val = self.metrics[key]
                bar = "█" * int(val * 25)
                print(f"  {label:<30} {val:>8.4f}  {bar}")

        # ── 3. Per-class classification report ────────────────────────────
        print(f"\n{sep}")
        print("  Per-Class Classification Report (sklearn)")
        print(sep)
        report = classification_report(
            self._y_true, self._y_pred, zero_division=0
        )
        for line in report.splitlines():
            print(f"    {line}")

        # ── 4. ASCII confusion matrix ──────────────────────────────────────
        print(f"\n{sep}")
        print("  Confusion Matrix")
        print(sep)
        cm     = confusion_matrix(self._y_true, self._y_pred, labels=self._classes)
        header = "           " + "  ".join(f"Pred {c}" for c in self._classes)
        print(f"    {header}")
        for i, row in enumerate(cm):
            row_str = "  ".join(f"{v:>9,}" for v in row)
            print(f"    True {self._classes[i]}  {row_str}")

        print(f"{sep}\n")
        logger.info("Evaluation report printed for '%s'.", self.model_name)

    def save_report(
        self,
        filename_stem: Optional[str] = None,
    ) -> Tuple[Path, Path]:
        """
        Save evaluation results to ``outputs/reports/``.

        Two files are written:

        1. **JSON** (``<stem>_metrics.json``) — machine-readable metrics dict.
        2. **TXT** (``<stem>_report.txt``)    — human-readable full report
                                                 including classification report
                                                 and confusion matrix.

        Args:
            filename_stem (str, optional):
                Base name for output files (without extension).
                Defaults to ``"{model_name}_{YYYYMMDD_HHMMSS}"``.

        Returns:
            Tuple[Path, Path]: ``(json_path, txt_path)`` of the saved files.

        Raises:
            RuntimeError: If :meth:`evaluate` has not been called first.
        """
        self._require_evaluated()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem      = filename_stem or f"{self.model_name}_{timestamp}"

        json_path = self.reports_dir / f"{stem}_metrics.json"
        txt_path  = self.reports_dir / f"{stem}_report.txt"

        # ── 1. JSON — scalar metrics ───────────────────────────────────────
        report_payload = {
            "model_name":    self.model_name,
            "timestamp":     timestamp,
            "n_samples":     int(len(self._y_true)),
            "n_classes":     int(len(self._classes)),
            "classes":       [str(c) for c in self._classes],
            "is_binary":     self._is_binary,
            "metrics":       {k: round(v, 6) for k, v in self.metrics.items()},
        }
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(report_payload, fh, indent=2)

        # ── 2. TXT — full human-readable report ───────────────────────────
        sep = "=" * 70
        cm  = confusion_matrix(self._y_true, self._y_pred, labels=self._classes)

        with txt_path.open("w", encoding="utf-8") as fh:
            fh.write(f"{sep}\n")
            fh.write(f"  MODEL EVALUATION REPORT — {self.model_name.upper()}\n")
            fh.write(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            fh.write(f"{sep}\n\n")

            # Model / data info
            fh.write(f"  Samples evaluated  : {len(self._y_true):,}\n")
            fh.write(f"  Number of classes  : {len(self._classes)}\n")
            fh.write(f"  Classes            : {list(self._classes)}\n")
            fh.write(f"  Problem type       : "
                     f"{'Binary' if self._is_binary else 'Multi-class'}\n\n")

            # Scalar metrics
            fh.write("Scalar Metrics\n")
            fh.write("-" * 40 + "\n")
            for k, v in self.metrics.items():
                fh.write(f"  {k:<30} {v:.6f}\n")

            # Per-class report
            fh.write("\nPer-Class Classification Report\n")
            fh.write("-" * 40 + "\n")
            fh.write(classification_report(
                self._y_true, self._y_pred, zero_division=0
            ))

            # Confusion matrix
            fh.write("\nConfusion Matrix\n")
            fh.write("-" * 40 + "\n")
            header = "           " + "  ".join(
                f"Pred {c}" for c in self._classes
            )
            fh.write(f"  {header}\n")
            for i, row in enumerate(cm):
                row_str = "  ".join(f"{v:>9,}" for v in row)
                fh.write(f"  True {self._classes[i]}  {row_str}\n")

            fh.write(f"\n{sep}\n")

        self._saved_reports.extend([json_path, txt_path])

        print(f"  ✓ JSON metrics → {json_path}")
        print(f"  ✓ TXT report   → {txt_path}")
        logger.info(
            "Reports saved: JSON=%s, TXT=%s.", json_path, txt_path
        )
        return json_path, txt_path

    # ------------------------------------------------------------------
    # Evaluation plots
    # ------------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        normalise: bool = False,
        title_suffix: str = "",
    ) -> plt.Figure:
        """
        Plot an annotated confusion matrix heatmap.

        Args:
            normalise (bool):
                If ``True``, show row-normalised proportions instead of counts.
                Particularly useful for imbalanced datasets.
                Defaults to ``False``.
            title_suffix (str):
                Optional text appended to the plot title. Defaults to ``""``.

        Returns:
            matplotlib.figure.Figure: The figure object.

        Side-effect:
            Saves the figure to ``{plots_dir}/confusion_matrix.png``.

        Raises:
            RuntimeError: If data has not been loaded via
                          :meth:`from_predictions` or :meth:`from_estimator`.
        """
        self._require_data()

        labels = self._classes
        cm     = confusion_matrix(self._y_true, self._y_pred, labels=labels)
        fmt    = ".2f" if normalise else "d"

        if normalise:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_data  = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)
            title    = f"Confusion Matrix (normalised){' — ' + title_suffix if title_suffix else ''}"
        else:
            cm_data = cm
            title   = f"Confusion Matrix (counts){' — ' + title_suffix if title_suffix else ''}"

        n      = len(labels)
        fig_sz = max(5, n * 0.9)
        fig, ax = plt.subplots(figsize=(fig_sz, max(4, fig_sz * 0.8)))

        sns.heatmap(
            cm_data,
            ax=ax,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=[str(l) for l in labels],
            yticklabels=[str(l) for l in labels],
            linewidths=0.5,
            linecolor="#DDDDDD",
            cbar_kws={"shrink": 0.8},
            annot_kws={"fontsize": max(8, 14 - n)},
        )
        ax.set_xlabel("Predicted Label", fontsize=10, labelpad=8)
        ax.set_ylabel("True Label",      fontsize=10, labelpad=8)
        ax.set_title(title,              fontsize=13, fontweight="bold", pad=12)

        plt.tight_layout()
        path = self._save_plot(fig, "confusion_matrix.png")
        plt.show()
        logger.info("Confusion matrix plot saved → %s.", path)
        return fig

    def plot_roc_curve(self) -> Optional[plt.Figure]:
        """
        Plot the ROC curve with AUC annotation.

        - **Binary**: Single ROC curve with AUC.
        - **Multi-class**: One-vs-Rest (OvR) ROC curve per class on one axes.

        Returns:
            matplotlib.figure.Figure | None:
                Figure object, or ``None`` if probability scores are unavailable.

        Side-effect:
            Saves the figure to ``{plots_dir}/roc_curve.png``.

        Raises:
            RuntimeError: If data has not been loaded.
        """
        self._require_data()

        if self._y_prob is None:
            print("  ROC curve: skipped — probability scores not available.")
            logger.info("ROC curve skipped: y_prob not provided.")
            return None

        fig, ax = plt.subplots(figsize=(7, 6))

        # ── Binary ────────────────────────────────────────────────────────
        if self._is_binary:
            fpr, tpr, _ = roc_curve(self._y_true, self._y_prob)
            roc_auc     = auc(fpr, tpr)

            ax.plot(fpr, tpr, color="#457B9D", linewidth=2.5,
                    label=f"{self.model_name} (AUC = {roc_auc:.4f})")
            ax.fill_between(fpr, tpr, alpha=0.10, color="#457B9D")

        # ── Multi-class OvR ───────────────────────────────────────────────
        else:
            palette   = sns.color_palette("muted", len(self._classes))
            y_bin     = label_binarize(self._y_true, classes=self._classes)

            for i, (cls, colour) in enumerate(zip(self._classes, palette)):
                prob_col = self._y_prob[:, i]
                fpr, tpr, _ = roc_curve(y_bin[:, i], prob_col)
                roc_auc     = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=colour, linewidth=1.8,
                        label=f"Class {cls} (AUC = {roc_auc:.3f})")

        # Random-chance baseline
        ax.plot([0, 1], [0, 1], linestyle="--", color="#AAAAAA",
                linewidth=1.2, label="Random (AUC = 0.50)")

        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate",  fontsize=10)
        ax.set_title(
            f"ROC Curve — {self.model_name}",
            fontsize=13, fontweight="bold", pad=12,
        )
        ax.legend(loc="lower right", fontsize=8, framealpha=0.8)
        ax.set_xlim([0.0, 1.01])
        ax.set_ylim([0.0, 1.01])
        ax.grid(alpha=0.3)

        plt.tight_layout()
        path = self._save_plot(fig, "roc_curve.png")
        plt.show()
        logger.info("ROC curve plot saved → %s.", path)
        return fig

    def plot_precision_recall_curve(self) -> Optional[plt.Figure]:
        """
        Plot the Precision-Recall curve (binary targets only).

        The PR curve is more informative than ROC for highly imbalanced
        fault-detection datasets where the negative class dominates.
        The iso-F1 contours are overlaid for reference.

        Returns:
            matplotlib.figure.Figure | None:
                Figure object, or ``None`` if conditions are not met.

        Side-effect:
            Saves the figure to ``{plots_dir}/precision_recall_curve.png``.

        Raises:
            RuntimeError: If data has not been loaded.
        """
        self._require_data()

        if self._y_prob is None:
            print("  PR curve: skipped — probability scores not available.")
            return None

        if not self._is_binary:
            print("  PR curve: skipped — binary targets only.")
            logger.info("PR curve skipped: multi-class target.")
            return None

        precision, recall, _ = precision_recall_curve(self._y_true, self._y_prob)
        pr_auc = auc(recall, precision)
        baseline = float(self._y_true.sum()) / len(self._y_true)

        fig, ax = plt.subplots(figsize=(7, 6))

        # ── Iso-F1 contours ───────────────────────────────────────────────
        f_scores = [0.2, 0.4, 0.6, 0.8]
        for f in f_scores:
            x   = np.linspace(0.01, 1.0, 200)
            y_f = f * x / (2 * x - f)
            mask = (y_f >= 0) & (y_f <= 1)
            ax.plot(x[mask], y_f[mask], linestyle=":", color="#CCCCCC",
                    linewidth=0.8, alpha=0.7)
            ax.annotate(
                f"F1={f}", xy=(x[mask][-1], y_f[mask][-1]),
                fontsize=6, color="#999999",
            )

        # ── Main PR curve ─────────────────────────────────────────────────
        ax.plot(recall, precision, color="#2A9D8F", linewidth=2.5,
                label=f"{self.model_name} (PR-AUC = {pr_auc:.4f})")
        ax.fill_between(recall, precision, alpha=0.10, color="#2A9D8F")
        ax.axhline(y=baseline, color="#E63946", linestyle="--",
                   linewidth=1.2, label=f"Baseline (prevalence = {baseline:.3f})")

        ax.set_xlabel("Recall",    fontsize=10)
        ax.set_ylabel("Precision", fontsize=10)
        ax.set_title(
            f"Precision-Recall Curve — {self.model_name}",
            fontsize=13, fontweight="bold", pad=12,
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
        ax.set_xlim([0.0, 1.01])
        ax.set_ylim([0.0, 1.05])
        ax.grid(alpha=0.3)

        plt.tight_layout()
        path = self._save_plot(fig, "precision_recall_curve.png")
        plt.show()
        logger.info("PR curve saved → %s.", path)
        return fig

    def plot_metrics_bar(self) -> plt.Figure:
        """
        Plot a horizontal bar chart of all scalar evaluation metrics.

        Useful as a compact, at-a-glance summary of model performance.

        Returns:
            matplotlib.figure.Figure: The figure object.

        Side-effect:
            Saves the figure to ``{plots_dir}/metrics_summary.png``.

        Raises:
            RuntimeError: If :meth:`evaluate` has not been called first.
        """
        self._require_evaluated()

        # Friendly labels for display
        label_map = {
            "accuracy":             "Accuracy",
            "precision_macro":      "Precision (macro)",
            "precision_weighted":   "Precision (weighted)",
            "recall_macro":         "Recall (macro)",
            "recall_weighted":      "Recall (weighted)",
            "f1_macro":             "F1 (macro)",
            "f1_weighted":          "F1 (weighted)",
            "roc_auc":              "ROC-AUC",
            "roc_auc_ovr_macro":    "ROC-AUC OvR",
            "avg_precision":        "Avg Precision (PR)",
        }
        labels = [label_map.get(k, k) for k in self.metrics]
        values = list(self.metrics.values())

        palette = [
            "#2A9D8F" if v >= 0.80
            else "#E9C46A" if v >= 0.60
            else "#E63946"
            for v in values
        ]

        fig, ax = plt.subplots(figsize=(9, max(4, len(values) * 0.55)))
        bars = ax.barh(labels, values, color=palette, edgecolor="white", height=0.65)

        # Score labels at bar ends
        for bar, val in zip(bars, values):
            ax.text(
                min(val + 0.01, 0.98),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center", fontsize=8.5,
                color="white" if val > 0.85 else "#333333",
            )

        ax.set_xlim([0.0, 1.05])
        ax.set_xlabel("Score", fontsize=10)
        ax.set_title(
            f"Evaluation Metrics Summary — {self.model_name}",
            fontsize=13, fontweight="bold", pad=12,
        )
        ax.axvline(x=0.80, color="#333333", linestyle="--",
                   linewidth=0.8, alpha=0.5, label="0.80 reference")
        ax.legend(fontsize=7, framealpha=0.6)
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        path = self._save_plot(fig, "metrics_summary.png")
        plt.show()
        logger.info("Metrics bar chart saved → %s.", path)
        return fig

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        estimator:  BaseEstimator,
        X:          Union[pd.DataFrame, np.ndarray],
        y:          ArrayLike,
        cv:         int    = 5,
        scoring:    str    = "f1_weighted",
        display:    bool   = True,
    ) -> Dict[str, float]:
        """
        Run stratified k-fold cross-validation and summarise the scores.

        This is a standalone utility — it does not affect :attr:`metrics` or
        any previously loaded predictions. It is provided so callers can
        validate model stability without a separate manual loop.

        Args:
            estimator (BaseEstimator): An *unfitted* (or re-fittable) estimator.
            X (DataFrame | ndarray):  Full feature matrix.
            y (array-like):           Full label vector.
            cv (int):                 Number of folds. Defaults to ``5``.
            scoring (str):            sklearn scoring string.
                                      Defaults to ``"f1_weighted"``.
            display (bool):           Print fold-level results. Defaults to ``True``.

        Returns:
            Dict[str, float]: Keys — ``mean``, ``std``, ``min``, ``max``,
                              plus one ``fold_{i}`` per fold.
        """
        skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=_D_RANDOM_STATE)
        scores = cross_val_score(
            estimator, X, y, cv=skf, scoring=scoring, n_jobs=-1
        )

        summary = {
            "mean": float(scores.mean()),
            "std":  float(scores.std()),
            "min":  float(scores.min()),
            "max":  float(scores.max()),
            **{f"fold_{i+1}": float(s) for i, s in enumerate(scores)},
        }

        if display:
            sep = "─" * 60
            print(f"\n{sep}")
            print(f"  Cross-Validation ({cv}-Fold Stratified) — {scoring}")
            print(sep)
            for i, s in enumerate(scores, 1):
                bar = "█" * int(s * 30)
                print(f"  Fold {i}: {s:.4f}  {bar}")
            print(sep)
            print(
                f"  Mean ± Std : {scores.mean():.4f} ± {scores.std():.4f}\n"
                f"  Min / Max  : {scores.min():.4f} / {scores.max():.4f}"
            )
            print(f"{sep}\n")

        logger.info(
            "Cross-validation (%d-fold, %s): mean=%.4f ± %.4f.",
            cv, scoring, scores.mean(), scores.std(),
        )
        return summary

    # ------------------------------------------------------------------
    # Convenience orchestrator
    # ------------------------------------------------------------------

    def run_full_evaluation(self, filename_stem: Optional[str] = None) -> MetricsDict:
        """
        Run the complete evaluation suite in a single call.

        Executes:
            1. :meth:`evaluate`
            2. :meth:`display_report`
            3. :meth:`plot_confusion_matrix` (counts)
            4. :meth:`plot_confusion_matrix` (normalised)
            5. :meth:`plot_roc_curve`
            6. :meth:`plot_precision_recall_curve`
            7. :meth:`plot_metrics_bar`
            8. :meth:`save_report`
            9. :meth:`display_saved_artifacts`

        Args:
            filename_stem (str, optional):
                Base name for saved report files. See :meth:`save_report`.

        Returns:
            MetricsDict: All computed scalar metrics.

        Raises:
            RuntimeError: If neither :meth:`from_predictions` nor
                          :meth:`from_estimator` has been called first.
        """
        logger.info("Running full evaluation pipeline for '%s' …", self.model_name)

        self.evaluate()
        self.display_report()
        self.plot_confusion_matrix(normalise=False)
        self.plot_confusion_matrix(normalise=True, title_suffix="normalised")
        self.plot_roc_curve()
        self.plot_precision_recall_curve()
        self.plot_metrics_bar()
        self.save_report(filename_stem=filename_stem)
        self.display_saved_artifacts()

        logger.info("Full evaluation complete — %d plots, %d reports saved.",
                    len(self._saved_plots), len(self._saved_reports))
        return self.metrics

    def display_saved_artifacts(self) -> None:
        """Print a summary of all plots and report files saved this session."""
        sep = "─" * 65
        print(f"\n{sep}")
        print(f"  Saved Artifacts — {self.model_name}")
        print(sep)
        for i, p in enumerate(self._saved_plots, 1):
            size_kb = p.stat().st_size / 1024
            print(f"  Plot   {i:>2} → {p.name:<40}  ({size_kb:.1f} KB)")
        for i, p in enumerate(self._saved_reports, 1):
            size_kb = p.stat().st_size / 1024
            print(f"  Report {i:>2} → {p.name:<40}  ({size_kb:.1f} KB)")
        print(f"{sep}\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_plot(self, fig: plt.Figure, filename: str, dpi: int = 150) -> Path:
        """Save *fig* to :attr:`plots_dir`, record the path, and return it."""
        path = self.plots_dir / filename
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        self._saved_plots.append(path)
        logger.debug("Plot saved → %s", path)
        return path

    def _require_data(self) -> None:
        """Raise RuntimeError if no prediction data has been loaded yet."""
        if self._y_true is None or self._y_pred is None:
            raise RuntimeError(
                "No prediction data loaded. Call from_predictions() or "
                "from_estimator() before running evaluation methods."
            )

    def _require_evaluated(self) -> None:
        """Raise RuntimeError if evaluate() has not been called yet."""
        if not self.metrics:
            raise RuntimeError(
                "Metrics not computed. Call ModelEvaluator.evaluate() first."
            )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        loaded    = "loaded"    if self._y_true  is not None else "no data"
        evaluated = "evaluated" if self.metrics              else "not evaluated"
        return (
            f"ModelEvaluator("
            f"model_name='{self.model_name}', "
            f"status='{loaded} / {evaluated}', "
            f"n_samples={len(self._y_true) if self._y_true is not None else 0})"
        )

    def __str__(self) -> str:
        if self.metrics:
            return (
                f"ModelEvaluator[{self.model_name}] — "
                f"acc={self.metrics.get('accuracy', 0):.4f} | "
                f"f1_w={self.metrics.get('f1_weighted', 0):.4f} | "
                f"plots={len(self._saved_plots)} | "
                f"reports={len(self._saved_reports)}"
            )
        return f"ModelEvaluator[{self.model_name}] — not evaluated"
