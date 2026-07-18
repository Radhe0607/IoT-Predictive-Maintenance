"""
baseline_model.py
=================
Baseline predictive maintenance model for the IoT Predictive Maintenance project.

This module provides a BaselineModel class that trains a Random Forest
classifier on an engineered feature DataFrame, evaluates it with a
comprehensive set of classification metrics, and persists the fitted model
to disk using joblib.

Pipeline responsibilities:
    1. Accept a feature-engineered DataFrame (from FeatureEngineer).
    2. Separate features (X) from the target label (y).
    3. Stratified train / test split to preserve class balance.
    4. Fit a RandomForestClassifier with configurable hyper-parameters.
    5. Evaluate on the hold-out test set:
           - Accuracy, Precision, Recall, F1-score (macro & weighted)
           - Full sklearn classification report
           - Confusion matrix
           - ROC-AUC score (for binary targets)
           - Feature importance ranking
    6. Save the trained model artifact using joblib.
    7. Generate and save evaluation plots:
           - Confusion matrix heatmap
           - Feature importance bar chart
           - ROC curve (binary targets)

Part of the Infotact Solutions Data Science & Machine Learning Internship project.

Note:
    This is a *baseline* implementation. Hyper-parameter tuning,
    cross-validation, and advanced model selection are left for subsequent
    iterations.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Suppress non-critical warnings
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plot aesthetics (consistent with the EDA module)
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
MetricsDict = Dict[str, float]

# ---------------------------------------------------------------------------
# Centralised configuration defaults
# ---------------------------------------------------------------------------
try:
    from src.configs.config import get_config as _get_cfg
    _cfg = _get_cfg()
    _D_TARGET_COL        = _cfg.model.target_col
    _D_TEST_SIZE         = _cfg.model.test_size
    _D_RANDOM_STATE      = _cfg.project.random_seed
    _D_N_ESTIMATORS      = _cfg.model.n_estimators
    _D_MAX_DEPTH         = _cfg.model.max_depth
    _D_MIN_SAMPLES_SPLIT = _cfg.model.min_samples_split
    _D_MIN_SAMPLES_LEAF  = _cfg.model.min_samples_leaf
    _D_CLASS_WEIGHT      = _cfg.model.class_weight
    _D_MODELS_DIR        = _cfg.paths.models_dir
    _D_PLOTS_DIR         = _cfg.paths.plots_dir
    _D_MODEL_FILENAME    = _cfg.paths.model_filename
    _D_TOP_N             = _cfg.evaluation.top_n_features
except Exception:   # fallback when running module in isolation
    _D_TARGET_COL        = "failure"
    _D_TEST_SIZE         = 0.20
    _D_RANDOM_STATE      = 42
    _D_N_ESTIMATORS      = 200
    _D_MAX_DEPTH         = None
    _D_MIN_SAMPLES_SPLIT = 5
    _D_MIN_SAMPLES_LEAF  = 2
    _D_CLASS_WEIGHT      = "balanced"
    _D_MODELS_DIR        = "outputs/models"
    _D_PLOTS_DIR         = "outputs/plots"
    _D_MODEL_FILENAME    = "random_forest_baseline.joblib"
    _D_TOP_N             = 20


# ---------------------------------------------------------------------------
# BaselineModel class
# ---------------------------------------------------------------------------


class BaselineModel:
    """
    Trains and evaluates a Random Forest baseline classifier for predictive
    maintenance.

    The class handles the full supervised-learning workflow from feature /
    target separation through to model persistence and evaluation reporting.
    It integrates cleanly with the upstream ``DataLoader → DataPreprocessor →
    FeatureEngineer`` pipeline.

    Attributes:
        target_col       (str):   Name of the label column in the DataFrame.
        test_size        (float): Fraction of data reserved for testing.
        random_state     (int):   Global random seed for reproducibility.
        n_estimators     (int):   Number of trees in the Random Forest.
        max_depth        (int|None): Max tree depth (``None`` = unlimited).
        class_weight     (str):   ``"balanced"`` or ``None``.
        models_dir       (Path):  Directory for saved model artifacts.
        plots_dir        (Path):  Directory for saved evaluation plots.
        model            (RandomForestClassifier): Fitted model (post-train).
        X_train, X_test  (DataFrame): Feature splits.
        y_train, y_test  (Series):    Label splits.
        feature_names    (List[str]):  Feature column names used for training.
        metrics          (MetricsDict): Evaluation metrics (post-evaluate).
        _is_binary       (bool):  Whether the target is binary (2 classes).
        _saved_model_path(Path):  Path of the serialised model file.

    Example::

        model = BaselineModel(
            target_col="failure",
            test_size=0.2,
            n_estimators=200,
        )
        model.train(enriched_df)
        model.evaluate()
        model.display_report()
        model.save_model()
        model.plot_confusion_matrix()
        model.plot_feature_importance()
    """

    def __init__(
        self,
        target_col:        str            = _D_TARGET_COL,
        test_size:         float          = _D_TEST_SIZE,
        random_state:      int            = _D_RANDOM_STATE,
        n_estimators:      int            = _D_N_ESTIMATORS,
        max_depth:         Optional[int]  = _D_MAX_DEPTH,
        min_samples_split: int            = _D_MIN_SAMPLES_SPLIT,
        min_samples_leaf:  int            = _D_MIN_SAMPLES_LEAF,
        class_weight:      Optional[str]  = _D_CLASS_WEIGHT,
        models_dir:        Union[str, Path] = _D_MODELS_DIR,
        plots_dir:         Union[str, Path] = _D_PLOTS_DIR,
    ) -> None:
        """
        Initialise the BaselineModel.

        Args:
            target_col (str):
                Column name of the binary/multi-class target label.
                Defaults to ``"failure"``.
            test_size (float):
                Proportion of the dataset to use as the test set.
                Must be in the range (0, 1). Defaults to ``0.20``.
            random_state (int):
                Seed for all randomised operations (split, forest).
                Defaults to ``42``.
            n_estimators (int):
                Number of decision trees in the Random Forest.
                Defaults to ``200``.
            max_depth (int, optional):
                Maximum depth of each tree. ``None`` grows trees until all
                leaves are pure or contain fewer than *min_samples_split*
                samples. Defaults to ``None``.
            min_samples_split (int):
                Minimum samples required to split an internal node.
                Defaults to ``5``.
            min_samples_leaf (int):
                Minimum samples required at each leaf node.
                Defaults to ``2``.
            class_weight (str, optional):
                ``"balanced"`` (default) adjusts weights inversely to class
                frequencies — important for imbalanced fault-detection data.
                Pass ``None`` to use uniform weights.
            models_dir (str | Path):
                Directory where the serialised model is saved.
                Defaults to ``"outputs/models"``.
            plots_dir (str | Path):
                Directory where evaluation plots are saved.
                Defaults to ``"outputs/plots"``.

        Raises:
            ValueError: If *test_size* is outside (0, 1).
        """
        if not (0.0 < test_size < 1.0):
            raise ValueError(
                f"test_size must be in (0, 1), got {test_size}."
            )

        self.target_col:        str            = target_col
        self.test_size:         float          = test_size
        self.random_state:      int            = random_state
        self.n_estimators:      int            = n_estimators
        self.max_depth:         Optional[int]  = max_depth
        self.min_samples_split: int            = min_samples_split
        self.min_samples_leaf:  int            = min_samples_leaf
        self.class_weight:      Optional[str]  = class_weight
        self.models_dir:        Path           = Path(models_dir).resolve()
        self.plots_dir:         Path           = Path(plots_dir).resolve()

        # State populated during train() and evaluate()
        self.model:         Optional[RandomForestClassifier] = None
        self.X_train:       Optional[pd.DataFrame]           = None
        self.X_test:        Optional[pd.DataFrame]           = None
        self.y_train:       Optional[pd.Series]              = None
        self.y_test:        Optional[pd.Series]              = None
        self.feature_names: List[str]                        = []
        self.metrics:       MetricsDict                      = {}
        self._is_binary:    bool                             = False
        self._saved_model_path: Optional[Path]               = None
        self._y_pred:       Optional[np.ndarray]             = None
        self._y_prob:       Optional[np.ndarray]             = None
        self._saved_plots:  List[Path]                       = []

        # Ensure output directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "BaselineModel initialised — target='%s', n_estimators=%d, "
            "test_size=%.2f, random_state=%d, class_weight=%s.",
            target_col, n_estimators, test_size, random_state, class_weight,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> "BaselineModel":
        """
        Prepare data, split into train/test sets, and fit the Random Forest.

        Steps performed:
            1. Validate that *target_col* is present in *df*.
            2. Drop non-numeric columns (except the target).
            3. Separate features (X) and target (y).
            4. Stratified train / test split (preserves class proportions).
            5. Fit ``RandomForestClassifier`` on the training set.

        Args:
            df (pd.DataFrame):
                Feature-engineered DataFrame containing both sensor features
                and the target label column.

        Returns:
            BaselineModel: ``self`` — enables method chaining.

        Raises:
            TypeError:  If *df* is not a ``pd.DataFrame``.
            ValueError: If *target_col* is not found in *df*, or if there
                        are fewer than 2 distinct classes in the target.
        """
        self._validate_dataframe(df)
        self._validate_target_col(df)

        logger.info(
            "train() called — input DataFrame: %d rows × %d cols.",
            df.shape[0], df.shape[1],
        )

        # ── 1. Separate features and target ──────────────────────────────
        feature_df = df.drop(columns=[self.target_col])

        # Keep only numeric features (datetime / object cols cannot be
        # passed to sklearn without further encoding)
        feature_df = feature_df.select_dtypes(include=[np.number])
        self.feature_names = list(feature_df.columns)

        X = feature_df
        y = df[self.target_col]

        n_classes = y.nunique()
        if n_classes < 2:
            raise ValueError(
                f"Target column '{self.target_col}' must have at least 2 "
                f"distinct classes, found {n_classes}."
            )
        self._is_binary = (n_classes == 2)

        logger.info(
            "Features: %d columns | Target classes: %s (binary=%s).",
            len(self.feature_names), list(y.unique()), self._is_binary,
        )

        # ── 2. Stratified train / test split ─────────────────────────────
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,              # preserves class distribution
        )

        logger.info(
            "Train/test split — train: %d rows, test: %d rows (%.0f%% / %.0f%%).",
            len(self.X_train), len(self.X_test),
            (1 - self.test_size) * 100, self.test_size * 100,
        )

        # ── 3. Build and fit the Random Forest ───────────────────────────
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,          # use all available CPU cores
            verbose=0,
        )

        logger.info("Fitting RandomForestClassifier …")
        self.model.fit(self.X_train, self.y_train)
        logger.info(
            "Model fitted successfully — %d trees, %d features.",
            self.n_estimators, len(self.feature_names),
        )

        return self   # enable chaining: model.train(df).evaluate().save_model()

    def evaluate(self) -> MetricsDict:
        """
        Run predictions on the test set and compute all evaluation metrics.

        Metrics computed:
            - ``accuracy``             — overall correct-prediction rate
            - ``precision_macro``      — macro-averaged precision
            - ``precision_weighted``   — weighted-averaged precision
            - ``recall_macro``         — macro-averaged recall
            - ``recall_weighted``      — weighted-averaged recall
            - ``f1_macro``             — macro-averaged F1-score
            - ``f1_weighted``          — weighted-averaged F1-score
            - ``roc_auc``              — area under the ROC curve (binary only)

        Returns:
            MetricsDict: Dictionary mapping metric name → float value.

        Raises:
            RuntimeError: If :meth:`train` has not been called first.
        """
        self._require_trained()

        # ── Generate predictions ──────────────────────────────────────────
        self._y_pred = self.model.predict(self.X_test)

        # Probability scores for ROC/AUC
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(self.X_test)
            self._y_prob = proba[:, 1] if self._is_binary else proba

        # ── Core classification metrics ───────────────────────────────────
        self.metrics = {
            "accuracy":           float(accuracy_score(self.y_test, self._y_pred)),
            "precision_macro":    float(precision_score(
                self.y_test, self._y_pred, average="macro",    zero_division=0)),
            "precision_weighted": float(precision_score(
                self.y_test, self._y_pred, average="weighted", zero_division=0)),
            "recall_macro":       float(recall_score(
                self.y_test, self._y_pred, average="macro",    zero_division=0)),
            "recall_weighted":    float(recall_score(
                self.y_test, self._y_pred, average="weighted", zero_division=0)),
            "f1_macro":           float(f1_score(
                self.y_test, self._y_pred, average="macro",    zero_division=0)),
            "f1_weighted":        float(f1_score(
                self.y_test, self._y_pred, average="weighted", zero_division=0)),
        }

        # ── ROC-AUC (binary targets only) ─────────────────────────────────
        if self._is_binary and self._y_prob is not None:
            try:
                self.metrics["roc_auc"] = float(
                    roc_auc_score(self.y_test, self._y_prob)
                )
            except ValueError as exc:
                logger.warning("Could not compute ROC-AUC: %s", exc)

        logger.info(
            "Evaluation — accuracy=%.4f | f1_weighted=%.4f | "
            "precision_weighted=%.4f | recall_weighted=%.4f.",
            self.metrics["accuracy"],
            self.metrics["f1_weighted"],
            self.metrics["precision_weighted"],
            self.metrics["recall_weighted"],
        )

        return self.metrics

    def display_report(self) -> None:
        """
        Print the full evaluation report to stdout.

        Displays:
            - Train / test split sizes
            - Per-metric scores (formatted table)
            - Full sklearn classification_report (per-class precision, recall, F1)
            - Confusion matrix (ASCII)

        Raises:
            RuntimeError: If :meth:`evaluate` has not been called first.
        """
        self._require_evaluated()

        sep = "─" * 68

        print(f"\n{sep}")
        print("  BASELINE MODEL — EVALUATION REPORT")
        print(sep)
        print(f"  Model          : RandomForestClassifier")
        print(f"  Target column  : '{self.target_col}'")
        print(f"  Features used  : {len(self.feature_names)}")
        print(f"  Train rows     : {len(self.X_train):,}")
        print(f"  Test rows      : {len(self.X_test):,}")
        print(f"  n_estimators   : {self.n_estimators}")
        print(f"  max_depth      : {self.max_depth or 'unlimited'}")
        print(f"  class_weight   : {self.class_weight}")
        print(sep)

        # ── Core metrics table ────────────────────────────────────────────
        print(f"\n  {'Metric':<30} {'Score':>10}")
        print(f"  {'──────':<30} {'─────':>10}")
        metric_labels = {
            "accuracy":           "Accuracy",
            "precision_macro":    "Precision (macro)",
            "precision_weighted": "Precision (weighted)",
            "recall_macro":       "Recall (macro)",
            "recall_weighted":    "Recall (weighted)",
            "f1_macro":           "F1-Score (macro)",
            "f1_weighted":        "F1-Score (weighted)",
            "roc_auc":            "ROC-AUC",
        }
        for key, label in metric_labels.items():
            if key in self.metrics:
                val = self.metrics[key]
                bar = "█" * int(val * 20)
                print(f"  {label:<30} {val:>10.4f}  {bar}")

        # ── sklearn classification report ─────────────────────────────────
        print(f"\n{sep}")
        print("  Per-Class Classification Report")
        print(sep)
        report = classification_report(
            self.y_test,
            self._y_pred,
            zero_division=0,
        )
        # Indent each line for visual consistency
        for line in report.splitlines():
            print(f"    {line}")

        # ── Confusion matrix (text) ───────────────────────────────────────
        print(f"\n{sep}")
        print("  Confusion Matrix")
        print(sep)
        cm     = confusion_matrix(self.y_test, self._y_pred)
        labels = sorted(self.y_test.unique())
        header = "        " + "  ".join(f"Pred {l}" for l in labels)
        print(f"    {header}")
        for i, row in enumerate(cm):
            row_str = "  ".join(f"{v:>9,}" for v in row)
            print(f"    True {labels[i]}  {row_str}")

        print(f"{sep}\n")

        logger.info("Evaluation report displayed.")

    def save_model(
        self,
        filename: str = _D_MODEL_FILENAME,
    ) -> Path:
        """
        Serialise the fitted Random Forest model to disk using joblib.

        The model is saved under :attr:`models_dir`. A companion metadata
        file (``<filename>_meta.txt``) recording the feature list and
        evaluation metrics is also written alongside.

        Args:
            filename (str):
                File name for the saved model artifact.
                Defaults to ``"random_forest_baseline.joblib"``.

        Returns:
            Path: Absolute path of the saved ``.joblib`` file.

        Raises:
            RuntimeError: If :meth:`train` has not been called first.
        """
        self._require_trained()

        model_path = self.models_dir / filename
        joblib.dump(self.model, model_path)
        self._saved_model_path = model_path

        # ── Write companion metadata file ─────────────────────────────────
        meta_path = model_path.with_suffix(".txt")
        with meta_path.open("w", encoding="utf-8") as fh:
            fh.write("Random Forest Baseline — Model Metadata\n")
            fh.write("=" * 50 + "\n")
            fh.write(f"Target column  : {self.target_col}\n")
            fh.write(f"n_estimators   : {self.n_estimators}\n")
            fh.write(f"max_depth      : {self.max_depth or 'unlimited'}\n")
            fh.write(f"class_weight   : {self.class_weight}\n")
            fh.write(f"train_rows     : {len(self.X_train)}\n")
            fh.write(f"test_rows      : {len(self.X_test)}\n")
            fh.write(f"n_features     : {len(self.feature_names)}\n\n")
            if self.metrics:
                fh.write("Evaluation Metrics\n")
                fh.write("-" * 30 + "\n")
                for k, v in self.metrics.items():
                    fh.write(f"  {k:<25} {v:.6f}\n")
            fh.write("\nFeature Names\n")
            fh.write("-" * 30 + "\n")
            for feat in self.feature_names:
                fh.write(f"  {feat}\n")

        print(f"\n  ✓ Model saved   → {model_path}")
        print(f"  ✓ Metadata saved → {meta_path}\n")
        logger.info(
            "Model serialised to %s  [%.1f KB].",
            model_path, model_path.stat().st_size / 1024,
        )
        return model_path

    @classmethod
    def load_model(cls, model_path: Union[str, Path]) -> RandomForestClassifier:
        """
        Load a previously saved joblib model artifact from disk.

        This is a convenience class-method; it returns the raw sklearn
        estimator rather than a full BaselineModel instance.

        Args:
            model_path (str | Path): Path to the ``.joblib`` file.

        Returns:
            RandomForestClassifier: The deserialised estimator.

        Raises:
            FileNotFoundError: If *model_path* does not exist.
        """
        path = Path(model_path).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: '{path}'."
            )
        estimator = joblib.load(path)
        logger.info("Model loaded from %s.", path)
        return estimator

    # ------------------------------------------------------------------
    # Evaluation plots
    # ------------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        normalise: bool = False,
    ) -> plt.Figure:
        """
        Plot and save an annotated confusion matrix heatmap.

        Args:
            normalise (bool):
                When ``True``, cells show row-normalised proportions [0, 1]
                instead of raw counts. Useful for imbalanced datasets.
                Defaults to ``False``.

        Returns:
            matplotlib.figure.Figure: The figure object.

        Raises:
            RuntimeError: If :meth:`evaluate` has not been called first.

        Side-effect:
            Saves the figure to ``{plots_dir}/confusion_matrix.png``.
        """
        self._require_evaluated()

        labels = sorted(self.y_test.unique())
        cm     = confusion_matrix(self.y_test, self._y_pred, labels=labels)
        fmt    = ".2f" if normalise else "d"

        if normalise:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_plot  = np.where(row_sums > 0, cm / row_sums, 0.0)
            title    = "Confusion Matrix (row-normalised)"
        else:
            cm_plot = cm
            title   = "Confusion Matrix (counts)"

        fig, ax = plt.subplots(figsize=(max(5, len(labels)), max(4, len(labels) - 1)))

        sns.heatmap(
            cm_plot,
            ax=ax,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            linewidths=0.5,
            linecolor="#DDDDDD",
            cbar=True,
        )
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label",      fontsize=10)
        ax.set_title(title,              fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()

        path = self._save_plot(fig, "confusion_matrix.png")
        plt.show()
        return fig

    def plot_feature_importance(
        self,
        top_n: int = 20,
    ) -> plt.Figure:
        """
        Plot and save a horizontal bar chart of the top-*n* feature importances.

        Importance values are the mean decrease in Gini impurity (MDI) from
        the fitted Random Forest.

        Args:
            top_n (int):
                Number of top features to display. Defaults to ``20``.

        Returns:
            matplotlib.figure.Figure: The figure object.

        Raises:
            RuntimeError: If :meth:`train` has not been called first.

        Side-effect:
            Saves the figure to ``{plots_dir}/feature_importance.png``.
        """
        self._require_trained()

        importances = self.model.feature_importances_
        indices     = np.argsort(importances)[::-1][:top_n]
        top_feats   = [self.feature_names[i] for i in indices]
        top_imp     = importances[indices]

        # Truncate long feature names for readability
        top_feats_disp = [f[:35] + "…" if len(f) > 36 else f for f in top_feats]

        fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.38)))
        palette = sns.color_palette("muted", top_n)

        bars = ax.barh(
            range(top_n), top_imp[::-1],
            color=palette[::-1], edgecolor="white", height=0.7,
        )
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_feats_disp[::-1], fontsize=8)
        ax.set_xlabel("Mean Decrease in Gini Impurity (MDI)", fontsize=10)
        ax.set_title(
            f"Top {top_n} Feature Importances — Random Forest",
            fontsize=13, fontweight="bold", pad=12,
        )
        ax.grid(axis="x", alpha=0.35)

        # Annotate bar ends
        for bar, val in zip(bars, top_imp[::-1]):
            ax.text(
                bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7,
            )

        plt.tight_layout()
        path = self._save_plot(fig, "feature_importance.png")
        plt.show()
        return fig

    def plot_roc_curve(self) -> Optional[plt.Figure]:
        """
        Plot and save the ROC curve for binary classification targets.

        The curve shows the trade-off between True Positive Rate and
        False Positive Rate across all classification thresholds. The
        AUC score is annotated on the plot.

        Returns:
            matplotlib.figure.Figure | None:
                The figure object, or ``None`` for multi-class targets.

        Raises:
            RuntimeError: If :meth:`evaluate` has not been called first.

        Side-effect:
            Saves the figure to ``{plots_dir}/roc_curve.png``.
        """
        self._require_evaluated()

        if not self._is_binary:
            logger.info(
                "ROC curve is only available for binary targets — skipping "
                "(target has >2 classes)."
            )
            print("  ROC curve: skipped (multi-class target).")
            return None

        fpr, tpr, _ = roc_curve(self.y_test, self._y_prob)
        auc_score   = self.metrics.get("roc_auc", float("nan"))

        fig, ax = plt.subplots(figsize=(7, 6))

        ax.plot(fpr, tpr, color="#457B9D", linewidth=2.5,
                label=f"Random Forest (AUC = {auc_score:.4f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#AAAAAA",
                linewidth=1.2, label="Random Classifier (AUC = 0.50)")
        ax.fill_between(fpr, tpr, alpha=0.12, color="#457B9D")

        ax.set_xlabel("False Positive Rate",  fontsize=10)
        ax.set_ylabel("True Positive Rate",   fontsize=10)
        ax.set_title(
            f"ROC Curve — '{self.target_col}'",
            fontsize=13, fontweight="bold", pad=12,
        )
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim([0.0, 1.01])
        ax.set_ylim([0.0, 1.01])
        ax.grid(alpha=0.3)

        plt.tight_layout()
        path = self._save_plot(fig, "roc_curve.png")
        plt.show()
        return fig

    def run_full_pipeline(
        self,
        df: pd.DataFrame,
        top_n_features: int = _D_TOP_N,
    ) -> MetricsDict:
        """
        Convenience method — runs the complete modelling pipeline in one call.

        Equivalent to calling:
            ``train(df) → evaluate() → display_report() →
              save_model() → plot_confusion_matrix() →
              plot_feature_importance() → plot_roc_curve()``

        Args:
            df (pd.DataFrame):   Feature-engineered DataFrame.
            top_n_features (int): Top-N importances to display. Defaults to 20.

        Returns:
            MetricsDict: Dictionary of all computed evaluation metrics.
        """
        self.train(df)
        self.evaluate()
        self.display_report()
        self.save_model()
        self.plot_confusion_matrix()
        self.plot_feature_importance(top_n=top_n_features)
        self.plot_roc_curve()
        self.display_saved_artifacts()
        return self.metrics

    def display_saved_artifacts(self) -> None:
        """Print a summary of all artifacts (model + plots) saved this session."""
        sep = "─" * 65
        print(f"\n{sep}")
        print("  Saved Artifacts")
        print(sep)
        if self._saved_model_path:
            size_kb = self._saved_model_path.stat().st_size / 1024
            print(f"  Model  → {self._saved_model_path}  ({size_kb:.1f} KB)")
        for i, p in enumerate(self._saved_plots, 1):
            size_kb = p.stat().st_size / 1024
            print(f"  Plot {i:>2} → {p.name:<40}  ({size_kb:.1f} KB)")
        print(f"{sep}\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_plot(self, fig: plt.Figure, filename: str, dpi: int = 150) -> Path:
        """Save *fig* to :attr:`plots_dir` and record the path."""
        path = self.plots_dir / filename
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        self._saved_plots.append(path)
        logger.debug("Plot saved → %s", path)
        return path

    def _validate_dataframe(self, df: object) -> None:
        """Raise TypeError if *df* is not a pandas DataFrame."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected a pd.DataFrame, got {type(df).__name__}."
            )

    def _validate_target_col(self, df: pd.DataFrame) -> None:
        """Raise ValueError if the target column is absent from *df*."""
        if self.target_col not in df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

    def _require_trained(self) -> None:
        """Raise RuntimeError if the model has not been trained yet."""
        if self.model is None:
            raise RuntimeError(
                "Model has not been trained. Call BaselineModel.train() first."
            )

    def _require_evaluated(self) -> None:
        """Raise RuntimeError if evaluate() has not been called yet."""
        if self._y_pred is None:
            raise RuntimeError(
                "Model has not been evaluated. Call BaselineModel.evaluate() first."
            )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        trained   = "trained"   if self.model      is not None else "untrained"
        evaluated = "evaluated" if self._y_pred    is not None else "not evaluated"
        return (
            f"BaselineModel("
            f"target='{self.target_col}', "
            f"n_estimators={self.n_estimators}, "
            f"test_size={self.test_size}, "
            f"status='{trained} / {evaluated}')"
        )

    def __str__(self) -> str:
        if self.metrics:
            return (
                f"BaselineModel [trained & evaluated] — "
                f"acc={self.metrics.get('accuracy', 0):.4f} | "
                f"f1_w={self.metrics.get('f1_weighted', 0):.4f} | "
                f"target='{self.target_col}'"
            )
        if self.model is not None:
            return (
                f"BaselineModel [trained, not evaluated] — "
                f"target='{self.target_col}', "
                f"features={len(self.feature_names)}"
            )
        return f"BaselineModel [untrained] — target='{self.target_col}'"
