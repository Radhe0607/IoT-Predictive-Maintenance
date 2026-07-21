"""
evaluation_pipeline.py
======================
Reusable model evaluation pipeline for the IoT Predictive Maintenance project.

This module provides an ``EvaluationPipeline`` class that orchestrates the
complete post-training evaluation workflow in a single, reusable component:

    Load model → Load test data → Generate predictions
    → Compute metrics → Print report → Save to outputs/reports/

Design goals:
    - **Reusability**: Model-agnostic; accepts any joblib-serialised
      sklearn-compatible estimator.
    - **Modularity**: Delegates metric computation entirely to the existing
      ``ModelEvaluator`` harness — no logic is duplicated.
    - **Integration**: Uses ``load_model`` from ``model_manager.py`` (the
      project's single source of truth for model persistence), and reads
      all default paths from ``config.yaml`` via the ``get_config`` singleton.
    - **Flexibility**: Accepts an explicit test-data path *or* falls back to
      the config-driven engineered feature file; supports CSV and Parquet.
    - **Safety**: Every public method raises an informative exception if
      called out of order; all directory creation is automatic.
    - **Traceability**: Structured logging throughout via the standard Python
      ``logging`` framework.

Typical usage (notebook / script):

    # One-liner convenience function
    from src.configs.evaluation.evaluation_pipeline import run_evaluation
    run_evaluation()

    # Fine-grained control
    from src.configs.evaluation.evaluation_pipeline import EvaluationPipeline

    pipeline = EvaluationPipeline(model_name="RandomForest_Baseline")
    pipeline.load_model()
    pipeline.load_test_data()
    pipeline.generate_predictions()
    pipeline.compute_metrics()
    pipeline.print_metrics()
    pipeline.save_results()

Part of the Infotact Solutions Data Science & Machine Learning Internship project.

References:
    - src/configs/utils/model_manager.py   — load_model / save_model
    - src/configs/evaluation/model_evaluation.py — ModelEvaluator
    - src/configs/config.py                — get_config / get_absolute_path
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Suppress non-critical sklearn / matplotlib warnings
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Centralised configuration defaults
# ---------------------------------------------------------------------------
# Graceful fallback when running in isolation (e.g. unit tests without the
# full project layout available on sys.path).
try:
    from src.configs.config import get_config as _get_cfg, get_absolute_path
    _cfg = _get_cfg()

    _DEFAULT_MODEL_PATH   = str(
        get_absolute_path(
            f"{_cfg.paths.models_dir}/{_cfg.paths.model_filename}"
        )
    )
    _DEFAULT_DATA_PATH    = str(get_absolute_path(_cfg.paths.engineered_data_file))
    _DEFAULT_TARGET_COL   = _cfg.model.target_col
    _DEFAULT_REPORTS_DIR  = str(get_absolute_path(_cfg.paths.reports_dir))
    _DEFAULT_PLOTS_DIR    = str(get_absolute_path(_cfg.paths.plots_dir))
    _DEFAULT_RANDOM_SEED  = _cfg.project.random_seed
    _DEFAULT_CV_FOLDS     = _cfg.evaluation.cv_folds
    _DEFAULT_CV_SCORING   = _cfg.evaluation.cv_scoring

except Exception:  # fallback when running module in isolation
    _DEFAULT_MODEL_PATH   = "outputs/models/random_forest_baseline.joblib"
    _DEFAULT_DATA_PATH    = "data/processed/features.csv"
    _DEFAULT_TARGET_COL   = "failure"
    _DEFAULT_REPORTS_DIR  = "outputs/reports"
    _DEFAULT_PLOTS_DIR    = "outputs/plots"
    _DEFAULT_RANDOM_SEED  = 42
    _DEFAULT_CV_FOLDS     = 5
    _DEFAULT_CV_SCORING   = "f1_weighted"

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
MetricsDict = Dict[str, float]


# ---------------------------------------------------------------------------
# EvaluationPipeline class
# ---------------------------------------------------------------------------


class EvaluationPipeline:
    """
    Orchestrates the complete model evaluation workflow for the IoT
    Predictive Maintenance project.

    The pipeline is designed to be model-agnostic — it accepts any
    joblib-serialised sklearn-compatible estimator and evaluates it
    against a labelled test dataset.  All heavy-lifting (metric computation,
    report saving, plotting) is delegated to the existing
    :class:`~src.configs.evaluation.model_evaluation.ModelEvaluator` harness
    so that no evaluation logic is duplicated.

    Call order::

        pipeline = EvaluationPipeline()
        pipeline.load_model()          # 1. Deserialise the trained model
        pipeline.load_test_data()      # 2. Load & split the feature dataset
        pipeline.generate_predictions()# 3. Run inference
        pipeline.compute_metrics()     # 4. Compute Acc / Pre / Rec / F1 …
        pipeline.print_metrics()       # 5. Pretty-print to stdout
        pipeline.save_results()        # 6. Persist JSON + TXT to reports dir

    Or use the convenience orchestrator::

        pipeline.run()                 # executes steps 1-6 in sequence

    Attributes:
        model_name   (str):       Human-readable identifier used in reports.
        model_path   (Path):      Path to the ``.joblib`` model artifact.
        data_path    (Path):      Path to the labelled feature CSV/Parquet.
        target_col   (str):       Name of the target/label column.
        reports_dir  (Path):      Directory where reports are persisted.
        plots_dir    (Path):      Directory where plots are persisted.
        model        (Any):       Deserialised model object (post load_model).
        X_test       (DataFrame): Feature matrix for evaluation.
        y_test       (Series):    Ground-truth labels for evaluation.
        y_pred       (ndarray):   Hard class predictions (post generate_predictions).
        y_prob       (ndarray):   Probability scores — None if unavailable.
        metrics      (MetricsDict): Scalar metrics (post compute_metrics).
        evaluator    (ModelEvaluator): Underlying evaluator instance.
    """

    def __init__(
        self,
        model_name:  str             = "RandomForest_Baseline",
        model_path:  Union[str, Path] = _DEFAULT_MODEL_PATH,
        data_path:   Union[str, Path] = _DEFAULT_DATA_PATH,
        target_col:  str             = _DEFAULT_TARGET_COL,
        reports_dir: Union[str, Path] = _DEFAULT_REPORTS_DIR,
        plots_dir:   Union[str, Path] = _DEFAULT_PLOTS_DIR,
    ) -> None:
        """
        Initialise the EvaluationPipeline.

        Args:
            model_name (str):
                Short identifier used in report headers and filenames.
                Defaults to ``"RandomForest_Baseline"``.
            model_path (str | Path):
                Path to the serialised ``.joblib`` model artifact produced by
                :func:`~src.configs.utils.model_manager.save_model` or
                ``BaselineModel.save_model()``.
                Defaults to the value from ``config.yaml``
                (``outputs/models/random_forest_baseline.joblib``).
            data_path (str | Path):
                Path to the labelled feature dataset (CSV or Parquet).
                Defaults to the config-driven engineered feature file
                (``data/processed/features.csv``).
            target_col (str):
                Column name of the binary/multi-class label.
                Defaults to ``"failure"`` (from ``config.yaml``).
            reports_dir (str | Path):
                Directory for JSON and TXT evaluation reports.
                Created automatically if absent.
                Defaults to ``"outputs/reports"`` (from ``config.yaml``).
            plots_dir (str | Path):
                Directory for evaluation plot PNGs.
                Created automatically if absent.
                Defaults to ``"outputs/plots"`` (from ``config.yaml``).
        """
        self.model_name:  str  = model_name
        self.model_path:  Path = Path(model_path).resolve()
        self.data_path:   Path = Path(data_path).resolve()
        self.target_col:  str  = target_col
        self.reports_dir: Path = Path(reports_dir).resolve()
        self.plots_dir:   Path = Path(plots_dir).resolve()

        # Ensure output directories exist
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # State — populated by successive pipeline steps
        self.model:    Optional[Any]            = None
        self.X_test:   Optional[pd.DataFrame]   = None
        self.y_test:   Optional[pd.Series]      = None
        self.y_pred:   Optional[np.ndarray]     = None
        self.y_prob:   Optional[np.ndarray]     = None
        self.metrics:  MetricsDict              = {}
        self.evaluator = None  # ModelEvaluator — instantiated at compute_metrics

        logger.info(
            "EvaluationPipeline initialised — model='%s', "
            "model_path='%s', data_path='%s'.",
            model_name, self.model_path, self.data_path,
        )

    # ------------------------------------------------------------------
    # Step 1 — Load the trained model
    # ------------------------------------------------------------------

    def load_model(self, model_path: Optional[Union[str, Path]] = None) -> "EvaluationPipeline":
        """
        Deserialise the trained model artifact from disk.

        Uses :func:`~src.configs.utils.model_manager.load_model` — the
        project's single source of truth for model persistence — to load
        any joblib-serialised sklearn-compatible estimator.

        Args:
            model_path (str | Path, optional):
                Override the model path provided at construction.
                Useful for evaluating different model versions without
                re-creating the pipeline.

        Returns:
            EvaluationPipeline: ``self`` — enables method chaining.

        Raises:
            FileNotFoundError:
                If the ``.joblib`` file does not exist on disk.
                Run the training pipeline first to generate the artifact.
        """
        try:
            from src.configs.utils.model_manager import load_model as _load
        except ImportError:
            import joblib

            def _load(path):
                p = Path(path).resolve()
                if not p.exists():
                    raise FileNotFoundError(
                        f"Model file not found: '{p}'.\n"
                        "Ensure the training pipeline has been run."
                    )
                model = joblib.load(p)
                size_kb = p.stat().st_size / 1024
                print(f"  ✓ Model loaded ← {p.name}  ({size_kb:.1f} KB)")
                return model

        path = Path(model_path).resolve() if model_path else self.model_path
        self.model_path = path

        logger.info("Loading model from '%s' …", path)
        self.model = _load(path)
        logger.info(
            "Model loaded — type=%s, path='%s'.",
            type(self.model).__name__, path,
        )
        return self

    # ------------------------------------------------------------------
    # Step 2 — Load the test dataset
    # ------------------------------------------------------------------

    def load_test_data(
        self,
        data_path:  Optional[Union[str, Path]] = None,
        target_col: Optional[str]              = None,
        test_size:  float                      = 0.20,
        random_state: int                      = _DEFAULT_RANDOM_SEED,
    ) -> "EvaluationPipeline":
        """
        Load the labelled feature dataset and extract the test split.

        The method supports both CSV and Parquet files (detected by suffix).
        When the dataset does not already contain a pre-split test subset,
        a stratified hold-out split matching the training configuration
        (``test_size=0.20``, ``random_state=42``) is applied so the test
        set is consistent with what ``BaselineModel.train()`` would produce.

        Args:
            data_path (str | Path, optional):
                Override the data path provided at construction.
                Defaults to the config-driven engineered feature file.
            target_col (str, optional):
                Override the target column name. Defaults to ``self.target_col``.
            test_size (float):
                Fraction of rows to use as the test set when a split is needed.
                Must match the value used during training. Defaults to ``0.20``.
            random_state (int):
                Random seed for the stratified split. Must match training.
                Defaults to the config seed (``42``).

        Returns:
            EvaluationPipeline: ``self`` — enables method chaining.

        Raises:
            FileNotFoundError:
                If the feature dataset file does not exist.
            ValueError:
                If ``target_col`` is not found in the dataset, or if the
                dataset is empty after loading.
            RuntimeError:
                If the file format is not CSV or Parquet.
        """
        from sklearn.model_selection import train_test_split

        path = Path(data_path).resolve() if data_path else self.data_path
        col  = target_col or self.target_col

        # ── 1. Validate file exists ───────────────────────────────────────
        if not path.exists():
            raise FileNotFoundError(
                f"Feature dataset not found: '{path}'.\n"
                "Run the feature engineering pipeline first to generate this file.\n"
                f"Expected location: '{path.parent}'."
            )

        # ── 2. Load based on file format ──────────────────────────────────
        suffix = path.suffix.lower()
        logger.info("Loading feature dataset from '%s' (format: %s) …", path, suffix)

        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in (".parquet", ".pq"):
            df = pd.read_parquet(path)
        else:
            raise RuntimeError(
                f"Unsupported file format '{suffix}'. "
                "Expected '.csv' or '.parquet'."
            )

        if df.empty:
            raise ValueError(
                f"The loaded dataset from '{path}' is empty. "
                "Ensure the feature engineering pipeline has produced valid output."
            )

        logger.info("Dataset loaded — shape: %s.", df.shape)

        # ── 3. Validate target column ─────────────────────────────────────
        if col not in df.columns:
            raise ValueError(
                f"Target column '{col}' not found in the dataset.\n"
                f"Available columns: {list(df.columns[:10])}{'…' if len(df.columns) > 10 else ''}.\n"
                "Set target_col to the correct label column name."
            )

        # ── 4. Separate features and target ───────────────────────────────
        X = df.drop(columns=[col])
        y = df[col]

        # Drop any non-numeric columns that the model cannot handle
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            logger.warning(
                "Dropping %d non-numeric column(s) from features: %s.",
                len(non_numeric), non_numeric,
            )
            X = X.drop(columns=non_numeric)

        # ── 5. Apply stratified hold-out split (mirrors BaselineModel) ────
        _, X_test, _, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        self.X_test = X_test.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)

        logger.info(
            "Test split extracted — %d samples, %d features, "
            "target='%s', classes=%s.",
            len(self.X_test), self.X_test.shape[1],
            col, list(y_test.unique()),
        )
        print(
            f"  ✓ Test data loaded — {len(self.X_test):,} samples × "
            f"{self.X_test.shape[1]:,} features  (target='{col}')"
        )
        return self

    # ------------------------------------------------------------------
    # Step 3 — Generate predictions
    # ------------------------------------------------------------------

    def generate_predictions(self) -> "EvaluationPipeline":
        """
        Run inference on the test set using the loaded model.

        Generates both hard class predictions (via ``model.predict``) and
        probability scores (via ``model.predict_proba``, when available).
        Probability scores are required for ROC-AUC and PR-curve metrics —
        the evaluator degrades gracefully when they are unavailable.

        Returns:
            EvaluationPipeline: ``self`` — enables method chaining.

        Raises:
            RuntimeError:
                If :meth:`load_model` or :meth:`load_test_data` has not
                been called first.
        """
        self._require_model()
        self._require_data()

        logger.info(
            "Generating predictions for %d samples …", len(self.X_test)
        )

        # ── Hard predictions ──────────────────────────────────────────────
        self.y_pred = self.model.predict(self.X_test)

        # ── Probability scores (optional) ─────────────────────────────────
        if hasattr(self.model, "predict_proba"):
            try:
                proba = self.model.predict_proba(self.X_test)
                n_classes = len(np.unique(np.asarray(self.y_test)))
                # Binary → positive-class column; multi-class → full matrix
                self.y_prob = proba[:, 1] if n_classes == 2 else proba
                logger.info(
                    "Probabilities extracted — shape: %s.", self.y_prob.shape
                )
            except Exception as exc:
                logger.warning(
                    "predict_proba() failed (%s). Continuing without "
                    "probability scores.", exc
                )
                self.y_prob = None
        else:
            logger.info(
                "Model does not expose predict_proba — ROC-AUC "
                "and PR-curve will be skipped."
            )
            self.y_prob = None

        logger.info(
            "Predictions generated — %d hard predictions, "
            "probabilities=%s.",
            len(self.y_pred), self.y_prob is not None,
        )
        print(
            f"  ✓ Predictions generated — {len(self.y_pred):,} samples  "
            f"(probabilities: {'yes' if self.y_prob is not None else 'no'})"
        )
        return self

    # ------------------------------------------------------------------
    # Step 4 — Compute evaluation metrics
    # ------------------------------------------------------------------

    def compute_metrics(self) -> MetricsDict:
        """
        Compute evaluation metrics using the :class:`ModelEvaluator` harness.

        Metrics computed:
            - ``accuracy``              — overall correct-prediction rate
            - ``precision_macro``       — macro-averaged precision
            - ``precision_weighted``    — sample-weighted precision
            - ``recall_macro``          — macro-averaged recall
            - ``recall_weighted``       — sample-weighted recall
            - ``f1_macro``              — macro-averaged F1-score
            - ``f1_weighted``           — sample-weighted F1-score
            - ``roc_auc``               — AUC-ROC (when probabilities available)
            - ``avg_precision``         — PR-AUC (binary, when probs available)

        Returns:
            MetricsDict: Mapping of metric name → float value.

        Raises:
            RuntimeError:
                If :meth:`generate_predictions` has not been called first.
        """
        self._require_predictions()

        # Import here to avoid circular imports at module load time
        from src.configs.evaluation.model_evaluation import ModelEvaluator

        logger.info("Computing evaluation metrics via ModelEvaluator …")

        self.evaluator = ModelEvaluator(
            model_name=self.model_name,
            reports_dir=self.reports_dir,
            plots_dir=self.plots_dir,
        )

        self.evaluator.from_predictions(
            y_true=self.y_test,
            y_pred=self.y_pred,
            y_prob=self.y_prob,
        )

        self.metrics = self.evaluator.evaluate()

        logger.info(
            "Metrics computed — accuracy=%.4f, f1_weighted=%.4f, "
            "precision_w=%.4f, recall_w=%.4f.",
            self.metrics.get("accuracy", 0),
            self.metrics.get("f1_weighted", 0),
            self.metrics.get("precision_weighted", 0),
            self.metrics.get("recall_weighted", 0),
        )
        return self.metrics

    # ------------------------------------------------------------------
    # Step 5 — Print metrics in readable format
    # ------------------------------------------------------------------

    def print_metrics(self) -> None:
        """
        Print evaluation metrics to stdout in a clean, human-readable format.

        Output sections:
            1. Header — model name, sample count, class info
            2. Metrics table — score value + ASCII bar for each metric
            3. Per-class classification report (sklearn format)
            4. ASCII confusion matrix

        Raises:
            RuntimeError: If :meth:`compute_metrics` has not been called first.
        """
        self._require_metrics()

        sep  = "─" * 70
        sep2 = "═" * 70

        # ── Header ───────────────────────────────────────────────────────
        print(f"\n{sep2}")
        print(f"  MODEL EVALUATION REPORT — {self.model_name.upper()}")
        print(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(sep2)
        print(f"  Model file         : {self.model_path.name}")
        print(f"  Data file          : {self.data_path.name}")
        print(f"  Samples evaluated  : {len(self.y_test):,}")
        print(f"  Target column      : '{self.target_col}'")
        print(sep)

        # ── Core metrics table ────────────────────────────────────────────
        _METRIC_LABELS = {
            "accuracy":             "Accuracy",
            "precision_macro":      "Precision       (macro)",
            "precision_weighted":   "Precision       (weighted)",
            "recall_macro":         "Recall          (macro)",
            "recall_weighted":      "Recall          (weighted)",
            "f1_macro":             "F1-Score        (macro)",
            "f1_weighted":          "F1-Score        (weighted)",
            "roc_auc":              "ROC-AUC         (binary)",
            "roc_auc_ovr_macro":    "ROC-AUC OvR     (macro)",
            "avg_precision":        "Avg Precision   (PR-AUC)",
        }

        print(f"\n  {'Metric':<32} {'Score':>8}   Visual")
        print(f"  {'──────':<32} {'─────':>8}   ──────")
        for key, label in _METRIC_LABELS.items():
            if key in self.metrics:
                val = self.metrics[key]
                filled = int(val * 30)
                bar    = "█" * filled + "░" * (30 - filled)
                # Colour-code: green ≥ 0.80, yellow ≥ 0.60, red otherwise
                if val >= 0.80:
                    tag = "✅"
                elif val >= 0.60:
                    tag = "⚠️ "
                else:
                    tag = "❌"
                print(f"  {label:<32} {val:>8.4f}   {tag} {bar}")

        # ── Per-class classification report ──────────────────────────────
        from sklearn.metrics import classification_report, confusion_matrix
        print(f"\n{sep}")
        print("  Per-Class Classification Report")
        print(sep)
        report = classification_report(self.y_test, self.y_pred, zero_division=0)
        for line in report.splitlines():
            print(f"    {line}")

        # ── ASCII confusion matrix ────────────────────────────────────────
        classes = np.unique(np.asarray(self.y_test))
        cm      = confusion_matrix(self.y_test, self.y_pred, labels=classes)
        print(f"\n{sep}")
        print("  Confusion Matrix")
        print(sep)
        header = "             " + "  ".join(f"Pred {c}" for c in classes)
        print(f"    {header}")
        for i, row in enumerate(cm):
            row_str = "  ".join(f"{v:>9,}" for v in row)
            print(f"    True {classes[i]}  {row_str}")

        print(f"\n{sep2}\n")
        logger.info("Metrics printed for model '%s'.", self.model_name)

    # ------------------------------------------------------------------
    # Step 6 — Save results
    # ------------------------------------------------------------------

    def save_results(
        self,
        filename_stem: Optional[str] = None,
        save_plots:    bool          = True,
    ) -> Tuple[Path, Path]:
        """
        Persist evaluation results to ``outputs/reports/``.

        Two files are always written:

        1. **JSON** (``<stem>_metrics.json``) — machine-readable metrics dict
           with model metadata, timestamp, and all scalar scores.
        2. **TXT** (``<stem>_report.txt``) — human-readable full report
           including the classification report and confusion matrix.

        Optionally, evaluation plots (confusion matrix, ROC curve, PR curve,
        metrics bar chart) are generated and saved to ``outputs/plots/``.

        Args:
            filename_stem (str, optional):
                Base name for the output files (without extension).
                Defaults to ``"{model_name}_{YYYYMMDD_HHMMSS}"``.
            save_plots (bool):
                If ``True``, generate and save evaluation plots via
                ``ModelEvaluator``. Defaults to ``True``.

        Returns:
            Tuple[Path, Path]: ``(json_path, txt_path)`` of the saved files.

        Raises:
            RuntimeError: If :meth:`compute_metrics` has not been called first.
        """
        self._require_metrics()

        logger.info(
            "Saving evaluation results to '%s' …", self.reports_dir
        )

        # ── Delegate report saving to ModelEvaluator ──────────────────────
        json_path, txt_path = self.evaluator.save_report(
            filename_stem=filename_stem
        )

        # ── Optionally generate & save plots ──────────────────────────────
        if save_plots:
            try:
                self.evaluator.plot_confusion_matrix(normalise=False)
                self.evaluator.plot_confusion_matrix(
                    normalise=True, title_suffix="normalised"
                )
                self.evaluator.plot_roc_curve()
                self.evaluator.plot_precision_recall_curve()
                self.evaluator.plot_metrics_bar()
                logger.info("Evaluation plots saved to '%s'.", self.plots_dir)
            except Exception as exc:
                logger.warning(
                    "Could not generate evaluation plots: %s. "
                    "Continuing without plots.", exc
                )

        logger.info(
            "Results saved — JSON: '%s', TXT: '%s'.", json_path, txt_path
        )
        return json_path, txt_path

    # ------------------------------------------------------------------
    # Convenience orchestrator
    # ------------------------------------------------------------------

    def run(
        self,
        model_path:    Optional[Union[str, Path]] = None,
        data_path:     Optional[Union[str, Path]] = None,
        target_col:    Optional[str]              = None,
        filename_stem: Optional[str]              = None,
        save_plots:    bool                       = True,
        test_size:     float                      = 0.20,
        random_state:  int                        = _DEFAULT_RANDOM_SEED,
    ) -> MetricsDict:
        """
        Run the complete evaluation pipeline in a single call.

        Executes the following steps in order:
            1. :meth:`load_model`
            2. :meth:`load_test_data`
            3. :meth:`generate_predictions`
            4. :meth:`compute_metrics`
            5. :meth:`print_metrics`
            6. :meth:`save_results`

        Args:
            model_path (str | Path, optional):
                Override the model path set at construction.
            data_path (str | Path, optional):
                Override the feature dataset path set at construction.
            target_col (str, optional):
                Override the target column name.
            filename_stem (str, optional):
                Base name for saved report files. See :meth:`save_results`.
            save_plots (bool):
                Whether to generate and save evaluation plots.
                Defaults to ``True``.
            test_size (float):
                Hold-out fraction for the stratified split.
                Defaults to ``0.20`` (mirrors ``BaselineModel``).
            random_state (int):
                Random seed for the split. Defaults to ``42``.

        Returns:
            MetricsDict: All computed scalar metrics.

        Raises:
            FileNotFoundError:
                If the model artifact or feature dataset is missing.
            ValueError:
                If the target column is absent from the dataset.
        """
        sep = "═" * 70

        print(f"\n{sep}")
        print(f"  IoT Predictive Maintenance — Model Evaluation Pipeline")
        print(f"  Model  : {self.model_name}")
        print(f"  Start  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{sep}\n")

        logger.info(
            "Starting full evaluation pipeline for '%s' …", self.model_name
        )

        try:
            self.load_model(model_path=model_path)
            self.load_test_data(
                data_path=data_path,
                target_col=target_col,
                test_size=test_size,
                random_state=random_state,
            )
            self.generate_predictions()
            self.compute_metrics()
            self.print_metrics()
            self.save_results(
                filename_stem=filename_stem,
                save_plots=save_plots,
            )

        except FileNotFoundError as exc:
            logger.error("Pipeline aborted — file not found: %s", exc)
            print(f"\n  ❌ ERROR — File not found:\n  {exc}\n")
            raise
        except ValueError as exc:
            logger.error("Pipeline aborted — validation error: %s", exc)
            print(f"\n  ❌ ERROR — Validation failed:\n  {exc}\n")
            raise
        except Exception as exc:
            logger.error(
                "Pipeline aborted — unexpected error: %s", exc, exc_info=True
            )
            print(f"\n  ❌ ERROR — Unexpected failure:\n  {exc}\n")
            raise

        print(f"\n{sep}")
        print("  Evaluation pipeline complete.")
        print(f"  Reports → {self.reports_dir}")
        if save_plots:
            print(f"  Plots   → {self.plots_dir}")
        print(f"{sep}\n")

        logger.info(
            "Evaluation pipeline complete for '%s' — "
            "accuracy=%.4f, f1_weighted=%.4f.",
            self.model_name,
            self.metrics.get("accuracy", 0),
            self.metrics.get("f1_weighted", 0),
        )
        return self.metrics

    # ------------------------------------------------------------------
    # Private guard helpers
    # ------------------------------------------------------------------

    def _require_model(self) -> None:
        """Raise RuntimeError if the model has not been loaded yet."""
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Call EvaluationPipeline.load_model() first."
            )

    def _require_data(self) -> None:
        """Raise RuntimeError if test data has not been loaded yet."""
        if self.X_test is None or self.y_test is None:
            raise RuntimeError(
                "Test data not loaded. Call EvaluationPipeline.load_test_data() first."
            )

    def _require_predictions(self) -> None:
        """Raise RuntimeError if predictions have not been generated yet."""
        if self.y_pred is None:
            raise RuntimeError(
                "Predictions not generated. Call EvaluationPipeline.generate_predictions() first."
            )

    def _require_metrics(self) -> None:
        """Raise RuntimeError if metrics have not been computed yet."""
        if not self.metrics:
            raise RuntimeError(
                "Metrics not computed. Call EvaluationPipeline.compute_metrics() first."
            )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        steps = []
        if self.model    is not None: steps.append("model_loaded")
        if self.X_test   is not None: steps.append("data_loaded")
        if self.y_pred   is not None: steps.append("predictions_ready")
        if self.metrics:              steps.append("metrics_computed")
        status = " → ".join(steps) if steps else "not started"
        return (
            f"EvaluationPipeline("
            f"model_name='{self.model_name}', "
            f"status='{status}')"
        )

    def __str__(self) -> str:
        if self.metrics:
            return (
                f"EvaluationPipeline[{self.model_name}] — "
                f"acc={self.metrics.get('accuracy', 0):.4f} | "
                f"f1_w={self.metrics.get('f1_weighted', 0):.4f} | "
                f"prec_w={self.metrics.get('precision_weighted', 0):.4f} | "
                f"rec_w={self.metrics.get('recall_weighted', 0):.4f}"
            )
        return f"EvaluationPipeline[{self.model_name}] — not evaluated"


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def run_evaluation(
    model_name:    str                       = "RandomForest_Baseline",
    model_path:    Optional[Union[str, Path]] = None,
    data_path:     Optional[Union[str, Path]] = None,
    target_col:    Optional[str]             = None,
    reports_dir:   Optional[Union[str, Path]] = None,
    plots_dir:     Optional[Union[str, Path]] = None,
    filename_stem: Optional[str]             = None,
    save_plots:    bool                      = True,
    test_size:     float                     = 0.20,
    random_state:  int                       = _DEFAULT_RANDOM_SEED,
) -> MetricsDict:
    """
    Convenience entry point — run the full model evaluation pipeline.

    Creates an :class:`EvaluationPipeline` with the given (or config-driven)
    settings and runs all steps: load model → load test data → generate
    predictions → compute metrics → print report → save to ``outputs/reports/``.

    This function is the simplest way to run evaluation from a notebook cell
    or a script without manual instantiation:

    .. code-block:: python

        from src.configs.evaluation.evaluation_pipeline import run_evaluation

        metrics = run_evaluation()
        print(metrics["accuracy"])

    Args:
        model_name (str):
            Human-readable identifier for the model. Used in report headers
            and output filenames. Defaults to ``"RandomForest_Baseline"``.
        model_path (str | Path, optional):
            Path to the ``.joblib`` model artifact.
            Defaults to the config-driven path
            (``outputs/models/random_forest_baseline.joblib``).
        data_path (str | Path, optional):
            Path to the labelled feature CSV/Parquet dataset.
            Defaults to the config-driven engineered feature file
            (``data/processed/features.csv``).
        target_col (str, optional):
            Name of the target/label column.
            Defaults to the config value (``"failure"``).
        reports_dir (str | Path, optional):
            Directory for JSON and TXT report files.
            Defaults to ``"outputs/reports"`` (from ``config.yaml``).
        plots_dir (str | Path, optional):
            Directory for evaluation plot PNGs.
            Defaults to ``"outputs/plots"`` (from ``config.yaml``).
        filename_stem (str, optional):
            Base name for saved report files (without extension).
            Defaults to ``"{model_name}_{YYYYMMDD_HHMMSS}"``.
        save_plots (bool):
            Whether to generate and save evaluation plots.
            Defaults to ``True``.
        test_size (float):
            Hold-out fraction for the stratified test split.
            Must match the value used during training. Defaults to ``0.20``.
        random_state (int):
            Random seed for reproducibility. Defaults to ``42``.

    Returns:
        MetricsDict: Mapping of metric name → float value for all computed
                     evaluation metrics.

    Raises:
        FileNotFoundError:
            If the model artifact or feature dataset cannot be found.
        ValueError:
            If ``target_col`` is missing from the dataset columns.
    """
    pipeline = EvaluationPipeline(
        model_name=model_name,
        model_path=model_path  or _DEFAULT_MODEL_PATH,
        data_path=data_path    or _DEFAULT_DATA_PATH,
        target_col=target_col  or _DEFAULT_TARGET_COL,
        reports_dir=reports_dir or _DEFAULT_REPORTS_DIR,
        plots_dir=plots_dir    or _DEFAULT_PLOTS_DIR,
    )

    return pipeline.run(
        filename_stem=filename_stem,
        save_plots=save_plots,
        test_size=test_size,
        random_state=random_state,
    )
