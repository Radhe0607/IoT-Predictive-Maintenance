"""
feature_engineering.py
======================
Modular feature engineering pipeline for the IoT Predictive Maintenance project.

This module provides a FeatureEngineer class that accepts a clean DataFrame
(produced by DataPreprocessor) and enriches it with domain-relevant derived
features before model training.

Feature groups generated (each independently toggleable):
    1. Rolling statistics     — rolling mean, min, max, std per sensor column
    2. Lag features           — 1-step and N-step lag values per sensor column
    3. Rate-of-change         — first-order difference (delta) per sensor column
    4. Interaction terms      — pairwise products / ratios between sensor columns
    5. Time-based features    — hour, day-of-week, month, is_weekend, cyclical
                                sin/cos encodings from a timestamp column
    6. Feature selection      — variance threshold + optional correlation pruning

After engineering, the resulting DataFrame can be persisted to CSV via
:meth:`save`.

Part of the Infotact Solutions Data Science & Machine Learning Internship project.

Note:
    Machine learning model training is intentionally out-of-scope here.
    This module only enriches the feature matrix and selects the most
    informative columns for the modelling layer.
"""

import logging
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ColList = List[str]

# ---------------------------------------------------------------------------
# Centralised configuration defaults
# ---------------------------------------------------------------------------
try:
    from src.configs.config import get_config as _get_cfg
    _cfg = _get_cfg()
    _D_ROLLING_WINDOWS   = _cfg.feature_engineering.rolling_windows
    _D_LAG_STEPS         = _cfg.feature_engineering.lag_steps
    _D_MAX_PAIRS         = _cfg.feature_engineering.max_interaction_pairs
    _D_VAR_THRESH        = _cfg.feature_engineering.variance_threshold
    _D_CORR_THRESH       = _cfg.feature_engineering.correlation_threshold
    _D_SAVE_FORMAT       = _cfg.feature_engineering.save_format
    _D_ENG_DATA_FILE     = _cfg.paths.engineered_data_file
except (ImportError, AttributeError, FileNotFoundError) as exc:   # fallback when running module in isolation
    logger.debug("Config singleton unavailable, using defaults: %s", exc)
    _D_ROLLING_WINDOWS   = [3, 5, 10]
    _D_LAG_STEPS         = [1, 3, 5]
    _D_MAX_PAIRS         = 10
    _D_VAR_THRESH        = 0.01
    _D_CORR_THRESH       = 0.95
    _D_SAVE_FORMAT       = "csv"
    _D_ENG_DATA_FILE     = "data/processed/features.csv"

# Supported output formats for save()
SaveFormat = Literal["csv", "parquet"]


# ---------------------------------------------------------------------------
# FeatureEngineer class
# ---------------------------------------------------------------------------


class FeatureEngineer:
    """
    Enriches a clean IoT sensor DataFrame with derived, domain-relevant features.

    The class is designed to sit directly after :class:`~src.data.preprocessing.DataPreprocessor`
    in the pipeline. It operates on a working copy of the DataFrame so the
    caller's original data is never mutated.

    Feature groups (all enabled by default, individually configurable):

    +-----------------------+--------------------------------------------------+
    | Group                 | What it produces                                 |
    +=======================+==================================================+
    | Rolling statistics    | ``{col}_roll_{stat}_{w}``                        |
    |                       | (stat ∈ {mean, min, max, std}; w = window sizes) |
    +-----------------------+--------------------------------------------------+
    | Lag features          | ``{col}_lag_{n}`` for each lag in *lag_steps*    |
    +-----------------------+--------------------------------------------------+
    | Rate-of-change        | ``{col}_delta`` — first-order difference         |
    +-----------------------+--------------------------------------------------+
    | Interaction terms     | ``{a}_x_{b}`` product and ``{a}_div_{b}`` ratio  |
    +-----------------------+--------------------------------------------------+
    | Time-based features   | hour, dow, month, is_weekend, hour_sin/cos,      |
    |                       | dow_sin/cos — derived from a datetime column     |
    +-----------------------+--------------------------------------------------+
    | Feature selection     | VarianceThreshold + optional correlation pruning |
    +-----------------------+--------------------------------------------------+

    Attributes:
        sensor_cols        (List[str]):  Sensor/numerical columns to process.
        timestamp_col      (str|None):   Name of the datetime column (optional).
        rolling_windows    (List[int]):  Window sizes for rolling features.
        lag_steps          (List[int]):  Lag steps to generate.
        interaction_pairs  (List[tuple]): Explicit (col_a, col_b) pairs for
                                          interaction terms. Auto-detected when empty.
        max_interaction_pairs (int):     Cap on auto-detected interaction pairs.
        variance_threshold (float):      Min variance below which a feature is dropped.
        correlation_threshold (float):   Features with abs-correlation above this
                                         value to an earlier feature are dropped.
                                         Set to 1.0 to disable.
        enable_rolling     (bool):       Toggle rolling-statistics group.
        enable_lags        (bool):       Toggle lag-feature group.
        enable_delta       (bool):       Toggle rate-of-change group.
        enable_interactions (bool):      Toggle interaction-term group.
        enable_time        (bool):       Toggle time-based feature group.
        enable_selection   (bool):       Toggle feature-selection step.
        _engineered_df     (pd.DataFrame|None): Result DataFrame after transform().
        _selected_features (List[str]):  Column names retained after selection.

    Example::

        engineer = FeatureEngineer(
            sensor_cols=["temperature", "vibration", "pressure"],
            timestamp_col="timestamp",
            rolling_windows=[3, 5, 10],
            lag_steps=[1, 3],
        )
        enriched_df = engineer.transform(clean_df)
        engineer.display_summary()
        engineer.save("data/processed/features.csv")
    """

    def __init__(
        self,
        sensor_cols:            Optional[ColList]          = None,
        timestamp_col:          Optional[str]              = None,
        rolling_windows:        Optional[List[int]]        = None,
        lag_steps:              Optional[List[int]]        = None,
        interaction_pairs:      Optional[List[Tuple[str, str]]] = None,
        max_interaction_pairs:  int                        = _D_MAX_PAIRS,
        variance_threshold:     float                      = _D_VAR_THRESH,
        correlation_threshold:  float                      = _D_CORR_THRESH,
        enable_rolling:         bool                       = True,
        enable_lags:            bool                       = True,
        enable_delta:           bool                       = True,
        enable_interactions:    bool                       = True,
        enable_time:            bool                       = True,
        enable_selection:       bool                       = True,
    ) -> None:
        """
        Initialise the FeatureEngineer.

        Args:
            sensor_cols (List[str], optional):
                Numerical sensor columns to derive features from.
                Auto-detected from numeric dtypes when ``None``.
            timestamp_col (str, optional):
                Name of the datetime or timestamp column.
                When provided, time-based features are extracted from it.
                When ``None``, the time-feature group is silently skipped even
                if *enable_time* is ``True``.
            rolling_windows (List[int], optional):
                List of integer window sizes for rolling statistics.
                Defaults to ``[3, 5, 10]``.
            lag_steps (List[int], optional):
                List of integer lag steps. Defaults to ``[1, 3, 5]``.
            interaction_pairs (List[tuple], optional):
                Explicit ``(col_a, col_b)`` pairs for interaction features.
                When ``None``, pairs are auto-selected up to *max_interaction_pairs*.
            max_interaction_pairs (int):
                Maximum number of auto-generated interaction pairs. Defaults to 10.
            variance_threshold (float):
                Features whose variance falls below this value are removed.
                Defaults to ``0.01``.
            correlation_threshold (float):
                When two features have an absolute Pearson correlation above this
                value, the later one is removed. Range [0, 1]. Defaults to ``0.95``.
                Set to ``1.0`` to disable correlation pruning.
            enable_rolling (bool):      Enable rolling statistics. Default ``True``.
            enable_lags (bool):         Enable lag features. Default ``True``.
            enable_delta (bool):        Enable rate-of-change features. Default ``True``.
            enable_interactions (bool): Enable interaction terms. Default ``True``.
            enable_time (bool):         Enable time-based features. Default ``True``.
            enable_selection (bool):    Enable feature selection step. Default ``True``.

        Raises:
            ValueError: If *variance_threshold* or *correlation_threshold* are
                        outside their valid ranges.
        """
        # User-supplied configuration
        self.sensor_cols:           Optional[ColList]               = sensor_cols
        self.timestamp_col:         Optional[str]                   = timestamp_col
        self.rolling_windows:       List[int]                       = rolling_windows or _D_ROLLING_WINDOWS
        self.lag_steps:             List[int]                       = lag_steps       or _D_LAG_STEPS
        self.interaction_pairs:     Optional[List[Tuple[str, str]]] = interaction_pairs
        self.max_interaction_pairs: int                             = max_interaction_pairs
        self.variance_threshold:    float                           = variance_threshold
        self.correlation_threshold: float                           = correlation_threshold

        # Feature-group toggles
        self.enable_rolling:      bool = enable_rolling
        self.enable_lags:         bool = enable_lags
        self.enable_delta:        bool = enable_delta
        self.enable_interactions: bool = enable_interactions
        self.enable_time:         bool = enable_time
        self.enable_selection:    bool = enable_selection

        # Internal state
        self._engineered_df:     Optional[pd.DataFrame] = None
        self._selected_features: List[str]              = []
        self._feature_log:       List[str]              = []   # records each group's output

        self._validate_config()
        logger.debug(
            "FeatureEngineer initialised — windows=%s, lags=%s, "
            "interactions=%s, time=%s, selection=%s.",
            self.rolling_windows, self.lag_steps,
            self.enable_interactions, self.enable_time, self.enable_selection,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full feature engineering pipeline on *df*.

        A deep copy is made immediately so the caller's DataFrame is never
        modified. The enriched result is cached in :attr:`_engineered_df` and
        also returned directly.

        Pipeline order:
            1. Resolve sensor columns (auto-detect when not specified).
            2. Detect / convert the timestamp column.
            3. Generate rolling-statistics features.
            4. Generate lag features.
            5. Generate rate-of-change (delta) features.
            6. Generate interaction-term features.
            7. Generate time-based features.
            8. Drop rows with NaN introduced by rolling/lag operations.
            9. Apply feature selection (variance + correlation).

        Args:
            df (pd.DataFrame): Clean DataFrame produced by DataPreprocessor.

        Returns:
            pd.DataFrame: Enriched DataFrame ready for model training.

        Raises:
            TypeError:  If *df* is not a ``pd.DataFrame``.
            ValueError: If no usable sensor columns can be found.
        """
        self._validate_dataframe(df)
        result = df.copy(deep=True)

        cols_before = result.shape[1]
        rows_before = result.shape[0]

        logger.info(
            "FeatureEngineer.transform() started — input: %d rows × %d cols.",
            rows_before, cols_before,
        )

        # ── Resolve columns ──────────────────────────────────────────────
        active_sensor_cols = self._resolve_sensor_cols(result)
        if not active_sensor_cols:
            raise ValueError(
                "No numerical sensor columns could be resolved. "
                "Provide them explicitly via the `sensor_cols` argument."
            )

        timestamp_col = self._resolve_timestamp_col(result)

        # ── Feature generation ───────────────────────────────────────────
        if self.enable_rolling:
            result = self._add_rolling_features(result, active_sensor_cols)

        if self.enable_lags:
            result = self._add_lag_features(result, active_sensor_cols)

        if self.enable_delta:
            result = self._add_delta_features(result, active_sensor_cols)

        if self.enable_interactions:
            result = self._add_interaction_features(result, active_sensor_cols)

        if self.enable_time and timestamp_col:
            result = self._add_time_features(result, timestamp_col)
        elif self.enable_time and not timestamp_col:
            logger.info(
                "Time-based features requested but no timestamp column found — skipping."
            )

        # ── Drop NaN rows introduced by rolling / lag ────────────────────
        rows_before_drop = len(result)
        result = result.dropna().reset_index(drop=True)
        rows_dropped = rows_before_drop - len(result)
        if rows_dropped:
            logger.info(
                "Dropped %d NaN row(s) introduced by rolling/lag operations.", rows_dropped
            )

        # ── Feature selection ────────────────────────────────────────────
        if self.enable_selection:
            result = self._select_features(result)

        self._engineered_df = result
        self._selected_features = list(result.columns)

        logger.info(
            "transform() complete — output: %d rows × %d cols "
            "(added %d features, dropped %d NaN rows).",
            result.shape[0], result.shape[1],
            result.shape[1] - cols_before,
            rows_dropped,
        )
        return result

    def display_summary(self) -> None:
        """
        Print a structured summary of the engineering pipeline output.

        Shows:
            - Input vs output shape
            - Feature groups generated and their column counts
            - Selected vs dropped feature counts (if selection was applied)
            - Top 10 columns by name

        Raises:
            RuntimeError: If :meth:`transform` has not been called yet.
        """
        self._require_transformed()

        df  = self._engineered_df
        sep = "─" * 65

        print(f"\n{sep}")
        print("  Feature Engineering Summary")
        print(sep)
        print(f"  Output shape   : {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        print(f"  Selected cols  : {len(self._selected_features):,}")
        print(sep)

        # Per-group summary
        for entry in self._feature_log:
            print(f"  {entry}")

        print(sep)
        print("  Final feature columns:")
        for i, col in enumerate(self._selected_features, 1):
            print(f"    {i:>3}. {col}")
        print(f"{sep}\n")

        logger.info(
            "FeatureEngineer summary: %d rows, %d columns, %d groups logged.",
            df.shape[0], df.shape[1], len(self._feature_log),
        )

    def save(
        self,
        output_path: Union[str, Path],
        fmt: SaveFormat = "csv",
        index: bool = False,
    ) -> Path:
        """
        Persist the engineered DataFrame to disk.

        Args:
            output_path (str | Path):
                Destination file path. Parent directories are created
                automatically if they do not exist.
            fmt (str):
                Output format — ``"csv"`` (default) or ``"parquet"``.
            index (bool):
                Whether to write the DataFrame index. Defaults to ``False``.

        Returns:
            Path: Resolved absolute path of the saved file.

        Raises:
            RuntimeError: If :meth:`transform` has not been called yet.
            ValueError:   If *fmt* is not ``"csv"`` or ``"parquet"``.
        """
        self._require_transformed()

        if fmt not in ("csv", "parquet"):
            raise ValueError(f"Unsupported format '{fmt}'. Choose 'csv' or 'parquet'.")

        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = self._engineered_df

        if fmt == "csv":
            df.to_csv(output_path, index=index)
        else:
            df.to_parquet(output_path, index=index)

        logger.info(
            "Engineered dataset saved → %s  [%d rows × %d cols, format=%s].",
            output_path, df.shape[0], df.shape[1], fmt,
        )
        return output_path

    @property
    def engineered_df(self) -> pd.DataFrame:
        """
        Return the cached engineered DataFrame.

        Returns:
            pd.DataFrame: The result of the last :meth:`transform` call.

        Raises:
            RuntimeError: If :meth:`transform` has not been called yet.
        """
        self._require_transformed()
        return self._engineered_df

    @property
    def selected_features(self) -> List[str]:
        """
        Return the list of column names retained after feature selection.

        Returns:
            List[str]: Selected feature column names.

        Raises:
            RuntimeError: If :meth:`transform` has not been called yet.
        """
        self._require_transformed()
        return list(self._selected_features)

    # ------------------------------------------------------------------
    # Feature group: Rolling statistics
    # ------------------------------------------------------------------

    def _add_rolling_features(
        self,
        df: pd.DataFrame,
        sensor_cols: ColList,
    ) -> pd.DataFrame:
        """
        Compute rolling mean, min, max, and std for each sensor column.

        Column naming convention: ``{col}_roll_{stat}_{window}``

        Rolling features capture temporal trends and volatility in sensor
        readings — e.g. a rising rolling mean in temperature may indicate
        thermal runaway before an actual fault occurs.

        Args:
            df (pd.DataFrame):   Working copy of the DataFrame.
            sensor_cols (list):  Sensor columns to process.

        Returns:
            pd.DataFrame: DataFrame with rolling-statistic columns appended.
        """
        new_cols: List[str] = []

        for col in sensor_cols:
            for w in self.rolling_windows:
                roll = df[col].rolling(window=w, min_periods=1)

                mean_col = f"{col}_roll_mean_{w}"
                min_col  = f"{col}_roll_min_{w}"
                max_col  = f"{col}_roll_max_{w}"
                std_col  = f"{col}_roll_std_{w}"

                df[mean_col] = roll.mean()
                df[min_col]  = roll.min()
                df[max_col]  = roll.max()
                df[std_col]  = roll.std()

                new_cols.extend([mean_col, min_col, max_col, std_col])

        msg = (
            f"Rolling stats  : +{len(new_cols):>4} cols "
            f"(windows={self.rolling_windows}, stats=[mean,min,max,std])"
        )
        self._feature_log.append(msg)
        logger.info(msg)
        return df

    # ------------------------------------------------------------------
    # Feature group: Lag features
    # ------------------------------------------------------------------

    def _add_lag_features(
        self,
        df: pd.DataFrame,
        sensor_cols: ColList,
    ) -> pd.DataFrame:
        """
        Generate time-lagged copies of each sensor column.

        Column naming convention: ``{col}_lag_{n}``

        Lag features allow the model to learn from the sensor's historical
        trajectory — essential for predicting failures that evolve gradually.

        Args:
            df (pd.DataFrame):   Working copy of the DataFrame.
            sensor_cols (list):  Sensor columns to lag.

        Returns:
            pd.DataFrame: DataFrame with lag columns appended.
        """
        new_cols: List[str] = []

        for col in sensor_cols:
            for lag in self.lag_steps:
                lag_col = f"{col}_lag_{lag}"
                df[lag_col] = df[col].shift(lag)
                new_cols.append(lag_col)

        msg = (
            f"Lag features   : +{len(new_cols):>4} cols "
            f"(steps={self.lag_steps})"
        )
        self._feature_log.append(msg)
        logger.info(msg)
        return df

    # ------------------------------------------------------------------
    # Feature group: Rate-of-change (delta)
    # ------------------------------------------------------------------

    def _add_delta_features(
        self,
        df: pd.DataFrame,
        sensor_cols: ColList,
    ) -> pd.DataFrame:
        """
        Compute the first-order difference (delta) for each sensor column.

        Column naming convention: ``{col}_delta``

        Delta features represent the instantaneous rate of change — a sudden
        spike in vibration delta, for example, may precede mechanical failure.

        Args:
            df (pd.DataFrame):   Working copy of the DataFrame.
            sensor_cols (list):  Columns to differentiate.

        Returns:
            pd.DataFrame: DataFrame with delta columns appended.
        """
        new_cols: List[str] = []

        for col in sensor_cols:
            delta_col = f"{col}_delta"
            df[delta_col] = df[col].diff()
            new_cols.append(delta_col)

        msg = (
            f"Delta (diff)   : +{len(new_cols):>4} cols "
            f"(1st-order difference)"
        )
        self._feature_log.append(msg)
        logger.info(msg)
        return df

    # ------------------------------------------------------------------
    # Feature group: Interaction terms
    # ------------------------------------------------------------------

    def _add_interaction_features(
        self,
        df: pd.DataFrame,
        sensor_cols: ColList,
    ) -> pd.DataFrame:
        """
        Generate pairwise product and ratio features between sensor columns.

        Column naming conventions:
            - Product : ``{col_a}_x_{col_b}``
            - Ratio   : ``{col_a}_div_{col_b}``  (division by zero → 0.0)

        Interaction terms help the model capture multi-sensor correlations
        — e.g. the product of temperature and pressure may be more predictive
        of failure than either signal alone.

        When :attr:`interaction_pairs` is ``None``, pairs are auto-selected
        from all combinations of *sensor_cols*, capped at
        :attr:`max_interaction_pairs` to avoid feature explosion.

        Args:
            df (pd.DataFrame):   Working copy of the DataFrame.
            sensor_cols (list):  Pool of columns from which pairs are drawn.

        Returns:
            pd.DataFrame: DataFrame with interaction columns appended.
        """
        from itertools import combinations

        if self.interaction_pairs is not None:
            pairs = self.interaction_pairs
        else:
            all_pairs = list(combinations(sensor_cols, 2))
            pairs     = all_pairs[: self.max_interaction_pairs]

        new_cols: List[str] = []

        for col_a, col_b in pairs:
            if col_a not in df.columns or col_b not in df.columns:
                logger.warning(
                    "Interaction pair (%s, %s) — one or both columns missing. Skipping.",
                    col_a, col_b,
                )
                continue

            # Product
            prod_col = f"{col_a}_x_{col_b}"
            df[prod_col] = df[col_a] * df[col_b]
            new_cols.append(prod_col)

            # Ratio (guard against division by zero)
            ratio_col = f"{col_a}_div_{col_b}"
            df[ratio_col] = df[col_a] / df[col_b].replace(0, np.nan)
            df[ratio_col] = df[ratio_col].fillna(0.0)
            new_cols.append(ratio_col)

        msg = (
            f"Interactions   : +{len(new_cols):>4} cols "
            f"({len(pairs)} pair(s) × [product, ratio])"
        )
        self._feature_log.append(msg)
        logger.info(msg)
        return df

    # ------------------------------------------------------------------
    # Feature group: Time-based features
    # ------------------------------------------------------------------

    def _add_time_features(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
    ) -> pd.DataFrame:
        """
        Extract calendar and cyclical features from a datetime column.

        Features generated:
            - ``hour``          — hour of day (0–23)
            - ``day_of_week``   — integer day (0=Monday … 6=Sunday)
            - ``month``         — month number (1–12)
            - ``is_weekend``    — binary 1 if Saturday/Sunday, else 0
            - ``hour_sin``      — sine encoding of hour (cyclical)
            - ``hour_cos``      — cosine encoding of hour (cyclical)
            - ``dow_sin``       — sine encoding of day-of-week (cyclical)
            - ``dow_cos``       — cosine encoding of day-of-week (cyclical)

        Cyclical sin/cos encodings prevent the model from treating hour 23
        and hour 0 as far apart — important for shift-based failure patterns.

        Args:
            df (pd.DataFrame):   Working copy of the DataFrame.
            timestamp_col (str): Name of the datetime column.

        Returns:
            pd.DataFrame: DataFrame with time-feature columns appended.
        """
        # Coerce to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

        ts = df[timestamp_col]
        new_cols: List[str] = []

        # Calendar scalars
        df["hour"]        = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek
        df["month"]       = ts.dt.month
        df["is_weekend"]  = ts.dt.dayofweek.isin([5, 6]).astype(int)
        new_cols.extend(["hour", "day_of_week", "month", "is_weekend"])

        # Cyclical encodings
        df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
        df["dow_sin"]  = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
        df["dow_cos"]  = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
        new_cols.extend(["hour_sin", "hour_cos", "dow_sin", "dow_cos"])

        msg = (
            f"Time features  : +{len(new_cols):>4} cols "
            f"(from '{timestamp_col}': calendar + cyclical sin/cos)"
        )
        self._feature_log.append(msg)
        logger.info(msg)
        return df

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove low-information and highly-redundant features.

        Two-stage selection:

        **Stage 1 — Variance threshold**
            Any numeric column whose variance is below :attr:`variance_threshold`
            is removed. Constant or near-constant features add no signal.

        **Stage 2 — Correlation pruning**
            From each pair of numeric features whose absolute Pearson
            correlation exceeds :attr:`correlation_threshold`, the *later*
            column (alphabetically) is dropped. This removes redundant
            duplicates while retaining one representative.

        Non-numeric columns (e.g. the timestamp) are passed through unchanged.

        Args:
            df (pd.DataFrame): Fully enriched DataFrame post-NaN-drop.

        Returns:
            pd.DataFrame: DataFrame with low-variance and highly-correlated
                          features removed.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric  = [c for c in df.columns if c not in numeric_cols]

        cols_before = len(numeric_cols)
        kept_cols   = list(numeric_cols)   # will be pruned below

        # ── Stage 1: Variance threshold ──────────────────────────────────
        if self.variance_threshold > 0.0:
            selector = VarianceThreshold(threshold=self.variance_threshold)
            selector.fit(df[numeric_cols])
            mask       = selector.get_support()
            low_var    = [c for c, m in zip(numeric_cols, mask) if not m]
            kept_cols  = [c for c, m in zip(numeric_cols, mask) if m]

            if low_var:
                logger.info(
                    "Variance threshold (%.4f): dropped %d low-variance column(s): %s",
                    self.variance_threshold, len(low_var), low_var,
                )

        # ── Stage 2: Correlation pruning ─────────────────────────────────
        dropped_corr: List[str] = []
        if self.correlation_threshold < 1.0 and len(kept_cols) > 1:
            corr_matrix = df[kept_cols].corr().abs()
            upper       = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [
                col for col in upper.columns
                if any(upper[col] > self.correlation_threshold)
            ]
            dropped_corr = to_drop
            kept_cols    = [c for c in kept_cols if c not in to_drop]

            if dropped_corr:
                logger.info(
                    "Correlation pruning (>%.2f): dropped %d highly-correlated column(s): %s",
                    self.correlation_threshold, len(dropped_corr), dropped_corr,
                )

        # ── Reassemble DataFrame ─────────────────────────────────────────
        final_cols = non_numeric + kept_cols
        df = df[[c for c in final_cols if c in df.columns]]

        total_dropped = cols_before - len(kept_cols)
        msg = (
            f"Feature select : -{total_dropped:>4} cols removed "
            f"(variance<{self.variance_threshold}, corr>{self.correlation_threshold})"
        )
        self._feature_log.append(msg)
        logger.info(
            "Feature selection: %d → %d numeric cols (%d dropped).",
            cols_before, len(kept_cols), total_dropped,
        )
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_sensor_cols(self, df: pd.DataFrame) -> ColList:
        """
        Return the active sensor column list.

        If :attr:`sensor_cols` is provided, validate those names exist in *df*.
        Otherwise, auto-detect all numeric-dtype columns, excluding any
        column whose name matches the configured timestamp column.

        Args:
            df (pd.DataFrame): Working copy of the DataFrame.

        Returns:
            List[str]: Resolved sensor column names.
        """
        if self.sensor_cols is not None:
            missing = [c for c in self.sensor_cols if c not in df.columns]
            if missing:
                raise ValueError(
                    f"The following sensor_cols are not in the DataFrame: {missing}"
                )
            return list(self.sensor_cols)

        # Auto-detect: numeric columns excluding any timestamp column
        auto = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.timestamp_col and self.timestamp_col in auto:
            auto.remove(self.timestamp_col)

        logger.debug("Auto-detected %d sensor column(s): %s", len(auto), auto)
        return auto

    def _resolve_timestamp_col(self, df: pd.DataFrame) -> Optional[str]:
        """
        Return the timestamp column name if it exists in *df*, else ``None``.

        If :attr:`timestamp_col` is explicitly set, validate its presence.
        Otherwise, attempt to auto-detect by looking for columns whose dtype
        is ``datetime64`` or whose name contains common timestamp keywords.

        Args:
            df (pd.DataFrame): Working copy of the DataFrame.

        Returns:
            str | None: Timestamp column name, or ``None`` if unavailable.
        """
        if self.timestamp_col is not None:
            if self.timestamp_col in df.columns:
                return self.timestamp_col
            logger.warning(
                "Configured timestamp_col '%s' not found in DataFrame — "
                "time-based features will be skipped.",
                self.timestamp_col,
            )
            return None

        # Auto-detect by dtype
        dt_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        if dt_cols:
            logger.debug("Auto-detected timestamp column: '%s'.", dt_cols[0])
            return dt_cols[0]

        # Auto-detect by name keywords
        keywords = ("timestamp", "datetime", "date", "time")
        for col in df.columns:
            if any(kw in col.lower() for kw in keywords):
                logger.debug("Auto-detected timestamp column by name: '%s'.", col)
                return col

        return None

    def _validate_config(self) -> None:
        """Raise ValueError for invalid constructor arguments."""
        if not (0.0 <= self.variance_threshold):
            raise ValueError(
                f"variance_threshold must be ≥ 0.0, got {self.variance_threshold}."
            )
        if not (0.0 <= self.correlation_threshold <= 1.0):
            raise ValueError(
                f"correlation_threshold must be in [0.0, 1.0], "
                f"got {self.correlation_threshold}."
            )
        if self.max_interaction_pairs < 1:
            raise ValueError(
                f"max_interaction_pairs must be ≥ 1, got {self.max_interaction_pairs}."
            )
        invalid_windows = [w for w in self.rolling_windows if w < 1]
        if invalid_windows:
            raise ValueError(
                f"All rolling_windows must be ≥ 1. Invalid: {invalid_windows}"
            )
        invalid_lags = [n for n in self.lag_steps if n < 1]
        if invalid_lags:
            raise ValueError(
                f"All lag_steps must be ≥ 1. Invalid: {invalid_lags}"
            )

    def _validate_dataframe(self, df: object) -> None:
        """Raise TypeError if *df* is not a pandas DataFrame."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected a pd.DataFrame, got {type(df).__name__}."
            )

    def _require_transformed(self) -> None:
        """Raise RuntimeError if transform() has not been called yet."""
        if self._engineered_df is None:
            raise RuntimeError(
                "No engineered DataFrame available. "
                "Call FeatureEngineer.transform() first."
            )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted = "transformed" if self._engineered_df is not None else "not transformed"
        return (
            f"FeatureEngineer("
            f"windows={self.rolling_windows}, "
            f"lags={self.lag_steps}, "
            f"interactions={self.enable_interactions}, "
            f"time={self.enable_time}, "
            f"selection={self.enable_selection}, "
            f"status='{fitted}')"
        )

    def __str__(self) -> str:
        fitted = "transformed" if self._engineered_df is not None else "not transformed"
        groups = ", ".join(
            g for g, flag in [
                ("rolling", self.enable_rolling),
                ("lags", self.enable_lags),
                ("delta", self.enable_delta),
                ("interactions", self.enable_interactions),
                ("time", self.enable_time),
            ] if flag
        )
        return f"FeatureEngineer [{fitted}] — groups: [{groups}]"
