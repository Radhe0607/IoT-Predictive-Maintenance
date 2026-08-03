"""
preprocessing.py
================
Modular data preprocessing pipeline for the IoT Predictive Maintenance project.

This module provides a DataPreprocessor class that transforms a raw pandas
DataFrame (produced by DataLoader) into a clean, ML-ready DataFrame.

Pipeline steps (each independently configurable):
    1. Drop exact duplicate rows
    2. Coerce columns to their correct data types
    3. Handle missing values  (numerical → median/mean/zero; categorical → mode/constant)
    4. Normalise numerical sensor features with sklearn StandardScaler
    5. Encode categorical columns  (label-encoding or one-hot, caller's choice)

Part of the Infotact Solutions Data Science & Machine Learning Internship project.

Note:
    Feature engineering and model training are intentionally out-of-scope here.
    This module only produces a *clean* DataFrame ready for the feature layer.
"""

import logging

from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# Maps column name → desired pandas dtype string, e.g. {"timestamp": "datetime64[ns]"}
DtypeMap = Dict[str, str]

# Strategy literals
NumericalFillStrategy  = Literal["median", "mean", "zero", "drop"]
CategoricalFillStrategy = Literal["mode", "constant", "drop"]
CategoricalEncoding     = Literal["label", "onehot", "none"]


# ---------------------------------------------------------------------------
# DataPreprocessor class
# ---------------------------------------------------------------------------


class DataPreprocessor:
    """
    Applies a configurable cleaning and normalisation pipeline to a raw DataFrame.

    The preprocessor is stateful: fitted scalers and encoders are retained after
    :meth:`fit_transform` so that the same transformations can be consistently
    applied to held-out data via :meth:`transform`.

    Attributes:
        numerical_cols       (List[str]):  Columns treated as numerical features.
        categorical_cols     (List[str]):  Columns treated as categorical features.
        dtype_map            (DtypeMap):   Column → target dtype coercions.
        num_fill_strategy    (str):        Missing-value strategy for numeric cols.
        cat_fill_strategy    (str):        Missing-value strategy for categorical cols.
        cat_fill_constant    (str):        Constant used when ``cat_fill_strategy="constant"``.
        cat_encoding         (str):        Categorical encoding method.
        drop_missing_thresh  (float):      Drop rows where fraction of NaNs exceeds this.
        _scaler              (StandardScaler):            Fitted scaler (post fit).
        _label_encoders      (Dict[str, LabelEncoder]):  Fitted label encoders per column.
        _onehot_cols         (List[str]):                 Columns expanded via one-hot.
        _fitted              (bool):                      Whether fit_transform was called.

    Example::

        preprocessor = DataPreprocessor(
            numerical_cols=["temperature", "vibration", "pressure"],
            categorical_cols=["machine_type", "failure_mode"],
            cat_encoding="label",
        )
        clean_df = preprocessor.fit_transform(raw_df)
        preprocessor.display_summary(raw_df, clean_df)
    """

    def __init__(
        self,
        numerical_cols:      Optional[List[str]] = None,
        categorical_cols:    Optional[List[str]] = None,
        dtype_map:           Optional[DtypeMap]  = None,
        num_fill_strategy:   NumericalFillStrategy   = "median",
        cat_fill_strategy:   CategoricalFillStrategy = "mode",
        cat_fill_constant:   str   = "UNKNOWN",
        cat_encoding:        CategoricalEncoding     = "label",
        drop_missing_thresh: float = 0.0,
        normalize:           bool  = True,
    ) -> None:
        """
        Initialise the DataPreprocessor.

        Args:
            numerical_cols (List[str], optional):
                Column names to treat as numerical features.
                If ``None``, all numeric-dtype columns in the DataFrame are used
                automatically at fit time.
            categorical_cols (List[str], optional):
                Column names to treat as categorical features.
                If ``None``, all object/category-dtype columns are used automatically.
            dtype_map (DtypeMap, optional):
                Mapping of ``{column_name: target_dtype}`` applied before any other
                step, e.g. ``{"timestamp": "datetime64[ns]", "sensor_id": "int32"}``.
            num_fill_strategy (str):
                How to impute missing values in numerical columns.
                One of ``"median"`` (default), ``"mean"``, ``"zero"``, ``"drop"``.
            cat_fill_strategy (str):
                How to impute missing values in categorical columns.
                One of ``"mode"`` (default), ``"constant"``, ``"drop"``.
            cat_fill_constant (str):
                Constant replacement used when ``cat_fill_strategy="constant"``.
                Defaults to ``"UNKNOWN"``.
            cat_encoding (str):
                Encoding scheme for categorical columns.
                One of ``"label"`` (default), ``"onehot"``, ``"none"``.
            drop_missing_thresh (float):
                Fraction of missing values per row above which the row is dropped
                *before* imputation. Range [0, 1]. ``0.0`` disables row-dropping.
            normalize (bool):
                Whether to apply ``StandardScaler`` to numerical columns.
                Defaults to ``True``.

        Raises:
            ValueError: If any strategy or encoding literal is invalid.
        """
        self.numerical_cols:      Optional[List[str]] = numerical_cols
        self.categorical_cols:    Optional[List[str]] = categorical_cols
        self.dtype_map:           DtypeMap  = dtype_map or {}
        self.num_fill_strategy:   str  = num_fill_strategy
        self.cat_fill_strategy:   str  = cat_fill_strategy
        self.cat_fill_constant:   str  = cat_fill_constant
        self.cat_encoding:        str  = cat_encoding
        self.drop_missing_thresh: float = drop_missing_thresh
        self.normalize:           bool  = normalize

        # State populated during fit_transform
        self._scaler:         Optional[StandardScaler]       = None
        self._label_encoders: Dict[str, LabelEncoder]        = {}
        self._onehot_cols:    List[str]                      = []
        self._fitted_num_cols: List[str]                     = []
        self._fitted_cat_cols: List[str]                     = []
        self._fitted:         bool                           = False

        self._validate_config()
        logger.debug("DataPreprocessor initialised with config: %s", self._config_repr())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the preprocessor on *df* and return the cleaned DataFrame.

        This is the primary entry point for training data. The fitted scalers
        and encoders are stored internally for later use by :meth:`transform`.

        Pipeline order:
            1. Deep-copy the input (original is never mutated).
            2. Coerce data types.
            3. Drop duplicate rows.
            4. Drop rows that exceed the missing-value threshold.
            5. Impute remaining missing values.
            6. Fit & apply StandardScaler to numerical columns.
            7. Fit & apply categorical encoding.

        Args:
            df (pd.DataFrame): The raw DataFrame produced by DataLoader.

        Returns:
            pd.DataFrame: Cleaned and transformed DataFrame.

        Raises:
            TypeError:  If *df* is not a ``pd.DataFrame``.
            ValueError: If specified columns are absent from *df*.
        """
        self._validate_dataframe(df)
        result = df.copy(deep=True)

        # Auto-detect columns when not explicitly supplied
        self._fitted_num_cols = self._resolve_numerical_cols(result)
        self._fitted_cat_cols = self._resolve_categorical_cols(result)

        self._validate_columns_exist(result)

        logger.info("Starting fit_transform — input shape: %s", result.shape)

        result = self._coerce_dtypes(result)
        result = self._drop_duplicates(result)
        result = self._drop_high_missing_rows(result)
        result = self._impute_missing(result)

        if self.normalize:
            result = self._fit_scale(result)

        result = self._fit_encode(result)

        self._fitted = True
        logger.info("fit_transform complete — output shape: %s", result.shape)
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the *already fitted* transformations to new data.

        Use this method for validation/test splits to ensure the same
        scaling and encoding parameters learned on training data are applied.

        Args:
            df (pd.DataFrame): New raw DataFrame with the same schema.

        Returns:
            pd.DataFrame: Transformed DataFrame (no fitting occurs).

        Raises:
            RuntimeError: If called before :meth:`fit_transform`.
            TypeError:    If *df* is not a ``pd.DataFrame``.
        """
        if not self._fitted:
            raise RuntimeError(
                "DataPreprocessor has not been fitted yet. "
                "Call fit_transform() on training data first."
            )
        self._validate_dataframe(df)
        result = df.copy(deep=True)

        logger.info("Starting transform — input shape: %s", result.shape)

        result = self._coerce_dtypes(result)
        result = self._drop_duplicates(result)
        result = self._drop_high_missing_rows(result)
        result = self._impute_missing(result)

        if self.normalize and self._scaler is not None:
            result = self._apply_scale(result)

        result = self._apply_encode(result)

        logger.info("transform complete — output shape: %s", result.shape)
        return result

    def display_summary(
        self,
        raw_df:   pd.DataFrame,
        clean_df: pd.DataFrame,
    ) -> None:
        """
        Print a before/after summary comparing the raw and cleaned DataFrames.

        Shows:
            - Row and column counts before and after
            - Duplicates removed
            - Missing values before and after
            - Columns dropped or added (e.g. from one-hot encoding)
            - Normalised columns
            - Encoded columns

        Args:
            raw_df   (pd.DataFrame): The original DataFrame (before preprocessing).
            clean_df (pd.DataFrame): The cleaned DataFrame (after preprocessing).
        """
        sep = "─" * 65

        raw_missing   = int(raw_df.isna().sum().sum())
        clean_missing = int(clean_df.isna().sum().sum())
        duplicates    = int(raw_df.duplicated().sum())

        print(f"\n{sep}")
        print("  Preprocessing Summary")
        print(sep)

        # Shape comparison
        print(f"  {'Metric':<35} {'Before':>10} {'After':>10}")
        print(f"  {'------':<35} {'------':>10} {'-----':>10}")
        print(f"  {'Rows':<35} {raw_df.shape[0]:>10,} {clean_df.shape[0]:>10,}")
        print(f"  {'Columns':<35} {raw_df.shape[1]:>10,} {clean_df.shape[1]:>10,}")
        print(f"  {'Missing values':<35} {raw_missing:>10,} {clean_missing:>10,}")
        print(f"  {'Duplicate rows (raw)':<35} {duplicates:>10,} {'—':>10}")

        # Normalisation
        print(sep)
        if self.normalize and self._fitted_num_cols:
            print(f"  Normalised ({len(self._fitted_num_cols)} cols):")
            for col in self._fitted_num_cols:
                print(f"    • {col}")
        else:
            print("  Normalisation: disabled")

        # Encoding
        print(sep)
        if self._fitted_cat_cols:
            enc_label = self.cat_encoding
            print(f"  Encoding [{enc_label}] ({len(self._fitted_cat_cols)} cols):")
            for col in self._fitted_cat_cols:
                print(f"    • {col}")
        else:
            print("  Encoding: no categorical columns found")

        # One-hot expansion
        if self._onehot_cols:
            new_cols = [c for c in clean_df.columns if c not in raw_df.columns]
            print(sep)
            print(f"  One-hot expanded {len(self._onehot_cols)} col(s) → "
                  f"{len(new_cols)} new binary column(s)")

        print(f"{sep}\n")

        logger.info(
            "Preprocessing summary: rows %d→%d, cols %d→%d, "
            "missing %d→%d, duplicates removed: %d.",
            raw_df.shape[0], clean_df.shape[0],
            raw_df.shape[1], clean_df.shape[1],
            raw_missing, clean_missing, duplicates,
        )

    # ------------------------------------------------------------------
    # Step 1 — dtype coercion
    # ------------------------------------------------------------------

    def _coerce_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cast columns to their target dtypes as specified in :attr:`dtype_map`.

        Columns that cannot be coerced are left as-is and a warning is logged.

        Args:
            df (pd.DataFrame): DataFrame to coerce in place (working copy).

        Returns:
            pd.DataFrame: DataFrame with corrected dtypes.
        """
        if not self.dtype_map:
            logger.debug("dtype_map is empty — skipping type coercion.")
            return df

        for col, target_dtype in self.dtype_map.items():
            if col not in df.columns:
                logger.warning(
                    "dtype_map references column '%s' which is not in the DataFrame. Skipping.", col
                )
                continue
            try:
                if "datetime" in target_dtype:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                else:
                    df[col] = df[col].astype(target_dtype)
                logger.debug("Coerced column '%s' to dtype '%s'.", col, target_dtype)
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Could not coerce column '%s' to dtype '%s': %s. Skipping.",
                    col, target_dtype, exc,
                )

        return df

    # ------------------------------------------------------------------
    # Step 2 — deduplication
    # ------------------------------------------------------------------

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove exact duplicate rows from the DataFrame.

        Args:
            df (pd.DataFrame): Working copy of the DataFrame.

        Returns:
            pd.DataFrame: Deduplicated DataFrame with reset index.
        """
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        removed = before - len(df)

        if removed:
            logger.info("Dropped %d duplicate row(s).", removed)
        else:
            logger.debug("No duplicate rows found.")

        return df

    # ------------------------------------------------------------------
    # Step 3a — high-missing row dropping
    # ------------------------------------------------------------------

    def _drop_high_missing_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows where the fraction of NaN values exceeds :attr:`drop_missing_thresh`.

        A threshold of ``0.0`` disables this step entirely.

        Args:
            df (pd.DataFrame): Working copy of the DataFrame.

        Returns:
            pd.DataFrame: DataFrame with sparsely-populated rows removed.
        """
        if self.drop_missing_thresh <= 0.0:
            logger.debug("Row-missing threshold ≤ 0 — skipping row-drop step.")
            return df

        min_valid = int(df.shape[1] * (1.0 - self.drop_missing_thresh))
        before = len(df)
        df = df.dropna(thresh=min_valid).reset_index(drop=True)
        removed = before - len(df)

        if removed:
            logger.info(
                "Dropped %d row(s) with more than %.0f%% missing values.",
                removed, self.drop_missing_thresh * 100,
            )

        return df

    # ------------------------------------------------------------------
    # Step 3b — imputation
    # ------------------------------------------------------------------

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute remaining missing values in numerical and categorical columns.

        Numerical strategies:
            - ``"median"`` : fill with per-column median
            - ``"mean"``   : fill with per-column mean
            - ``"zero"``   : fill with 0
            - ``"drop"``   : drop rows that still contain NaN in numeric cols

        Categorical strategies:
            - ``"mode"``     : fill with most-frequent value (first mode)
            - ``"constant"`` : fill with :attr:`cat_fill_constant`
            - ``"drop"``     : drop rows that still contain NaN in categorical cols

        Args:
            df (pd.DataFrame): Working copy of the DataFrame.

        Returns:
            pd.DataFrame: DataFrame with missing values handled.
        """
        # --- numerical ---
        num_cols_present = [c for c in self._fitted_num_cols if c in df.columns]
        for col in num_cols_present:
            n_missing = int(df[col].isna().sum())
            if n_missing == 0:
                continue

            if self.num_fill_strategy == "median":
                fill_val = df[col].median()
                df[col] = df[col].fillna(fill_val)
            elif self.num_fill_strategy == "mean":
                fill_val = df[col].mean()
                df[col] = df[col].fillna(fill_val)
            elif self.num_fill_strategy == "zero":
                df[col] = df[col].fillna(0)
            elif self.num_fill_strategy == "drop":
                df = df.dropna(subset=[col]).reset_index(drop=True)

            logger.debug(
                "Imputed %d missing value(s) in numerical column '%s' (strategy=%s).",
                n_missing, col, self.num_fill_strategy,
            )

        # --- categorical ---
        cat_cols_present = [c for c in self._fitted_cat_cols if c in df.columns]
        for col in cat_cols_present:
            n_missing = int(df[col].isna().sum())
            if n_missing == 0:
                continue

            if self.cat_fill_strategy == "mode":
                mode_vals = df[col].mode()
                fill_val  = mode_vals.iloc[0] if not mode_vals.empty else self.cat_fill_constant
                df[col]   = df[col].fillna(fill_val)
            elif self.cat_fill_strategy == "constant":
                df[col] = df[col].fillna(self.cat_fill_constant)
            elif self.cat_fill_strategy == "drop":
                df = df.dropna(subset=[col]).reset_index(drop=True)

            logger.debug(
                "Imputed %d missing value(s) in categorical column '%s' (strategy=%s).",
                n_missing, col, self.cat_fill_strategy,
            )

        total_remaining = int(df.isna().sum().sum())
        logger.info("Imputation complete — %d missing value(s) remain.", total_remaining)
        return df

    # ------------------------------------------------------------------
    # Step 4 — normalisation (fit)
    # ------------------------------------------------------------------

    def _fit_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit a :class:`~sklearn.preprocessing.StandardScaler` on numerical columns
        and return the scaled DataFrame.

        The fitted scaler is stored in :attr:`_scaler` for later use by
        :meth:`_apply_scale`.

        Args:
            df (pd.DataFrame): Working copy of the DataFrame post-imputation.

        Returns:
            pd.DataFrame: DataFrame with numerical columns standardised (μ=0, σ=1).
        """
        cols = [c for c in self._fitted_num_cols if c in df.columns]
        if not cols:
            logger.warning("No numerical columns available for scaling.")
            return df

        self._scaler = StandardScaler()
        df[cols] = self._scaler.fit_transform(df[cols])
        logger.info("StandardScaler fitted and applied to %d column(s): %s.", len(cols), cols)
        return df

    def _apply_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the pre-fitted :attr:`_scaler` to numerical columns (no refitting).

        Args:
            df (pd.DataFrame): Working copy of the DataFrame.

        Returns:
            pd.DataFrame: Scaled DataFrame.
        """
        cols = [c for c in self._fitted_num_cols if c in df.columns]
        if not cols:
            return df
        df[cols] = self._scaler.transform(df[cols])
        logger.debug("Applied pre-fitted StandardScaler to %d column(s).", len(cols))
        return df

    # ------------------------------------------------------------------
    # Step 5 — categorical encoding (fit)
    # ------------------------------------------------------------------

    def _fit_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and apply the configured categorical encoding.

        - ``"label"``  : :class:`~sklearn.preprocessing.LabelEncoder` per column
                         (integer codes, suitable for tree-based models).
        - ``"onehot"`` : ``pd.get_dummies`` with ``drop_first=True``
                         (binary columns, suitable for linear models).
        - ``"none"``   : No encoding is applied.

        Args:
            df (pd.DataFrame): Working copy of the DataFrame.

        Returns:
            pd.DataFrame: DataFrame with categorical columns encoded.
        """
        cols = [c for c in self._fitted_cat_cols if c in df.columns]
        if not cols or self.cat_encoding == "none":
            logger.debug("Categorical encoding skipped (cols=%s, encoding=%s).", cols, self.cat_encoding)
            return df

        if self.cat_encoding == "label":
            for col in cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self._label_encoders[col] = le
                logger.debug("LabelEncoder fitted for column '%s' — classes: %s.", col, list(le.classes_))
            logger.info("Label encoding applied to %d column(s).", len(cols))

        elif self.cat_encoding == "onehot":
            self._onehot_cols = cols
            df = pd.get_dummies(df, columns=cols, drop_first=True, dtype=int)
            logger.info("One-hot encoding applied to %d column(s).", len(cols))

        return df

    def _apply_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply pre-fitted encoders to new data (no refitting).

        Unseen categories in label-encoded columns are mapped to ``-1``.
        For one-hot encoding, ``pd.get_dummies`` is re-applied and the
        column set is aligned to match the training schema.

        Args:
            df (pd.DataFrame): Working copy of the DataFrame.

        Returns:
            pd.DataFrame: Encoded DataFrame.
        """
        if self.cat_encoding == "label" and self._label_encoders:
            for col, le in self._label_encoders.items():
                if col not in df.columns:
                    continue
                # Map unseen categories to -1 gracefully
                df[col] = df[col].astype(str).map(
                    lambda x, le=le: le.transform([x])[0]   # noqa: E731
                    if x in le.classes_ else -1
                )

        elif self.cat_encoding == "onehot" and self._onehot_cols:
            existing = [c for c in self._onehot_cols if c in df.columns]
            df = pd.get_dummies(df, columns=existing, drop_first=True, dtype=int)

        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_numerical_cols(self, df: pd.DataFrame) -> List[str]:
        """
        Return the final list of numerical columns to use.

        If :attr:`numerical_cols` was provided by the caller, validate and
        return it. Otherwise, auto-detect all numeric-dtype columns.
        """
        if self.numerical_cols is not None:
            return list(self.numerical_cols)
        # Auto-detect: all numeric-dtype columns
        auto = df.select_dtypes(include=[np.number]).columns.tolist()
        logger.debug("Auto-detected %d numerical column(s): %s", len(auto), auto)
        return auto

    def _resolve_categorical_cols(self, df: pd.DataFrame) -> List[str]:
        """
        Return the final list of categorical columns to use.

        If :attr:`categorical_cols` was provided, validate and return it.
        Otherwise, auto-detect object and category dtype columns.
        """
        if self.categorical_cols is not None:
            return list(self.categorical_cols)
        # Auto-detect: object + category dtype columns
        auto = df.select_dtypes(include=["object", "category"]).columns.tolist()
        logger.debug("Auto-detected %d categorical column(s): %s", len(auto), auto)
        return auto

    def _validate_config(self) -> None:
        """Raise ValueError for any invalid constructor argument."""
        valid_num  = {"median", "mean", "zero", "drop"}
        valid_cat  = {"mode", "constant", "drop"}
        valid_enc  = {"label", "onehot", "none"}

        if self.num_fill_strategy not in valid_num:
            raise ValueError(
                f"Invalid num_fill_strategy '{self.num_fill_strategy}'. "
                f"Choose one of: {valid_num}"
            )
        if self.cat_fill_strategy not in valid_cat:
            raise ValueError(
                f"Invalid cat_fill_strategy '{self.cat_fill_strategy}'. "
                f"Choose one of: {valid_cat}"
            )
        if self.cat_encoding not in valid_enc:
            raise ValueError(
                f"Invalid cat_encoding '{self.cat_encoding}'. "
                f"Choose one of: {valid_enc}"
            )
        if not (0.0 <= self.drop_missing_thresh <= 1.0):
            raise ValueError(
                f"drop_missing_thresh must be in [0.0, 1.0], got {self.drop_missing_thresh}."
            )

    def _validate_dataframe(self, df: object) -> None:
        """Raise TypeError if *df* is not a pandas DataFrame."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected a pd.DataFrame, got {type(df).__name__}."
            )

    def _validate_columns_exist(self, df: pd.DataFrame) -> None:
        """Raise ValueError for any explicitly listed column absent from *df*."""
        all_explicit = []
        if self.numerical_cols is not None:
            all_explicit.extend(self.numerical_cols)
        if self.categorical_cols is not None:
            all_explicit.extend(self.categorical_cols)
        missing = [c for c in all_explicit if c not in df.columns]
        if missing:
            raise ValueError(
                f"The following specified columns are not in the DataFrame: {missing}"
            )

    def _config_repr(self) -> str:
        """Return a compact config string for debug logging."""
        return (
            f"num_strategy={self.num_fill_strategy}, "
            f"cat_strategy={self.cat_fill_strategy}, "
            f"encoding={self.cat_encoding}, "
            f"normalize={self.normalize}, "
            f"thresh={self.drop_missing_thresh}"
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted_str = "fitted" if self._fitted else "not fitted"
        return (
            f"DataPreprocessor("
            f"num_strategy='{self.num_fill_strategy}', "
            f"cat_strategy='{self.cat_fill_strategy}', "
            f"encoding='{self.cat_encoding}', "
            f"normalize={self.normalize}, "
            f"status='{fitted_str}')"
        )

    def __str__(self) -> str:
        fitted_str = "fitted" if self._fitted else "not fitted"
        return (
            f"DataPreprocessor [{fitted_str}] — "
            f"num:{self.num_fill_strategy} / "
            f"cat:{self.cat_fill_strategy} / "
            f"enc:{self.cat_encoding} / "
            f"normalize:{self.normalize}"
        )
