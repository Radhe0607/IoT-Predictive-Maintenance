"""
data_loader.py
==============
Modular dataset loading component for the IoT Predictive Maintenance project.

This module provides a reusable DataLoader class that handles:
    - CSV dataset loading via pandas
    - File existence validation
    - Dataset shape and column inspection
    - Graceful error handling

Part of the Infotact Solutions Data Science & Machine Learning Internship project.

Note:
    Preprocessing logic is intentionally excluded from this module.
    This module is solely responsible for raw data ingestion and basic inspection.
"""

import logging
from pathlib import Path
from typing import Optional, List

import pandas as pd

# ---------------------------------------------------------------------------
# Centralised configuration
# ---------------------------------------------------------------------------
try:
    from src.configs.config import get_config as _get_config
    _cfg = _get_config()
    _DEFAULT_ENCODING  = _cfg.data.encoding
    _DEFAULT_SEPARATOR = _cfg.data.separator
except Exception:   # fallback when running module in isolation
    _DEFAULT_ENCODING  = "utf-8"
    _DEFAULT_SEPARATOR = ","

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DataLoader class
# ---------------------------------------------------------------------------


class DataLoader:
    """
    A reusable component for loading CSV datasets into pandas DataFrames.

    This class encapsulates all raw data ingestion logic for the IoT
    Predictive Maintenance pipeline. It validates the target file,
    loads the data, and surfaces structural metadata (shape, dtypes,
    column names) without applying any transformations.

    Attributes:
        file_path (Path): Resolved absolute path to the CSV file.
        encoding  (str):  Character encoding used when reading the file.
        separator (str):  Column delimiter used in the CSV file.
        _df       (Optional[pd.DataFrame]): Cached DataFrame after loading.

    Example::

        loader = DataLoader("data/raw/sensor_readings.csv")
        df = loader.load()
        loader.display_info()
    """

    def __init__(
        self,
        file_path: str | Path,
        encoding:  str = _DEFAULT_ENCODING,
        separator: str = _DEFAULT_SEPARATOR,
    ) -> None:
        """
        Initialise the DataLoader.

        Args:
            file_path (str | Path): Path to the CSV file to load.
            encoding  (str):        File encoding. Defaults to ``"utf-8"``.
            separator (str):        CSV delimiter character. Defaults to ``","``.
        """
        self.file_path: Path = Path(file_path).resolve()
        self.encoding: str = encoding
        self.separator: str = separator
        self._df: Optional[pd.DataFrame] = None

        logger.debug("DataLoader initialised for: %s", self.file_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """
        Validate the target file and load it into a pandas DataFrame.

        Returns:
            pd.DataFrame: The loaded dataset.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError:        If the file is not a CSV (wrong extension).
            RuntimeError:      If pandas fails to parse the file.
        """
        self._validate_file()

        logger.info("Loading dataset from: %s", self.file_path)

        try:
            self._df = pd.read_csv(
                self.file_path,
                encoding=self.encoding,
                sep=self.separator,
            )
        except pd.errors.EmptyDataError as exc:
            raise RuntimeError(
                f"The file '{self.file_path}' is empty and cannot be loaded."
            ) from exc
        except pd.errors.ParserError as exc:
            raise RuntimeError(
                f"Failed to parse '{self.file_path}'. "
                "Ensure it is a valid CSV with the correct separator."
            ) from exc
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                f"Encoding error while reading '{self.file_path}'. "
                f"Try a different encoding (current: '{self.encoding}')."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected error while loading '{self.file_path}': {exc}"
            ) from exc

        logger.info(
            "Dataset loaded successfully — shape: %s", self._df.shape
        )
        return self._df

    def display_info(self) -> None:
        """
        Print a structured summary of the loaded dataset to stdout/log.

        Displays:
            - File path
            - Number of rows and columns
            - Column names with their dtypes
            - Missing-value counts per column

        Raises:
            RuntimeError: If :meth:`load` has not been called first.
        """
        self._require_loaded()

        df = self._df
        separator_line = "─" * 60

        print(f"\n{separator_line}")
        print(f"  Dataset Info — {self.file_path.name}")
        print(separator_line)
        print(f"  File path   : {self.file_path}")
        print(f"  Rows        : {df.shape[0]:,}")
        print(f"  Columns     : {df.shape[1]:,}")
        print(separator_line)
        print(f"  {'Column':<30} {'Dtype':<15} {'Missing':>8}")
        print(f"  {'------':<30} {'-----':<15} {'-------':>8}")

        for col in df.columns:
            missing = int(df[col].isna().sum())
            print(
                f"  {col:<30} {str(df[col].dtype):<15} {missing:>8,}"
            )

        print(separator_line)
        total_missing = int(df.isna().sum().sum())
        total_cells = df.shape[0] * df.shape[1]
        missing_pct = (total_missing / total_cells * 100) if total_cells else 0
        print(f"  Total missing values: {total_missing:,} ({missing_pct:.2f}%)")
        print(f"{separator_line}\n")

        logger.info(
            "Dataset summary displayed — %d rows, %d columns, "
            "%d total missing values (%.2f%%).",
            df.shape[0],
            df.shape[1],
            total_missing,
            missing_pct,
        )

    @property
    def dataframe(self) -> pd.DataFrame:
        """
        Return the cached DataFrame.

        Returns:
            pd.DataFrame: The loaded dataset.

        Raises:
            RuntimeError: If :meth:`load` has not been called yet.
        """
        self._require_loaded()
        return self._df

    @property
    def columns(self) -> List[str]:
        """
        Return the list of column names in the dataset.

        Returns:
            List[str]: Column name strings.

        Raises:
            RuntimeError: If :meth:`load` has not been called yet.
        """
        self._require_loaded()
        return list(self._df.columns)

    @property
    def shape(self) -> tuple[int, int]:
        """
        Return the (rows, columns) shape of the dataset.

        Returns:
            tuple[int, int]: A ``(num_rows, num_columns)`` tuple.

        Raises:
            RuntimeError: If :meth:`load` has not been called yet.
        """
        self._require_loaded()
        return self._df.shape

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_file(self) -> None:
        """
        Verify that the target path points to a readable CSV file.

        Raises:
            FileNotFoundError: If the path does not exist on disk.
            ValueError:        If the path exists but is not a ``.csv`` file.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: '{self.file_path}'. "
                "Please verify the path and try again."
            )

        if not self.file_path.is_file():
            raise FileNotFoundError(
                f"Expected a file but found a directory: '{self.file_path}'."
            )

        if self.file_path.suffix.lower() != ".csv":
            raise ValueError(
                f"Expected a .csv file, but got '{self.file_path.suffix}' "
                f"for path: '{self.file_path}'."
            )

        logger.debug("File validation passed: %s", self.file_path)

    def _require_loaded(self) -> None:
        """
        Raise an error if the dataset has not been loaded yet.

        Raises:
            RuntimeError: If :attr:`_df` is ``None``.
        """
        if self._df is None:
            raise RuntimeError(
                "No dataset has been loaded. "
                "Call DataLoader.load() before accessing this property."
            )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "loaded" if self._df is not None else "not loaded"
        return (
            f"DataLoader("
            f"file_path='{self.file_path}', "
            f"encoding='{self.encoding}', "
            f"separator='{self.separator}', "
            f"status='{status}')"
        )

    def __str__(self) -> str:
        if self._df is not None:
            return (
                f"DataLoader → '{self.file_path.name}' "
                f"[{self._df.shape[0]} rows × {self._df.shape[1]} cols]"
            )
        return f"DataLoader → '{self.file_path.name}' [not loaded]"
