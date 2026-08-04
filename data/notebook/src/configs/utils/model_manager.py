"""
model_manager.py
================
Reusable model persistence utility for the IoT Predictive Maintenance project.

This module provides two stand-alone functions — ``save_model`` and
``load_model`` — that act as a single, consistent interface for serialising
and deserialising any scikit-learn-compatible estimator (or pipeline) to/from
disk using ``joblib``.

Design goals:
    - **Reusability**: The functions are model-agnostic; they accept any object
      that joblib can serialise (sklearn estimator, Pipeline, custom class …).
    - **Safety**: ``save_model`` creates the destination directory tree on
      demand and writes to a temporary file first, renaming atomically on
      success.  ``load_model`` raises an informative ``FileNotFoundError``
      rather than letting joblib's opaque IOError bubble up.
    - **Traceability**: Every save / load operation is logged through the
      standard Python ``logging`` framework so it integrates with the rest of
      the pipeline's log stream without any additional setup.
    - **Configurability**: Both functions accept an explicit ``file_path``
      argument, so callers are never forced to rely on hard-coded paths.  The
      module-level constant ``DEFAULT_MODELS_DIR`` provides a sensible default
      that mirrors the project's ``config.yaml`` setting.

Typical usage::

    from src.configs.utils.model_manager import save_model, load_model

    # --- After training -------------------------------------------------------
    saved_path = save_model(fitted_rf, "outputs/models/random_forest_baseline.joblib")

    # --- At inference time ----------------------------------------------------
    model = load_model("outputs/models/random_forest_baseline.joblib")
    predictions = model.predict(X_test)

Integration with BaselineModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``BaselineModel.save_model()`` (in ``baseline_model.py``) calls this utility
internally instead of calling ``joblib.dump`` directly.  This ensures that all
model-persistence logic lives in one place and benefits from the safety
wrappers (temp-file write, directory auto-creation, structured logging).

Part of the Infotact Solutions Data Science & Machine Learning Internship project.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Union

import joblib

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default output directory (mirrors config.yaml → paths.models_dir)
# ---------------------------------------------------------------------------
# Fall back gracefully when the config layer is not importable (e.g. during
# isolated unit tests or when running the module standalone).
try:
    from src.configs.config import get_config as _get_cfg

    _cfg = _get_cfg()
    DEFAULT_MODELS_DIR: str = _cfg.paths.models_dir   # e.g. "outputs/models"
except (ImportError, AttributeError, FileNotFoundError) as exc:
    logger.debug("Config singleton unavailable, using defaults: %s", exc)
    DEFAULT_MODELS_DIR = "outputs/models"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_model(
    model: Any,
    file_path: Union[str, Path],
    *,
    compress: int = 3,
    overwrite: bool = True,
) -> Path:
    """
    Serialise *model* to *file_path* using :func:`joblib.dump`.

    The function:
        1. Resolves *file_path* to an absolute ``pathlib.Path``.
        2. Creates any missing parent directories automatically.
        3. Writes the model to a sibling temporary file first.
        4. Renames (atomically on POSIX; best-effort on Windows) the
           temporary file to the final path.
        5. Logs the saved file size.

    Args:
        model (Any):
            The trained model object to persist.  Typically a fitted
            scikit-learn estimator or Pipeline, but any joblib-serialisable
            object is accepted.
        file_path (str | Path):
            Destination file path for the serialised model.
            Use a ``.joblib`` extension by convention (e.g.
            ``"outputs/models/random_forest_baseline.joblib"``).
        compress (int):
            joblib compression level (0 = no compression, 9 = maximum).
            Defaults to ``3`` — a reasonable balance between size and speed.
        overwrite (bool):
            If ``False`` and *file_path* already exists, raises
            ``FileExistsError`` instead of overwriting.
            Defaults to ``True``.

    Returns:
        Path: The resolved absolute path of the saved ``.joblib`` file.

    Raises:
        FileExistsError:
            If *overwrite* is ``False`` and the target file already exists.
        OSError:
            If the filesystem operation (directory creation / file write /
            rename) fails.

    Example::

        from src.configs.utils.model_manager import save_model

        path = save_model(fitted_model, "outputs/models/rf_v1.joblib")
        print(f"Model persisted to: {path}")
    """
    # ── 1. Resolve and validate the destination path ──────────────────────
    dest = Path(file_path).resolve()

    if not overwrite and dest.exists():
        raise FileExistsError(
            f"Model file already exists and overwrite=False: '{dest}'. "
            "Set overwrite=True or choose a different file_path."
        )

    # ── 2. Ensure the destination directory exists ────────────────────────
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("Output directory ensured: '%s'.", dest.parent)

    # ── 3. Write to a temporary file in the same directory ────────────────
    #       Using the same directory as the destination ensures that
    #       os.replace() is an atomic rename on POSIX (same filesystem).
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        dir=dest.parent,
        prefix=f".{dest.stem}_tmp_",
        suffix=".joblib",
    )
    tmp_path = Path(tmp_path_str)

    try:
        os.close(tmp_fd)                       # joblib opens the path itself
        joblib.dump(model, tmp_path, compress=compress)

        # ── 4. Atomically rename to the final destination ─────────────────
        os.replace(tmp_path, dest)             # replaces dest if it exists

    except Exception:
        # Clean up the orphaned temp file on failure
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    # ── 5. Log the outcome ────────────────────────────────────────────────
    size_kb = dest.stat().st_size / 1024
    logger.info(
        "Model saved → '%s'  [%.1f KB, compress=%d, type=%s].",
        dest, size_kb, compress, type(model).__name__,
    )
    return dest


def load_model(file_path: Union[str, Path]) -> Any:
    """
    Deserialise a joblib model artifact from *file_path*.

    Args:
        file_path (str | Path):
            Absolute or relative path to the ``.joblib`` file produced by
            :func:`save_model` (or by ``joblib.dump`` directly).

    Returns:
        Any: The deserialised model object (typically a fitted sklearn
             estimator or Pipeline).

    Raises:
        FileNotFoundError:
            If *file_path* does not exist on disk.  The error message
            includes the resolved absolute path and a hint to run the
            training pipeline first.
        joblib.externals.loky.process_executor.TerminatedWorkerError:
            Re-raised as-is if joblib's internal worker crashes during
            deserialisation (rare; usually caused by a corrupt file).

    Example::

        from src.configs.utils.model_manager import load_model

        model = load_model("outputs/models/random_forest_baseline.joblib")
        predictions = model.predict(X_new)
    """
    # ── 1. Resolve the path and check existence ───────────────────────────
    src = Path(file_path).resolve()

    if not src.exists():
        raise FileNotFoundError(
            f"Model file not found: '{src}'.\n"
            "Ensure the training pipeline has been run and the model has "
            f"been saved to the expected location.  Expected directory: "
            f"'{src.parent}'."
        )

    # ── 2. Deserialise with joblib ────────────────────────────────────────
    logger.debug("Loading model from '%s' …", src)
    model = joblib.load(src)

    # ── 3. Log the outcome ────────────────────────────────────────────────
    size_kb = src.stat().st_size / 1024
    logger.info(
        "Model loaded ← '%s'  [%.1f KB, type=%s].",
        src, size_kb, type(model).__name__,
    )
    return model
