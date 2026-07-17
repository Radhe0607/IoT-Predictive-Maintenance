"""
src/models/__init__.py
======================
Exposes the public interface of the models sub-package.

Exports:
    BaselineModel — Random Forest classifier with full evaluation suite,
                    joblib model persistence, and evaluation plot generation.
"""

from .baseline_model import BaselineModel

__all__ = ["BaselineModel"]
