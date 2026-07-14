"""
src/data/__init__.py
====================
Exposes the public interface of the data ingestion sub-package.

Exports:
    DataLoader      — raw CSV ingestion and inspection.
    DataPreprocessor — cleaning, normalisation and encoding pipeline.
"""

from .data_loader import DataLoader
from .preprocessing import DataPreprocessor

__all__ = ["DataLoader", "DataPreprocessor"]
