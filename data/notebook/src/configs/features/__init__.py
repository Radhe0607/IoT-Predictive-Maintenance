"""
src/features/__init__.py
========================
Exposes the public interface of the feature engineering sub-package.

Exports:
    FeatureEngineer — derives rolling stats, lag, delta, interaction,
                      and time-based features; performs feature selection;
                      and persists the enriched dataset.
"""

from .feature_engineering import FeatureEngineer

__all__ = ["FeatureEngineer"]
