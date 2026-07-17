"""
src/analysis/__init__.py
========================
Exposes the public interface of the EDA sub-package.

Exports:
    EDAAnalyser — full exploratory data analysis suite:
                  overview, statistics, missing-value analysis,
                  distributions, box plots, correlation heatmap,
                  target distribution, and plot persistence.
"""

from .eda import EDAAnalyser

__all__ = ["EDAAnalyser"]
