"""
src/evaluation/__init__.py
==========================
Exposes the public interface of the evaluation sub-package.

Exports:
    EDAAnalyser     — full exploratory data analysis suite:
                      overview, statistics, missing-value analysis,
                      distributions, box plots, correlation heatmap,
                      target distribution, and plot persistence.

    ModelEvaluator  — model-agnostic evaluation harness:
                      confusion matrix, ROC/PR curves, metrics bar,
                      JSON + TXT report persistence, cross-validation.
"""

from .eda import EDAAnalyser
from .model_evaluation import ModelEvaluator

__all__ = ["EDAAnalyser", "ModelEvaluator"]
