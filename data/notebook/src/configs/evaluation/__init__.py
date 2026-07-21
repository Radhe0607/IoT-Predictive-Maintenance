"""
src/evaluation/__init__.py
==========================
Exposes the public interface of the evaluation sub-package.

Exports:
    EDAAnalyser        — full exploratory data analysis suite:
                         overview, statistics, missing-value analysis,
                         distributions, box plots, correlation heatmap,
                         target distribution, and plot persistence.

    ModelEvaluator     — model-agnostic evaluation harness:
                         confusion matrix, ROC/PR curves, metrics bar,
                         JSON + TXT report persistence, cross-validation.

    EvaluationPipeline — reusable end-to-end evaluation orchestrator:
                         load model → load test data → generate predictions
                         → compute metrics (Accuracy, Precision, Recall,
                         F1-Score) → print report → save to outputs/reports/.

    run_evaluation     — convenience function that creates an
                         EvaluationPipeline and runs the full pipeline
                         in a single call, using config-driven defaults.
"""

from .eda import EDAAnalyser
from .model_evaluation import ModelEvaluator
from .evaluation_pipeline import EvaluationPipeline, run_evaluation

__all__ = [
    "EDAAnalyser",
    "ModelEvaluator",
    "EvaluationPipeline",
    "run_evaluation",
]
