"""
src/configs/reports/__init__.py
================================
Public interface for the IoT Predictive Maintenance reporting sub-package.

This package provides a reusable, modular reporting module for generating
structured prediction and evaluation reports after model inference.

Usage::

    # Standalone convenience function (recommended for notebooks / scripts)
    from src.configs.reports import generate_report

    report = generate_report(inference_result, save=True)

    # Class-based usage
    from src.configs.reports import ReportGenerator, PredictiveMaintenanceReport

    generator = ReportGenerator(reports_dir="outputs/reports")
    report = generator.generate_report(inference_result, y_true=y_test)
    report.save_both("custom_report_name")

Exported symbols:
    ReportGenerator               — Main report generation engine. Accepts inference
                                    results, ground truth, or metrics; calculates
                                    summaries & evaluation statistics.
    PredictiveMaintenanceReport   — Container object holding report data with
                                    methods to export to JSON and TXT.
    generate_report               — Convenience function for one-liner report
                                    generation and export.
"""

from .report_generator import (
    PredictiveMaintenanceReport,
    ReportGenerator,
    generate_report,
)

__all__ = [
    "PredictiveMaintenanceReport",
    "ReportGenerator",
    "generate_report",
]
