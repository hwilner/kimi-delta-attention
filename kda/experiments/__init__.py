"""Pre-registered experiment runners (Phase 4)."""

from .calibration import (
    CalibrationConfig,
    CalibrationSamples,
    SIGNAL_SPECS,
    average_precision,
    binary_auroc,
    binned_recall,
    bootstrap_ci,
    calibration_curve,
    collect_samples_for_batch,
    collect_variant_samples,
    expected_calibration_error,
    run_calibration_experiment,
)

__all__ = [
    "CalibrationConfig",
    "CalibrationSamples",
    "SIGNAL_SPECS",
    "average_precision",
    "binary_auroc",
    "binned_recall",
    "bootstrap_ci",
    "calibration_curve",
    "collect_samples_for_batch",
    "collect_variant_samples",
    "expected_calibration_error",
    "run_calibration_experiment",
]
