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
from .calibration_robustness import (
    CalibrationRobustnessConfig,
    regime_cells,
    run_calibration_robustness,
)
from .report_experiments import (
    CapacityConfig,
    OverwriteConfig,
    edit_locality_metrics,
    run_capacity_experiment,
    run_overwrite_experiment,
)
from .routing import (
    RoutingExperimentConfig,
    RoutingVariantData,
    collect_routing_data,
    evaluate_operating_point,
    run_routing_experiment,
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
    "CalibrationRobustnessConfig",
    "regime_cells",
    "run_calibration_robustness",
    "RoutingExperimentConfig",
    "RoutingVariantData",
    "collect_routing_data",
    "evaluate_operating_point",
    "run_routing_experiment",
    "CapacityConfig",
    "OverwriteConfig",
    "edit_locality_metrics",
    "run_capacity_experiment",
    "run_overwrite_experiment",
]
