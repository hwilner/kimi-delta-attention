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
from .precision_gated_ablation import (
    PrecisionGatedAblationConfig,
    ablation_cells,
    is_default_cell,
    run_precision_gated_ablation,
)
from .report_consistency import (
    check_report_consistency,
    default_ablation_config,
    default_calibration_robustness_config,
    default_routing_robustness_config,
    parse_pipe_tables,
)
from .report_experiments import (
    CapacityConfig,
    OverwriteConfig,
    edit_locality_metrics,
    run_capacity_experiment,
    run_overwrite_experiment,
)
from .rls_lite_ablation import (
    RLSLiteAblationConfig,
    is_rls_lite_default_cell,
    rls_lite_cells,
    run_rls_lite_ablation,
)
from .routing import (
    RoutingExperimentConfig,
    RoutingVariantData,
    collect_routing_data,
    evaluate_operating_point,
    run_routing_experiment,
)
from .routing_robustness import (
    RoutingRobustnessConfig,
    routing_regime_cells,
    run_routing_robustness,
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
    "PrecisionGatedAblationConfig",
    "ablation_cells",
    "is_default_cell",
    "run_precision_gated_ablation",
    "RLSLiteAblationConfig",
    "is_rls_lite_default_cell",
    "rls_lite_cells",
    "run_rls_lite_ablation",
    "RoutingExperimentConfig",
    "RoutingVariantData",
    "collect_routing_data",
    "evaluate_operating_point",
    "run_routing_experiment",
    "RoutingRobustnessConfig",
    "routing_regime_cells",
    "run_routing_robustness",
    "check_report_consistency",
    "default_ablation_config",
    "default_calibration_robustness_config",
    "default_routing_robustness_config",
    "parse_pipe_tables",
    "CapacityConfig",
    "OverwriteConfig",
    "edit_locality_metrics",
    "run_capacity_experiment",
    "run_overwrite_experiment",
]
