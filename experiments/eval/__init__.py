"""Evaluation modules for Step 2 subspace diagnostics."""

from .subspace_metrics import (
    SubspaceMetrics,
    compute_factor_subspace,
    subspace_explained_variance,
    subspace_effective_rank,
    subspace_coordinate_sparsity,
    subspace_overlap,
    subspace_stability,
)

from .inv_eq_metrics import (
    InvEqMetricsCollector,
    compute_inv_eq_metrics,
    compute_rank_metrics,
    invariance_error,
    change_energy,
)

from .group_probes import (
    LinearProbe,
    MultiFactorActionProbe,
    CrossLeakageProbes,
)

from .viz import (
    plot_factor_sweep_trajectory,
    plot_subspace_overlap_heatmap,
    plot_eigenvalue_spectrum,
    plot_rank_monitor,
    log_visualizations_to_wandb,
    close_all_figures,
)

__all__ = [
    # Subspace metrics
    "SubspaceMetrics",
    "compute_factor_subspace",
    "subspace_explained_variance",
    "subspace_effective_rank",
    "subspace_coordinate_sparsity",
    "subspace_overlap",
    "subspace_stability",
    # Inv/Eq metrics
    "InvEqMetricsCollector",
    "compute_inv_eq_metrics",
    "compute_rank_metrics",
    "invariance_error",
    "change_energy",
    # Probes
    "LinearProbe",
    "MultiFactorActionProbe",
    "CrossLeakageProbes",
    # Visualizations
    "plot_factor_sweep_trajectory",
    "plot_subspace_overlap_heatmap",
    "plot_eigenvalue_spectrum",
    "plot_rank_monitor",
    "log_visualizations_to_wandb",
    "close_all_figures",
]
