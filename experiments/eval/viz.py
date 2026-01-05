"""
W&B visualizations for subspace diagnostics.

Generates:
- Factor sweep trajectories (2D projection onto factor subspace)
- Subspace overlap heatmaps
- Eigenvalue spectra per factor
- SVD rank monitors
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def _check_matplotlib():
    """Ensure matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualizations. "
            "Install with: pip install matplotlib"
        )


def plot_factor_sweep_trajectory(
    delta_z: torch.Tensor,
    factor_name: str,
    num_anchors: int = 8,
    figsize: Tuple[int, int] = (6, 6),
) -> "plt.Figure":
    """
    Plot factor sweep trajectories projected onto top-2 PCs.

    For rotation, this should trace a circle/loop.
    For translation, this should trace lines.

    Args:
        delta_z: Factor-induced changes, shape (N, d).
        factor_name: Name of the factor for labeling.
        num_anchors: Number of anchor images to show trajectories for.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    _check_matplotlib()

    # Center and compute PCA
    delta_z = delta_z - delta_z.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(delta_z, full_matrices=False)
    basis = Vh[:2].T  # (d, 2)

    # Project to 2D
    proj = (delta_z @ basis).numpy()  # (N, 2)

    fig, ax = plt.subplots(figsize=figsize)

    # Sample and plot trajectories
    N = proj.shape[0]
    samples_per_anchor = N // num_anchors if num_anchors > 0 else N

    colors = plt.cm.tab10(np.linspace(0, 1, min(num_anchors, 10)))

    for i in range(min(num_anchors, N // max(1, samples_per_anchor))):
        start = i * samples_per_anchor
        end = min(start + samples_per_anchor, N)
        traj = proj[start:end]

        color = colors[i % len(colors)]
        ax.plot(traj[:, 0], traj[:, 1], "-", color=color, alpha=0.7, linewidth=1.5)
        ax.scatter(
            traj[0, 0], traj[0, 1], color=color, s=50, marker="o", zorder=5
        )  # Start
        ax.scatter(
            traj[-1, 0], traj[-1, 1], color=color, s=50, marker="s", zorder=5
        )  # End

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{factor_name.capitalize()} Sweep Trajectories")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_subspace_overlap_heatmap(
    overlaps: Dict[str, float],
    factor_names: List[str],
    figsize: Tuple[int, int] = (6, 5),
) -> "plt.Figure":
    """
    Plot heatmap of subspace overlaps between factors.

    Args:
        overlaps: Dict mapping "f1_f2" to overlap value.
        factor_names: List of factor names.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    _check_matplotlib()

    n = len(factor_names)
    matrix = np.eye(n)  # Diagonal is 1 (self-overlap)

    # Fill in off-diagonal
    for i, f1 in enumerate(factor_names):
        for j, f2 in enumerate(factor_names):
            if i < j:
                key = f"subspace_overlap_{f1}_{f2}"
                alt_key = f"subspace_overlap_{f2}_{f1}"
                val = overlaps.get(key, overlaps.get(alt_key, 0.0))
                matrix[i, j] = val
                matrix[j, i] = val

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(factor_names, rotation=45, ha="right")
    ax.set_yticklabels(factor_names)

    # Add text annotations
    for i in range(n):
        for j in range(n):
            text = ax.text(
                j, i, f"{matrix[i, j]:.2f}",
                ha="center", va="center", fontsize=10,
                color="white" if matrix[i, j] > 0.5 else "black"
            )

    ax.set_title("Subspace Overlap (lower is better)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    return fig


def plot_eigenvalue_spectrum(
    delta_z: torch.Tensor,
    factor_name: str,
    top_k: int = 32,
    expected_dim: int = 2,
    figsize: Tuple[int, int] = (8, 4),
) -> "plt.Figure":
    """
    Plot eigenvalue spectrum for a factor.

    Args:
        delta_z: Factor-induced changes, shape (N, d).
        factor_name: Name of the factor.
        top_k: Number of eigenvalues to show.
        expected_dim: Expected intrinsic dimension (for marking).
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    _check_matplotlib()

    delta_z = delta_z - delta_z.mean(dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(delta_z, full_matrices=False)

    # Eigenvalues of covariance are proportional to S^2
    eigvals = (S**2).numpy()[:top_k]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Linear scale
    ax1.bar(range(len(eigvals)), eigvals, color="steelblue", alpha=0.8)
    ax1.axvline(expected_dim - 0.5, color="red", linestyle="--", label=f"Expected dim={expected_dim}")
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Eigenvalue")
    ax1.set_title(f"{factor_name.capitalize()} Eigenspectrum")
    ax1.legend()

    # Log scale
    ax2.bar(range(len(eigvals)), eigvals, color="steelblue", alpha=0.8)
    ax2.axvline(expected_dim - 0.5, color="red", linestyle="--")
    ax2.set_yscale("log")
    ax2.set_xlabel("Component")
    ax2.set_ylabel("Eigenvalue (log)")
    ax2.set_title(f"{factor_name.capitalize()} Eigenspectrum (log)")

    fig.tight_layout()
    return fig


def plot_rank_monitor(
    z_eq: torch.Tensor,
    z_inv: torch.Tensor,
    figsize: Tuple[int, int] = (10, 4),
) -> "plt.Figure":
    """
    Plot SVD spectra for collapse monitoring.

    Args:
        z_eq: Equivariant representations, shape (N, d_eq).
        z_inv: Invariant representations, shape (N, d_inv).
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    _check_matplotlib()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for ax, z, name in [(ax1, z_eq, "z_eq"), (ax2, z_inv, "z_inv")]:
        z_centered = z - z.mean(dim=0, keepdim=True)
        _, S, _ = torch.linalg.svd(z_centered, full_matrices=False)
        S = S.numpy()

        # Normalize
        S_norm = S / (S.sum() + 1e-8)

        ax.bar(range(min(50, len(S))), S_norm[:50], color="steelblue", alpha=0.8)
        ax.set_xlabel("Component")
        ax.set_ylabel("Normalized Singular Value")
        ax.set_title(f"{name} SVD Spectrum")

        # Add effective rank annotation
        sum_s = S.sum()
        sum_s_sq = (S**2).sum()
        eff_rank = (sum_s**2) / (sum_s_sq + 1e-8)
        ax.annotate(
            f"Eff. rank: {eff_rank:.1f}",
            xy=(0.95, 0.95), xycoords="axes fraction",
            ha="right", va="top",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        )

    fig.tight_layout()
    return fig


def log_visualizations_to_wandb(
    metrics: Dict[str, float],
    delta_z_by_factor: Optional[Dict[str, torch.Tensor]] = None,
    z_eq: Optional[torch.Tensor] = None,
    z_inv: Optional[torch.Tensor] = None,
    factor_dims: Optional[Dict[str, int]] = None,
    step: Optional[int] = None,
    prefix: str = "viz",
):
    """
    Log all visualizations to W&B.

    Args:
        metrics: Dict of subspace metrics (for overlap heatmap).
        delta_z_by_factor: Dict mapping factor name to delta-z tensor.
        z_eq: Equivariant representations for rank monitor.
        z_inv: Invariant representations for rank monitor.
        factor_dims: Expected dimensions per factor.
        step: W&B step.
        prefix: Prefix for logged keys.
    """
    if not HAS_WANDB or not HAS_MATPLOTLIB:
        return

    factor_dims = factor_dims or {"rot": 2, "trans_x": 1, "trans_y": 1, "scale": 1}
    log_dict = {}

    # Factor sweep trajectories
    if delta_z_by_factor:
        # Individual factor plots
        for factor, delta_z in delta_z_by_factor.items():
            if delta_z is not None and delta_z.shape[0] >= 8:
                try:
                    fig = plot_factor_sweep_trajectory(delta_z, factor)
                    log_dict[f"{prefix}/sweep_traj_{factor}"] = wandb.Image(fig)
                    plt.close(fig)
                except Exception:
                    pass
        
        # Combined factor trajectory plot
        try:
            fig = plot_combined_factor_trajectories(delta_z_by_factor)
            log_dict[f"{prefix}/combined_trajectories"] = wandb.Image(fig)
            plt.close(fig)
        except Exception:
            pass
        
        # Special rotation circle plot if rotation is available
        if "rot" in delta_z_by_factor and delta_z_by_factor["rot"] is not None:
            try:
                fig = plot_rotation_circle(delta_z_by_factor["rot"])
                log_dict[f"{prefix}/rotation_circle"] = wandb.Image(fig)
                plt.close(fig)
            except Exception:
                pass

        # Eigenvalue spectra
        for factor, delta_z in delta_z_by_factor.items():
            if delta_z is not None and delta_z.shape[0] >= 8:
                try:
                    expected = factor_dims.get(factor, 2)
                    fig = plot_eigenvalue_spectrum(delta_z, factor, expected_dim=expected)
                    log_dict[f"{prefix}/eigs_{factor}"] = wandb.Image(fig)
                    plt.close(fig)
                except Exception:
                    pass

    # Subspace overlap heatmap
    overlap_keys = [k for k in metrics.keys() if k.startswith("subspace_overlap_")]
    if overlap_keys:
        # Extract factor names from metrics
        factors = set()
        for k in overlap_keys:
            parts = k.replace("subspace_overlap_", "").split("_")
            if len(parts) >= 2:
                factors.add(parts[0])
                factors.add(parts[1])
        factors = sorted(factors)

        if len(factors) >= 2:
            try:
                overlaps = {k: metrics[k] for k in overlap_keys}
                fig = plot_subspace_overlap_heatmap(overlaps, factors)
                log_dict[f"{prefix}/overlap_heatmap"] = wandb.Image(fig)
                plt.close(fig)
            except Exception:
                pass

    # Rank monitor
    if z_eq is not None and z_inv is not None:
        try:
            fig = plot_rank_monitor(z_eq, z_inv)
            log_dict[f"{prefix}/rank_monitor"] = wandb.Image(fig)
            plt.close(fig)
        except Exception:
            pass

    # Log to W&B
    if log_dict:
        wandb.log(log_dict, step=step)


def plot_rotation_circle(
    delta_z: torch.Tensor,
    angles: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (6, 6),
) -> "plt.Figure":
    """
    Plot rotation-induced changes as a 2D trajectory.
    
    For clean rotation, this should trace a circle/ellipse.
    Points are colored by rotation angle for intuition.
    
    Args:
        delta_z: Rotation-induced changes, shape (N, d).
        angles: Optional rotation angles in degrees for coloring.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure.
    """
    _check_matplotlib()
    
    # Center and compute PCA
    delta_z = delta_z - delta_z.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(delta_z, full_matrices=False)
    basis = Vh[:2].T  # (d, 2)
    
    # Project to 2D
    proj = (delta_z @ basis).numpy()  # (N, 2)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color by angle if provided
    if angles is not None:
        scatter = ax.scatter(proj[:, 0], proj[:, 1], c=angles, cmap='hsv', 
                            s=30, alpha=0.7, edgecolors='none')
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Rotation angle (°)')
    else:
        ax.scatter(proj[:, 0], proj[:, 1], c='steelblue', s=30, alpha=0.7)
    
    # Add unit circle for reference
    theta = np.linspace(0, 2*np.pi, 100)
    r = max(np.abs(proj).max(), 1.0)
    ax.plot(r * np.cos(theta), r * np.sin(theta), 'k--', alpha=0.2, linewidth=1)
    
    ax.set_xlabel("PC1 (expected: cos θ)")
    ax.set_ylabel("PC2 (expected: sin θ)")
    ax.set_title("Rotation Subspace (should be circular)")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


def plot_translation_lines(
    delta_z_x: torch.Tensor,
    delta_z_y: torch.Tensor,
    figsize: Tuple[int, int] = (10, 5),
) -> "plt.Figure":
    """
    Plot translation-induced changes.
    
    X and Y translation should produce orthogonal linear trajectories.
    
    Args:
        delta_z_x: X-translation changes, shape (N, d).
        delta_z_y: Y-translation changes, shape (N, d).
        figsize: Figure size.
    
    Returns:
        Matplotlib figure.
    """
    _check_matplotlib()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    for ax, delta_z, label in [(ax1, delta_z_x, "X-translation"), 
                                (ax2, delta_z_y, "Y-translation")]:
        delta_z = delta_z - delta_z.mean(dim=0, keepdim=True)
        _, _, Vh = torch.linalg.svd(delta_z, full_matrices=False)
        basis = Vh[:2].T
        proj = (delta_z @ basis).numpy()
        
        ax.scatter(proj[:, 0], proj[:, 1], c=np.arange(len(proj)), cmap='viridis', 
                   s=30, alpha=0.7)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"{label} Subspace")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle("Translation Subspaces (should be linear)")
    fig.tight_layout()
    return fig


def plot_combined_factor_trajectories(
    delta_z_by_factor: Dict[str, torch.Tensor],
    figsize: Tuple[int, int] = (12, 4),
) -> "plt.Figure":
    """
    Plot all factor trajectories side by side.
    
    Args:
        delta_z_by_factor: Dict mapping factor name to delta-z tensor.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure.
    """
    _check_matplotlib()
    
    factors = [f for f in delta_z_by_factor.keys() if delta_z_by_factor[f] is not None 
               and len(delta_z_by_factor[f]) > 0]
    n_factors = len(factors)
    
    if n_factors == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No factor data available", ha='center', va='center')
        return fig
    
    fig, axes = plt.subplots(1, n_factors, figsize=(4*n_factors, 4))
    if n_factors == 1:
        axes = [axes]
    
    expected_shapes = {
        "rot": "circle",
        "trans_x": "line",
        "trans_y": "line", 
        "scale": "line",
        "trans": "plane",
    }
    
    for ax, factor in zip(axes, factors):
        delta_z = delta_z_by_factor[factor]
        delta_z = delta_z - delta_z.mean(dim=0, keepdim=True)
        
        try:
            _, _, Vh = torch.linalg.svd(delta_z, full_matrices=False)
            basis = Vh[:2].T
            proj = (delta_z @ basis).numpy()
            
            # Color gradient based on sample order
            colors = plt.cm.viridis(np.linspace(0, 1, len(proj)))
            ax.scatter(proj[:, 0], proj[:, 1], c=colors, s=20, alpha=0.7)
            
            # Connect points to show trajectory
            ax.plot(proj[:, 0], proj[:, 1], 'k-', alpha=0.2, linewidth=0.5)
            
            expected = expected_shapes.get(factor, "unknown")
            ax.set_title(f"{factor.upper()}\n(expected: {expected})")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        except Exception:
            ax.text(0.5, 0.5, f"Error: {factor}", ha='center', va='center', transform=ax.transAxes)
    
    fig.suptitle("Factor Subspace Trajectories", fontsize=12)
    fig.tight_layout()
    return fig


def close_all_figures():
    """Close all matplotlib figures to free memory."""
    if HAS_MATPLOTLIB:
        plt.close("all")

