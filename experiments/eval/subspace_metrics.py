"""
Subspace metrics for factor disentanglement analysis.

These metrics measure whether each transformation factor (rotation, translation,
scale) lives in a low-dimensional, stable, sparse subspace within the
equivariant representation.

References:
- Disentangling images with Lie group transformations (arXiv:2012.12071)
- Towards a Definition of Disentangled Representations (arXiv:1812.02230)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def compute_factor_subspace(
    delta_z: torch.Tensor,
    k: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract factor subspace via PCA on delta-z representations.

    Args:
        delta_z: Change in equivariant representation due to factor,
                 shape (N, d) where N is number of samples.
        k: Number of principal components to extract.

    Returns:
        U: Orthonormal basis vectors, shape (d, k).
        S: Singular values, shape (min(N, d),).
        explained_variance_ratio: Fraction of variance explained by top-k.
    """
    # Center the delta
    delta_z = delta_z - delta_z.mean(dim=0, keepdim=True)

    # SVD
    U, S, Vh = torch.linalg.svd(delta_z, full_matrices=False)

    # Explained variance
    var_explained = S**2
    total_var = var_explained.sum()
    explained_variance_ratio = var_explained[:k].sum() / (total_var + 1e-8)

    # Return top-k right singular vectors (columns of V = rows of Vh transposed)
    basis = Vh[:k].T  # (d, k)

    return basis, S, explained_variance_ratio


def subspace_explained_variance(
    delta_z: torch.Tensor,
    k: int,
) -> float:
    """
    Compute fraction of variance explained by top-k dimensions.

    Args:
        delta_z: Factor-induced changes, shape (N, d).
        k: Number of dimensions to consider.

    Returns:
        Explained variance ratio in [0, 1].
    """
    _, _, ev_ratio = compute_factor_subspace(delta_z, k)
    return float(ev_ratio.item())


def subspace_effective_rank(delta_z: torch.Tensor) -> float:
    """
    Compute effective rank of the factor-change covariance.

    Uses participation ratio: (sum λ)^2 / (sum λ^2).
    Low effective rank (near expected k) indicates clean subspace.

    Args:
        delta_z: Factor-induced changes, shape (N, d).

    Returns:
        Effective rank (continuous value).
    """
    delta_z = delta_z - delta_z.mean(dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(delta_z, full_matrices=False)

    # Eigenvalues of covariance are S^2 / N
    eigvals = S**2
    eigvals = eigvals + 1e-10  # Stability

    sum_eig = eigvals.sum()
    sum_eig_sq = (eigvals**2).sum()

    eff_rank = (sum_eig**2) / (sum_eig_sq + 1e-10)
    return float(eff_rank.item())


def subspace_coordinate_sparsity(
    basis: torch.Tensor,
) -> float:
    """
    Measure coordinate sparsity of subspace basis.

    Computes participation ratio of per-dimension energy.
    Low value (near k) means subspace aligns with few coordinates.

    Args:
        basis: Orthonormal basis, shape (d, k).

    Returns:
        Coordinate participation ratio.
    """
    # Per-dim energy: sum of squared basis loadings
    energy = (basis**2).sum(dim=1)  # (d,)

    # Participation ratio
    sum_e = energy.sum()
    sum_e_sq = (energy**2).sum()

    coord_pr = (sum_e**2) / (sum_e_sq + 1e-10)
    return float(coord_pr.item())


def subspace_overlap(
    basis_f: torch.Tensor,
    basis_g: torch.Tensor,
) -> float:
    """
    Compute normalized overlap between two factor subspaces.

    Args:
        basis_f: Basis for factor f, shape (d, k_f).
        basis_g: Basis for factor g, shape (d, k_g).

    Returns:
        Normalized overlap in [0, 1]. Near 0 means orthogonal/disentangled.
    """
    k_f = basis_f.shape[1]
    k_g = basis_g.shape[1]

    # Overlap matrix
    overlap_matrix = basis_f.T @ basis_g  # (k_f, k_g)

    # Frobenius norm squared, normalized
    overlap = (overlap_matrix**2).sum() / (k_f * k_g + 1e-10)
    return float(overlap.item())


def subspace_stability(
    basis_current: torch.Tensor,
    basis_ref: torch.Tensor,
) -> float:
    """
    Measure stability of subspace compared to reference.

    Args:
        basis_current: Current basis, shape (d, k).
        basis_ref: Reference basis (e.g., from previous epoch), shape (d, k).

    Returns:
        Stability score in [0, 1]. Near 1 means stable.
    """
    k = basis_current.shape[1]

    # Subspace similarity via principal angles
    overlap_matrix = basis_current.T @ basis_ref  # (k, k)
    similarity = (overlap_matrix**2).sum() / k

    return float(similarity.item())


class SubspaceMetrics:
    """
    Collector for subspace metrics across factors.

    Computes metrics for each factor (rot, trans_x, trans_y, scale)
    and cross-factor overlaps.
    """

    def __init__(
        self,
        factor_dims: Optional[Dict[str, int]] = None,
        stability_momentum: float = 0.9,
    ):
        """
        Args:
            factor_dims: Expected intrinsic dimension per factor.
                         Default: {"rot": 2, "trans_x": 1, "trans_y": 1, "scale": 1}
            stability_momentum: EMA momentum for reference basis updates.
        """
        self.factor_dims = factor_dims or {
            "rot": 2,
            "trans_x": 1,
            "trans_y": 1,
            "scale": 1,
        }
        self.stability_momentum = stability_momentum

        # Reference bases for stability tracking
        self.ref_bases: Dict[str, Optional[torch.Tensor]] = {
            f: None for f in self.factor_dims
        }

    def compute_all(
        self,
        delta_z_by_factor: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Compute all subspace metrics for given factor deltas.

        Args:
            delta_z_by_factor: Dict mapping factor name to delta-z tensor (N, d).

        Returns:
            Dict of metric names to values.
        """
        metrics = {}
        bases = {}

        # Per-factor metrics
        for factor, delta_z in delta_z_by_factor.items():
            if delta_z is None or delta_z.shape[0] < 2:
                continue

            k = self.factor_dims.get(factor, 2)

            # Explained variance
            ev = subspace_explained_variance(delta_z, k)
            metrics[f"subspace_ev_{factor}_{k}"] = ev

            # Effective rank
            eff_rank = subspace_effective_rank(delta_z)
            metrics[f"subspace_effrank_{factor}"] = eff_rank

            # Extract basis for further metrics
            basis, _, _ = compute_factor_subspace(delta_z, k)
            bases[factor] = basis

            # Coordinate sparsity
            coord_pr = subspace_coordinate_sparsity(basis)
            metrics[f"subspace_coordpr_{factor}"] = coord_pr

            # Stability
            if self.ref_bases[factor] is not None:
                stability = subspace_stability(basis, self.ref_bases[factor])
                metrics[f"subspace_stability_{factor}"] = stability

            # Update reference with EMA
            if self.ref_bases[factor] is None:
                self.ref_bases[factor] = basis.clone()
            else:
                self.ref_bases[factor] = (
                    self.stability_momentum * self.ref_bases[factor]
                    + (1 - self.stability_momentum) * basis
                )
                # Re-orthonormalize
                U, _, Vh = torch.linalg.svd(self.ref_bases[factor], full_matrices=False)
                self.ref_bases[factor] = U @ Vh

        # Cross-factor overlaps
        factor_names = list(bases.keys())
        for i, f1 in enumerate(factor_names):
            for f2 in factor_names[i + 1:]:
                if f1 in bases and f2 in bases:
                    overlap = subspace_overlap(bases[f1], bases[f2])
                    metrics[f"subspace_overlap_{f1}_{f2}"] = overlap

        return metrics

    def reset_references(self):
        """Reset reference bases for stability tracking."""
        self.ref_bases = {f: None for f in self.factor_dims}

