"""
Direct invariance and equivariance error metrics.

These metrics directly measure whether representations behave as expected:
- Invariant representations should NOT change under transformations
- Equivariant representations SHOULD change predictably
"""

from typing import Dict, Tuple

import torch


def invariance_error(
    z_base: torch.Tensor,
    z_transformed: torch.Tensor,
) -> float:
    """
    Compute invariance error: how much z changes under transformation.

    Args:
        z_base: Representation of base image, shape (N, d).
        z_transformed: Representation of transformed image, shape (N, d).

    Returns:
        Normalized MSE: ||z_t - z_b||^2 / ||z_b||^2.
        Lower is better for invariant representations.
    """
    diff = z_transformed - z_base
    diff_norm_sq = (diff**2).sum(dim=1).mean()
    base_norm_sq = (z_base**2).sum(dim=1).mean() + 1e-8

    return float((diff_norm_sq / base_norm_sq).item())


def change_energy(
    z_base: torch.Tensor,
    z_transformed: torch.Tensor,
) -> float:
    """
    Compute change energy: how much z changes under transformation.

    Args:
        z_base: Representation of base image, shape (N, d).
        z_transformed: Representation of transformed image, shape (N, d).

    Returns:
        Normalized change: ||z_t - z_b||^2 / ||z_b||^2.
        Should be non-trivial for equivariant representations.
    """
    # Same computation as invariance_error, different interpretation
    return invariance_error(z_base, z_transformed)


def compute_inv_eq_metrics(
    z_inv_base: torch.Tensor,
    z_inv_transformed: torch.Tensor,
    z_eq_base: torch.Tensor,
    z_eq_transformed: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute all invariance/equivariance metrics for a paired batch.

    Args:
        z_inv_base: Invariant rep (z_AGG) of base images, shape (N, d_inv).
        z_inv_transformed: Invariant rep of transformed images, shape (N, d_inv).
        z_eq_base: Equivariant rep (z_t) of base images, shape (N, d_eq).
        z_eq_transformed: Equivariant rep of transformed images, shape (N, d_eq).

    Returns:
        Dict with keys:
            - inv_error_inv: Invariance error on z_inv (want low)
            - inv_error_eq: Invariance error on z_eq (cross-check, want higher)
            - eq_change_eq: Change energy on z_eq (want non-trivial)
            - eq_change_inv: Change energy on z_inv (cross-check, want low)
    """
    return {
        "inv_error_inv": invariance_error(z_inv_base, z_inv_transformed),
        "inv_error_eq": invariance_error(z_eq_base, z_eq_transformed),
        "eq_change_eq": change_energy(z_eq_base, z_eq_transformed),
        "eq_change_inv": change_energy(z_inv_base, z_inv_transformed),
    }


def compute_rank_metrics(
    z_inv: torch.Tensor,
    z_eq: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute effective rank metrics for collapse detection.

    Args:
        z_inv: Invariant representations, shape (N, d_inv).
        z_eq: Equivariant representations, shape (N, d_eq).

    Returns:
        Dict with effective ranks and variance metrics.
    """
    def effective_rank(z: torch.Tensor) -> float:
        """Effective rank via participation ratio of singular values."""
        z_centered = z - z.mean(dim=0, keepdim=True)
        _, S, _ = torch.linalg.svd(z_centered, full_matrices=False)
        S = S + 1e-10
        sum_s = S.sum()
        sum_s_sq = (S**2).sum()
        return float((sum_s**2 / sum_s_sq).item())

    def mean_variance(z: torch.Tensor) -> float:
        """Mean per-dimension variance."""
        z_centered = z - z.mean(dim=0, keepdim=True)
        var_per_dim = (z_centered**2).mean(dim=0)
        return float(var_per_dim.mean().item())

    return {
        "rank_eq_eff": effective_rank(z_eq),
        "rank_inv_eff": effective_rank(z_inv),
        "var_eq_mean": mean_variance(z_eq),
        "var_inv_mean": mean_variance(z_inv),
    }


class InvEqMetricsCollector:
    """
    Collector for invariance/equivariance metrics over a dataset.

    Accumulates representations and computes metrics at the end.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset accumulated data."""
        self.z_inv_base_list = []
        self.z_inv_trans_list = []
        self.z_eq_base_list = []
        self.z_eq_trans_list = []

    def add_batch(
        self,
        z_inv_base: torch.Tensor,
        z_inv_transformed: torch.Tensor,
        z_eq_base: torch.Tensor,
        z_eq_transformed: torch.Tensor,
    ):
        """Add a batch of paired representations."""
        self.z_inv_base_list.append(z_inv_base.detach().cpu())
        self.z_inv_trans_list.append(z_inv_transformed.detach().cpu())
        self.z_eq_base_list.append(z_eq_base.detach().cpu())
        self.z_eq_trans_list.append(z_eq_transformed.detach().cpu())

    def compute(self) -> Dict[str, float]:
        """Compute metrics over all accumulated data."""
        if not self.z_inv_base_list:
            return {}

        z_inv_base = torch.cat(self.z_inv_base_list, dim=0)
        z_inv_trans = torch.cat(self.z_inv_trans_list, dim=0)
        z_eq_base = torch.cat(self.z_eq_base_list, dim=0)
        z_eq_trans = torch.cat(self.z_eq_trans_list, dim=0)

        metrics = compute_inv_eq_metrics(
            z_inv_base, z_inv_trans, z_eq_base, z_eq_trans
        )

        # Add rank metrics using base representations
        rank_metrics = compute_rank_metrics(z_inv_base, z_eq_base)
        metrics.update(rank_metrics)

        return metrics

