"""
Coding-rate regularization based on MCR² framework.

References:
- Learning Diverse and Discriminative Representations via the Principle of
  Maximum Code Rate Reduction, 2020 (arXiv:2006.08558)
- Simplifying DINO via Coding Rate Regularization, 2025 (arXiv:2502.10385)
"""

import torch
import torch.nn as nn


def coding_rate_loss(
    Z: torch.Tensor,
    alpha: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Compute log-det coding rate (expansion term from MCR²).

    This term encourages the representation to span a high-dimensional space,
    preventing collapse without requiring an EMA teacher.

    Args:
        Z: Feature matrix of shape (B, d) - batch of d-dimensional features.
        alpha: Scaling factor for the covariance matrix.
        eps: Small constant for numerical stability.

    Returns:
        Scalar loss tensor. Minimizing this loss maximizes the coding rate,
        encouraging diverse representations.

    Note:
        The loss is -R(Z) where R(Z) = log det(I + alpha * C) and C = Z^T Z / B.
        We return the negative so that minimizing the loss maximizes the rate.
    """
    B, d = Z.shape

    # Center features (optional but often helps)
    Z_centered = Z - Z.mean(dim=0, keepdim=True)

    # Compute covariance: C = Z^T Z / B
    C = (Z_centered.T @ Z_centered) / B

    # Add scaled identity for stability and compute log-det
    # R(Z) = log det(I + alpha * C)
    I = torch.eye(d, device=Z.device, dtype=Z.dtype)
    R = torch.logdet(I + alpha * C + eps * I)

    # Return negative rate (minimize loss = maximize rate)
    return -R


def coding_rate_loss_stable(
    Z: torch.Tensor,
    alpha: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Numerically stable version using eigenvalue decomposition.

    Use this if logdet produces NaN on certain hardware/batch sizes.
    """
    B, d = Z.shape

    Z_centered = Z - Z.mean(dim=0, keepdim=True)
    C = (Z_centered.T @ Z_centered) / B

    # Eigenvalue decomposition
    eigvals = torch.linalg.eigvalsh(C)
    # Clamp eigenvalues for stability
    eigvals = torch.clamp(eigvals, min=eps)

    # R(Z) = sum log(1 + alpha * lambda_i)
    R = torch.sum(torch.log(1.0 + alpha * eigvals))

    return -R


class CodingRateLoss(nn.Module):
    """
    Module wrapper for coding-rate loss.

    Args:
        alpha: Scaling factor for covariance matrix (default: 1.0).
        eps: Numerical stability constant (default: 1e-4).
        target: Which representation to apply to ('agg_out', 'z_t', 'both').
        lambda_rate: Weight for the rate loss term.
        use_stable: Use eigenvalue-based stable computation.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        eps: float = 1e-4,
        target: str = "agg_out",
        lambda_rate: float = 0.01,
        use_stable: bool = False,
    ):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.target = target
        self.lambda_rate = lambda_rate
        self.use_stable = use_stable

    def forward(
        self,
        agg_out: torch.Tensor = None,
        z_t: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute weighted coding-rate loss on specified targets.

        Args:
            agg_out: Aggregator output (z_AGG), shape (B, d_agg).
            z_t: Encoder output for a single timestep, shape (B, d_enc).

        Returns:
            Weighted loss scalar.
        """
        loss = 0.0
        loss_fn = coding_rate_loss_stable if self.use_stable else coding_rate_loss

        if self.target in ("agg_out", "both") and agg_out is not None:
            loss = loss + loss_fn(agg_out, self.alpha, self.eps)

        if self.target in ("z_t", "both") and z_t is not None:
            loss = loss + loss_fn(z_t, self.alpha, self.eps)

        return self.lambda_rate * loss

    def extra_repr(self) -> str:
        return (
            f"alpha={self.alpha}, eps={self.eps}, target={self.target}, "
            f"lambda_rate={self.lambda_rate}, use_stable={self.use_stable}"
        )

