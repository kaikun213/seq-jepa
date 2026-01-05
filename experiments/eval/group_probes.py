"""
Multi-factor group probes for action regression and cross-leakage detection.

These probes measure:
- How well each transformation factor can be predicted from z_eq (equivariant)
- How much factor information leaks into z_inv (invariant)
- How much class information leaks into z_eq

The gap between these indicates quality of the equivariant/invariant split.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, accuracy_score


class LinearProbe(nn.Module):
    """Simple linear probe for regression or classification."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MultiFactorActionProbe:
    """
    Probes for predicting multiple action factors from representations.

    Trains separate linear probes for each factor and measures R² scores.
    Uses delta-z (z(t+1) - z(t)) for better interpretability.
    """

    def __init__(
        self,
        feature_dim: int,
        factor_dims: Dict[str, int],
        lr: float = 0.001,
        device: torch.device = None,
    ):
        """
        Args:
            feature_dim: Dimension of input features (z_eq or z_inv).
            factor_dims: Dict mapping factor name to output dimension.
                         E.g., {"rot": 2, "trans_x": 1, "trans_y": 1, "scale": 1}
            lr: Learning rate for probe training.
            device: Target device.
        """
        self.feature_dim = feature_dim
        self.factor_dims = factor_dims
        self.device = device or torch.device("cpu")
        self.lr = lr

        # Create a probe for each factor
        self.probes: Dict[str, LinearProbe] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}

        for factor, out_dim in factor_dims.items():
            probe = LinearProbe(feature_dim, out_dim).to(self.device)
            self.probes[factor] = probe
            self.optimizers[factor] = torch.optim.Adam(probe.parameters(), lr=lr)

        self.mse_loss = nn.MSELoss()

    def train_step(
        self,
        features: torch.Tensor,
        targets_by_factor: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Single training step for all factor probes.

        Args:
            features: Input features, shape (B, feature_dim).
            targets_by_factor: Dict mapping factor name to target tensor.

        Returns:
            Dict of per-factor MSE losses.
        """
        losses = {}

        for factor, probe in self.probes.items():
            if factor not in targets_by_factor:
                continue

            target = targets_by_factor[factor].to(self.device)
            probe.train()

            self.optimizers[factor].zero_grad()
            pred = probe(features)
            loss = self.mse_loss(pred, target)
            loss.backward()
            self.optimizers[factor].step()

            losses[f"probe_loss_{factor}"] = loss.item()

        return losses

    @torch.no_grad()
    def evaluate(
        self,
        features: torch.Tensor,
        targets_by_factor: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Evaluate R² scores for all factors.

        Args:
            features: Input features, shape (N, feature_dim).
            targets_by_factor: Dict mapping factor name to target tensor.

        Returns:
            Dict of per-factor R² scores.
        """
        r2_scores = {}

        for factor, probe in self.probes.items():
            if factor not in targets_by_factor:
                continue

            target = targets_by_factor[factor].to(self.device)
            probe.eval()

            pred = probe(features)

            # Move to CPU for sklearn
            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()

            # Handle multi-dimensional targets
            if pred_np.ndim == 1:
                pred_np = pred_np.reshape(-1, 1)
                target_np = target_np.reshape(-1, 1)

            r2 = r2_score(target_np, pred_np, multioutput="uniform_average")
            r2_scores[f"probe_action_r2_{factor}"] = r2

        # Compute mean R²
        if r2_scores:
            r2_scores["probe_action_r2_mean"] = np.mean(list(r2_scores.values()))

        return r2_scores

    def reset(self):
        """Reset probe weights."""
        for factor in self.probes:
            probe = LinearProbe(self.feature_dim, self.factor_dims[factor])
            self.probes[factor] = probe.to(self.device)
            self.optimizers[factor] = torch.optim.Adam(probe.parameters(), lr=self.lr)


class CrossLeakageProbes:
    """
    Probes for detecting information leakage across representations.

    Measures:
    - Class accuracy from z_inv (should be high) vs z_eq (should be low)
    - Action R² from z_eq (should be high) vs z_inv (should be low)
    """

    def __init__(
        self,
        inv_dim: int,
        eq_dim: int,
        num_classes: int,
        action_dim: int,
        lr: float = 0.001,
        device: torch.device = None,
    ):
        """
        Args:
            inv_dim: Dimension of invariant representation (z_AGG).
            eq_dim: Dimension of equivariant representation (z_t).
            num_classes: Number of classes for classification.
            action_dim: Dimension of action vector.
            lr: Learning rate.
            device: Target device.
        """
        self.device = device or torch.device("cpu")
        self.lr = lr

        # Class probes
        self.class_from_inv = LinearProbe(inv_dim, num_classes).to(self.device)
        self.class_from_eq = LinearProbe(eq_dim, num_classes).to(self.device)

        # Action probes
        self.action_from_eq = LinearProbe(eq_dim, action_dim).to(self.device)
        self.action_from_inv = LinearProbe(inv_dim, action_dim).to(self.device)

        # Optimizers
        self.opt_class_inv = torch.optim.Adam(self.class_from_inv.parameters(), lr=lr)
        self.opt_class_eq = torch.optim.Adam(self.class_from_eq.parameters(), lr=lr)
        self.opt_action_eq = torch.optim.Adam(self.action_from_eq.parameters(), lr=lr)
        self.opt_action_inv = torch.optim.Adam(self.action_from_inv.parameters(), lr=lr)

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def train_step(
        self,
        z_inv: torch.Tensor,
        z_eq: torch.Tensor,
        labels: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Train all probes on a batch.

        Args:
            z_inv: Invariant representations, shape (B, inv_dim).
            z_eq: Equivariant representations, shape (B, eq_dim).
            labels: Class labels, shape (B,).
            actions: Action vectors, shape (B, action_dim).

        Returns:
            Dict of probe losses.
        """
        losses = {}

        z_inv = z_inv.detach().to(self.device)
        z_eq = z_eq.detach().to(self.device)
        labels = labels.to(self.device)
        actions = actions.to(self.device)

        # Class from inv
        self.class_from_inv.train()
        self.opt_class_inv.zero_grad()
        logits = self.class_from_inv(z_inv)
        loss = self.ce_loss(logits, labels)
        loss.backward()
        self.opt_class_inv.step()
        losses["probe_class_loss_inv"] = loss.item()

        # Class from eq
        self.class_from_eq.train()
        self.opt_class_eq.zero_grad()
        logits = self.class_from_eq(z_eq)
        loss = self.ce_loss(logits, labels)
        loss.backward()
        self.opt_class_eq.step()
        losses["probe_class_loss_eq"] = loss.item()

        # Action from eq
        self.action_from_eq.train()
        self.opt_action_eq.zero_grad()
        pred = self.action_from_eq(z_eq)
        loss = self.mse_loss(pred, actions)
        loss.backward()
        self.opt_action_eq.step()
        losses["probe_action_loss_eq"] = loss.item()

        # Action from inv
        self.action_from_inv.train()
        self.opt_action_inv.zero_grad()
        pred = self.action_from_inv(z_inv)
        loss = self.mse_loss(pred, actions)
        loss.backward()
        self.opt_action_inv.step()
        losses["probe_action_loss_inv"] = loss.item()

        return losses

    @torch.no_grad()
    def evaluate(
        self,
        z_inv: torch.Tensor,
        z_eq: torch.Tensor,
        labels: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Evaluate all probes.

        Returns:
            Dict with accuracy and R² metrics plus gap metrics.
        """
        z_inv = z_inv.to(self.device)
        z_eq = z_eq.to(self.device)
        labels = labels.to(self.device)
        actions = actions.to(self.device)

        metrics = {}

        # Class accuracy from inv
        self.class_from_inv.eval()
        logits = self.class_from_inv(z_inv)
        preds = logits.argmax(dim=1)
        acc_inv = (preds == labels).float().mean().item() * 100
        metrics["probe_class_acc_inv"] = acc_inv

        # Class accuracy from eq
        self.class_from_eq.eval()
        logits = self.class_from_eq(z_eq)
        preds = logits.argmax(dim=1)
        acc_eq = (preds == labels).float().mean().item() * 100
        metrics["probe_class_acc_eq"] = acc_eq

        # Action R² from eq
        self.action_from_eq.eval()
        pred = self.action_from_eq(z_eq)
        r2_eq = r2_score(
            actions.cpu().numpy(),
            pred.cpu().numpy(),
            multioutput="uniform_average",
        )
        metrics["probe_action_r2_eq"] = r2_eq

        # Action R² from inv
        self.action_from_inv.eval()
        pred = self.action_from_inv(z_inv)
        r2_inv = r2_score(
            actions.cpu().numpy(),
            pred.cpu().numpy(),
            multioutput="uniform_average",
        )
        metrics["probe_action_r2_inv"] = r2_inv

        # Gap metrics (higher is better - indicates good separation)
        metrics["gap_class"] = acc_inv - acc_eq
        metrics["gap_action"] = r2_eq - r2_inv

        return metrics

    def reset(self):
        """Reset all probe weights."""
        inv_dim = self.class_from_inv.linear.in_features
        eq_dim = self.class_from_eq.linear.in_features
        num_classes = self.class_from_inv.linear.out_features
        action_dim = self.action_from_eq.linear.out_features

        self.class_from_inv = LinearProbe(inv_dim, num_classes).to(self.device)
        self.class_from_eq = LinearProbe(eq_dim, num_classes).to(self.device)
        self.action_from_eq = LinearProbe(eq_dim, action_dim).to(self.device)
        self.action_from_inv = LinearProbe(inv_dim, action_dim).to(self.device)

        self.opt_class_inv = torch.optim.Adam(
            self.class_from_inv.parameters(), lr=self.lr
        )
        self.opt_class_eq = torch.optim.Adam(
            self.class_from_eq.parameters(), lr=self.lr
        )
        self.opt_action_eq = torch.optim.Adam(
            self.action_from_eq.parameters(), lr=self.lr
        )
        self.opt_action_inv = torch.optim.Adam(
            self.action_from_inv.parameters(), lr=self.lr
        )

