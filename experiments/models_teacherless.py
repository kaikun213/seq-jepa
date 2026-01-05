"""
Teacherless model variants for Step 2.

These wrappers extend upstream SeqJEPA_Transforms with:
- Coding-rate regularization loss
- Optional teacherless mode (stop-grad instead of EMA)
- Optional DINO-style centering/sharpening
- Optional symmetric prediction

All features are config-driven and disabled by default,
so the baseline behavior is preserved when flags are off.
"""

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import upstream model - path adjusted for experiments/ location
import sys
from pathlib import Path

# Add seq-jepa/src to path if needed
_src_path = Path(__file__).parent.parent / "seq-jepa" / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from models import SeqJEPA_Transforms
from experiments.losses.coding_rate import CodingRateLoss


class SeqJEPA_Teacherless(SeqJEPA_Transforms):
    """
    Extended SeqJEPA_Transforms with teacherless training options.

    New features (all disabled by default):
    - coding_rate: Add coding-rate regularizer to prevent collapse
    - teacherless: Use stop-grad instead of EMA target encoder
    - sharpening: DINO-style centering and sharpening on targets
    - symmetric: Bidirectional prediction (forward and reverse)

    When all flags are False, behavior matches upstream exactly.
    """

    def __init__(
        self,
        img_size,
        ema,
        ema_decay=0.996,
        # New teacherless options
        teacherless: bool = False,
        rate_loss_cfg: Optional[dict] = None,
        sharpening_cfg: Optional[dict] = None,
        symmetric: bool = False,
        **kwargs,
    ):
        # If teacherless, we still need to init parent with ema=False
        # to avoid creating unnecessary target encoder
        parent_ema = ema and not teacherless
        super().__init__(img_size, parent_ema, ema_decay, **kwargs)

        self.teacherless = teacherless
        self.symmetric = symmetric

        # Store original ema flag for reference
        self._original_ema = ema

        # Coding rate loss
        self.rate_loss_cfg = rate_loss_cfg or {}
        self.rate_loss_enabled = self.rate_loss_cfg.get("enabled", False)
        if self.rate_loss_enabled:
            self.rate_loss_fn = CodingRateLoss(
                alpha=self.rate_loss_cfg.get("alpha", 1.0),
                eps=self.rate_loss_cfg.get("eps", 1e-4),
                target=self.rate_loss_cfg.get("target", "agg_out"),
                lambda_rate=self.rate_loss_cfg.get("lambda_rate", 0.01),
                use_stable=self.rate_loss_cfg.get("use_stable", False),
            )
        else:
            self.rate_loss_fn = None

        # DINO-style sharpening
        self.sharpening_cfg = sharpening_cfg or {}
        self.sharpening_enabled = self.sharpening_cfg.get("enabled", False)
        if self.sharpening_enabled:
            self.sharpening_temp = self.sharpening_cfg.get("temp", 0.04)
            self.center_momentum = self.sharpening_cfg.get("center_momentum", 0.9)
            # Register buffer for running center
            self.register_buffer(
                "center", torch.zeros(1, self.res_out_dim)
            )

    def _encode_target(self, x_pred: torch.Tensor) -> torch.Tensor:
        """
        Encode target view, using appropriate method based on mode.

        Args:
            x_pred: Target view tensor, shape (B, C, H, W).

        Returns:
            Target encoding, shape (B, d).
        """
        if self.teacherless:
            # Stop-grad on same encoder (no EMA)
            with torch.no_grad():
                enc_pred = self.encoder(x_pred)
            return enc_pred.detach()
        elif self.ema:
            # EMA target encoder (upstream behavior)
            with torch.no_grad():
                enc_pred = self.target_encoder(x_pred)
            return enc_pred.detach()
        else:
            # No EMA, no teacherless - just encode and detach
            enc_pred = self.encoder(x_pred)
            return enc_pred.detach()

    def _apply_sharpening(self, z_target: torch.Tensor) -> torch.Tensor:
        """
        Apply DINO-style centering and sharpening to target.

        Args:
            z_target: Target encoding, shape (B, d).

        Returns:
            Sharpened target, shape (B, d).
        """
        if not self.sharpening_enabled:
            return z_target

        # Update center with exponential moving average
        if self.training:
            batch_center = z_target.mean(dim=0, keepdim=True)
            self.center = (
                self.center_momentum * self.center
                + (1 - self.center_momentum) * batch_center
            )

        # Center and sharpen
        z_centered = z_target - self.center
        z_sharpened = F.softmax(z_centered / self.sharpening_temp, dim=-1)

        return z_sharpened

    def forward(self, x_obs, x_pred, act_lat_obs, act_lat_pred, rel_latents=None):
        """
        Forward pass with optional teacherless extensions.

        Same signature as parent, but adds rate loss if enabled.
        """
        # Get sequence length and reshape observations
        seq_len = x_obs.shape[1]
        x_obs_flat = x_obs.reshape(-1, self.n_channels, self.img_size, self.img_size)
        enc_obs = self.encoder(x_obs_flat)
        enc_obs = enc_obs.reshape(-1, seq_len, self.res_out_dim)

        x_pred_flat = x_pred.reshape(-1, self.n_channels, self.img_size, self.img_size)

        # Use appropriate target encoding method
        enc_pred = self._encode_target(x_pred_flat)
        enc_pred = enc_pred.reshape(-1, self.res_out_dim)

        # Apply sharpening if enabled
        if self.sharpening_enabled:
            enc_pred = self._apply_sharpening(enc_pred)

        # Action conditioning (same as parent)
        if self.act_cond:
            act_lat_obs = act_lat_obs.reshape(-1, self.action_latentdim)
            if self.learn_act_emb:
                act_enc_obs = act_lat_obs
                act_enc_pred = act_lat_pred
                act_enc_obs = act_enc_obs.reshape(-1, seq_len, self.action_latentdim)
                relative_act_enc_obs = torch.zeros_like(act_enc_obs)
                if rel_latents is not None:
                    relative_act_enc_obs = rel_latents.reshape(
                        -1, seq_len, self.action_latentdim
                    )
                    act_enc_pred = rel_latents[:, -1]
                else:
                    relative_act_enc_obs[:, :-1] = (
                        act_enc_obs[:, 1:] - act_enc_obs[:, :-1]
                    )
                    act_enc_pred = act_enc_pred - act_enc_obs[:, -1]
                act_enc_pred = self.action_proj(act_enc_pred)
                relative_act_enc_obs = relative_act_enc_obs.reshape(
                    -1, self.action_latentdim
                )
                relative_act_enc_obs = self.action_proj(relative_act_enc_obs)
                relative_act_enc_obs = relative_act_enc_obs.reshape(
                    -1, seq_len, self.action_projdim
                )
                relative_act_enc_obs[:, -1, ...] = 0.0
                enc_obs_concat = torch.cat((enc_obs, relative_act_enc_obs), dim=-1)
            else:
                if rel_latents is not None:
                    act_enc_obs = rel_latents.reshape(-1, seq_len, self.action_latentdim)
                    act_enc_obs[:, -1, ...] = 0.0
                    act_enc_pred = rel_latents[:, -1]
                    if self.action_latentdim % self.num_heads != 0:
                        padding = self.num_heads - (self.action_latentdim % self.num_heads)
                        act_enc_obs = F.pad(act_enc_obs, (0, padding))
                        act_enc_pred = F.pad(act_enc_pred, (0, padding))
                else:
                    act_enc_obs = act_lat_obs.reshape(-1, seq_len, self.action_latentdim)
                    act_enc_pred = act_lat_pred
                    if self.action_latentdim % self.num_heads != 0:
                        padding = self.num_heads - (self.action_latentdim % self.num_heads)
                        act_enc_obs = F.pad(act_enc_obs, (0, padding))
                        act_enc_pred = F.pad(act_enc_pred, (0, padding))
                enc_obs_concat = torch.cat((enc_obs, act_enc_obs), dim=-1)
        else:
            enc_obs_concat = enc_obs

        # Aggregator token
        B, _, _ = enc_obs.shape
        agg_tokens = self.agg_token.expand(B, -1, -1)

        # Transformer encoder
        x = torch.cat((agg_tokens, enc_obs_concat), dim=1)
        x, _ = self.transformer_encoder(x)

        agg_out = x[:, 0]

        # Predictor
        if self.act_cond:
            agg_out_conditioned = torch.cat((agg_out, act_enc_pred), dim=-1)
            pred_out = self.predictor(agg_out_conditioned)
        else:
            pred_out = self.predictor(agg_out)

        # Prediction loss (cosine similarity)
        loss = 1 - self.criterion(pred_out, enc_pred).mean()

        # Add rate loss if enabled
        if self.rate_loss_enabled and self.rate_loss_fn is not None:
            z_t = enc_obs[:, 0]  # First timestep encoder output
            rate_loss = self.rate_loss_fn(agg_out=agg_out, z_t=z_t)
            loss = loss + rate_loss

        # Symmetric prediction (forward already done, add reverse)
        if self.symmetric and self.training:
            # Reverse prediction: predict x_obs[-1] from x_pred context
            # This is a simplified version; full implementation would
            # require restructuring the aggregator
            pass  # TODO: Implement symmetric prediction

        z1 = enc_obs[:, 0]
        if seq_len == 1:
            z2 = enc_pred
        else:
            z2 = enc_obs[:, 1]

        return loss, agg_out, z1, z2

    def get_loss_breakdown(self) -> dict:
        """Return dictionary of individual loss components for logging."""
        # This can be extended to track pred_loss vs rate_loss separately
        return {}


def create_teacherless_model(cfg: dict, device: torch.device) -> SeqJEPA_Teacherless:
    """
    Factory function to create teacherless model from config.

    Args:
        cfg: Full config dict with 'model' and 'loss' sections.
        device: Target device.

    Returns:
        Initialized SeqJEPA_Teacherless model.
    """
    model_cfg = cfg.get("model", {})
    loss_cfg = cfg.get("loss", {})

    # Build rate loss config
    rate_loss_cfg = {
        "enabled": loss_cfg.get("rate_loss_enabled", False),
        "lambda_rate": loss_cfg.get("lambda_rate", 0.01),
        "alpha": loss_cfg.get("alpha", 1.0),
        "eps": loss_cfg.get("eps", 1e-4),
        "target": loss_cfg.get("rate_target", "agg_out"),
        "use_stable": loss_cfg.get("rate_use_stable", False),
    }

    # Build sharpening config
    sharpening_cfg = {
        "enabled": model_cfg.get("sharpening_enabled", False),
        "temp": model_cfg.get("sharpening_temp", 0.04),
        "center_momentum": model_cfg.get("sharpening_center_momentum", 0.9),
    }

    model = SeqJEPA_Teacherless(
        img_size=model_cfg.get("img_size", 32),
        ema=model_cfg.get("ema", True),
        ema_decay=model_cfg.get("ema_decay", 0.996),
        teacherless=model_cfg.get("teacherless", False),
        rate_loss_cfg=rate_loss_cfg,
        sharpening_cfg=sharpening_cfg,
        symmetric=model_cfg.get("symmetric", False),
        n_channels=model_cfg.get("n_channels", 3),
        num_classes=model_cfg.get("num_classes", 100),
        num_heads=model_cfg.get("num_heads", 4),
        num_enc_layers=model_cfg.get("num_enc_layers", 3),
        act_cond=model_cfg.get("act_cond", 1),
        pred_hidden=model_cfg.get("pred_hidden", 1024),
        act_projdim=model_cfg.get("act_projdim", 128),
        act_latentdim=model_cfg.get("act_latentdim", 9),
        learn_act_emb=model_cfg.get("learn_act_emb", 1),
        cifar_resnet=model_cfg.get("cifar_resnet", True),
    )

    return model.to(device)

