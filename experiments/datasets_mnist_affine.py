"""
MNIST with controlled affine transformations for subspace diagnostics.

Supports modes:
- rot_only: Only rotation transforms
- trans_only: Only translation transforms
- scale_only: Only scale transforms
- multi: All transforms combined

Ground-truth parameters are returned for precise subspace analysis.
"""

import math
import random
from typing import List, Literal, Optional, Tuple

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class MNISTAffineSequence(Dataset):
    """
    MNIST with controlled affine transforms for equivariance testing.

    Each sample is a sequence of views of the same digit under different
    transformations. Ground-truth transformation parameters are returned
    for subspace analysis.
    """

    def __init__(
        self,
        root: str,
        split: str,
        seq_len: int,
        mode: Literal["rot_only", "trans_only", "scale_only", "multi"] = "multi",
        rot_range: Tuple[float, float] = (-75.0, 75.0),
        trans_range: Tuple[int, int] = (-7, 7),
        scale_range: Tuple[float, float] = (0.5, 1.0),
        normalize: bool = True,
        download: bool = True,
        pad_size: int = 40,  # Pad to allow transforms without cropping
    ) -> None:
        """
        Args:
            root: Data root directory.
            split: 'train' or 'test'.
            seq_len: Number of actions (views = seq_len + 1).
            mode: Transform mode - which factors to vary.
            rot_range: (min, max) rotation in degrees.
            trans_range: (min, max) translation in pixels.
            scale_range: (min, max) scale factor.
            normalize: Apply normalization.
            download: Download if not present.
            pad_size: Size to pad images to (allows room for transforms).
        """
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")

        self.seq_len = seq_len
        self.num_views = seq_len + 1
        self.mode = mode
        self.rot_range = rot_range
        self.trans_range = trans_range
        self.scale_range = scale_range
        self.pad_size = pad_size

        self.base = torchvision.datasets.MNIST(
            root=root,
            train=(split == "train"),
            download=download,
        )

        # Normalization
        mean = (0.1307,)
        std = (0.3081,)
        if normalize:
            self.normalize = T.Normalize(mean, std)
        else:
            self.normalize = None

    def __len__(self) -> int:
        return len(self.base)

    def _sample_params(self) -> List[dict]:
        """Sample transformation parameters for each view."""
        params_list = []

        for _ in range(self.num_views):
            params = {"rot": 0.0, "tx": 0, "ty": 0, "scale": 1.0}

            if self.mode in ("rot_only", "multi"):
                params["rot"] = random.uniform(*self.rot_range)

            if self.mode in ("trans_only", "multi"):
                params["tx"] = random.randint(*self.trans_range)
                params["ty"] = random.randint(*self.trans_range)

            if self.mode in ("scale_only", "multi"):
                # Log-uniform sampling for scale
                log_min = math.log(self.scale_range[0])
                log_max = math.log(self.scale_range[1])
                params["scale"] = math.exp(random.uniform(log_min, log_max))

            params_list.append(params)

        # First view is identity (reference)
        params_list[0] = {"rot": 0.0, "tx": 0, "ty": 0, "scale": 1.0}

        return params_list

    def _apply_transform(
        self,
        img: torch.Tensor,
        params: dict,
    ) -> torch.Tensor:
        """Apply affine transformation to image."""
        # img is (C, H, W), already a tensor

        # Pad image to allow for transforms
        orig_size = img.shape[-1]  # Assuming square
        pad_amount = (self.pad_size - orig_size) // 2
        if pad_amount > 0:
            img = TF.pad(img, pad_amount, fill=0)

        # Apply scale (resize and center)
        if params["scale"] != 1.0:
            new_size = int(self.pad_size * params["scale"])
            img = TF.resize(img, [new_size, new_size])
            # Center crop or pad back to pad_size
            if new_size > self.pad_size:
                img = TF.center_crop(img, self.pad_size)
            elif new_size < self.pad_size:
                pad_needed = self.pad_size - new_size
                pad_left = pad_needed // 2
                pad_right = pad_needed - pad_left
                img = TF.pad(img, [pad_left, pad_left, pad_right, pad_right], fill=0)

        # Apply rotation
        if params["rot"] != 0.0:
            img = TF.rotate(img, params["rot"], fill=0)

        # Apply translation
        if params["tx"] != 0 or params["ty"] != 0:
            img = TF.affine(
                img,
                angle=0,
                translate=[params["tx"], params["ty"]],
                scale=1.0,
                shear=0,
                fill=0,
            )

        # Center crop back to original size
        img = TF.center_crop(img, orig_size)

        return img

    def _params_to_tensor(self, params: dict) -> torch.Tensor:
        """Convert parameters to tensor for action encoding."""
        # Encode rotation as sin/cos
        theta = math.radians(params["rot"])
        sin_rot = math.sin(theta)
        cos_rot = math.cos(theta)

        # Normalize translation
        tx_norm = params["tx"] / max(abs(self.trans_range[0]), abs(self.trans_range[1]))
        ty_norm = params["ty"] / max(abs(self.trans_range[0]), abs(self.trans_range[1]))

        # Log scale
        log_scale = math.log(params["scale"])

        return torch.tensor(
            [sin_rot, cos_rot, tx_norm, ty_norm, log_scale],
            dtype=torch.float32,
        )

    def _compute_relative_params(
        self,
        params_curr: dict,
        params_next: dict,
    ) -> torch.Tensor:
        """Compute relative transformation parameters."""
        # Relative rotation
        delta_rot = params_next["rot"] - params_curr["rot"]
        # Wrap to [-180, 180]
        delta_rot = ((delta_rot + 180) % 360) - 180
        theta = math.radians(delta_rot)
        sin_rot = math.sin(theta)
        cos_rot = math.cos(theta)

        # Relative translation
        delta_tx = params_next["tx"] - params_curr["tx"]
        delta_ty = params_next["ty"] - params_curr["ty"]
        tx_norm = delta_tx / max(abs(self.trans_range[0]), abs(self.trans_range[1]))
        ty_norm = delta_ty / max(abs(self.trans_range[0]), abs(self.trans_range[1]))

        # Relative scale (ratio in log space)
        delta_log_scale = math.log(params_next["scale"]) - math.log(params_curr["scale"])

        return torch.tensor(
            [sin_rot, cos_rot, tx_norm, ty_norm, delta_log_scale],
            dtype=torch.float32,
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Returns:
            views: Stacked views, shape (num_views, C, H, W).
            actions_abs: Absolute action parameters, shape (num_views, action_dim).
            actions_rel: Relative action parameters, shape (seq_len, action_dim).
            label: Class label.
            orig_img: Original untransformed image.
        """
        img, label = self.base[idx]

        # Convert to tensor
        img_tensor = T.ToTensor()(img)  # (1, 28, 28)

        # Sample parameters
        params_list = self._sample_params()

        views = []
        actions_abs = []

        for params in params_list:
            # Apply transform
            view = self._apply_transform(img_tensor.clone(), params)

            # Normalize if enabled
            if self.normalize is not None:
                view = self.normalize(view)

            views.append(view)
            actions_abs.append(self._params_to_tensor(params))

        # Compute relative actions
        actions_rel = []
        for i in range(self.seq_len):
            rel = self._compute_relative_params(params_list[i], params_list[i + 1])
            actions_rel.append(rel)

        # Original image
        orig_img = img_tensor.clone()
        if self.normalize is not None:
            orig_img = self.normalize(orig_img)

        return (
            torch.stack(views, dim=0),
            torch.stack(actions_abs, dim=0),
            torch.stack(actions_rel, dim=0),
            label,
            orig_img,
        )

    @property
    def action_dim(self) -> int:
        """Dimension of action vector."""
        return 5  # sin_rot, cos_rot, tx_norm, ty_norm, log_scale

    @property
    def factor_dims(self) -> dict:
        """Expected subspace dimensions per factor."""
        return {
            "rot": 2,  # sin, cos -> 2D subspace
            "trans_x": 1,
            "trans_y": 1,
            "scale": 1,
        }

