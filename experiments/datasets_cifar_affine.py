"""
CIFAR-10/100 with controlled affine transformations for subspace diagnostics.

Supports mild transforms with padding to avoid content destruction:
- Rotation: +/-30 degrees (continuous)
- Translation: +/-4 pixels
- Scale: 0.8-1.2

Ground-truth parameters are returned for precise subspace analysis.
"""

import math
import random
from typing import Literal, List, Tuple

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class CIFARAffineSequence(Dataset):
    """
    CIFAR with controlled affine transforms for equivariance testing.

    Uses mild transforms with reflection padding to preserve content.
    """

    def __init__(
        self,
        root: str,
        split: str,
        seq_len: int,
        dataset: Literal["cifar10", "cifar100"] = "cifar10",
        mode: Literal["rot_only", "trans_only", "scale_only", "multi"] = "multi",
        rot_range: Tuple[float, float] = (-30.0, 30.0),
        trans_range: Tuple[int, int] = (-4, 4),
        scale_range: Tuple[float, float] = (0.8, 1.2),
        normalize: bool = True,
        download: bool = True,
        pad_mode: str = "reflect",  # Use reflect padding to preserve content
    ) -> None:
        """
        Args:
            root: Data root directory.
            split: 'train' or 'test'.
            seq_len: Number of actions (views = seq_len + 1).
            dataset: 'cifar10' or 'cifar100'.
            mode: Transform mode - which factors to vary.
            rot_range: (min, max) rotation in degrees.
            trans_range: (min, max) translation in pixels.
            scale_range: (min, max) scale factor.
            normalize: Apply normalization.
            download: Download if not present.
            pad_mode: Padding mode for transforms ('reflect', 'constant', 'replicate').
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
        self.pad_mode = pad_mode
        self.img_size = 32

        # Load base dataset
        if dataset == "cifar10":
            self.base = torchvision.datasets.CIFAR10(
                root=root,
                train=(split == "train"),
                download=download,
            )
            self.num_classes = 10
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2470, 0.2435, 0.2616)
        elif dataset == "cifar100":
            self.base = torchvision.datasets.CIFAR100(
                root=root,
                train=(split == "train"),
                download=download,
            )
            self.num_classes = 100
            mean = (0.5071, 0.4867, 0.4408)
            std = (0.2675, 0.2565, 0.2761)
        else:
            raise ValueError("dataset must be 'cifar10' or 'cifar100'")

        # Normalization
        if normalize:
            self.normalize = T.Normalize(mean, std)
        else:
            self.normalize = None

        self.to_tensor = T.ToTensor()

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
                # Uniform sampling for scale (range is mild)
                params["scale"] = random.uniform(*self.scale_range)

            params_list.append(params)

        # First view is identity (reference)
        params_list[0] = {"rot": 0.0, "tx": 0, "ty": 0, "scale": 1.0}

        return params_list

    def _apply_transform(
        self,
        img: torch.Tensor,
        params: dict,
    ) -> torch.Tensor:
        """Apply affine transformation to image with padding."""
        # img is (C, H, W)

        # Pad image to allow for transforms
        pad_amount = 8  # Extra padding for transforms
        if self.pad_mode == "reflect":
            img = TF.pad(img, pad_amount, padding_mode="reflect")
        elif self.pad_mode == "replicate":
            # Use edge replication
            img = torch.nn.functional.pad(
                img.unsqueeze(0), [pad_amount] * 4, mode="replicate"
            ).squeeze(0)
        else:
            img = TF.pad(img, pad_amount, fill=0)

        padded_size = img.shape[-1]

        # Apply scale
        if params["scale"] != 1.0:
            new_size = int(padded_size * params["scale"])
            img = TF.resize(img, [new_size, new_size], antialias=True)
            # Center crop or pad back
            if new_size > padded_size:
                img = TF.center_crop(img, padded_size)
            elif new_size < padded_size:
                pad_needed = padded_size - new_size
                pad_left = pad_needed // 2
                pad_right = pad_needed - pad_left
                if self.pad_mode == "reflect":
                    img = TF.pad(img, [pad_left, pad_left, pad_right, pad_right], padding_mode="reflect")
                else:
                    img = TF.pad(img, [pad_left, pad_left, pad_right, pad_right], fill=0)

        # Apply rotation
        if params["rot"] != 0.0:
            img = TF.rotate(img, params["rot"], fill=0)

        # Apply translation via affine
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
        img = TF.center_crop(img, self.img_size)

        return img

    def _params_to_tensor(self, params: dict) -> torch.Tensor:
        """Convert parameters to tensor for action encoding."""
        # Encode rotation as sin/cos
        theta = math.radians(params["rot"])
        sin_rot = math.sin(theta)
        cos_rot = math.cos(theta)

        # Normalize translation
        max_trans = max(abs(self.trans_range[0]), abs(self.trans_range[1]))
        tx_norm = params["tx"] / max_trans if max_trans > 0 else 0.0
        ty_norm = params["ty"] / max_trans if max_trans > 0 else 0.0

        # Log scale (centered around 0 for scale=1)
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
        theta = math.radians(delta_rot)
        sin_rot = math.sin(theta)
        cos_rot = math.cos(theta)

        # Relative translation
        max_trans = max(abs(self.trans_range[0]), abs(self.trans_range[1]))
        delta_tx = params_next["tx"] - params_curr["tx"]
        delta_ty = params_next["ty"] - params_curr["ty"]
        tx_norm = delta_tx / max_trans if max_trans > 0 else 0.0
        ty_norm = delta_ty / max_trans if max_trans > 0 else 0.0

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
        img_tensor = self.to_tensor(img)  # (3, 32, 32)

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

        # Original image (normalized)
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


# Convenience aliases
class CIFAR10AffineSequence(CIFARAffineSequence):
    """CIFAR-10 with controlled affine transforms."""

    def __init__(self, root: str, split: str, seq_len: int, **kwargs):
        super().__init__(root, split, seq_len, dataset="cifar10", **kwargs)


class CIFAR100AffineSequence(CIFARAffineSequence):
    """CIFAR-100 with controlled affine transforms."""

    def __init__(self, root: str, split: str, seq_len: int, **kwargs):
        super().__init__(root, split, seq_len, dataset="cifar100", **kwargs)

