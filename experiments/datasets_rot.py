import math
import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F


class CIFAR10RotationSequence(Dataset):
    """CIFAR-10 views under discrete rotations for seq-JEPA transforms."""

    def __init__(
        self,
        root: str,
        split: str,
        seq_len: int,
        rotations: List[int],
        normalize: bool = True,
        download: bool = True,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if not rotations:
            raise ValueError("rotations must be non-empty")

        self.seq_len = seq_len
        self.num_views = seq_len + 1
        self.rotations = list(rotations)

        self.base = torchvision.datasets.CIFAR10(
            root=root,
            train=(split == "train"),
            download=download,
        )

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        self.transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)]) if normalize else T.ToTensor()

    def __len__(self) -> int:
        return len(self.base)

    def _angle_to_sincos(self, angle_deg: float) -> torch.Tensor:
        theta = math.radians(angle_deg)
        return torch.tensor([math.sin(theta), math.cos(theta)], dtype=torch.float32)

    def _sample_angles(self) -> List[int]:
        # Fix the first view to 0 deg to define an absolute reference.
        angles = [0]
        for _ in range(self.seq_len):
            angles.append(random.choice(self.rotations))
        return angles

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        img, label = self.base[idx]
        angles = self._sample_angles()
        views = []
        actions_abs = []
        for angle in angles:
            rotated = F.rotate(img, angle)
            views.append(self.transform(rotated))
            actions_abs.append(self._angle_to_sincos(angle))

        actions_rel = []
        for i in range(self.seq_len):
            delta = angles[i + 1] - angles[i]
            delta = ((delta + 180) % 360) - 180
            actions_rel.append(self._angle_to_sincos(delta))

        stacked_views = torch.stack(views, dim=0)
        stacked_actions_abs = torch.stack(actions_abs, dim=0)
        stacked_actions_rel = torch.stack(actions_rel, dim=0)
        orig_img = self.transform(img)
        return stacked_views, stacked_actions_abs, stacked_actions_rel, label, orig_img
