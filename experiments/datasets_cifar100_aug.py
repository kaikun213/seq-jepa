import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T

from augmentations import (
    ParamCompose,
    ParamRandomResizedCrop,
    ParamColorJitter,
    ParamGaussianBlur,
    ParamSolarize,
)


class CIFAR100AugSequence(Dataset):
    """CIFAR-100 views with augmentation parameter actions."""

    def __init__(
        self,
        root: str,
        split: str,
        seq_len: int,
        download: bool = True,
        aug: bool = True,
        no_blur: bool = False,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if not aug:
            raise ValueError("aug must be True for CIFAR100AugSequence")

        self.seq_len = seq_len
        self.num_views = seq_len + 1
        self.aug = aug
        self.no_blur = no_blur

        self.dataset = torchvision.datasets.CIFAR100(
            root=root,
            train=(split == "train"),
            download=download,
            transform=None,
        )

        norm_params = {"mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761]}

        self.orig_img_trans = T.Compose(
            [
                T.Resize((32, 32)),
                T.ToTensor(),
                T.Normalize(norm_params["mean"], norm_params["std"]),
            ]
        )

        crop = ParamRandomResizedCrop(
            pflip=0.5,
            size=(32, 32),
            scale=(0.2, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
        )
        jitter = ParamColorJitter(
            pjitter=0.8,
            pgray=0.2,
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1,
        )
        solarize_obs = ParamSolarize(threshold=128.0, p=0.0)
        solarize_pred = ParamSolarize(threshold=128.0, p=0.2)

        if self.no_blur:
            obs_params = [crop, jitter, solarize_obs]
            pred_params = [crop, jitter, solarize_pred]
        else:
            blur_obs = ParamGaussianBlur(
                pblur=1.0,
                kernel_size=tuple(2 * round((x - 10) / 20) + 1 for x in (32, 32)),
                sigma=(0.1, 2.0),
            )
            blur_pred = ParamGaussianBlur(
                pblur=0.1,
                kernel_size=tuple(2 * round((x - 10) / 20) + 1 for x in (32, 32)),
                sigma=(0.1, 2.0),
            )
            obs_params = [crop, jitter, blur_obs, solarize_obs]
            pred_params = [crop, jitter, blur_pred, solarize_pred]

        self.trans = ParamCompose(
            obs_params,
            [T.ToTensor(), T.Normalize(norm_params["mean"], norm_params["std"])],
        )
        self.pred_trans = ParamCompose(
            pred_params,
            [T.ToTensor(), T.Normalize(norm_params["mean"], norm_params["std"])],
        )
        self.num_params = self.trans.nb_params
        self.num_classes = 100

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, label = self.dataset[idx]

        augmented_images = []
        augmented_params = []
        orig_img = self.orig_img_trans(img)

        for _ in range(self.num_views - 1):
            aug_img, aug_params = self.trans(img)
            augmented_images.append(aug_img)
            augmented_params.append(aug_params)

        aug_img, aug_params = self.pred_trans(img)
        augmented_images.append(aug_img)
        augmented_params.append(aug_params)

        return (
            torch.stack(augmented_images, dim=0),
            torch.stack(augmented_params, dim=0),
            label,
            orig_img,
        )
