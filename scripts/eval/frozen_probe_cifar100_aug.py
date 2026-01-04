#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Frozen probe eval for CIFAR-100 aug")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint (optional)")
    parser.add_argument("--output", type=str, default="", help="Output JSON path (optional)")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_repo_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    seq_jepa_src = repo_root / "seq-jepa" / "src"
    if str(seq_jepa_src) not in sys.path:
        sys.path.insert(0, str(seq_jepa_src))
    return repo_root


def pick_device(requested: str):
    import torch

    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def unpack_batch(batch, device, action_norm: bool):
    import torch

    if len(batch) == 5:
        views, actions_abs, actions_rel, labels, _ = batch
    elif len(batch) == 4:
        views, actions_abs, labels, _ = batch
        actions_rel = None
    else:
        raise ValueError("Unexpected batch structure from dataset.")

    views = views.to(device)
    actions_abs = actions_abs.to(device)
    labels = labels.to(device)

    if action_norm:
        actions_abs = torch.nn.functional.normalize(actions_abs, p=2, dim=1)

    if actions_rel is None:
        actions_rel = actions_abs[:, 1:] - actions_abs[:, :-1]
    else:
        actions_rel = actions_rel.to(device)

    return views, actions_abs, actions_rel, labels


def compute_r2(targets, preds) -> float:
    import numpy as np

    if not targets:
        return float("nan")
    y_true = np.concatenate(targets, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    mask = ss_tot > 0
    if not mask.any():
        return float("nan")
    r2 = 1.0 - (ss_res[mask] / ss_tot[mask])
    return float(r2.mean())


def main():
    args = parse_args()
    cfg = load_config(args.config)
    repo_root = ensure_repo_paths()

    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import accuracy_score

    from models import SeqJEPA_Transforms
    from experiments.datasets_cifar100_aug import CIFAR100AugSequence

    run_cfg = cfg.get("run", {})
    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    eval_cfg = cfg.get("eval", {})

    device = pick_device(run_cfg.get("device", "auto"))

    data_root = dataset_cfg.get("data_root", str(repo_root / "data"))
    download = bool(dataset_cfg.get("download", True))
    action_norm = bool(dataset_cfg.get("action_norm", True))
    use_rel_latents = bool(dataset_cfg.get("use_rel_latents", False))

    eval_seq_len = int(eval_cfg.get("seq_len", dataset_cfg.get("seq_len", 3)))

    train_ds = CIFAR100AugSequence(
        root=data_root,
        split="train",
        seq_len=eval_seq_len,
        download=download,
        aug=True,
        no_blur=bool(dataset_cfg.get("no_blur", False)),
    )
    val_ds = CIFAR100AugSequence(
        root=data_root,
        split="test",
        seq_len=eval_seq_len,
        download=download,
        aug=True,
        no_blur=bool(dataset_cfg.get("no_blur", False)),
    )

    batch_size = int(eval_cfg.get("batch_size", 256))
    num_workers = int(eval_cfg.get("num_workers", 4))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    act_latentdim = int(model_cfg.get("act_latentdim", train_ds.num_params))

    model = SeqJEPA_Transforms(
        model_cfg.get("img_size", 32),
        bool(model_cfg.get("ema", False)),
        n_channels=int(model_cfg.get("n_channels", 3)),
        num_classes=100,
        num_heads=int(model_cfg.get("num_heads", 4)),
        num_enc_layers=int(model_cfg.get("num_enc_layers", 3)),
        act_cond=int(model_cfg.get("act_cond", 1)),
        pred_hidden=int(model_cfg.get("pred_hidden", 1024)),
        act_projdim=int(model_cfg.get("act_projdim", 128)),
        act_latentdim=act_latentdim,
        learn_act_emb=int(model_cfg.get("learn_act_emb", 1)),
        ema_decay=float(model_cfg.get("ema_decay", 0.996)),
        cifar_resnet=bool(model_cfg.get("cifar_resnet", True)),
    ).to(device)

    output_dir = Path(run_cfg.get("output_dir", str(repo_root / "runs" / run_cfg.get("name", "run"))))
    checkpoint_path = args.checkpoint or str(output_dir / "checkpoints" / "last.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    clf = torch.nn.Linear(model.emb_dim, 100).to(device)
    reg_crop = torch.nn.Linear(model.res_out_dim * 2, 4).to(device)
    reg_jitter = torch.nn.Linear(model.res_out_dim * 2, 4).to(device)
    reg_blur = torch.nn.Linear(model.res_out_dim * 2, 1).to(device)

    probe_epochs = int(eval_cfg.get("probe_epochs", 100))
    probe_lr = float(eval_cfg.get("probe_lr", 0.1))

    clf_opt = torch.optim.SGD(clf.parameters(), lr=probe_lr, momentum=0.9)
    reg_opt = torch.optim.SGD(
        list(reg_crop.parameters()) + list(reg_jitter.parameters()) + list(reg_blur.parameters()),
        lr=probe_lr,
        momentum=0.9,
    )

    ce_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()

    for epoch in range(probe_epochs):
        clf.train()
        reg_crop.train()
        reg_jitter.train()
        reg_blur.train()
        for batch in train_loader:
            views, actions_abs, actions_rel, labels = unpack_batch(batch, device, action_norm)
            rel_latents = actions_rel if use_rel_latents else None
            with torch.no_grad():
                _, agg_out, z1, z2 = model(
                    views[:, :-1],
                    views[:, -1],
                    actions_abs[:, :-1],
                    actions_abs[:, -1],
                    rel_latents=rel_latents,
                )
            clf_opt.zero_grad()
            logits = clf(agg_out.detach())
            loss = ce_loss(logits, labels)
            loss.backward()
            clf_opt.step()

            reg_opt.zero_grad()
            embs_concat = torch.cat((z1.detach(), z2.detach()), dim=1)
            target = actions_rel[:, 0]
            pred_crop = reg_crop(embs_concat)
            pred_jitter = reg_jitter(embs_concat)
            pred_blur = reg_blur(embs_concat)
            loss_reg = (
                mse_loss(pred_crop, target[:, :4])
                + mse_loss(pred_jitter, target[:, 4:8])
                + mse_loss(pred_blur, target[:, 8:9])
            )
            loss_reg.backward()
            reg_opt.step()

    clf.eval()
    reg_crop.eval()
    reg_jitter.eval()
    reg_blur.eval()

    lin_preds = []
    lin_targets = []
    crop_preds = []
    crop_targets = []
    jitter_preds = []
    jitter_targets = []
    blur_preds = []
    blur_targets = []

    with torch.no_grad():
        for batch in val_loader:
            views, actions_abs, actions_rel, labels = unpack_batch(batch, device, action_norm)
            rel_latents = actions_rel if use_rel_latents else None
            _, agg_out, z1, z2 = model(
                views[:, :-1],
                views[:, -1],
                actions_abs[:, :-1],
                actions_abs[:, -1],
                rel_latents=rel_latents,
            )
            logits = clf(agg_out)
            lin_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
            lin_targets.extend(labels.detach().cpu().numpy())

            embs_concat = torch.cat((z1, z2), dim=1)
            target = actions_rel[:, 0]
            crop_preds.append(reg_crop(embs_concat).detach().cpu().numpy())
            crop_targets.append(target[:, :4].detach().cpu().numpy())
            jitter_preds.append(reg_jitter(embs_concat).detach().cpu().numpy())
            jitter_targets.append(target[:, 4:8].detach().cpu().numpy())
            blur_preds.append(reg_blur(embs_concat).detach().cpu().numpy())
            blur_targets.append(target[:, 8:9].detach().cpu().numpy())

    lin_acc = accuracy_score(lin_targets, lin_preds) * 100.0 if lin_targets else float("nan")
    crop_r2 = compute_r2(crop_targets, crop_preds)
    jitter_r2 = compute_r2(jitter_targets, jitter_preds)
    blur_r2 = compute_r2(blur_targets, blur_preds)

    results = {
        "linacc_top1": lin_acc,
        "crop_r2": crop_r2,
        "jitter_r2": jitter_r2,
        "blur_r2": blur_r2,
        "eval_seq_len": eval_seq_len,
        "probe_epochs": probe_epochs,
    }

    output_path = args.output
    if not output_path:
        output_path = str(output_dir / "eval" / "frozen_probe_metrics.json")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
