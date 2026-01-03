#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Seq-JEPA local runner (wrapper)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def ensure_repo_paths() -> Path:
    repo_root = Path(__file__).resolve().parent
    seq_jepa_src = repo_root / "seq-jepa" / "src"
    if str(seq_jepa_src) not in sys.path:
        sys.path.insert(0, str(seq_jepa_src))
    return repo_root


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(requested: str):
    import torch

    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cap_subset_size(subset_size, max_steps, batch_size):
    if not max_steps:
        return subset_size
    max_samples = max_steps * batch_size
    if not subset_size or subset_size <= 0:
        return max_samples
    return min(subset_size, max_samples)


def make_subset(dataset, subset_size, seed):
    from torch.utils.data import Subset

    if not subset_size or subset_size <= 0 or subset_size >= len(dataset):
        return dataset
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return Subset(dataset, indices[:subset_size])


def setup_wandb(cfg: dict, run_name: str, output_dir: Path):
    wandb_cfg = cfg.get("logging", {}).get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None

    mode = wandb_cfg.get("mode")
    if mode:
        os.environ["WANDB_MODE"] = mode
    os.environ.setdefault("WANDB__SERVICE_WAIT", "300")
    os.environ.setdefault("WANDB_DIR", str(output_dir))
    os.environ.setdefault("WANDB_CACHE_DIR", str(output_dir / "wandb_cache"))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(output_dir / "wandb_config"))
    os.environ.setdefault("TMPDIR", str(output_dir / "tmp"))
    os.environ.setdefault("TEMP", str(output_dir / "tmp"))
    os.environ.setdefault("TMP", str(output_dir / "tmp"))

    (output_dir / "wandb_cache").mkdir(parents=True, exist_ok=True)
    (output_dir / "wandb_config").mkdir(parents=True, exist_ok=True)
    (output_dir / "tmp").mkdir(parents=True, exist_ok=True)

    api_key = wandb_cfg.get("api_key")
    if api_key:
        os.environ.setdefault("WANDB_API_KEY", api_key)

    import wandb

    if api_key:
        wandb.login(key=api_key, relogin=False)

    run = wandb.init(
        project=wandb_cfg.get("project"),
        entity=wandb_cfg.get("entity"),
        group=wandb_cfg.get("group"),
        name=run_name,
        config=cfg,
        dir=str(output_dir),
    )
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")
    return run


def save_metrics(metrics, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "metrics.jsonl"
    csv_path = output_dir / "metrics.csv"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in metrics:
            f.write(json.dumps(row) + "\n")

    if metrics:
        keys = sorted(metrics[0].keys())
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for row in metrics:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")

    try:
        import matplotlib.pyplot as plt

        if metrics:
            epochs = [m["epoch"] for m in metrics]
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.plot(epochs, [m["ep_loss"] for m in metrics], label="ep_loss")
            ax.plot(epochs, [m["online_linacc_test"] for m in metrics], label="linacc_test")
            ax.plot(epochs, [m["online_r2_test"] for m in metrics], label="equi_r2_test")
            ax.set_xlabel("epoch")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "metrics.png", dpi=160)
            plt.close(fig)
    except Exception:
        pass


def evaluate_gates(latest: dict, gating_cfg: dict) -> tuple[bool, list[str]]:
    reasons = []
    if not gating_cfg:
        return True, reasons

    lin_min = gating_cfg.get("linacc_test_min")
    if lin_min is not None and latest.get("online_linacc_test") is not None:
        if latest["online_linacc_test"] < lin_min:
            reasons.append(f"linacc_test<{lin_min}")

    r2_min = gating_cfg.get("r2_test_min")
    if r2_min is not None and latest.get("online_r2_test") is not None:
        if latest["online_r2_test"] < r2_min:
            reasons.append(f"r2_test<{r2_min}")

    loss_max = gating_cfg.get("loss_max")
    if loss_max is not None and latest.get("ep_loss") is not None:
        if latest["ep_loss"] > loss_max:
            reasons.append(f"ep_loss>{loss_max}")

    leakage_linacc_max = gating_cfg.get("leakage_linacc_test_max")
    if leakage_linacc_max is not None and latest.get("leakage_linacc_test") is not None:
        if latest["leakage_linacc_test"] > leakage_linacc_max:
            reasons.append(f"leakage_linacc_test>{leakage_linacc_max}")

    leakage_r2_max = gating_cfg.get("leakage_r2_test_max")
    if leakage_r2_max is not None and latest.get("leakage_r2_test") is not None:
        if latest["leakage_r2_test"] > leakage_r2_max:
            reasons.append(f"leakage_r2_test>{leakage_r2_max}")

    lin_gap_min = gating_cfg.get("leakage_linacc_gap_min")
    if lin_gap_min is not None:
        online_lin = latest.get("online_linacc_test")
        leak_lin = latest.get("leakage_linacc_test")
        if online_lin is not None and leak_lin is not None:
            if (online_lin - leak_lin) < lin_gap_min:
                reasons.append(f"linacc_gap<{lin_gap_min}")

    r2_gap_min = gating_cfg.get("leakage_r2_gap_min")
    if r2_gap_min is not None:
        online_r2 = latest.get("online_r2_test")
        leak_r2 = latest.get("leakage_r2_test")
        if online_r2 is not None and leak_r2 is not None:
            if (online_r2 - leak_r2) < r2_gap_min:
                reasons.append(f"r2_gap<{r2_gap_min}")

    return len(reasons) == 0, reasons


def _compute_r2(targets, preds) -> float:
    import numpy as np
    from sklearn.metrics import r2_score

    if not targets:
        return float("nan")
    targets_np = np.concatenate(targets, axis=0)
    preds_np = np.concatenate(preds, axis=0)
    return r2_score(targets_np, preds_np)


def train_one_epoch_local(
    model,
    linprobe,
    equiprobe,
    leakage_linprobe,
    leakage_actprobe,
    optimizer,
    optimizer_lin,
    optimizer_equi,
    optimizer_leakage_lin,
    optimizer_leakage_act,
    device,
    train_loader,
    ema: bool,
) -> dict:
    import torch
    from sklearn.metrics import accuracy_score

    model.train()
    linprobe.train()
    equiprobe.train()

    ce_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()

    epoch_loss = 0.0
    total_samples = 0

    lin_preds = []
    lin_targets = []
    leak_lin_preds = []
    leak_lin_targets = []
    equi_preds = []
    equi_targets = []
    leak_act_preds = []
    leak_act_targets = []
    lin_loss_sum = 0.0
    equi_loss_sum = 0.0
    leak_lin_loss_sum = 0.0
    leak_act_loss_sum = 0.0

    for batch, actions_abs, actions_rel, labels, _ in train_loader:
        batch = batch.to(device)
        actions_abs = actions_abs.to(device)
        actions_rel = actions_rel.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss, agg_out, z1, z2 = model(
            batch[:, :-1],
            batch[:, -1],
            actions_abs[:, :-1],
            actions_abs[:, -1],
            rel_latents=actions_rel,
        )
        loss.backward()
        optimizer.step()
        if ema:
            model.update_moving_average()

        batch_size = labels.size(0)
        epoch_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        optimizer_lin.zero_grad()
        logits = linprobe(agg_out.detach())
        lin_loss = ce_loss(logits, labels)
        lin_loss.backward()
        optimizer_lin.step()
        lin_loss_sum += float(lin_loss.item())
        lin_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
        lin_targets.extend(labels.detach().cpu().numpy())

        optimizer_leakage_lin.zero_grad()
        leak_logits = leakage_linprobe(z1.detach())
        leak_lin_loss = ce_loss(leak_logits, labels)
        leak_lin_loss.backward()
        optimizer_leakage_lin.step()
        leak_lin_loss_sum += float(leak_lin_loss.item())
        leak_lin_preds.extend(torch.argmax(leak_logits, dim=1).detach().cpu().numpy())
        leak_lin_targets.extend(labels.detach().cpu().numpy())

        optimizer_equi.zero_grad()
        embs_concat = torch.cat((z1.detach(), z2.detach()), dim=1)
        pred_rel = equiprobe(embs_concat)
        target_rel = actions_rel[:, 0]
        equi_loss = mse_loss(pred_rel, target_rel)
        equi_loss.backward()
        optimizer_equi.step()
        equi_loss_sum += float(equi_loss.item())
        equi_preds.append(pred_rel.detach().cpu().numpy())
        equi_targets.append(target_rel.detach().cpu().numpy())

        optimizer_leakage_act.zero_grad()
        leak_pred_rel = leakage_actprobe(agg_out.detach())
        leak_act_loss = mse_loss(leak_pred_rel, target_rel)
        leak_act_loss.backward()
        optimizer_leakage_act.step()
        leak_act_loss_sum += float(leak_act_loss.item())
        leak_act_preds.append(leak_pred_rel.detach().cpu().numpy())
        leak_act_targets.append(target_rel.detach().cpu().numpy())

    lin_acc = accuracy_score(lin_targets, lin_preds) if lin_targets else float("nan")
    leak_lin_acc = accuracy_score(leak_lin_targets, leak_lin_preds) if leak_lin_targets else float("nan")
    equi_r2 = _compute_r2(equi_targets, equi_preds)
    leak_act_r2 = _compute_r2(leak_act_targets, leak_act_preds)

    results = {
        "ep_loss": epoch_loss / max(total_samples, 1),
        "online_linacc_train": lin_acc * 100.0,
        "online_r2_train": equi_r2,
        "online_linloss_train": lin_loss_sum / max(len(train_loader), 1),
        "online_r2_loss_train": equi_loss_sum / max(len(train_loader), 1),
        "leakage_linacc_train": leak_lin_acc * 100.0,
        "leakage_r2_train": leak_act_r2,
        "leakage_linloss_train": leak_lin_loss_sum / max(len(train_loader), 1),
        "leakage_r2_loss_train": leak_act_loss_sum / max(len(train_loader), 1),
    }
    return results


def eval_one_epoch_local(
    model,
    linprobe,
    equiprobe,
    leakage_linprobe,
    leakage_actprobe,
    device,
    val_loader,
) -> dict:
    import torch
    from sklearn.metrics import accuracy_score

    model.eval()
    linprobe.eval()
    equiprobe.eval()

    ce_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()

    lin_preds = []
    lin_targets = []
    leak_lin_preds = []
    leak_lin_targets = []
    equi_preds = []
    equi_targets = []
    leak_act_preds = []
    leak_act_targets = []
    lin_loss_sum = 0.0
    equi_loss_sum = 0.0
    leak_lin_loss_sum = 0.0
    leak_act_loss_sum = 0.0

    with torch.no_grad():
        for batch, actions_abs, actions_rel, labels, _ in val_loader:
            batch = batch.to(device)
            actions_abs = actions_abs.to(device)
            actions_rel = actions_rel.to(device)
            labels = labels.to(device)

            _, agg_out, z1, z2 = model(
                batch[:, :-1],
                batch[:, -1],
                actions_abs[:, :-1],
                actions_abs[:, -1],
                rel_latents=actions_rel,
            )

            logits = linprobe(agg_out)
            lin_loss = ce_loss(logits, labels)
            lin_loss_sum += float(lin_loss.item())
            lin_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
            lin_targets.extend(labels.detach().cpu().numpy())

            leak_logits = leakage_linprobe(z1)
            leak_lin_loss = ce_loss(leak_logits, labels)
            leak_lin_loss_sum += float(leak_lin_loss.item())
            leak_lin_preds.extend(torch.argmax(leak_logits, dim=1).detach().cpu().numpy())
            leak_lin_targets.extend(labels.detach().cpu().numpy())

            embs_concat = torch.cat((z1, z2), dim=1)
            pred_rel = equiprobe(embs_concat)
            target_rel = actions_rel[:, 0]
            equi_loss = mse_loss(pred_rel, target_rel)
            equi_loss_sum += float(equi_loss.item())
            equi_preds.append(pred_rel.detach().cpu().numpy())
            equi_targets.append(target_rel.detach().cpu().numpy())

            leak_pred_rel = leakage_actprobe(agg_out)
            leak_act_loss = mse_loss(leak_pred_rel, target_rel)
            leak_act_loss_sum += float(leak_act_loss.item())
            leak_act_preds.append(leak_pred_rel.detach().cpu().numpy())
            leak_act_targets.append(target_rel.detach().cpu().numpy())

    lin_acc = accuracy_score(lin_targets, lin_preds) if lin_targets else float("nan")
    leak_lin_acc = accuracy_score(leak_lin_targets, leak_lin_preds) if leak_lin_targets else float("nan")
    equi_r2 = _compute_r2(equi_targets, equi_preds)
    leak_act_r2 = _compute_r2(leak_act_targets, leak_act_preds)

    results = {
        "online_linacc_test": lin_acc * 100.0,
        "online_r2_test": equi_r2,
        "online_linloss_test": lin_loss_sum / max(len(val_loader), 1),
        "online_r2_loss_test": equi_loss_sum / max(len(val_loader), 1),
        "leakage_linacc_test": leak_lin_acc * 100.0,
        "leakage_r2_test": leak_act_r2,
        "leakage_linloss_test": leak_lin_loss_sum / max(len(val_loader), 1),
        "leakage_r2_loss_test": leak_act_loss_sum / max(len(val_loader), 1),
    }
    return results


def main():
    args = parse_args()
    cfg = load_config(args.config)

    run_cfg = cfg.get("run", {})
    mps_fallback = run_cfg.get("mps_fallback", False)
    if mps_fallback:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    repo_root = ensure_repo_paths()
    load_dotenv(repo_root / ".env")

    import torch
    from torch.utils.data import DataLoader

    from utils import seed_worker
    from models import SeqJEPA_Transforms
    from experiments.datasets_rot import CIFAR10RotationSequence

    seed = int(run_cfg.get("seed", 42))
    seed_everything(seed)

    device = pick_device(run_cfg.get("device", "auto"))
    print(f"Using device: {device}")

    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    optim_cfg = cfg.get("optim", {})

    if dataset_cfg.get("name") != "cifar10_rot":
        raise ValueError("Only dataset.name='cifar10_rot' is supported by this runner.")

    seq_len = int(dataset_cfg.get("seq_len", 2))
    rotations = dataset_cfg.get("rotations", [0, 90, 180, 270])
    data_root = dataset_cfg.get("data_root", str(repo_root / "data"))
    download = bool(dataset_cfg.get("download", True))

    train_ds = CIFAR10RotationSequence(
        root=data_root,
        split="train",
        seq_len=seq_len,
        rotations=rotations,
        download=download,
    )
    val_ds = CIFAR10RotationSequence(
        root=data_root,
        split="test",
        seq_len=seq_len,
        rotations=rotations,
        download=download,
    )

    batch_size = int(dataset_cfg.get("batch_size", 32))
    max_steps = run_cfg.get("max_steps")
    max_val_steps = run_cfg.get("max_val_steps")

    subset_train = cap_subset_size(dataset_cfg.get("subset_train"), max_steps, batch_size)
    subset_val = cap_subset_size(dataset_cfg.get("subset_val"), max_val_steps, batch_size)

    subset_seed = int(dataset_cfg.get("subset_seed", seed))
    train_ds = make_subset(train_ds, subset_train, subset_seed)
    val_ds = make_subset(val_ds, subset_val, subset_seed + 1)

    num_workers = int(dataset_cfg.get("num_workers", 0))
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=bool(run_cfg.get("drop_last", True)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
    )

    num_classes = 10
    act_latentdim = int(model_cfg.get("act_latentdim", len(rotations)))

    model = SeqJEPA_Transforms(
        model_cfg.get("img_size", 32),
        bool(model_cfg.get("ema", False)),
        n_channels=int(model_cfg.get("n_channels", 3)),
        num_classes=num_classes,
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

    external_linprobe = torch.nn.Sequential(
        torch.nn.Linear(model.emb_dim, num_classes)
    ).to(device)
    external_equiprobe = torch.nn.Sequential(
        torch.nn.Linear(model.res_out_dim * 2, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, act_latentdim),
    ).to(device)
    leakage_linprobe = torch.nn.Sequential(
        torch.nn.Linear(model.res_out_dim, num_classes)
    ).to(device)
    leakage_actprobe = torch.nn.Sequential(
        torch.nn.Linear(model.emb_dim, act_latentdim)
    ).to(device)

    lr = float(optim_cfg.get("lr", 0.001))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    probe_lr = float(optim_cfg.get("probe_lr", lr))
    optimizer_linprobe = torch.optim.Adam(external_linprobe.parameters(), lr=probe_lr)
    optimizer_equiprobe = torch.optim.Adam(external_equiprobe.parameters(), lr=probe_lr)
    optimizer_leakage_lin = torch.optim.Adam(leakage_linprobe.parameters(), lr=probe_lr)
    optimizer_leakage_act = torch.optim.Adam(leakage_actprobe.parameters(), lr=probe_lr)

    epochs = int(run_cfg.get("epochs", 1))
    run_name = run_cfg.get("name", Path(args.config).stem)
    output_dir = Path(run_cfg.get("output_dir", str(repo_root / "runs" / run_name)))
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    wandb_run = setup_wandb(cfg, run_name, output_dir)

    metrics = []
    for epoch in range(epochs):
        start_time = time.time()
        train_result = train_one_epoch_local(
            model,
            external_linprobe,
            external_equiprobe,
            leakage_linprobe,
            leakage_actprobe,
            optimizer,
            optimizer_linprobe,
            optimizer_equiprobe,
            optimizer_leakage_lin,
            optimizer_leakage_act,
            device,
            train_loader,
            bool(model_cfg.get("ema", False)),
        )
        val_result = eval_one_epoch_local(
            model,
            external_linprobe,
            external_equiprobe,
            leakage_linprobe,
            leakage_actprobe,
            device,
            val_loader,
        )
        result_row = {**train_result, **val_result}
        result_row.update({
            "epoch": epoch + 1,
            "ep_time": time.time() - start_time,
        })
        result_row["wall_time"] = time.time() - start_time
        metrics.append(result_row)
        print(
            "Epoch {}/{} - loss {:.4f} linacc_test {:.2f} r2_test {:.4f}".format(
                epoch + 1,
                epochs,
                result_row["ep_loss"],
                result_row["online_linacc_test"],
                result_row["online_r2_test"],
            )
        )
        if wandb_run:
            wandb_run.log(result_row, step=epoch)

    save_metrics(metrics, output_dir)

    if wandb_run and metrics:
        import wandb
        gating_cfg = cfg.get("gating", {})
        gate_ok, gate_reasons = evaluate_gates(metrics[-1], gating_cfg)
        wandb_run.summary["gate_pass"] = gate_ok
        if gate_reasons:
            wandb_run.summary["gate_reasons"] = ", ".join(gate_reasons)

        metrics_png = output_dir / "metrics.png"
        if metrics_png.exists():
            wandb_run.log({"metrics_plot": wandb.Image(str(metrics_png))}, step=epochs)

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
