# seq-jepa research workspace

This repo hosts the upstream seq-JEPA code alongside research notes and additive experiments.

## Repo structure
- seq-jepa/: upstream code snapshot
  - src/: core model, training, and utilities
  - scripts/: entrypoints and helper scripts
  - data/: local datasets
- docs/: planning notes and step breakdowns
- assets/, static/, index.html: supporting assets

## Modification policy
- Do not edit existing files under `seq-jepa/src` or `seq-jepa/scripts`.
- Implement changes as new, separate modules and wrapper entrypoints so the original code remains runnable.
- Document changes in `docs` and keep `docs/README.md` up to date.

## Quick start (local)
- Create a venv and install requirements (see `docs/020-practical-setup.md`).
- If macOS cannot install `requirements.txt`, use `requirements-local.txt`.
- Copy `.env.example` to `.env` and set `WANDB_API_KEY` if you want logging.
- See `docs/021-metrics-and-gates.md` for metric definitions and gate thresholds.
- Use the wrapper runner and configs:
  - `python train.py --config configs/quick/cifar100_aug_smoke.yaml`
  - `python train.py --config configs/quick/cifar100_aug_quick.yaml`
  - `python train.py --config configs/quick/cifar10_rot_smoke.yaml` (fast fallback)
  - `python train.py --config configs/quick/cifar10_rot_quick.yaml` (fast fallback)
