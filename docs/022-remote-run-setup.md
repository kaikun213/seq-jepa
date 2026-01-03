# Remote run setup (Vast.ai + SLURM)

## Goal
- Run Step 1 baseline reproduction with minimal overhead using the lightest wired dataset.
- Keep configs consistent across local and remote runs.

## Dataset note
- The wrapper runner currently supports `dataset.name=cifar10_rot` only.
- STL-10 is not wired yet, so use CIFAR-10 rotation for Step 1 unless we add an STL-10 dataset.

## Recommended configs
- Baseline: `configs/remote/cifar10_rot_baseline.yaml`
- Shakedown: `configs/quick/cifar10_rot_smoke.yaml` or `configs/quick/cifar10_rot_quick.yaml`

## Common prerequisites
- W&B API key: set `WANDB_API_KEY` or place it in `.env` (loaded by `train.py`).
- Python env with `requirements.txt` installed.
- CIFAR-10 will download into `data/` on first run. If compute nodes have no internet, pre-populate `data/` before submission.

## Vast.ai
- Suggested instance: 1x GPU (RTX 3060 12GB or A10 24GB), 8 vCPU, 30-60GB disk.
- Run:
  - `git clone` the repo
  - `pip install -r requirements.txt`
  - `scripts/vast/run_cifar10_rot_baseline.sh`
- Optional: run the smoke or quick config first to validate the environment.

## SLURM
- Template: `scripts/slurm/seqjepa_cifar10_rot_baseline.sbatch`
- Fill in partition/account, module loads, and env activation.
- Submit with `sbatch scripts/slurm/seqjepa_cifar10_rot_baseline.sbatch`

## Rough runtimes (very approximate)
- Mac M3 Max: quick config ~8 min (based on recent 5-10 min runs).
- Vast.ai 1x RTX 3060: quick ~1-2 min, baseline ~4-6 min.
- First run adds data download time (~1-3 min) if not cached.

## If you hit issues
- CUDA OOM: lower `dataset.batch_size` to 64.
- Slow dataloading: increase `dataset.num_workers` to 4-8 on GPU nodes.
