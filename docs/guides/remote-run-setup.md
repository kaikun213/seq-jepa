# Remote run setup (Vast.ai + SLURM)

## Goal
- Run Step 1 baseline reproduction with minimal overhead using the lightest wired dataset.
- Keep configs consistent across local and remote runs.

## Dataset note
- The wrapper runner supports `dataset.name=cifar100_aug` (paper-aligned transforms) and `dataset.name=cifar10_rot` (fast sanity).
- Step 1 baseline uses CIFAR-100 augmentation actions; CIFAR-10 rotations stay as a quick fallback.

## Recommended configs
- Baseline (CIFAR-100 aug): `configs/remote/cifar100_aug_baseline.yaml`
- Shakedown (CIFAR-100 aug): `configs/quick/cifar100_aug_smoke.yaml` or `configs/quick/cifar100_aug_quick.yaml`
- Optional fast fallback: `configs/quick/cifar10_rot_smoke.yaml` or `configs/quick/cifar10_rot_quick.yaml`

## Common prerequisites
- W&B API key: set `WANDB_API_KEY` or place it in `.env` (loaded by `train.py`).
- Python env with `requirements.txt` installed.
- CIFAR-100 will download into `data/` on first run. If compute nodes have no internet, pre-populate `data/` before submission.
  
## Private repo access (Vast.ai)
- Preferred: add your GitHub SSH key to the instance (`~/.ssh/id_ed25519`) and clone via SSH.
- HTTPS fallback: set `GITHUB_TOKEN` and clone with `https://github.com/kaikun213/seq-jepa-streaming.git`.
- If using the Vast.ai CLI, set `VAST_API_KEY`.

## Vast.ai
- Suggested instance: 1x GPU (RTX 3060 12GB or A10 24GB), 8 vCPU, 30-60GB disk.
- Run:
  - `git clone` the repo
  - `pip install -r requirements.txt`
  - `scripts/vast/run_cifar100_aug_baseline.sh`
- Optional: run the CIFAR-100 smoke/quick config first to validate the environment.

### Automated CLI run (recommended)
Use the launcher script to search offers and create an instance with the onstart script:

```bash
export GITHUB_SSH_KEY_PATH=~/.ssh/private_github
scripts/vast/launch_cifar100_baseline.sh
```

Notes:
- The script prefers SSH. If you instead want HTTPS, set `GITHUB_TOKEN` and unset `GITHUB_SSH_KEY_PATH`.
- Default GPU list is `RTX_3060`, `A10`, `RTX_3070`, `RTX_3080`, `RTX_3090`. Override with `VAST_GPU_LIST`.
- The onstart script clones the repo, installs requirements, and runs the baseline config.

## SLURM
- Template: `scripts/slurm/seqjepa_cifar10_rot_baseline.sbatch` (update to CIFAR-100 when needed)
- Fill in partition/account, module loads, and env activation.
- Submit with `sbatch scripts/slurm/seqjepa_cifar10_rot_baseline.sbatch`

## Rough runtimes (very approximate)
- Mac M3 Max: quick config ~8 min (based on recent 5-10 min runs).
- Vast.ai 1x RTX 3060: quick ~1-2 min, baseline ~4-6 min.
- First run adds data download time (~1-3 min) if not cached.

## If you hit issues
- CUDA OOM: lower `dataset.batch_size` to 64.
- Slow dataloading: increase `dataset.num_workers` to 4-8 on GPU nodes.
