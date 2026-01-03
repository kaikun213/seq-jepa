#!/usr/bin/env bash
#SBATCH --job-name=seqjepa-cifar10-rot
#SBATCH --partition=REPLACE_ME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# module load cuda/11.8 python/3.10
# source /path/to/venv/bin/activate
# export WANDB_API_KEY=...  # or rely on .env in the repo

CONFIG_PATH="${1:-configs/remote/cifar10_rot_baseline.yaml}"
REPO_ROOT="${REPO_ROOT:-$PWD}"

cd "$REPO_ROOT"

echo "Using config: $CONFIG_PATH"
python train.py --config "$CONFIG_PATH"
