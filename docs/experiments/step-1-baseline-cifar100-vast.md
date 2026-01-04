# Experiment: Step 1 CIFAR-100 Baseline Gate

**Date**: 2026-01-04  
**Step**: Step 1 - Baseline Reproduction  
**Status**: 🔄 In Progress

## Objective
Establish a mid-cost CIFAR-100 gate (200 epochs) that is closer to the paper settings than smoke/quick, so model changes (e.g., teacherless) have a stable invariance/equivariance baseline.

## Setup
- Config (remote gate): `configs/remote/cifar100_aug_baseline.yaml`
- Config (local gate): `configs/local/cifar100_aug_baseline_gate.yaml`
- Script (local): `scripts/local/run_cifar100_aug_baseline_gate.sh`
- Script (remote): `scripts/vast/run_cifar100_aug_baseline.sh`
- Dataset: CIFAR-100 augmentation actions (crop + jitter + blur), `seq_len=3`
- Training: 200 epochs, batch_size=128, AdamW + cosine, warmup 10
- Output (local): `runs/local/cifar100-aug-baseline-gate/`
- Output (remote): `runs/remote/cifar100-aug-baseline-gate/`

Note: The earlier 10-epoch shakedown was archived at
`docs/experiments/archive/step-1-baseline-cifar100-vast-10ep.md` and
`configs/archive/remote/cifar100_aug_baseline_10ep.yaml`.

## Results
- Local gate run started on MPS; metrics pending.
- W&B group: `local-gate`.

## Analysis
- Gate uses online probes during training for fast regression detection.
- Frozen probe evaluation is optional and used for paper-level comparisons.

## Decisions
- Use this gate as the baseline for Step 2 model changes once results are recorded.

## Files Changed
- None (run-only)
