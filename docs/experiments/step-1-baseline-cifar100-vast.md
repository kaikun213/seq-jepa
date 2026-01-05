# Experiment: Step 1 CIFAR-100 Baseline Gate

**Date**: 2026-01-04  
**Step**: Step 1 - Baseline Reproduction  
**Status**: 🔄 In Progress

## Objective
Establish a mid-cost CIFAR-100 gate (local 30 epochs) that is closer to the paper settings than smoke/quick, so model changes (e.g., teacherless) have a stable invariance/equivariance baseline.

## Setup
- Config (remote gate): `configs/remote/cifar100_aug_baseline.yaml`
- Config (local gate): `configs/local/cifar100_aug_baseline_gate.yaml`
- Script (local): `scripts/local/run_cifar100_aug_baseline_gate.sh`
- Script (remote): `scripts/vast/run_cifar100_aug_baseline.sh`
- Dataset: CIFAR-100 augmentation actions (crop + jitter + blur), `seq_len=3`
- Training: 30 epochs (local gate), batch_size=128, AdamW + cosine, warmup 10
- Output (local): `runs/local/cifar100-aug-baseline-gate/`
- Output (remote): `runs/remote/cifar100-aug-baseline-gate/`

Note: The earlier 10-epoch shakedown was archived at
`docs/experiments/archive/step-1-baseline-cifar100-vast-10ep.md` and
`configs/archive/remote/cifar100_aug_baseline_10ep.yaml`.

## Results
- Previous local run was interrupted at epoch 110 after confirming a stable trajectory.
- Metrics snapshot (from output log):
  - Epoch 30: `linacc_test=24.71`, `r2_test=0.4892`
  - Epoch 40: `linacc_test=28.72`, `r2_test=0.5462`
  - Epoch 50: `linacc_test=31.46`, `r2_test=0.5900`
- New gate thresholds (local 30-epoch run):
  - `linacc_test_min=23.0`
  - `r2_test_min=0.45`
- W&B group: `local-gate`.

## Analysis
- Gate uses online probes during training for fast regression detection.
- Frozen probe evaluation is optional and used for paper-level comparisons.

## Decisions
- Use the 30-epoch local gate as the baseline for Step 2 model changes.
- Re-run the local gate with the updated thresholds to confirm pass/fail behavior.

## Files Changed
- None (run-only)
