# Experiment: Step 1 CIFAR-100 Baseline (Vast, Clean Run)

**Date**: 2026-01-04  
**Step**: Step 1 - Baseline Reproduction  
**Status**: ✅ Completed (results below paper)

## Objective
Run a clean, from-scratch CIFAR-100 augmentation baseline on Vast.ai to validate the remote pipeline and compare against seq-JEPA paper results.

## Setup
- Config: `configs/remote/cifar100_aug_baseline.yaml`
- Dataset: CIFAR-100 augmentation actions (crop + jitter + blur), `seq_len=3`
- Training:
  - epochs: 10
  - batch_size: 128
  - optimizer: Adam (per wrapper defaults)
  - action_norm: true
- Hardware: Vast.ai 1x RTX 3060 (12GB)
- Output: `runs/remote/cifar100-aug-baseline/metrics.jsonl`

## Results
From epoch 10 (test metrics):
- `online_linacc_test`: 2.21
- `online_r2_test`: 0.1054
- `ep_loss`: 0.1719
- `leakage_linacc_test`: 2.81
- `leakage_r2_test`: -0.0427

Note: Metrics are from the online probe trained jointly during training (not a frozen-feature probe).

## Comparison to Paper (seq-JEPA)
Paper reference (Table 2, CIFAR-100, Crop+Jitter+Blur, seq-JEPA M_tr=3):
- Top-1 classification: 58.33
- Crop R2: 0.79
- Jitter R2: 0.63
- Blur R2: 0.92

Our run is far below paper results. This is expected because the setup is not comparable:
- Training is 10 epochs vs. 2000 epochs in the paper.
- Batch size is 128 vs. 512.
- RTX 3060 vs. A100 40GB.
- Our evaluation uses an online probe trained jointly; paper uses frozen representations + linear probe.
- Our equivariance score is a single regression over combined augmentation parameters, not per-augmentation R2.

## Analysis
- Pipeline is validated end-to-end on Vast (clone, install, run, W&B sync, output files).
- Representation quality is still near-random for classification, which is consistent with short training.
- Equivariance R2 is low but non-zero, again expected under the reduced training regime.

## Decisions
- Treat this as a pipeline validation run, not a paper-level reproduction.
- Next follow-up should align with paper protocol: longer training, larger batch size, frozen probe evaluation, and per-augmentation R2.

## Files Changed
- None (run-only)
