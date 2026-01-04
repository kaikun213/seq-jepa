# Experiment: Step 1 CIFAR-100 Paper Repro (Vast)

**Date**: 2026-01-04  
**Step**: Step 1 - Baseline Reproduction  
**Status**: 🔄 In Progress

## Objective
Run a paper-aligned CIFAR-100 augmentation experiment (long training + frozen-probe eval) on Vast.ai.

## Setup
- Config: `configs/remote/cifar100_aug_paper.yaml`
- Dataset: CIFAR-100 augmentation actions (crop + jitter + blur), `seq_len=3`
- Eval sequence length: `M_val=5` (frozen probe)
- Training:
  - epochs: 2000
  - batch_size: 512
  - optimizer: AdamW, cosine decay, warmup 20 epochs, min lr 1e-5
- Frozen-probe eval:
  - classifier on `z_agg`
  - regressors on encoder pairs for crop/jitter/blur R2
- Hardware: Vast.ai 1x GPU (paper-aligned run)
- Output:
  - Training metrics: `runs/remote/cifar100-aug-paper/metrics.jsonl`
  - Checkpoint: `runs/remote/cifar100-aug-paper/checkpoints/last.pt`
  - Eval: `runs/remote/cifar100-aug-paper/eval/frozen_probe_metrics.json`

## Results
- Pending (track via W&B).

## Analysis
- Pending.

## Decisions
- None yet.

## Files Changed
- None (run-only).
