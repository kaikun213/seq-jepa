# Metrics and gates

This document covers two gate tiers:
- Local CIFAR-10 rotations for fast iteration sanity checks.
- Remote CIFAR-100 augmentations for paper-aligned comparisons and model changes.

## Local CIFAR-10 rotations (fast iteration)
### Dataset and task
- Dataset: CIFAR-10.
- Each sample is a short sequence of the same image under rotations.
- Rotations are discrete angles: {0, 90, 180, 270} degrees.
- Sequence length: M+1 = 4 views (three actions) in current configs.

### Action encoding (aligned with seq-JEPA paper style)
- Actions are treated as continuous transformation parameters.
- For rotations, we encode an angle as sin/cos:
  - action vector = [sin(theta), cos(theta)]
- Relative actions are computed between consecutive views:
  - delta = angle_{t+1} - angle_t (wrapped to [-180, 180])
  - rel_action = [sin(delta), cos(delta)]

This mirrors the paper’s use of continuous transformation parameters (e.g., quaternions for 3D rotations, saccade deltas for patches, and augmentation parameters).

### Runs (local)
- Smoke: `configs/quick/cifar10_rot_smoke.yaml`
  - subset_train=128, subset_val=64, epochs=1
  - Purpose: ensure forward/backward runs and loss is finite.
- Quick: `configs/quick/cifar10_rot_quick.yaml`
  - subset_train=5000, subset_val=1000, epochs=5
  - Purpose: fast signal on learning trends.
- Baseline-lite: `configs/quick/cifar10_rot_baseline.yaml`
  - subset_train=20000, subset_val=2000, epochs=10
  - Purpose: stronger evidence before remote runs.

### Metrics logged to W&B
- `ep_loss`: JEPA cosine prediction loss (lower is better).
- `online_linacc_*`: linear probe accuracy on `agg_out` (invariance proxy).
- `online_r2_*`: R^2 of action regression from encoder features (equivariance proxy).
- `online_linloss_*`: cross-entropy for the linear probe (invariance loss).
- `online_r2_loss_*`: MSE for the action regression (equivariance loss).
- `leakage_linacc_*`: class accuracy from encoder output `z_t` (leakage check).
- `leakage_r2_*`: action regression from `agg_out` (leakage check).
- `leakage_linloss_*`: cross-entropy for leakage class probe.
- `leakage_r2_loss_*`: MSE for leakage action probe.

Note: `online_r2_loss_*` is not the inverse of R^2; it is the raw MSE used by the regression probe.

### What "good" looks like
- `ep_loss` decreases steadily.
- `online_linacc_test` rises above random (10% for CIFAR-10).
- `online_r2_test` becomes positive and increases.
- `leakage_linacc_test` stays lower than `online_linacc_test`.
- `leakage_r2_test` stays lower than `online_r2_test`.

Why lower leakage matters:
- The encoder output `z_t` should preserve pose/action (equivariant), so it should not be the best place for class identity.
- The aggregator output `agg_out` should be invariant to pose, so it should not retain action prediction power.

### Gates (pass/fail thresholds)
Gates are thresholds to decide whether to proceed to more expensive runs.

- Smoke gate (minimal):
  - `online_linacc_test >= 12.0`
  - `online_r2_test >= -0.3`
- Quick gate:
  - `online_linacc_test >= 20.0`
  - `online_r2_test >= 0.0`
  - `leakage_linacc_gap_min >= 0.0` (online >= leakage)
  - `leakage_r2_gap_min >= 0.0` (online >= leakage)
- Baseline-lite gate:
  - `online_linacc_test >= 30.0`
  - `online_r2_test >= 0.1`
  - `leakage_linacc_gap_min >= 0.0` (online >= leakage)
  - `leakage_r2_gap_min >= 0.0` (online >= leakage)

Gate status is written to W&B summary as `gate_pass` and `gate_reasons`.

## Local CIFAR-100 gate (mid-cost)
### Runs (local)
- Gate (mid): `configs/local/cifar100_aug_baseline_gate.yaml`
- Script: `scripts/local/run_cifar100_aug_baseline_gate.sh`

### Metrics logged to W&B
- Same online metrics as the CIFAR-10 gates (`online_*`, `leakage_*`).
- This run is meant to be the local regression gate for model changes.

### Gate guidance
- Treat the first completed run as the baseline.
- Update thresholds to match that baseline and flag regressions if they drop.

## Remote CIFAR-100 augmentations (paper-aligned gates)
### Dataset and task
- Dataset: CIFAR-100 with augmentation actions (crop + jitter + blur).
- Action parameters: 9 dims (crop 4, jitter 4, blur 1).
- Training uses `seq_len=3`; frozen probe eval uses `seq_len=5` by default.

### Runs (remote)
- Baseline gate (mid): `configs/remote/cifar100_aug_baseline.yaml`
- Paper-aligned (long): `configs/remote/cifar100_aug_paper.yaml`

### Metrics logged to W&B
- Same online metrics as the local gates (`online_*`, `leakage_*`).
- For paper comparability, run the frozen probe eval:
  - `scripts/eval/frozen_probe_cifar100_aug.py --config <config.yaml>`
  - Outputs top-1 accuracy and per-augmentation R2 (crop/jitter/blur).

### Gate guidance
- Use the CIFAR-100 baseline gate to compare model changes (e.g., teacherless).
- Current default thresholds (from config):
  - `online_linacc_test >= 5.0`
  - `online_r2_test >= 0.1`
  - `leakage_linacc_gap_min >= 0.0`
  - `leakage_r2_gap_min >= 0.0`
- After the first gate run, update thresholds to match the observed baseline and treat any drop as a regression.

## Future metrics (planned)
- Subspace usage: sparsity, codebook usage, and head specialization.
- Continual learning: forgetting curves, replay ablations, stability metrics.
