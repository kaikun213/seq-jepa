# Step 2 - Remote Baselines (CIFAR-100, 30 epochs)

**Date**: 2026-01-05  
**Status**: 🟡 Running on Vast.ai (v2 with paper-aligned configs)

## Objective

Compare EMA+Rate vs Teacherless+Rate on full CIFAR-100 dataset to validate local findings at scale.

## Experiments

### Exp A: EMA + Rate Loss
- **Config**: `configs/remote/step2_exp_a_ema_rate.yaml`
- **Model**: EMA target encoder + coding rate regularization
- **λ_rate**: 0.01
- **Epochs**: 30

### Exp B: Teacherless + Rate Loss
- **Config**: `configs/remote/step2_exp_b_teacherless.yaml`
- **Model**: Stop-grad (no EMA) + coding rate regularization
- **λ_rate**: 0.03 (stronger for teacherless stability)
- **Epochs**: 30

## Config Changes (v2)

Fixed config divergence from paper baseline (v1 had issues):

| Parameter | v1 (problematic) | v2 (paper-aligned) |
|-----------|------------------|-------------------|
| batch_size | 128 | **512** |
| lr | 0.001 | **0.0004** |
| weight_decay | 0.0 | **0.001** |
| warmup_epochs | 3 | **5** |
| optimizer | Adam (implicit) | **AdamW** |

## Expected Results (based on paper at 30 epochs)

| Metric | Exp A (EMA) | Exp B (Teacherless) |
|--------|-------------|---------------------|
| linacc_test | ~20-25% | ~18-22% |
| r2_test | ~0.4-0.5 | ~0.3-0.5 |
| gate_pass | ✅ | ✅ |

Note: At 30 epochs (1.5% of paper's 2000 epochs), we expect ~40% of final accuracy.

## Launch Details

- **GPU**: RTX 4090 (preferred), RTX 3090 (backup)
- **Parallel execution**: Two instances, one per experiment
- **Estimated Runtime**: ~30-60 min each on RTX 4090

### v1 (old config, still running)
- Instance 29522248 - let complete to see if recovers

### v2 (paper-aligned) ✅ Launched
- **Exp A (EMA)**: Instance 29527387
- **Exp B (Teacherless)**: Instance 29527388
- GPU offers: 16320064, 28889471
- Launched: 2026-01-05 ~18:30 UTC

## W&B

Runs will appear in: https://wandb.ai/kaikun213/seq-jepa-streaming
- Group: `step2-remote`

## Results

### v1 (problematic config) - Instance 29522248
*Still running - monitoring for recovery*

### v2 (paper-aligned) - Running
- Exp A (EMA): Instance 29527387 - *Running*
- Exp B (Teacherless): Instance 29527388 - *Running*

## Next Steps

1. Compare v1 vs v2 results to confirm config fix
2. Compare EMA vs Teacherless performance at scale
3. If teacherless competitive, proceed with step-3 CRATE integration
4. Final paper-aligned run (200 epochs) after validating architecture

