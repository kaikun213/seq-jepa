# Step 2 - Remote Baselines (CIFAR-100, 30 epochs)

**Date**: 2026-01-05  
**Status**: 🟡 Running on Vast.ai

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
- **λ_rate**: 0.03 (tuned for teacherless stability)
- **Epochs**: 30

## Expected Results (based on local experiments)

| Metric | Exp A (EMA) | Exp B (Teacherless) |
|--------|-------------|---------------------|
| linacc_test | ~30-35% | ~28-32% |
| r2_test | ~0.3-0.4 | ~0.3-0.5 |
| gate_pass | ✅ | ✅ |

## Launch Details

- **Vast.ai Instance**: 29521734
- **GPU**: RTX 3090 (preferred) or similar
- **Estimated Runtime**: ~2-4 hours total

## W&B

Runs will appear in: https://wandb.ai/kaikun213/seq-jepa-streaming
- Group: `step2-remote`

## Results

*Pending - will update after completion*

## Next Steps

1. Compare EMA vs Teacherless performance at scale
2. If teacherless competitive, proceed with step-3 CRATE integration
3. Final paper-aligned run (200 epochs) after validating architecture

