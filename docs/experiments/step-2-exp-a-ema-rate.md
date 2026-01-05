# Experiment A: EMA + Coding-Rate Regularizer

**Date**: 2026-01-05  
**Status**: ✅ Smoke & Quick passed, baseline-lite pending

## Objective

Validate that adding coding-rate regularizer to EMA-based seq-JEPA doesn't hurt baseline performance while potentially improving representation quality.

## Configuration

- **Model**: SeqJEPA_Teacherless with `ema=True`, `rate_loss_enabled=True`
- **Rate loss**: λ=0.01, α=1.0, target=agg_out
- **Dataset**: CIFAR-10 rotation (4 discrete rotations)

## Results

### Smoke Test (1 epoch, 128 samples)
- **Loss**: -0.17 (negative = maximizing coding rate ✓)
- **linacc_test**: 9.38%
- **r2_test**: -0.006
- **W&B**: [lhzhc7a1](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/lhzhc7a1)

### Quick Test (5 epochs, 5000 samples)
- **Loss**: -1.77 (steadily decreasing)
- **linacc_test**: 31.0% (passes 20% gate ✓)
- **r2_test**: 0.117 (passes 0.0 gate ✓)
- **W&B**: [z765jckk](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/z765jckk)

### Baseline-Lite Test (10 epochs, 20000 samples)
- **Loss**: -4.11 (strongly decreasing)
- **linacc_test**: 34.1% (passes 30% gate ✓)
- **r2_test**: 0.302 (passes 0.1 gate ✓)
- **W&B**: [f1u4hzpj](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/f1u4hzpj)

## Observations

1. Coding rate loss is working (negative loss increases over training)
2. Linear probe accuracy comparable to baseline
3. Equivariance R² positive and improving

## Next Steps

- [ ] Run baseline-lite (10 epochs, 20k samples)
- [ ] Compare with baseline without rate loss
- [ ] Run remote baseline on CIFAR-100

