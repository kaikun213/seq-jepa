# Experiment B: Stop-Grad (Teacherless)

**Date**: 2026-01-05  
**Status**: ✅ Smoke & Quick passed, baseline-lite pending

## Objective

Test whether we can remove the EMA teacher entirely using stop-gradient + coding-rate regularizer to prevent collapse.

## Configuration

- **Model**: SeqJEPA_Teacherless with `teacherless=True`, `ema=False`, `rate_loss_enabled=True`
- **Rate loss**: λ=0.01, α=1.0, target=agg_out
- **Dataset**: CIFAR-10 rotation (4 discrete rotations)

## Results

### Smoke Test (1 epoch, 128 samples)
- **Loss**: -0.16
- **linacc_test**: 12.5% (passes 12% gate ✓)
- **r2_test**: -0.010
- **gate_pass**: True ✓
- **W&B**: [dumlzy8r](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/dumlzy8r)

### Quick Test (5 epochs, 5000 samples)
- **Loss**: -1.78 (steadily decreasing)
- **linacc_test**: 30.8% (passes 20% gate ✓)
- **r2_test**: 0.065 (passes 0.0 gate ✓)
- **W&B**: [tj58cbh1](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/tj58cbh1)

## Comparison with Experiment A

| Metric | Exp A (EMA+Rate) | Exp B (StopGrad) | Δ |
|--------|------------------|------------------|---|
| Loss | -1.77 | -1.78 | -0.01 |
| linacc_test | 31.0% | 30.8% | -0.2% |
| r2_test | 0.117 | 0.065 | -0.052 |

### Baseline-Lite Test (10 epochs, 20000 samples)
- **Loss**: -4.09 (strongly decreasing)
- **linacc_test**: 27.8% (fails 30% gate ✗)
- **r2_test**: 0.144 (passes 0.1 gate ✓)
- **Peak linacc**: 32.55% at epoch 4
- **W&B**: [khgtp6bt](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/khgtp6bt)

## Key Findings

1. **Coding rate prevents collapse** - Loss matches EMA version (-4.09 vs -4.11)
2. **Stability issue** - Accuracy degrades in later epochs (32.55% → 27.8%)
3. **EMA still helps** - Provides smoother learning dynamics
4. **Quick test promising** - Early epochs competitive with EMA

## Analysis

The stop-gradient approach successfully prevents collapse (coding rate keeps increasing), but without EMA:
- Representations become less stable over longer training
- May need: different λ_rate, LR schedule, or warmup period

## Next Steps

- [ ] Try higher λ_rate (0.05, 0.1) for stronger anti-collapse
- [ ] Add LR warmup/cosine schedule
- [ ] Run remote baseline on CIFAR-100
- [ ] Compare with Exp D (DINO sharpening)

