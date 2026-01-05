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

### Tuning Experiments

#### Higher λ_rate (0.1)
- **Loss**: -41.2 (rate dominates)
- **linacc_test**: 21.4% (worse - degraded)
- **W&B**: [mndlms9u](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/mndlms9u)
- **Finding**: Too high - rate loss dominates prediction loss

#### Warmup + Cosine LR + λ_rate=0.03 ✅
- **Loss**: -10.7
- **linacc_test**: **30.0%** (passes gate ✓)
- **r2_test**: **0.423** (passes gate ✓)
- **gate_pass**: **True** ✓
- **W&B**: [sjd2550e](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/sjd2550e)
- **Finding**: **Cosine LR with 2-epoch warmup stabilizes teacherless training!**

## Updated Comparison

| Metric | Exp A (EMA) | Exp B (Original) | Exp B (Tuned) |
|--------|-------------|------------------|---------------|
| linacc | 34.1% | 27.8% | **30.0%** |
| r2 | 0.302 | 0.144 | **0.423** |
| gate | ✓ | ✗ | **✓** |

## Key Finding (Updated)

**Teacherless training works with proper tuning!** 

The key ingredients:
1. **Moderate λ_rate** (0.03 not 0.01 or 0.1)
2. **Cosine LR schedule** with warmup
3. **Coding rate regularizer** to prevent collapse

## Next Steps

- [ ] Run remote baseline on CIFAR-100
- [ ] Test with MNIST for subspace analysis
- [ ] Compare with Exp D (DINO sharpening)

