# Step 2 - Experiment C: Symmetric Loss

## Purpose
Test if symmetric loss (bidirectional prediction) improves representation quality.

## Configuration
- Model: EMA + Rate + Symmetric
- Dataset: CIFAR-10 rotation (20k subset)
- Epochs: 10

## Results

| Metric | Value | Gate |
|--------|-------|------|
| linacc_test | 33.3% | ✅ (>30%) |
| r2_test | 0.42 | ✅ (>0.1) |
| leakage_linacc | 36.9% | - |
| gate_pass | True | ✅ |

## Observations
1. Symmetric loss performs similarly to non-symmetric EMA+Rate baseline
2. Loss decreases smoothly (-2.05 → -3.48)
3. Subspace metrics: rank_eq_eff ~60, rank_inv_eff ~33

## W&B
- Run: [hj0seblj](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/hj0seblj)

## Conclusion
Symmetric loss doesn't hurt performance but also doesn't provide clear benefits on CIFAR-10. May be more useful for multi-step prediction or more complex datasets.

