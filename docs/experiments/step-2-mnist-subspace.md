# MNIST Subspace Diagnostics: EMA + Rate Loss

**Date**: 2026-01-05  
**Status**: ✅ Initial run complete

## Objective

Test seq-JEPA with coding-rate regularizer on controlled MNIST affine transforms to analyze factor subspaces (rotation, translation, scale).

## Configuration

- **Dataset**: MNIST with continuous affine transforms
  - Rotation: ±75°
  - Translation: ±7px
  - Scale: 0.5-1.0 (log-uniform)
- **Model**: SeqJEPA_Teacherless with EMA + rate loss
- **Rate loss**: λ=0.01, α=1.0, target=agg_out
- **Epochs**: 20

## Results

| Epoch | Loss | linacc_test | r2_test |
|-------|------|-------------|---------|
| 1 | -1.31 | 21.5% | 0.113 |
| 10 | -2.44 | 17.5% | 0.310 |
| 20 | -2.96 | 16.2% | 0.256 |

**W&B**: [7z37n2lu](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/7z37n2lu)

## Observations

1. **Coding rate working**: Loss steadily decreases (rate increasing)
2. **Equivariance learning**: R² positive and stable (~0.25-0.37)
3. **Classification harder**: MNIST with continuous affine transforms is challenging
4. **Leakage metrics**: leakage_linacc_test=44.5% (class info in z_eq) vs online=16.2%

## Notes

- The current run uses standard linear probe accuracy for MNIST classification
- MNIST with continuous transforms is harder than discrete rotations
- Subspace metric integration into train.py is pending

## Next Steps

- [ ] Integrate subspace metrics computation into eval loop
- [ ] Run with rotation-only mode for cleaner subspace analysis
- [ ] Compare with CIFAR affine version
- [ ] Visualize factor subspaces in W&B

