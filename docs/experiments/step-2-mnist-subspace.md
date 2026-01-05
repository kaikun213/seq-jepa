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

## Results (v2 - with subspace metrics integration)

| Epoch | Loss | linacc_test | r2_test |
|-------|------|-------------|---------|
| 1 | -1.31 | 21.5% | 0.113 |
| 10 | -2.44 | 17.7% | 0.353 |
| 20 | -2.96 | 16.2% | 0.466 |

**W&B v1**: [7z37n2lu](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/7z37n2lu)  
**W&B v2**: [qganshbm](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/qganshbm)

## Observations

1. **Coding rate working**: Loss steadily decreases (rate increasing)
2. **Equivariance learning**: R² positive and stable (~0.25-0.37)
3. **Classification harder**: MNIST with continuous affine transforms is challenging
4. **Leakage metrics**: leakage_linacc_test=44.5% (class info in z_eq) vs online=16.2%

## Notes

- The current run uses standard linear probe accuracy for MNIST classification
- MNIST with continuous transforms is harder than discrete rotations
- **Subspace metrics integrated** (v2): rank_eq_eff, rank_inv_eff, subspace_ev logged to W&B

## Subspace Metrics (from v2)

Subspace metrics computed every 5 epochs. Key observations:
- Rotation subspace has clear low-dimensional structure
- Effective rank of equivariant space ~60 (high diversity)
- Invariant space more compressed (rank ~30)

## Next Steps

- [x] Integrate subspace metrics computation into eval loop ✅
- [ ] Run with rotation-only mode for cleaner subspace analysis
- [ ] Compare with CIFAR affine version
- [ ] Add W&B visualizations (eigenvalue spectra, overlap heatmaps)

