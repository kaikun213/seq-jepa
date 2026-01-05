# MNIST Subspace Diagnostics: EMA + Rate Loss

**Date**: 2026-01-05  
**Status**: ✅ Complete with visualizations

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

## Results (v3 - with W&B visualizations)

| Epoch | Loss | linacc_test | r2_test |
|-------|------|-------------|---------|
| 1 | -1.31 | 21.5% | 0.113 |
| 10 | -2.44 | 17.7% | 0.353 |
| 20 | -2.96 | 16.2% | 0.466 |

**W&B v3 (with viz)**: [xr99gop0](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/xr99gop0)  
**W&B v2**: [qganshbm](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/qganshbm)  
**W&B v1**: [7z37n2lu](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/7z37n2lu)

## W&B Visualizations Logged

The following visualizations are now logged to W&B every `subspace_frequency` epochs:

1. **viz/sweep_traj_rot** - Rotation sweep trajectories in 2D PCA space
2. **viz/rotation_circle** - Rotation changes with color-coded angles
3. **viz/combined_trajectories** - All factor trajectories side by side
4. **viz/eigs_rot** - Eigenvalue spectrum for rotation subspace
5. **viz/rank_monitor** - SVD spectra for collapse monitoring

## Observations

1. **Coding rate working**: Loss steadily decreases (rate increasing) → representations expanding
2. **Equivariance learning**: R² = 0.47 indicates action prediction from delta-z works
3. **Classification harder**: MNIST with continuous affine transforms is challenging (16.2%)
4. **Leakage metrics**: leakage_linacc=55.4% (class info in z_eq) vs online=16.2% (class from z_inv)
5. **Subspace structure**: Rotation changes concentrate in ~2-3 dimensions (see eigenvalue spectrum)

## Subspace Metrics

| Metric | Value |
|--------|-------|
| rank_eq_eff | ~60 (high diversity) |
| rank_inv_eff | ~30 (more compressed) |
| subspace_ev_rot | concentrated in top-2 PCs |

## Next Steps

- [x] Integrate subspace metrics computation into eval loop ✅
- [x] Add W&B visualizations (sweep trajectories, eigenspectra) ✅
- [ ] Run rotation-only mode for cleaner circular trajectories
- [ ] Test teacherless version on MNIST
- [ ] Compare with CIFAR affine version

