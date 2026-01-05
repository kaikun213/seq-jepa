# MNIST Subspace Diagnostics: EMA + Rate Loss

**Date**: 2026-01-05  
**Status**: ✅ Complete with visualizations

## Objective

Test seq-JEPA with coding-rate regularizer on controlled MNIST affine transforms to analyze factor subspaces (rotation, translation, scale).

## Configuration

- **Dataset**: MNIST with continuous affine transforms
  - Rotation: ±75° (multi-factor) / ±180° (rotation-only)
  - Translation: ±7px
  - Scale: 0.5-1.0 (log-uniform)
- **Model**: SeqJEPA_Teacherless with EMA + rate loss
- **Rate loss**: λ=0.01, α=1.0, target=agg_out
- **Epochs**: 20

## Results

### Multi-Factor Run (v3) - [xr99gop0](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/xr99gop0)

| Epoch | Loss | linacc_test | r2_test |
|-------|------|-------------|---------|
| 1 | -1.31 | 21.5% | 0.113 |
| 10 | -2.44 | 17.7% | 0.353 |
| 20 | -2.96 | 16.2% | 0.466 |

### Rotation-Only Run - [d288j027](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/d288j027)

| Epoch | Loss | linacc_test | r2_test |
|-------|------|-------------|---------|
| 1 | -1.33 | 24.1% | 0.009 |
| 10 | -2.46 | 36.3% | 0.071 |
| 20 | -2.96 | 35.0% | 0.033 |

## W&B Visualizations Logged

The following visualizations are logged every `subspace_frequency` epochs:

1. **viz/sweep_traj_rot** - Rotation sweep trajectories in 2D PCA space
2. **viz/rotation_circle** - Δz projected onto top-2 PCs (should form a circle)
3. **viz/combined_trajectories** - All factor trajectories side by side
4. **viz/eigs_rot** - Eigenvalue spectrum for rotation-induced Δz
5. **viz/rank_monitor** - SVD spectra for collapse monitoring

## Observations

1. **Coding rate working**: Loss steadily decreases (rate increasing) → representations expanding
2. **Equivariance learning**: R² = 0.47 (multi-factor) indicates action prediction from Δz works
3. **Rotation-only harder**: R² only 0.03 - single factor without diverse actions is harder to predict
4. **Classification trade-off**: Rotation-only gets higher linacc (35%) vs multi-factor (16%) - less content variation
5. **Leakage metrics**: leakage_linacc=55.4% (class in z_eq) vs online=16.2% (class in z_inv)

## Visualization Interpretation

### What the plots SHOULD show (ideal case, per [Lie group paper](https://arxiv.org/abs/2012.12071)):

| Plot | Ideal Result | Meaning |
|------|-------------|---------|
| **Rotation Circle** | Clean circle/ellipse | Rotation encoded as (sin θ, cos θ) in 2D subspace |
| **Eigenvalue Spectrum** | Sharp drop after k=2 | 99% of rotation variance in 2 dimensions |
| **Factor Trajectories** | Circular paths for rotation | Each anchor traces same loop under rotation |

### What we ACTUALLY see (current results):

| Plot | Actual Result | Interpretation |
|------|--------------|----------------|
| **Rotation Circle** | Diffuse blob/cloud | Rotation NOT cleanly encoded in 2D; spread across many dims |
| **Eigenvalue Spectrum** | Gradual decay, no sharp drop | ~30-40% variance in top-2, rest spread across 30+ dims |
| **Factor Trajectories** | Scattered points with weak structure | No clear circular trajectories; different anchors don't trace same path |

### Why the gap?

1. **Training too short**: 20 epochs may not be enough for clean subspace emergence
2. **No explicit group constraint**: Unlike the Lie group paper which enforces torus structure, we only use prediction + rate loss
3. **Encoder architecture**: ResNet may not naturally produce group-equivariant features
4. **Rate loss insufficient**: MCR² encourages diversity but not explicit group structure

## Preliminary Analysis & Interpretation

### Comparison to [Chau et al. 2020](https://arxiv.org/abs/2012.12071) (Lie Group + Sparse Coding)

The Lie group paper learns clean disentangled representations where:
- Rotations trace perfect circles in a 2D latent subspace
- Translations trace lines
- Subspaces are orthogonal and coordinate-sparse

**Key differences in their approach:**
1. **Explicit group constraint**: They constrain transformations to form a representation of an n-dimensional torus
2. **Sparse coding**: Dictionary learning encourages coordinate sparsity
3. **Generative model**: Bayesian approach with explicit factorization

**Our current approach (seq-JEPA + rate loss):**
1. **No explicit group structure**: We rely on prediction loss to implicitly learn equivariance
2. **Rate loss only**: Prevents collapse but doesn't enforce sparsity or group structure
3. **Discriminative**: Encoder learns useful features, not generative model

### Implications for Step 3 (CRATE Integration)

The gap between our results and the Lie group paper suggests that **CRATE components (MSSA + ISTA)** from Step 3 may be necessary to achieve clean subspace structure:

- **MSSA (Multi-head Subspace Self-Attention)**: Could enforce that different heads attend to different group factors
- **ISTA sparsity**: Would encourage coordinate-sparse subspace alignments

### Current Status Assessment

| Goal | Status | Evidence |
|------|--------|----------|
| Prevent collapse | ✅ Achieved | Negative loss (rate increasing), rank_eq_eff ~60 |
| Learn equivariance | ⚠️ Partial | R² = 0.47 for action prediction |
| Clean factor subspaces | ❌ Not yet | Diffuse rotation plot, gradual eigenvalue decay |
| Disentanglement | ⚠️ Weak | Class leakage into z_eq (55% vs 16%) |

**Conclusion**: The coding-rate regularizer successfully prevents collapse and enables some equivariant learning, but does not produce the clean group-theoretic subspace structure seen in specialized approaches. This motivates the CRATE integration in Step 3.

## Subspace Metrics

| Metric | Multi-Factor | Rotation-Only |
|--------|-------------|---------------|
| rank_eq_eff | ~60 | ~55 |
| rank_inv_eff | ~30 | ~28 |
| subspace_ev_rot_2 | ~35% | ~40% |

## Next Steps

- [x] Integrate subspace metrics computation into eval loop ✅
- [x] Add W&B visualizations (sweep trajectories, eigenspectra) ✅
- [x] Run rotation-only mode for cleaner circular trajectories ✅
- [ ] Test teacherless version on MNIST
- [ ] Compare with CIFAR affine version
- [ ] Step 3: Add CRATE components (MSSA/ISTA) for explicit subspace structure
