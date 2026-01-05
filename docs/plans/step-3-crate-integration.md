# Step 3 - CRATE integration

**Key references**:
- CRATE: Coding Rate Reduction via Transformers ([website](https://ma-lab-berkeley.github.io/CRATE/)) - open loop, sparse rate reduction via MSSA + ISTA
- CRATE-MAE: Closed-Loop Training of Masked Autoencoders, 2024 ([arXiv:2404.02446](https://arxiv.org/abs/2404.02446)) - adds predictive objective (closed loop), ~30% fewer params, prevents some open-loop issues
- CRATE-alpha: Scaling CRATE, 2024 ([arXiv:2405.20299](https://arxiv.org/abs/2405.20299)) - fixes scaling issues and artifacts in CRATE (open loop, applicable to CRATE-MAE)
- Lie group + sparse coding: [arXiv:2012.12071](https://arxiv.org/abs/2012.12071) - achieves clean circular subspaces via explicit group constraints

## Motivation (from Step 2 findings)

Step 2 showed that **coding-rate regularization alone** prevents collapse but does NOT produce clean group-theoretic subspace structure:
- Rotation changes spread across ~30 dimensions instead of 2
- No circular trajectories in learned representation
- Gradual eigenvalue decay instead of sharp cutoff

**Why CRATE may help:**
- MSSA (Multi-head Subspace Self-Attention): Different heads can specialize to different group factors
- ISTA sparsity: Encourages coordinate-sparse subspace alignments
- This mirrors the explicit group constraints in [Chau et al. 2020](https://arxiv.org/abs/2012.12071)

## Goal
- Introduce sparse coding / codebook structure without changing seq-JEPA routing
- Achieve cleaner factor subspaces (rotation → 2D, translation → 2D, scale → 1D)

## Integration A: replace aggregator g_phi
- Tokens: u_i = concat(z_i, emb(a_i)) for i=1..M-1, token z_M.
- Add [AGG] token and time positional embeddings.
- Replace transformer blocks with L CRATE blocks (MSSA + ISTA).

## Integration B: replace predictor h_psi
- Two-token sequence: token0 = z_AGG, token1 = emb(a_M).
- Apply small CRATE block; output token0 as z_hat_{M+1}.

## When to use CRATE-alpha
- If vanilla CRATE diverges or scales poorly on longer sequences, switch.

## Measurements
- Same invariance/equivariance metrics as baseline.
- Sparsity after ISTA (fraction near zero).
- Head/codebook specialization vs action dimensions or object factors.

## Subspace Diagnostics (from Step 2)
Use the visualization infrastructure built in Step 2:
- **rotation_circle**: Should become circular with CRATE
- **eigs_rot**: Should show sharp drop after expected dimension
- **sweep_traj_rot**: Anchor trajectories should trace parallel loops
- **rank_monitor**: Continue monitoring collapse prevention

## Tasks
1. Implement MSSA block (multi-head subspace self-attention)
2. Implement ISTA sparsity layer
3. Replace aggregator with CRATE blocks (Integration A)
4. Test on MNIST-AffineSeq with rotation-only mode
5. Compare subspace metrics to Step 2 baseline (rate-only)
6. If successful, test on CIFAR affine transforms
7. Validate teacherless mode with CRATE
