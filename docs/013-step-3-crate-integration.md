# Step 3 - CRATE integration

**Key references**:
- CRATE: Coding Rate Reduction via Transformers ([website](https://ma-lab-berkeley.github.io/CRATE/)) - open loop, sparse rate reduction via MSSA + ISTA
- CRATE-MAE: Closed-Loop Training of Masked Autoencoders, 2024 ([arXiv:2404.02446](https://arxiv.org/abs/2404.02446)) - adds predictive objective (closed loop), ~30% fewer params, prevents some open-loop issues
- CRATE-alpha: Scaling CRATE, 2024 ([arXiv:2405.20299](https://arxiv.org/abs/2405.20299)) - fixes scaling issues and artifacts in CRATE (open loop, applicable to CRATE-MAE)

## Goal
- Introduce sparse coding / codebook structure without changing seq-JEPA routing.

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
