# Baseline architecture (seq-JEPA)

**Reference**: seq-JEPA: Autoregressive Predictive Learning of Invariant-Equivariant World Models ([arXiv:2505.03176](https://arxiv.org/abs/2505.03176))

## Data flow
- Encode first M views: z_1..z_M = f_theta(x_1..x_M).
- Build tokens u_i = concat(z_i, emb(a_i)) for i=1..M-1 and token z_M (no action).
- Add learnable [AGG] token, apply sequence aggregator g_phi.
- Output z_AGG from [AGG] token.
- Predictor h_psi takes (z_AGG, a_M) and outputs z_hat_{M+1}.

## Target and loss
- Target encoder is EMA of f_theta; target is stop-grad.
- Loss: 1 - cosine_similarity(z_hat_{M+1}, z_{M+1}).

## Representation expectations
- z_t (encoder outputs) should be more equivariant.
- z_AGG (aggregator output) should be more invariant.

## Action conditioning
- Action embeddings are required to encourage equivariance in z_t.
- Removing action conditioning is a baseline ablation (see Step 1).

## Notes
- Keep this routing intact while experimenting with rate-reduction and sparse blocks.
