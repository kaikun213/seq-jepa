# Milestones (M1-M6)

## M1 - Baseline seq-JEPA reproduction
- Scripts/configs for 1 fast dataset and 1 structural dataset.
- Invariance/equivariance metrics for z_t and z_AGG.
- Action conditioning ablation.

## M2 - Coding-rate regularizer (EMA on)
- Stability diagnostics and lambda_rate sweep.
- No collapse regressions.

## M3 - Teacherless attempt (EMA off)
- Either success with metrics and diagnostics or documented failure + fallback.

## M4 - CRATE/CRATE-alpha aggregator
- Invariance/equivariance metrics.
- Sparsity and codebook usage stats.
- Comparison vs baseline g_phi.

## M5 - Streaming protocol + replay
- Non-stationary stream benchmark definition.
- Forgetting curves and replay ablations.

## M6 - ToST swap (if needed)
- Compute/memory vs performance tradeoff.
- Running-statistics behavior.

## Out of scope
- M7 (Monty integration) is tracked in another repo.
