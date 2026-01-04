# Step 6 - Local learning approximations

**Key references**:
- LPL (Halvagal & Zenke 2023) - local predictive + Hebbian plasticity (Nature)
- ESPP (Graf et al. 2024) - spiking LPL-like rule for streaming data (arXiv)
- Learning what matters (Brito & Gerstner 2024) - correlation-invariant sparsity, invariance to second-order input correlations (PLOS)
- Memory consolidation and improvement by synaptic tagging and capture in recurrent neural networks (Luboeinski & Tetzlaff, 2021) ([Nature](https://www.nature.com/articles/s42003-021-01778-y.pdf)) - STC helps with "what should we remember"

## Goal
- Add local/plasticity-inspired learning without full rewrite.

## Stage 1: auxiliary local losses
- Keep backprop for main objective.
- Add layerwise predictive or Hebbian-style losses.

## Stage 2: freeze backbone
- Pretrain encoder f_theta.
- Train only aggregator/predictor with local-ish updates.

## Stage 3: fully local variants (future)
- ESPP-inspired online local rules if needed.

## Stabilizers for correlation shifts
- Use "Learning what matters" style decorrelation-invariant updates as regularizer.
