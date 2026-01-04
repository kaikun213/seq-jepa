# Step 4 - ToST for scaling and streaming

**Key reference**: Token Statistics Transformer (ToST), 2024 ([arXiv:2412.17810](https://arxiv.org/abs/2412.17810)) - linear scaled attention block via variational rate reduction

## Goal
- Make attention and coding-rate computations linear in sequence length.

## Use case 1: ToST attention
- Replace g_phi transformer with Token Statistics Self-Attention (TSSA).
- Keep outer routing intact.

## Use case 2: approximate coding-rate regularizer
- Use ToST variational stats to avoid explicit logdet on large D.

## Local Coding Rate (LCR)
- Maintain a temporal window of recent sequences.
- Compute coding-rate regularization on the window only.

## Deliverables
- Compute and memory comparison vs baseline.
- Stability and performance impacts on long sequences.
