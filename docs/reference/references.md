# Paper References

This document provides a comprehensive list of papers relevant to the seq-JEPA + rate-reduction implementation project.

## Core seq-JEPA

- **seq-JEPA: Autoregressive Predictive Learning of Invariant-Equivariant World Models** ([arXiv:2505.03176](https://arxiv.org/abs/2505.03176))
  - Baseline architecture: encoder → action-conditioned aggregator → predictor
  - Uses EMA target encoder for stability
  - Separates equivariant (z_t) from invariant (z_AGG) representations via routing

## Rate Reduction and Coding Theory

- **Learning Diverse and Discriminative Representations via the Principle of Maximum Code Rate Reduction**, 2020 ([arXiv:2006.08558](https://arxiv.org/abs/2006.08558))
  - MCR² framework: promotes diversity via coding rate maximization
  - Foundation for coding-rate regularization approaches

- **Simplifying DINO via Coding Rate Regularization**, 2025 ([arXiv:2502.10385](https://arxiv.org/pdf/2502.10385))
  - JEPA-like training without teacher using coding-rate term based on covariance
  - Prevents collapse without EMA target

## CRATE Family

- **CRATE: Coding Rate Reduction via Transformers** ([website](https://ma-lab-berkeley.github.io/CRATE/))
  - Open loop: sparse rate reduction via MSSA + ISTA
  - Transformer-like blocks interpretable as optimization steps

- **CRATE-MAE: Closed-Loop Training of Masked Autoencoders**, 2024 ([arXiv:2404.02446](https://arxiv.org/abs/2404.02446))
  - Adds predictive objective (closed loop) to CRATE
  - ~30% fewer parameters than baseline
  - Prevents some issues present in open-loop CRATE

- **CRATE-alpha: Scaling CRATE**, 2024 ([arXiv:2405.20299](https://arxiv.org/abs/2405.20299))
  - Fixes scaling issues and "artifacts" in CRATE
  - Open loop but applicable to CRATE-MAE as well

## Efficient Attention and Scaling

- **Token Statistics Transformer (ToST)**, 2024 ([arXiv:2412.17810](https://arxiv.org/abs/2412.17810))
  - Linear scaled attention block via variational rate reduction
  - Same research group as CRATE
  - Enables streaming and long sequences

## Continual Learning and Memory

- **Incremental Learning of Structured Memory via Closed-Loop Transcription**, 2022 ([arXiv:2202.05411](https://arxiv.org/pdf/2202.05411v2))
  - i-CTRL: minmax game between encoder-decoder
  - Effectively alleviates catastrophic forgetting
  - Designed for incremental class learning

- **Continual Learning with Memory Cascades**, 2021 ([OpenReview](https://openreview.net/forum?id=E1xIZf0E7qr))
  - Links EWC's hierarchical Bayesian prior and Benna-Fusi cascade model
  - Shows cascade on hidden parameters equivalent to hierarchical prior
  - Tested on permuted-MNIST with and without task boundaries

- **Context selectivity with dynamic availability enables lifelong continual learning** (Barry & Gerstner, 2024) ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0893608025006082))
  - Metaplasticity based on recent activation history (neuron-level)
  - n-GateON: decides how relevant a synapse is and if to adjust
  - Consolidation/metaplasticity similar to STC but different mechanism

- **Memory consolidation and improvement by synaptic tagging and capture in recurrent neural networks** (Luboeinski & Tetzlaff, 2021) ([Nature](https://www.nature.com/articles/s42003-021-01778-y.pdf))
  - STC helps with "what should we remember" (vs. just inference)
  - Synaptic tagging and capture mechanism

## Local Learning and Plasticity

- **LPL (Halvagal & Zenke 2023)** - Nature
  - Local predictive + Hebbian plasticity
  - Learns invariant representations in deep sensory networks

- **ESPP (Graf et al. 2024)** - arXiv
  - Spiking LPL-like rule for streaming data
  - Online local learning rule with predictive and contrastive coding

- **Learning what matters** (Brito & Gerstner 2024) - PLOS
  - Correlation-invariant sparsity
  - Invariance to second-order input correlations (no need for whitening)

- **Breaking Balance / BCP** (Rossbroich & Zenke 2025) - BioRxiv
  - 3-factor rule with E/I imbalance being the 3rd factor
  - Online credit assignment

- **Fast adaptation with neuronal surprise** (Barry & Gerstner 2024) ([lcnwww.epfl.ch](https://lcnwww.epfl.ch))
  - Learn fast at surprise points but otherwise stable
  - 3-factor rule (pre x post x surprise)

- **Two-factor synaptic consolidation** (Iatropoulos, Gerstner, Brea 2025) - PNAS
  - Sleep-like memory consolidation through multiple internal "factors" per synapse
  - Noise robustness via "tagging"

## Related Work (For Later Reference)

- **Learn in dynamic environment disentangled representations**, 2020 ([NeurIPS](https://proceedings.neurips.cc/paper/2020/file/e449b9317dad920c0dd5ad0a2a2d5e49-Paper.pdf))
  - Similar ideas to seq-JEPA
  - Shows good performance for long-planning prediction
  - Correlation between disentanglement and prediction quality
  - **Limits**: Requires action labels as input (known interactions), no active perception

- **Learning to act without actions**, 2023 ([arXiv:2312.10812](https://arxiv.org/abs/2312.10812))
  - Tries seq-JEPA-like approach without action labels
  - Multiple steps to predict actions and build codebook of action types
  - New version (Dec 2025): LAWM and CLAM - learn action space (transformations) from demonstrations

- **Iterative Latent Equilibrium**, 2025 ([arXiv:2511.21882](https://arxiv.org/html/2511.21882v1))
  - Potentially interesting EBM model
  - May be relevant for future exploration

