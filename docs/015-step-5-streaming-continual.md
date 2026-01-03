# Step 5 - Streaming and continual learning

**Key references**:
- Incremental Learning of Structured Memory via Closed-Loop Transcription, 2022 ([arXiv:2202.05411](https://arxiv.org/pdf/2202.05411v2)) - minmax game between encoder-decoder for continual learning, alleviates catastrophic forgetting
- Continual Learning with Memory Cascades, 2021 ([OpenReview](https://openreview.net/forum?id=E1xIZf0E7qr)) - links EWC's hierarchical Bayesian prior and Benna-Fusi cascade model
- Context selectivity with dynamic availability enables lifelong continual learning (Barry & Gerstner, 2024) ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0893608025006082)) - metaplasticity based on recent activation history, n-GateON mechanism

## Goal
- Define streaming protocols and measure forgetting.

## Protocols
- Protocol A: stationary stream, measure stability and convergence.
- Protocol B: non-stationary stream (new objects, actions, or lighting), measure forgetting.

## Replay buffer
- Store (x_t, a_t, x_{t+1}) or sequences.
- Try reservoir sampling or FIFO.
- Ablate buffer size and sampling mix.

## Optional mitigation
- Add SI or EWC only after measurable forgetting.
- Context gating if stream has latent regime shifts.

## Deliverables
- Forgetting curves and replay ablations.
