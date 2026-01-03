You’re not crazy to try this. Architecturally, seq‑JEPA is already doing something very “column‑like”: it **separates** (i) an *equivariant* per‑view representation from (ii) an *action‑integrated*, more *invariant* aggregate representation purely by **how information is routed** through the system (encoder → action‑conditioned sequence aggregator → action‑conditioned predictor).

What you’re proposing is essentially:

* keep that **routing / inductive bias** (because it’s the core “invariant–equivariant split”),
* but swap in **rate‑reduction / sparse‑coding machinery** (CRATE/ToST/CTRL‑ish) to get:

  * explicit *subspaces / codebooks*,
  * potentially more **parameter efficiency** and reuse,
  * and ideally better **streaming stability** (less teacher dependence, less forgetting).

That can make sense—but several parts are genuinely experimental, and a few assumptions are easy to accidentally violate. I’ll spell those out, then give you a complete Phase‑1 → Phase‑2 plan with concrete implementation steps starting from the `hafezgh/seq-jepa` repo.

---

## First-principles sanity check: what “should” work vs what’s speculative

### What is well-grounded / evidenced

1. **seq‑JEPA’s architectural split can yield equivariant vs invariant representations without explicit equivariance losses** (equivariant at the encoder outputs, more invariant at the aggregator output) because the model must predict the next view representation using action-conditioned sequential context.

2. **CRATE shows that rate‑reduction + sparse coding can be implemented as a transformer-like block** (MSSA + ISTA) where each layer is interpretable as an optimization step of a sparse rate reduction objective; and it can be used in SSL/autoencoding (CRATE‑MAE). See CRATE ([website](https://ma-lab-berkeley.github.io/CRATE/)) and CRATE-MAE ([arXiv:2404.02446](https://arxiv.org/abs/2404.02446)).

3. **ToST shows a path to linear-time "attention" via variational rate reduction (token statistics self-attention)**—relevant if your sequence length grows or you move to streaming/running stats. See ToST ([arXiv:2412.17810](https://arxiv.org/abs/2412.17810)).

4. **LPL / ESPP / "Learning what matters" are real demonstrations that local/predictive/plasticity-inspired rules can produce invariances and work online** (in their respective settings). See references in Step 6 and `018-references.md`.

### What is genuinely experimental / likely to break first

1. **Removing the EMA/target encoder in seq‑JEPA**
   seq‑JEPA explicitly uses a target encoder that is an EMA of the online encoder, plus stop‑grad, to stabilize the prediction target and prevent collapse.
   Replacing that with “coding‑rate regularization” *might* work, but it’s not guaranteed unless the regularizer is strong and well-conditioned in your regime (batch size, feature dim, sequence length, optimizer, etc.).

2. **Porting CTRL/i‑CTRL’s “closed loop” idea to step‑ahead latent prediction**
   CTRL/i‑CTRL’s “closed loop transcription” is a *minimax equilibrium between encoder and decoder* to map classes to subspaces (LDR) and support incremental class learning.
   seq‑JEPA is *not* a minmax game; it’s one-way predictive modeling across time. So you can borrow **rate‑reduction structure** from CTRL/MCR², but the “closed loop” mechanism is different (details below).

3. **“Codebooks = factorized cortical variables” is plausible but unproven**
   CRATE’s codebook/subspace story is compelling and empirically shows structured attention patterns and sparsity.
   But whether that yields the *specific* factorization you want (object identity vs pose vs lighting etc.) under seq‑JEPA’s sequential/action objective—especially at *small model sizes*—is an open empirical question.

---

## Clarifying the “closed loop” confusion: i‑CTRL vs CRATE‑MAE vs seq‑JEPA

They are all “closed loop” in a loose sense, but **the loops are not the same**.

### 1) CTRL / i‑CTRL "closed-loop transcription"

**Reference**: Incremental Learning of Structured Memory via Closed-Loop Transcription, 2022 ([arXiv:2202.05411](https://arxiv.org/pdf/2202.05411v2))

* Loop meaning: **data ↔ representation ↔ data** via an encoder–decoder pair, where training is framed as a **two-player minimax game** over a rate‑reduction utility to reach an equilibrium representation (classes ↔ subspaces).
* Incremental learning: the method is explicitly designed for **incremental class learning / catastrophic forgetting** and reports strong results with a fixed-capacity autoencoder-like system.
* Key trick: the *minimax formulation* and the explicit **subspace structure (LDR)** are central.

### 2) CRATE‑MAE "closed loop"

**Reference**: CRATE-MAE: Closed-Loop Training of Masked Autoencoders, 2024 ([arXiv:2404.02446](https://arxiv.org/abs/2404.02446))

* Loop meaning: **masked completion / autoencoding** (encode visible tokens → decode masked tokens), with layers derived/interpreted via diffusion/compression connections.
* Not a minmax game: it's trained like an autoencoder/MAE recipe (no adversarial equilibrium in the CTRL sense).
* Key trick: architectural unrolling of sparse rate reduction steps (MSSA+ISTA) giving sparse/structured representations.

### 3) seq‑JEPA "closed loop" (if you want to call it that)

**Reference**: seq-JEPA: Autoregressive Predictive Learning of Invariant-Equivariant World Models ([arXiv:2505.03176](https://arxiv.org/abs/2505.03176))

* Loop meaning: **predict next latent** from past action‑conditioned context (world model style), and compare to a target representation from an EMA target encoder.
* There is no decoder and no minmax game; collapse prevention is via **EMA target + stop-grad**.

**So:**

* i‑CTRL’s loop is *equilibrium transcription for memory*,
* CRATE‑MAE’s loop is *masked reconstruction/completion*,
* seq‑JEPA’s loop is *step-ahead latent prediction*.

What you’re proposing is a new hybrid: **seq‑JEPA’s temporal/action structure + rate‑reduction/sparse coding constraints**.

---

## A complete Phase 1 → Phase 2 plan (self-contained)

### The guiding principle for the whole plan

Don’t try to make everything “white-box + streaming + continual + cortical” at once.

Instead, enforce this invariant:

> At every step you must be able to answer:
> (A) does the model still learn the **invariant–equivariant split**?
> (B) does it **avoid collapse**?
> (C) does it remain **trainable** (no insane conditioning / exploding stats)?
> (D) does it improve **streaming stability** measurably?

Everything below is built around those checkpoints.

---

# Phase 1 — from baseline seq‑JEPA → teacherless + coding rate → CRATE/ToST blocks → streaming

## Step 0 — Set up the repo and create a reproducible experiment harness

Repo:

```text
https://github.com/hafezgh/seq-jepa/tree/main
```

### 0.1 Create a clean dev environment

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

### 0.2 Identify entrypoints + core modules (do this once)

Because I can’t reliably read the repo’s file contents via the browser interface (GitHub is returning “error while loading” for file views in this environment), the most robust way is to locate things locally with search.

Run these from repo root:

**Find training entrypoints**

```bash
rg -n "if __name__ == .__main__." -S seq-jepa
rg -n "argparse|hydra|omegaconf|click" -S seq-jepa
find seq-jepa -maxdepth 3 -type f -name "*.py"
```

**Find seq‑JEPA components by semantic anchors from the paper**

* `[AGG]` token / CLS-like aggregator token

```bash
rg -n "AGG|\\[AGG\\]|CLS" -S seq-jepa
```

* target encoder / EMA

```bash
rg -n "EMA|moving average|momentum encoder|target encoder" -S seq-jepa
```

* cosine similarity loss

```bash
rg -n "cosine|CosineSimilarity|F\\.cosine_similarity" -S seq-jepa
```

* action embedding + concatenation of (z, a)

```bash
rg -n "action|a_M|relative|delta|transform" -S seq-jepa
rg -n "torch\\.cat\\(|concat" -S seq-jepa
```

**Find evaluation scripts**

```bash
rg -n "linear probe|linear_probe|eval|evaluation|downstream" -S seq-jepa
```

### 0.3 Add standardized logging & checkpoints (if not already present)

You want every run to log:

* seed
* dataset + transformation/action set
* sequence length (train & inference)
* architecture dims
* loss components and weights
* collapse diagnostics (below)

Minimum: TensorBoard or WandB.

### 0.4 Add “collapse + structure diagnostics” (mandatory)

Add a small function that logs, per batch:

* feature mean/std per dimension for:

  * encoder outputs `z_t` (equivariant candidate)
  * aggregator output `z_AGG` (invariant candidate)
* effective rank / singular values of batch feature matrix
* average cosine similarity distribution
* (later) sparsity stats if using CRATE/ISTA (e.g., fraction of zeros)

This is the early warning system for every later step.

---

## Step 1 — Reproduce baseline seq‑JEPA (paper objective) as your control

**Reference**: seq-JEPA: Autoregressive Predictive Learning of Invariant-Equivariant World Models ([arXiv:2505.03176](https://arxiv.org/abs/2505.03176))

You need this baseline working before you replace anything.

### 1.1 Baseline objective and architecture (make sure it matches paper)

Paper summary:

* encode first `M` views with encoder `f_θ` → `z_1...z_M`
* concatenate `(z_i, a_i)` tokens (except last)
* sequence transformer encoder `g_ϕ` with learnable `[AGG]` token outputs `z_AGG`
* predictor `h_ψ` takes `(z_AGG, a_M)` to predict `\hat z_{M+1}`
* target encoder is EMA of `f_θ`, target is stop-grad
* loss is **1 − cosine_similarity** between `\hat z_{M+1}` and `z_{M+1}`

### 1.2 Datasets/tasks to run first (choose one “fast”, one “structural”)

From the seq‑JEPA paper, the canonical settings include:

* 3DIEBench for rotation equivariance/invariance comparisons
* CIFAR100/TinyImageNet with handcrafted augmentation actions
* STL‑10 “saccade patches” setting (sequence of glimpses with actions as relative patch displacement)

**Fast sanity task**: STL‑10 saccades or CIFAR100 aug actions
**Structural task**: 3DIEBench (because it cleanly tests equivariance to SO(3) and invariance tasks in the same benchmark)

### 1.3 Baseline measurements you must report

For each dataset, report:

1. **Equivariance metric** computed on `z_t` (encoder outputs).
2. **Invariance metric** (e.g., linear probe) computed on `z_AGG` (aggregator output).
3. A “leakage test”: how well does `z_AGG` do on equivariance vs how well does `z_t` do on invariance? (You want specialization, not total entanglement.)

Also log how performance changes with **sequence length at inference**, because seq‑JEPA claims both invariant and equivariant downstream performance can improve with more observations.

### 1.4 Minimal ablation grid for baseline

* Train sequence length `M_train ∈ {1, 2, 4}`
* Eval/inference sequence length `M_eval ∈ {1, 2, 4, 8}` (if supported)
* with/without action conditioning (remove `a` embedding) to confirm action-conditioning’s role in promoting equivariance in the encoder, as the paper reports.

**Stop condition**: baseline reproduces *qualitatively*:

* `z_t` is more equivariant than `z_AGG`
* `z_AGG` is more invariant than `z_t`

---

## Step 2 — Replace the EMA teacher with "predictive + coding-rate regularizer" (teacherless stability)

**Key references**:
- Learning Diverse and Discriminative Representations via the Principle of Maximum Code Rate Reduction, 2020 ([arXiv:2006.08558](https://arxiv.org/abs/2006.08558)) - MCR² framework
- Simplifying DINO via Coding Rate Regularization, 2025 ([arXiv:2502.10385](https://arxiv.org/pdf/2502.10385)) - JEPA-like training without teacher using coding-rate term based on covariance

This is the first high-risk step.

### 2.1 Decide what “teacherless” means precisely

You have three progressively riskier options:

**Option A (least risky): keep EMA but add coding-rate regularizer first**

* Purpose: verify your coding-rate machinery doesn’t kill learning.
* Then later try removing EMA.

**Option B: remove EMA, but keep stop-grad target from the *same* encoder path**

* Similar spirit to SimSiam-style teacherless setups: online encoder provides target with stop-grad; predictor must prevent collapse.
* seq‑JEPA already has a predictor `h_ψ`, which helps, but it wasn’t validated without EMA.

**Option C: symmetric prediction (two directions)**

* Predict `t+1` from context and also predict `t` from reversed context (or from `t+1`), doubling constraints.
* Often improves stability, but changes the training task.

### 2.2 Implement the new loss as: prediction + rate regularizer (no minmax)

You explicitly said you don’t want a minmax game here; you want step-ahead prediction. Great—do rate reduction as a **regularizer**, not as a two-player equilibrium.

Define:

* `L_pred = 1 - cos( z_hat_{t+1}, sg(z_{t+1}) )` (same as paper but with target possibly from same encoder instead of EMA)
* `L_rate = - R(Z)` (expansion / diversity) or a full MCR²-style disparity term if you can define partitions.

Then optimize:

* `L_total = L_pred + λ_rate * L_rate`

#### 2.2.1 A practical coding-rate regularizer that is teacherless-friendly

Start with a **pure expansion / anti-collapse** term on features:

* let `Z` be a batch of features (shape `[B, D]`) for either `z_AGG` or `z_t`
* compute covariance `C = (Zᵀ Z) / B`
* use `R(Z) = log det( I + α C )`

This is in the spirit of coding-rate objectives (log-det promotes spread across dimensions). It’s also the piece you can compute reliably without labels.

**Where to apply it?**

* Apply `R( z_AGG_batch )` first (because collapse in z_AGG kills invariance).
* Optionally also apply to `z_t_batch` (but be careful: too strong can hurt equivariance learning by forcing isotropy everywhere).

**Hyperparameters**

* `λ_rate`: start tiny (e.g., 1e-3) and sweep log-scale up to 1e-1.
* `α`: tie to feature scale; you can set `α = D / (ε^2)` with ε like in MCR²-style formulas, but treat it as tunable initially.

**Numerical stability**
Use `torch.linalg.slogdet` on a `D×D` matrix; if D is large, this is heavy. That’s exactly where ToST-style variational / token-statistics tricks later become relevant.

### 2.3 Collapse diagnostics specific to teacherless experiments

For every run, plot:

* feature std histogram over dims (should not collapse to ~0)
* eigenvalue spectrum of covariance
* mean cosine similarity between different samples (should not go to 1.0 globally)
* training loss decomposition

### 2.4 The decisive experiment

Run baseline vs teacherless+rate on the same dataset:

* compare invariance/equivariance metrics on `z_AGG` and `z_t`
* compare stability (no collapse)
* compare speed/compute

**If teacherless collapses**, don’t force it. Fall back to:

* keep EMA but **slow its update** (higher momentum)
* keep EMA but add rate regularizer to reduce dependence on EMA over time
* or add a VICReg-style variance/cov penalty as a “stabilizer” (even if your main story is coding rate)

---

## Step 3 — Add a CRATE/CRATE‑α "codebook" structure inside seq‑JEPA (without changing the outer JEPA routing)

**Key references**:
- CRATE: Coding Rate Reduction via Transformers ([website](https://ma-lab-berkeley.github.io/CRATE/)) - open loop, sparse rate reduction via MSSA + ISTA
- CRATE-MAE: Closed-Loop Training of Masked Autoencoders, 2024 ([arXiv:2404.02446](https://arxiv.org/abs/2404.02446)) - adds predictive objective (closed loop), ~30% fewer params, prevents some open-loop issues
- CRATE-alpha: Scaling CRATE, 2024 ([arXiv:2405.20299](https://arxiv.org/abs/2405.20299)) - fixes scaling issues and artifacts in CRATE (open loop, applicable to CRATE-MAE)

Here you aim to replace (or augment) the **sequence aggregator** and/or **predictor** with structured sparse coding blocks.

### 3.1 Why CRATE fits your goal (and what to be careful about)

CRATE layers implement MSSA + ISTA updates, designed to optimize a **sparse rate reduction** objective with codebooks/subspaces, and empirically show structured attention maps and sparsity.

But: CRATE is typically applied to tokens from images (patch tokens) and trained either supervised or via MAE-style autoencoding; applying it to *action-conditioned latent tokens* is new.

### 3.2 Integration point A: Replace the sequence transformer `g_ϕ` with a CRATE-style block

In seq‑JEPA:

* `g_ϕ` consumes tokens: `[ (z_1,a_1), ..., (z_{M-1},a_{M-1}), z_M ]` plus `[AGG]` and outputs `z_AGG`.

Plan:

* Construct tokens `u_i = concat(z_i, emb(a_i))` (keep exactly as paper)
* Add learnable `[AGG]` token as token 0 (same)
* Feed tokens into **L layers of CRATE** (each is MSSA+ISTA) rather than standard self-attention transformer.

**Key engineering decision**: CRATE’s “token semantics” may depend on patch geometry; your tokens are time-ordered. That’s okay—just add:

* positional embeddings for time index i
* and optionally action embedding already included.

### 3.3 Integration point B: Replace predictor MLP `h_ψ` with structured sparse predictor

In seq‑JEPA:

* predictor `h_ψ(z_AGG, a_M)` outputs `\hat z_{M+1}`.

Plan:

* Instead of an MLP, use a small CRATE-like block acting on a 2-token sequence:

  * token0 = z_AGG
  * token1 = emb(a_M)
  * output token0 transformed = predicted next latent
    This is a structured way to force “action-conditioned transformation” to be sparse/subspace-like.

### 3.4 CRATE‑α choice (scaling/performance)

If you see training instabilities or scaling issues with CRATE blocks, CRATE‑α specifically proposes minimal modifications to the sparse coding block + a training recipe to improve scalability.

**Pragmatic rule**:

* Start with vanilla CRATE block (simpler).
* If it underperforms or diverges when you go beyond tiny models or longer sequences, switch to CRATE‑α.

### 3.5 What to measure (to validate the “codebook” story)

Beyond invariance/equivariance:

* sparsity level after ISTA (fraction of ~0 activations)
* per-head / per-codebook specialization:

  * do different heads correlate with different action dimensions?
  * do certain heads stabilize across objects (reusable codebooks)?
* “multiple hypothesis” capability:

  * if you introduce ambiguity (e.g., occluded glimpses), does the representation retain multimodal structure or collapse?

This is the point where your cortical-column hypothesis becomes testable.

---

## Step 4 — Use ToST ideas to make coding-rate / attention compatible with streaming and long sequences

**Key reference**: Token Statistics Transformer (ToST), 2024 ([arXiv:2412.17810](https://arxiv.org/abs/2412.17810)) - linear scaled attention block via variational rate reduction

ToST replaces pairwise attention with a token-statistics operator derived from a **variational form of MCR²**, achieving linear time/memory in tokens.

### 4.1 Two concrete uses of ToST in your pipeline

**Use case 1: Replace `g_ϕ` with ToST attention**

* Keep seq‑JEPA’s outer architecture identical.
* Swap the transformer encoder inside `g_ϕ` to Token Statistics Self-Attention (TSSA).
* Benefits: scaling to large M or streaming windows.

**Use case 2: Replace expensive log-det coding-rate regularizer**

* If you are using `log det(I + αC)` for collapse prevention, ToST’s variational approach provides a route to approximate/maintain relevant statistics incrementally.

### 4.2 Your “Local Coding Rate (LCR)” idea (temporal window)

This is a sensible streaming constraint:

* Maintain a buffer of last N sequences or last N transitions
* Compute coding-rate regularization only on that window (not global dataset)

That *does* violate the “global” spirit of many coding-rate objectives, but it matches the streaming requirement. Treat it as an approximation and test empirically.

---

## Step 5 — Make it truly streaming + continual: catastrophic forgetting tests and mitigations

**Key references**:
- Incremental Learning of Structured Memory via Closed-Loop Transcription, 2022 ([arXiv:2202.05411](https://arxiv.org/pdf/2202.05411v2)) - minmax game between encoder-decoder for continual learning, alleviates catastrophic forgetting
- Continual Learning with Memory Cascades, 2021 ([OpenReview](https://openreview.net/forum?id=E1xIZf0E7qr)) - links EWC's hierarchical Bayesian prior and Benna-Fusi cascade model
- Context selectivity with dynamic availability enables lifelong continual learning (Barry & Gerstner, 2024) ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0893608025006082)) - metaplasticity based on recent activation history, n-GateON mechanism

### 5.1 Define streaming protocols (you need explicit benchmarks)

Pick 2 protocols:

**Protocol A: stationary stream**

* random transitions from a fixed distribution (baseline online training)
* measures: stability, convergence under online updates

**Protocol B: non-stationary stream (continual learning)**
Examples:

* object identities introduced over time
* action distributions shift over time (e.g., new rotations)
* lighting/hue changes later, etc.

For each, measure:

* current performance
* performance on past segments (forgetting)

### 5.2 Add a replay buffer (do this before SI/EWC)

Replay is the simplest high-leverage intervention.

Implementation:

* store `(x_t, a_t, x_{t+1})` triplets (or sequences)
* train online on current + sampled replay
* try reservoir sampling or FIFO
* ablate buffer size

### 5.3 Add Synaptic Intelligence (SI) or EWC as a second layer

SI is a classic per-parameter importance regularizer for continual learning.

Do not implement SI/EWC until you have a clear forgetting problem measured—otherwise you’ll add complexity without diagnosis.

### 5.4 Context gating (optional but often effective)

If your stream has latent context changes, gating (or a small context encoder) can prevent interference.

---

## Step 6 — Localize learning rules (LPL / "Learning what matters" / ESPP) as incremental approximations, not a full rewrite

**Key references**:
- LPL (Halvagal & Zenke 2023) - local predictive + Hebbian plasticity (Nature)
- ESPP (Graf et al. 2024) - spiking LPL-like rule for streaming data (arXiv)
- Learning what matters (Brito & Gerstner 2024) - correlation-invariant sparsity, invariance to second-order input correlations (PLOS)
- Memory consolidation and improvement by synaptic tagging and capture in recurrent neural networks (Luboeinski & Tetzlaff, 2021) ([Nature](https://www.nature.com/articles/s42003-021-01778-y.pdf)) - STC helps with "what should we remember"

### 6.1 Don't jump straight to fully local learning for transformers

Transformers trained fully by local rules is a major research project. Instead, localize in stages:

**Stage 1: Auxiliary local losses**

* Keep backprop for the main objective.
* Add a local predictive/plasticity-inspired auxiliary loss at intermediate layers, e.g.:

  * predict next-layer activity from current activity (layerwise predictive coding)
  * Hebbian-like term

This is where LPL connects: it combines Hebbian and predictive plasticity and learns invariant representations in deep sensory networks.

**Stage 2: Freeze backbone, local-train only the “column module”**

* Treat your seq‑JEPA+CRATE block as the “column”
* Freeze `f` (or pretrain it)
* Learn only the aggregator/predictor online with local-ish rules

**Stage 3: Fully online spiking/local variant**
ESPP is explicitly an online local learning rule for spiking nets with predictive and contrastive coding.
That’s a plausible inspiration if you later want a spiking Monty module.

### 6.2 Use "Learning what matters" as a stabilizer for correlation shifts

Brito & Gerstner's work (Learning what matters, 2024) develops sparse coding / plasticity rules invariant to second-order input correlations—relevant when streaming correlations change and naive Hebbian learning drifts.

Use it as:

* a regularizer or normalization layer behavior
* or to motivate a decorrelation-invariant sparse update inside your codebook learning

---

# Phase 2 — integrate into Monty as a replicated learning module

You said you already assume each module knows its own action transformations / mapping to “object transformations” and feeds seq‑JEPA the right action signal. That’s a good decomposition.

## Step 7 — Define the module API (must be stable before integration)

Your “column module” should expose:

**Forward**

* inputs: observation `x_t`, action `a_t` (to next observation)
* outputs:

  * `z_t` (equivariant)
  * `z_AGG_t` (invariant integrated state)
  * `z_hat_{t+1}` (prediction)

**Update**

* takes `(x_t, a_t, x_{t+1})` (or sequences)
* performs one update step online
* returns diagnostics (loss, collapse stats, sparsity stats)

**State**

* maintains internal memory for the last N tokens (streaming window)

## Step 8 — Single-module Monty experiment

Before scaling to many columns:

* Use 1 module, feed it Monty’s observation stream + actions.
* Evaluate:

  * object ID invariance via `z_AGG`
  * pose/equivariance via `z_t`
  * path integration capability (seq‑JEPA paper highlights sequential aggregation enabling path integration-like tasks).

## Step 9 — Multi-module scaling experiment

Now replicate across columns.

Key design choices:

* shared weights vs per-column weights
* shared codebooks vs per-column codebooks
* whether `z_AGG` is pooled across columns (higher-level area) or kept local

Your “codebook reuse across objects” hypothesis becomes measurable here:

* do columns converge to similar codebooks?
* do they specialize by receptive field / action statistics?

---

# Concrete experiment schedule (so you can actually execute this)

## Milestone M1: baseline seq‑JEPA reproduction

Deliverables:

* scripts/configs for 1 fast dataset + 1 structural dataset
* invariance/equivariance metrics separated for `z_t` and `z_AGG`
* ablation: with/without action conditioning

## Milestone M2: coding-rate regularizer added (EMA still on)

Deliverables:

* stability diagnostics
* no performance collapse
* sweep λ_rate

## Milestone M3: teacherless attempt (remove EMA)

Deliverables:

* either success with diagnostics + metrics, **or** documented failure mode + fallback (EMA retained)

## Milestone M4: CRATE/CRATE‑α aggregator

Deliverables:

* same invariance/equivariance metrics
* sparsity / codebook usage stats
* comparison to baseline transformer g_ϕ

## Milestone M5: streaming protocol + replay

Deliverables:

* defined non-stationary stream benchmark
* forgetting curves
* replay ablations (buffer size, sampling)

## Milestone M6: ToST swap (if sequences long / rate computations heavy)

Deliverables:

* compute/memory vs performance tradeoff
* running-statistics behavior

## Milestone M7: Monty single-module integration

Deliverables:

* stable API + baseline Monty task metrics

---

## The two biggest “assumption violations” to watch

1. **Rate reduction objectives are often global/batch-based.**
   If you go streaming + tiny batch sizes, your covariance/log-det estimates become noisy, and the regularizer can destabilize learning. This is exactly where ToST-style token-statistics and temporal windows become necessary, but you should expect sensitivity.

2. **CRATE’s interpretability/structure depends on token geometry and training recipe.**
   You’re changing the token semantics to (latent+action) time tokens. It may still work, but don’t assume the same emergent “segmentation” phenomena will directly translate. What you *can* test is whether heads/codebooks specialize to action dimensions or object factors.

---

## If you want one “best starting point” (minimal risk, maximal signal)

Do this first:

1. Baseline seq‑JEPA on 3DIEBench-like task (or whatever the repo supports closest)
2. Add **coding-rate expansion regularizer on z_AGG only**, keep EMA
3. Add replay buffer + online stream protocol
4. Only then attempt teacher removal
5. Only then swap g_ϕ for CRATE/ToST

This sequence minimizes the chance you lose the core invariant–equivariant split while adding too many moving parts.

---

If you want, I can also give you:

* a precise set of metrics + plots to include in a paper-quality ablation (collapse, specialization, sparsity, forgetting), and
* suggested hyperparameter ranges for λ_rate / ε / ISTA thresholds / replay schedules—tailored to whether you’re using small “column-sized” models or larger backbones.
