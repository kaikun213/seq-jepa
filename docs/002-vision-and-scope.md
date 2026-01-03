# Vision and scope

## Goal
- Keep seq-JEPA's invariant/equivariant split while adding rate-reduction and sparse coding for efficiency and streaming stability.
- Maintain trainability and avoid collapse at every step.

## Guiding checks (must hold at each step)
- z_t stays more equivariant than z_AGG.
- z_AGG stays more invariant than z_t.
- No collapse (feature variance and rank stay healthy).
- Training remains stable and data-efficient.
- Streaming stability improves or at least does not regress.

## What is grounded vs speculative
Grounded:
- seq-JEPA routing can yield equivariant vs invariant representations.
- CRATE blocks can implement sparse rate-reduction steps.
- ToST provides linear-time token statistics attention.

Speculative / high-risk:
- Removing EMA teacher without new stabilization.
- Porting CTRL closed-loop ideas into step-ahead prediction.
- Codebooks necessarily align with desired object factors.

## Closed-loop terminology (do not conflate)
- CTRL/i-CTRL: encoder-decoder minimax equilibrium for class/subspace memory.
- CRATE-MAE: masked reconstruction, no minimax game.
- seq-JEPA: step-ahead latent prediction with EMA target.

## Repo policy
- Keep upstream `seq-jepa/src` and `seq-jepa/scripts` unchanged.
- Implement changes in new modules and wrapper entrypoints so the original code remains runnable.

## Scope
- Plan covers Phase 1 steps 0-6 from `docs/001-initial-plan`.
- Monty integration milestone M7 is out of scope for this repo.
