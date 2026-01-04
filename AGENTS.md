# Agent notes

## Repo overview
- `seq-jepa/` contains the upstream code snapshot.
  - `seq-jepa/src/` holds core models, training, and utilities.
  - `seq-jepa/scripts/` contains entrypoints and helper scripts.
  - `seq-jepa/data/` is for local datasets.
- `docs/` contains planning notes and milestones.
- `assets/`, `static/`, and `index.html` are supporting assets.

## Change policy
- Preserve the upstream `seq-jepa/` code so it can run unmodified.
- Add changes in separate files/modules and create wrapper entrypoints where needed.
- Avoid editing files under `seq-jepa/src` and `seq-jepa/scripts` unless explicitly requested.

## Docs hygiene
- **Status tracking**: Always update `docs/status-tracker.md` when starting/completing work
- **Experiment logs**: Document experiments in `docs/experiments/` following the template
- **Archiving**: When experiments are superseded, move configs to `configs/archive/` and logs to `docs/experiments/archive/` with a short note about what replaced them
- **Decisions**: Log significant decisions in `docs/decisions/` with rationale
- **Index**: Keep `docs/README.md` updated when adding or renaming notes
- See `docs/doc-structure-guide.md` for full documentation best practices
