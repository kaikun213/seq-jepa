# Documentation Index

Planning notes derived from `docs/plans/initial-plan.md`.
See `README.md` and `AGENTS.md` for repo policy.

## Quick Navigation

- **📊 [Status Tracker](status-tracker.md)**: Current state of all work (START HERE)
- **📝 [Plans](plans/)**: Step plans, milestones, and vision
- **📚 [Reference](reference/)**: Architecture, metrics, and paper references
- **🛠️ [Guides](guides/)**: Setup and practical guides
- **🧪 [Experiments](experiments/)**: Experiment logs and results
- **💡 [Decisions](decisions/)**: Decision log with rationale

## Documentation Structure

### Status & Tracking
- **status-tracker.md**: Current status of all milestones and steps (agents should update this)
- **experiments/**: Individual experiment logs with results and analysis
- **decisions/**: Important decisions with rationale and alternatives

### Plans (`plans/`)
- **initial-plan.md**: Original consolidated plan and rationale
- **vision-and-scope.md**: Vision, scope, and risk boundaries
- **architecture-baseline.md**: Baseline seq-JEPA routing and targets
- **step-0-repo-setup.md**: Environment, code discovery, diagnostics
- **step-1-baseline-repro.md**: Baseline reproduction and ablations
- **step-2-teacherless-rate.md**: Teacherless options and rate regularizer
- **step-3-crate-integration.md**: CRATE integration plan and metrics
- **step-4-tost-streaming.md**: ToST scaling and streaming notes
- **step-5-streaming-continual.md**: Streaming protocols and replay
- **step-6-local-learning.md**: Local-learning approximations
- **milestones-m1-m6.md**: Milestones and deliverables

### Reference (`reference/`)
- **repo-overview.md**: Repository structure, architecture, and where to make changes
- **references.md**: Comprehensive paper references organized by topic
- **metrics-and-gates.md**: Metrics definitions and gate thresholds

### Guides (`guides/`)
- **practical-setup.md**: Local/remote setup and fast iteration suite
- **remote-run-setup.md**: Vast.ai + SLURM setup templates and runtime estimates

## For Agents

When starting work:
1. Check `status-tracker.md` for current state
2. Read relevant step plan (e.g., `plans/step-1-baseline-repro.md`)
3. Update status tracker when starting/completing work
4. Log experiments in `experiments/` directory
5. Document decisions in `decisions/` if significant

See `AGENTS.md` for full agent guidelines.
