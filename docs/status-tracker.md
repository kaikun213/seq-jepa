# Status Tracker

Last updated: 2026-01-04

## Current Focus

**Active Step**: Step 1 - Baseline Reproduction  
**Next Step**: Step 2 - Teacherless Rate

## Milestone Status

| Milestone | Status | Progress | Notes |
|-----------|--------|----------|-------|
| M1 (Baseline) | 🟡 In Progress | 95% | Paper-aligned CIFAR-100 run in progress |
| M2 (Teacherless) | ⚪ Not Started | 0% | |
| M3 (CRATE) | ⚪ Not Started | 0% | |
| M4 (ToST) | ⚪ Not Started | 0% | |
| M5 (Streaming) | ⚪ Not Started | 0% | |
| M6 (Local) | ⚪ Not Started | 0% | |

## Step Status

| Step | Status | Blocking Issues |
|------|--------|-----------------|
| Step 0 - Repo Setup | ✅ Complete | — |
| Step 1 - Baseline Repro | 🟡 In Progress | Paper-aligned run in progress; eval pending |
| Step 2 - Teacherless Rate | ⚪ Pending | Waiting on Step 1 |
| Step 3 - CRATE Integration | ⚪ Pending | |
| Step 4 - ToST Streaming | ⚪ Pending | |
| Step 5 - Streaming Continual | ⚪ Pending | |
| Step 6 - Local Learning | ⚪ Pending | |

## Recent Work

### 2026-01-04
- Reorganized documentation structure into subdirectories
- Set up local git config for private email
- Ran clean Vast CIFAR-100 baseline; results far below paper targets
- Started paper-aligned CIFAR-100 run on Vast (long, eval pending)

### 2026-01-03
- Smoke and quick runs passing on CIFAR-10 rotations
- W&B integration working
- Created metrics-and-gates.md with threshold definitions

### 2026-01-02
- Initial repo setup complete
- Created experiments/ directory structure
- Added CIFAR-100 augmentation support

## Active Blockers

- Baseline performance far below paper; need longer training and paper-aligned probe evaluation

## Notes for Agents

When updating this file:
1. Update the "Last updated" date
2. Adjust milestone/step status as work progresses
3. Add entries to "Recent Work" with date headers
4. Document any blockers in "Active Blockers"
