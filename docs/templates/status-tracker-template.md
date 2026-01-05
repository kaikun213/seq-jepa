# Status Tracker Template

> **Usage**: This is the template for `docs/status-tracker.md`.
> The status tracker is the **single source of truth** for project state.
> Update it when starting/completing work, encountering blockers, or launching remote jobs.

---

# Status Tracker

Last updated: YYYY-MM-DD

## Current Focus

**Active Step**: Step N - [Step Name]  
**Next Step**: [What comes after current work]

## Active Remote Jobs

_Track all running Vast.ai instances to prevent forgotten charges._

| Instance ID | Label | GPU | Status | W&B Run | Est. Cost | Notes |
|-------------|-------|-----|--------|---------|-----------|-------|
| NNNNNN | job-label | RTX 3090 | 🟢 Running | [run-id](link) | $X.XX/hr | Description |
| NNNNNN | job-label | RTX 3060 | 🟡 Starting | — | $X.XX/hr | Description |

**Cost tracking**:
- Current hourly burn rate: $X.XX/hr
- Instances to destroy after completion: [list]

## Milestone Status

_High-level progress on major milestones._

| Milestone | Status | Progress | Key Results |
|-----------|--------|----------|-------------|
| M1 (Baseline) | ✅ Complete | 100% | linacc: XX%, R²: X.XX |
| M2 (Teacherless) | 🟡 In Progress | XX% | [current state] |
| M3 (CRATE) | ⚪ Not Started | 0% | |
| M4 (ToST) | ⚪ Not Started | 0% | |
| M5 (Streaming) | ⚪ Not Started | 0% | |
| M6 (Local) | ⚪ Not Started | 0% | |

### Status Legend
- ✅ Complete
- 🟡 In Progress
- ⚪ Not Started
- ❌ Blocked
- ⏸️ Paused

## Step Status

_Detailed status of individual steps._

| Step | Status | Key Metrics | Blocking Issues |
|------|--------|-------------|-----------------|
| Step 0 - Repo Setup | ✅ Complete | — | — |
| Step 1 - Baseline Repro | ✅ Complete | [W&B](link): XX% linacc | — |
| Step 2 - Teacherless Rate | 🟡 In Progress | [W&B](link): XX% linacc | [blocker if any] |
| Step 3 - CRATE Integration | ⚪ Pending | | |
| Step 4 - ToST Streaming | ⚪ Pending | | |
| Step 5 - Streaming Continual | ⚪ Pending | | |
| Step 6 - Local Learning | ⚪ Pending | | |

## Recent Work

_Changelog-style entries with dates. Most recent first._

### YYYY-MM-DD
- **[Major accomplishment]**:
  - W&B: [run-id](link)
  - Key metrics: [summary]
  - Instance: [if remote, note instance ID and whether destroyed]
- **[Other work]**: Description

### YYYY-MM-DD
- **[Previous day's work]**: Description

## Active Blockers

_What's preventing progress? Empty if none._

- [Blocker 1]: Description and mitigation plan
- [Blocker 2]: Description and mitigation plan

Or: _None currently_

## Upcoming Priorities

_What should be done next, in priority order._

1. **[Priority 1]**: Description
2. **[Priority 2]**: Description
3. **[Priority 3]**: Description

## Notes for Agents

When updating this file:

1. **Always update "Last updated" date**
2. **Remote jobs**: Add to "Active Remote Jobs" when launching, remove when destroyed
3. **Milestones**: Update progress % and status as work completes
4. **Steps**: Link to W&B runs when available
5. **Recent Work**: Add dated entries with key metrics and links
6. **Blockers**: Keep current; remove resolved blockers

### Quick Commands

```bash
# Check running Vast instances
vastai show instances --raw --api-key "$VAST_API_KEY"

# Destroy instance
vastai destroy instance NNNNNN --api-key "$VAST_API_KEY"
```

