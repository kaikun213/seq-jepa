# Documentation Structure Guide

## Best Practices for Agent-Friendly Documentation

This guide explains the documentation structure and best practices for maintaining it, especially when working with AI agents.

## Core Principles

### 1. **Separation of Concerns**
- **Plans** (`plans/`): What we want to do
- **Reference** (`reference/`): Architecture, metrics, and paper references
- **Guides** (`guides/`): Setup and practical guides
- **Status** (`status-tracker.md`): What's actually happening
- **Experiments** (`experiments/`): What was tried and results
- **Decisions** (`decisions/`): Why choices were made

### 2. **Single Source of Truth**
- **Status Tracker** (`status-tracker.md`) is the authoritative state
- Agents should update it when starting/completing work
- Other docs reference it but don't duplicate status

### 3. **Machine-Readable Format**
- Use consistent status indicators: ✅ 🔄 ⏸️ 📋 ❌ 🔍
- Structured tables for easy parsing
- Clear section headers for agent navigation

### 4. **Progressive Detail**
- High-level overview in status tracker
- Detailed plans in `plans/` directory
- Experiment-specific details in `experiments/`
- Reference material in `reference/`
- Setup guides in `guides/`

## Directory Structure

```
docs/
├── plans/            # Planning documents (what to do)
│   ├── initial-plan.md
│   ├── vision-and-scope.md
│   ├── architecture-baseline.md
│   ├── step-0-repo-setup.md
│   ├── step-1-baseline-repro.md
│   └── ...
├── reference/        # Reference material
│   ├── repo-overview.md
│   ├── references.md
│   └── metrics-and-gates.md
├── guides/          # Setup and practical guides
│   ├── practical-setup.md
│   └── remote-run-setup.md
├── templates/        # Templates for documentation
│   ├── experiment-template.md   # Copy for new experiments
│   └── status-tracker-template.md  # Reference for status structure
├── status-tracker.md  # Current state (START HERE)
├── experiments/      # Experiment logs (what was tried)
│   ├── README.md
│   ├── step-1-baseline-*.md
│   └── ...
├── decisions/        # Decision log (why choices were made)
│   └── README.md
└── README.md         # This index
```

## Status Indicators

Use these consistently across all docs:

- ✅ **Complete**: Fully implemented and validated
- 🔄 **In Progress**: Actively being worked on
- ⏸️ **Paused**: Started but temporarily stopped
- 📋 **Planned**: Not yet started
- ❌ **Blocked**: Cannot proceed due to dependencies
- 🔍 **Investigating**: Debugging or exploring approach

## Workflow for Agents

### Starting New Work

1. **Read status tracker**: `docs/status-tracker.md`
   - Check what's currently in progress
   - Identify next task
   - Note any blockers

2. **Read relevant plan**: e.g., `docs/plans/step-1-baseline-repro.md`
   - Understand objectives
   - Note requirements
   - Check dependencies

3. **Update status tracker**:
   - Mark task as "🔄 In Progress"
   - Add to "Currently Working On"
   - Note start date and owner

### During Work

4. **Create experiment log** (if doing experiments):
   - Copy `templates/experiment-template.md` to `experiments/step-N-name.md`
   - Fill in all sections, especially **Preliminary Analysis**
   - Link to W&B runs and configs

5. **Document decisions** (if significant):
   - Add to `decisions/` directory
   - Explain rationale and alternatives
   - Note impact

### Launching Remote Jobs

5. **Track remote instances** (critical for cost control):
   - Add to "Active Remote Jobs" table in status tracker
   - Include: Instance ID, label, GPU, W&B link, estimated cost
   - After completion: **destroy instance** and remove from table
   - Never leave instances running untracked

### Completing Work

6. **Update status tracker**:
   - Mark task as "✅ Complete"
   - Move to "Recently Completed"
   - Update progress percentages
   - Note completion date
   - **Destroy any remote instances** used for this work

7. **Update experiment log**:
   - Mark status as complete
   - Add final results and analysis
   - **Fill in Preliminary Analysis** (required)
   - Note any follow-ups needed

## File Naming Conventions

### Planning Documents (`plans/`)
- `initial-plan.md`: Original plan
- `vision-and-scope.md`: Vision and scope
- `architecture-baseline.md`: Architecture overview
- `step-0-*.md`: Step 0 documents
- `step-1-*.md`: Step 1 documents
- etc.

### Reference Documents (`reference/`)
- `repo-overview.md`: Repository structure and architecture
- `references.md`: Paper references
- `metrics-and-gates.md`: Metrics definitions

### Guides (`guides/`)
- `practical-setup.md`: Local setup guide
- `remote-run-setup.md`: Remote execution guide

### Status Documents
- `status-tracker.md`: Main status document

### Experiment Logs
- `step-1-baseline-smoke.md`: Specific experiment
- `step-2-teacherless-lambda-sweep.md`: Hyperparameter sweep
- Use descriptive names that indicate step and variant

### Decision Logs
- `decision-001-use-wrapper-train.md`: Numbered decisions
- Or use dates: `2025-01-02-separate-experiments-dir.md`

## What Goes Where?

### Status Tracker (`status-tracker.md`)
- Current state of all milestones
- What's actively being worked on
- What's blocking
- High-level progress

**Do NOT put here**:
- Detailed experiment results
- Code changes
- Detailed analysis

### Step Plans (`plans/`)
- What needs to be done
- Requirements and dependencies
- Success criteria
- Implementation approach

**Do NOT put here**:
- Current status (use status tracker)
- Experiment results (use experiments/)
- Decisions (use decisions/)

### Experiment Logs (`experiments/`)
- What was tried
- Configs and hyperparameters
- Results and metrics
- Analysis and observations
- What worked/didn't work

**Do NOT put here**:
- General plans (use step docs)
- Status updates (use status tracker)
- Architecture decisions (use decisions/)

### Decision Log (`decisions/`)
- Why choices were made
- Alternatives considered
- Rationale
- Impact assessment

**Do NOT put here**:
- Implementation details (use code/comments)
- Experiment results (use experiments/)
- Current status (use status tracker)

## Maintenance Guidelines

### For Agents

1. **Always update status tracker** when:
   - Starting new work
   - Completing work
   - Encountering blockers
   - Changing priorities

2. **Create experiment logs** for:
   - Any hyperparameter sweeps
   - Architecture changes
   - Significant debugging sessions
   - Failed attempts (document why)

3. **Document decisions** when:
   - Choosing between alternatives
   - Making architectural changes
   - Changing approach mid-step
   - Rejecting a planned approach

4. **Keep plans updated** when:
   - Requirements change
   - Dependencies shift
   - Success criteria evolve

### For Humans

- Review status tracker weekly
- Clean up completed experiments periodically
- Consolidate related decisions
- Update plans when scope changes

## Example: Complete Workflow

### Scenario: Starting Step 2 (Teacherless + Coding Rate)

1. **Agent reads** `status-tracker.md`:
   - Sees Step 1 is complete
   - Step 2 is "📋 Planned"
   - No blockers

2. **Agent reads** `plans/step-2-teacherless-rate.md`:
   - Understands three options (A, B, C)
   - Notes hyperparameters to tune
   - Sees success criteria

3. **Agent updates** `status-tracker.md`:
   - Changes Step 2 status to "🔄 In Progress"
   - Adds to "Currently Working On"
   - Sets progress to 10%

4. **Agent implements** Option A (keep EMA, add rate):
   - Creates `experiments/step-2-teacherless-option-a.md`
   - Documents config and hyperparameters
   - Runs experiment

5. **Agent documents results**:
   - Updates experiment log with metrics
   - Analyzes what worked
   - Notes any issues

6. **Agent makes decision**:
   - Creates `decisions/decision-002-option-a-over-b.md`
   - Explains why Option A was chosen
   - Documents alternatives considered

7. **Agent completes**:
   - Updates status tracker to "✅ Complete"
   - Sets progress to 100%
   - Moves to "Recently Completed"

## Benefits of This Structure

1. **For Agents**:
   - Easy to find current state
   - Clear what to do next
   - Know where to document work
   - Understand context quickly

2. **For Humans**:
   - See progress at a glance
   - Understand what was tried
   - Know why decisions were made
   - Easy to onboard new contributors

3. **For Project**:
   - Maintains context over time
   - Avoids re-exploring dead ends
   - Tracks what worked/didn't
   - Clear audit trail

## Anti-Patterns to Avoid

❌ **Don't**: Put status in planning docs
✅ **Do**: Keep plans static, status in tracker

❌ **Don't**: Mix experiment results with plans
✅ **Do**: Separate experiments/ directory

❌ **Don't**: Forget to update status tracker
✅ **Do**: Make it a habit when starting/completing

❌ **Don't**: Document every tiny decision
✅ **Do**: Only significant architectural/approach decisions

❌ **Don't**: Duplicate information across files
✅ **Do**: Reference other docs, don't copy

## Questions?

- **What's the current state?** → Check `status-tracker.md`
- **What should I do?** → Check status tracker + relevant step plan in `plans/`
- **What was tried?** → Check `experiments/` directory
- **Why was X chosen?** → Check `decisions/` directory
- **How does Y work?** → Check `reference/repo-overview.md` or step plans in `plans/`

