# Decision Log

This directory tracks important decisions made during development, including rationale and alternatives considered.

## Purpose

- **For agents**: Understand why code/architecture choices were made
- **For humans**: Maintain context on trade-offs and constraints
- **For future**: Avoid re-exploring dead ends

## Decision Template

```markdown
# Decision: [Short Title]

**Date**: YYYY-MM-DD
**Context**: [What problem/question prompted this]
**Decision**: [What was decided]
**Rationale**: [Why this choice]
**Alternatives Considered**:
- [Option A]: [Why rejected]
- [Option B]: [Why rejected]
**Impact**: [What this affects]
**Status**: ✅ Implemented | 🔄 In Progress | 📋 Planned
```

## Index

| Decision | Date | Status | Impact |
|----------|------|--------|--------|
| Use wrapper train.py instead of editing upstream | - | ✅ | All training goes through wrapper |
| Separate experiments/ directory for modifications | - | ✅ | Upstream code preserved |
| CIFAR-10 rotations as fast iteration dataset | - | ✅ | Quick feedback loop established |

