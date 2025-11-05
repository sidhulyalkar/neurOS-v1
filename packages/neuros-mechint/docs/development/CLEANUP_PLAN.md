# Repository Cleanup Plan

## Current State Analysis

**Root Directory Issues:**
- 15 markdown files cluttering the root
- Many are outdated planning/summary documents
- Redundant documentation
- Unclear what's current vs historical

## Cleanup Actions

### 1. Keep at Root (Essential Files)
- ✅ README.md - Main package documentation
- ✅ CONTRIBUTING.md - Contribution guidelines
- ✅ LICENSE - License file
- ✅ pyproject.toml - Package configuration
- **ADD:** CHANGELOG.md - Version history

### 2. Move to docs/archive/ (Historical Documents)
These are completed work summaries and planning docs:
- EXPANSION_PHASE2_SUMMARY.md
- EXPANSION_PLAN.md
- EXPANSION_SUMMARY.md
- GRAPH_BUILDER_IMPLEMENTATION.md
- IMPORT_FIX_PLAN.md
- IMPORT_FIXES.md
- IMPORT_FIXES_SUMMARY.md
- NOTEBOOKS_FIX_GUIDE.md
- PACKAGE_CREATION_SUMMARY.md
- SESSION_HANDOFF.md
- STATUS.md (outdated)
- VALIDATION_REPORT.md

### 3. Move to docs/planning/ (Future Work)
- PACKAGE_REORGANIZATION_PLAN.md

### 4. Create New Documentation Structure

```
neuros-mechint/
├── README.md                    # Main documentation
├── CONTRIBUTING.md              # How to contribute
├── CHANGELOG.md                 # NEW: Version history
├── LICENSE                      # License
├── pyproject.toml              # Package config
│
├── docs/
│   ├── archive/                # Completed work summaries
│   │   ├── phase1/
│   │   │   └── PACKAGE_CREATION_SUMMARY.md
│   │   └── phase2/
│   │       ├── EXPANSION_SUMMARY.md
│   │       ├── EXPANSION_PHASE2_SUMMARY.md
│   │       ├── IMPORT_FIXES_SUMMARY.md
│   │       ├── NOTEBOOKS_FIX_GUIDE.md
│   │       └── VALIDATION_REPORT.md
│   │
│   ├── planning/               # Future work plans
│   │   └── PACKAGE_REORGANIZATION_PLAN.md
│   │
│   ├── development/            # Development notes
│   │   ├── IMPORT_FIX_PLAN.md
│   │   ├── GRAPH_BUILDER_IMPLEMENTATION.md
│   │   └── SESSION_HANDOFF.md
│   │
│   └── guides/                 # User guides
│       └── (future tutorials)
│
├── examples/                   # Jupyter notebooks (already organized)
├── src/                       # Source code
└── tests/                     # Tests
```

## Execution Steps

1. Create docs subdirectories
2. Move files to appropriate locations
3. Create comprehensive CHANGELOG.md
4. Update README.md with current status
5. Remove redundant files
6. Update any broken links

## Benefits

- ✅ Clean root directory (only 5 essential files)
- ✅ Clear separation: current vs historical vs future
- ✅ Easy to find relevant documentation
- ✅ Professional repository structure
- ✅ Clear version history via CHANGELOG
