# Subagent Workflow Guide

Quick reference for using specialized agents to boost productivity on the Camden codebase.

## Agent Selection Matrix

### 🔍 Explore Agent
**When**: You need to understand existing code, find implementations, or map the codebase structure

**Ideal Queries**:
- "Where is the ScanConfig builder implemented and how is it used?"
- "How does classification flow through the three crates?"
- "Find all places where OpenCV image processing happens"
- "Show me how the frontend GUI updates scan results"
- "Where are detector models loaded and configured?"

**Thoroughness Levels**:
- `quick` - Fast file pattern matching (find detector implementations)
- `medium` - Moderate exploration (trace a feature across crates)
- `very thorough` - Comprehensive analysis (understand entire feature ecosystem)

**Output**: File locations, implementation patterns, architectural understanding

---

### 📐 Plan Agent
**When**: Implementing features that touch multiple crates or need CLI/Frontend parity (MANDATORY)

**Ideal Queries**:
- "Plan how to add a new scan classification model option (must sync CLI + frontend)"
- "Design refactoring of error handling across all three crates"
- "Plan thumbnail caching system that works in both CLI and GUI"
- "How should we implement a new --rename-to-guid variant?"

**Critical**: Always use Plan for ANY feature that affects both CLI and frontend to ensure parity upfront

**Output**: Step-by-step implementation plan, critical files to modify, architectural decisions

---

### 🛠️ General-Purpose Agent
**When**: Complex multi-step tasks, research, or understanding dependencies

**Ideal Queries**:
- "Search the codebase for all uses of rayon parallelism and show me patterns"
- "Find performance bottlenecks in the image processing pipeline"
- "Help me understand the ONNX Runtime integration for classification"
- "Refactor the config loading system to support new file formats"

**Output**: Comprehensive findings, code locations, implementation insights

---

## Recommended Workflow

### For New Features
1. **Explore** → Understand what already exists
2. **Plan** → Design multi-crate solution with CLI/Frontend parity in mind
3. **Build** → Implement with confidence

### For Bug Fixes
1. **Explore** → Find where the bug manifests
2. **Direct implementation** → Fix it

### For Performance/Refactoring
1. **General-Purpose** → Research current patterns
2. **Plan** → If it spans multiple crates
3. **Implement** → With clear architecture

---

## Key Constraints for This Codebase

- **CLI/Frontend Parity**: Both interfaces must stay in sync. Always plan this upfront.
- **Three Crates**: Changes often ripple through camden, camden-core, and camden-frontend
- **Shared Patterns**: ScanConfig builder, ONNX models, OpenCV integration
- **Dependencies**: OpenCV (image), xxHash (checksums), rayon (parallelism), Slint (GUI)

Use agents to ensure you understand these constraints before coding.
