# CLAUDE.md

Guidance for working with this Rust workspace.

## Build & Development

Always use `task` commands (not direct cargo) to load environment variables:

| Command         | Purpose            |
| --------------- | ------------------ |
| `task check`    | Workspace analysis |
| `task build`    | Debug build        |
| `task release`  | Optimized build    |
| `task test`     | Run tests          |
| `task fmt`      | Format code        |
| `task frontend` | Launch Slint GUI   |

## Project Structure

Three-crate Rust workspace:

- **camden** (src/) - CLI binary with command dispatch, uses `indicatif` progress bars
- **camden-core** (core/) - Scanner, detector, classifier, operations, snapshots, thumbnails
- **camden-frontend** - Slint-based GUI application

Dependencies: OpenCV (image processing), xxHash (checksums), rayon (parallelism)

## Critical Directives

### CLI/Frontend Parity (MANDATORY)

When implementing new features, ensure both CLI and frontend remain in sync:

- CLI flags should mirror frontend UI options (e.g., `--enable-classification` ↔ checkbox)
- Use shared `ScanConfig` builder from `camden-core` for both
- Standard flags: `--rename-to-guid`, `--detect-low-resolution`, `--enable-classification`
- Test features in both interfaces before considering implementation complete
- UX must feel consistent: same defaults, behavior, output formats

### Code Style (Rust)

- Edition 2021, format with `rustfmt`, lint with `clippy`
- Use `Result<T, E>` with `?` operator; **no `panic!` for recoverable errors**
- Prefer immutability; use `thiserror`/`anyhow` for error types
- Keep modules small and focused; document public APIs with `///` doc comments
- Naming: snake_case (functions/variables), PascalCase (types), SCREAMING_SNAKE (constants)

### Development Workflow

- Ground yourself in context before coding: understand configs, dependencies, and requirements
- Challenge assumptions upfront: clarify inputs, outputs, constraints, and UX/maintainability impacts
- Keep code modular, testable, DRY, and idiomatic; use self-documenting patterns
- Think beyond single files: align frontend, backend, and tooling interactions
- Watch for scope creep before executing

## @TODO.md Protocol

- Parse @TODO.md: `@[agent]: [task]`
- Execute agents sequentially by tag order
- qa-agent ALWAYS final gatekeeper
- Mark completed: `✅ [task] done by [agent]`
- Report progress.md summary

### AI Classification

The `classification` feature (default-on) uses ONNX Runtime:

- Models in `.vendor/models/` (ONNX format)
- Configuration via `camden-classifier.toml`
- Check `layout` field (NCHW vs NHWC) for tensor compatibility
