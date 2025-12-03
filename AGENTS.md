# AGENTS.md - Camden Image Duplicate Finder

## Commands (prefer `task` over raw cargo)
Taskfile sets environment variables for isolated dependencies (OpenCV, vcpkg in `.vendor/`).
- **Check**: `task check`
- **Build**: `task build` (static OpenCV via vcpkg)
- **Release**: `task release`
- **Test all**: `task test`
- **Test single**: `cargo test -p camden-core test_name` (no task wrapper)
- **Format**: `task fmt`
- **Frontend**: `task frontend` (runs Slint GUI)
- **Setup deps**: `task deps` then `task install-opencv-static`

## Architecture
- **camden** (root): CLI entry point using `camden-core`, with `indicatif` progress bars
- **camden-core**: Core library with scanner, detector, reporting, operations, snapshots, thumbnails
- **camden-frontend**: Slint-based GUI (requires vcpkg/OpenCV static build)
- Uses OpenCV for image processing, xxHash checksums, rayon for parallelism

## CLI / Frontend Feature Parity
**IMPORTANT**: When implementing new features, ensure both CLI and frontend remain in sync:
- CLI flags should mirror frontend UI options (e.g., `--enable-classification` ↔ checkbox)
- Use shared `ScanConfig` builder methods from `camden-core` for both
- CLI flags: `--rename-to-guid`, `--detect-low-resolution`, `--enable-classification` (or `--classify`)
- Test features in both interfaces before considering implementation complete
- UX should feel consistent: same defaults, same behavior, same output formats

## Code Style (Rust)
- Edition 2021, format with `rustfmt`, lint with `clippy`
- Use `Result<T, E>` for errors, `?` operator for propagation; avoid `panic!` for recoverable errors
- Prefer immutability; use `thiserror`/`anyhow` for error types
- Keep modules small and focused; document public APIs with `///` doc comments
- Naming: snake_case for functions/variables, PascalCase for types, SCREAMING_SNAKE for constants

## Workflow & Investigation Principles
- Prefer `task` targets (`task check`, `task build`, `task test`, `task fmt`, `task frontend`, etc.) so dotenv-provided environment variables (OpenCV, vcpkg) load correctly; only run cargo/bun directly if the Taskfile lacks the needed target and the user agrees.
- Always ground yourself in context before coding: list files, understand configs/env vars/dependencies, question unclear requirements, and plan what you are changing and why.
- Challenge assumptions up front by clarifying inputs, outputs, and constraints, and consider UX/maintainability impacts across CLI/frontend.
- Keep code modular, testable, DRY, and idiomatic; prefer self-documenting patterns, handle edge cases cleanly, and avoid unnecessary comments unless existing style relies on them for public API documentation.
- Think beyond single files; align frontend, backend, and tooling interactions, and watch for scope creep before executing.
