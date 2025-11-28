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

## Code Style (Rust)
- Edition 2021, format with `rustfmt`, lint with `clippy`
- Use `Result<T, E>` for errors, `?` operator for propagation; avoid `panic!` for recoverable errors
- Prefer immutability; use `thiserror`/`anyhow` for error types
- Keep modules small and focused; document public APIs with `///` doc comments
- Naming: snake_case for functions/variables, PascalCase for types, SCREAMING_SNAKE for constants
