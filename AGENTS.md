# AGENTS.md - Camden Image Duplicate Finder

## Commands
- **Check**: `cargo check --workspace --all-targets`
- **Build**: `cargo build --workspace` (or use `task build` for static OpenCV)
- **Test all**: `cargo test --workspace --all-targets`
- **Test single**: `cargo test -p camden-core test_name` (replace `test_name`)
- **Format**: `cargo fmt --all`
- **Lint**: `cargo clippy --workspace --all-targets`

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
