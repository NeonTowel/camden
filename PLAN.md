## Vision

Deliver a Windows-first visual review experience for Camden that showcases duplicate image groups with a polished, minimal UI, enables manual selection of files to relocate, and reuses the existing Rust scanning core while keeping the door open for future cross-platform support.

## Guiding Principles

- Maintain a clear separation between core duplicate-detection logic and presentation layers.
- Prefer incremental deliverables that are testable end-to-end (CLI ↔ core ↔ UI).
- Keep artefacts (thumbnails, caches) deterministic and easily purgeable.
- Align all build tooling with the existing `task` runner, `.envrc`, and vcpkg setup.

## Phase 0 — Foundations & Environment

1. Verify OpenCV availability via `task check` and document the exact vcpkg triplet in `.envrc.example`.
2. Add a `task preview` scaffold that will later run the Tauri dev server and watcher.
3. Introduce a dedicated `camden-core` library crate (within the workspace) that contains scanning, detection, and future metadata utilities; update the CLI to depend on it.
4. Update CI (or local scripts) to build both CLI and core library.

## Phase 1 — Core Refactor & API Surface

1. Move existing scanning and detector modules into `camden-core`, exposing:
   - `scan_with_progress(root, cfg) -> DuplicateSummary`
   - `DuplicateGroup` structures with file paths, hashes, and similarity metrics.
2. Preserve CLI behaviour by consuming `camden-core` APIs and keeping current JSON output working (backward compatibility).
3. Add serde-friendly DTOs for UI consumption (`GroupPayload`, `FileEntryPayload`).
4. Document the public API in crate-level docs and keep unit tests green (`task check`).

## Phase 2 — Metadata Enrichment

1. Extend `ImageFeatures` with:
   - EXIF timestamps (when available).
   - File size, dimensions, dominant color swatch.
   - Confidence score derived from hash + heuristics.
2. Implement `enrich_metadata(&[PathBuf]) -> Vec<FileMetadata>` using Rayon.
3. Update DTOs to include metadata and add targeted unit tests (mock fixtures under `tests/assets`).
4. Ensure metadata extraction is optional (feature flag `metadata` default-on).

## Phase 3 — Thumbnail Pipeline

1. Create `thumbnails` module in `camden-core`:
   - Generate 512px WebP thumbnails via OpenCV for supported formats.
   - Fall back to `image`/WIC pipeline for formats OpenCV can’t decode.
2. Store thumbnails under `%LOCALAPPDATA%/Camden/thumbnails/<fingerprint>.webp`.
3. Provide cache management helpers (`ensure_thumbnail`, `purge_thumbnails`).
4. Add integration test that verifies thumbnail generation and cache reuse.

## Phase 4 — Persistence & IPC Contract

1. Define `GroupSnapshot` structure and serialise scan results to `AppData/Local/Camden/cache/groups.json`.
2. Create a thin command-line entry (`camden preview-scan <root>`) that runs scan + metadata + thumbnail generation and writes the snapshot. Use this in manual testing until UI is ready.
3. Specify IPC payloads for the UI (e.g., `load_groups`, `toggle_selection`, `commit_move`).
4. Document the contract in `docs/api-preview.md`.

## Phase 5 — Slint Frontend Scaffold (Windows focus)

1. Add a new `camden-frontend` crate to the workspace that depends on `camden-core` and `slint`.
2. Create initial `.slint` markup defining the app shell (window, root selection controls, placeholder list).
3. Provide a `task frontend` (or similar) that runs `cargo run -p camden-frontend`.
4. Document development workflow for the Slint UI in `docs/frontend.md`.

## Phase 6 — Slint UI Implementation

1. Layout:
   - Header: root chooser, target directory, scan button, status text.
   - Split view: left pane lists duplicate groups; right pane shows thumbnails + metadata.
   - Footer: action buttons (`Move selected`, `Open in Explorer`, etc.).
2. Bind Slint models to `ScanSummary` data structures (converted into Slint-friendly arrays/structs).
3. Load thumbnail images from the cache and display them with fallback placeholders.
4. Ensure the UI remains responsive by running scans and file operations on background threads and updating the model via the Slint event loop.

## Phase 7 — Selection & Move Flow

1. Track per-file selection state inside Slint models; surface counts in header/footer.
2. Implement keyboard shortcuts (arrow navigation, space to toggle selection, enter to open detail).
3. Hook “Move selected” to `camden_core::move_paths`, streaming progress back to the UI.
4. Provide success/error notifications and write a post-move summary.

## Phase 8 — Polish & QA

1. Add filters (file type chips, confidence slider, size range) with live model updates.
2. Surface key metadata (dimensions, captured time, size) directly in the list items or detail pane.
3. Harden error handling (thumbnail load failures, IO errors) with retry/skip options.
4. Capture screenshots, usage docs, and troubleshooting notes for the Slint UI.
5. Identify follow-up work for macOS/Linux parity or alternate frontends if needed.

## Acceptance Criteria

- `task check` continues to pass across all phases, ensuring the Rust core remains buildable.
- `task frontend` (or equivalent) launches the Slint application, loads cached scan data, and renders duplicate groups with smooth interaction.
- Users can visually inspect groups, toggle selections, and relocate chosen files through the UI, with observable progress feedback and resulting filesystem changes.
- Documentation reflects setup, development workflow, and troubleshooting steps for the preview feature.

