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

## Phase 5 — Tauri + React App Scaffold (Windows focus)

1. Generate a new Tauri app under `ui/previewer` using React + TypeScript + Tailwind.
2. Configure Vite dev server, Prettier, ESLint, and include `task preview` to run `cargo tauri dev`.
3. Implement Rust-side Tauri commands that wrap `camden-core` functions (load snapshot, stream progress, execute moves).
4. Add smoke tests for commands (`cargo test -p previewer`).

## Phase 6 — UI Implementation

1. Layout:
   - Header: root path, target selector, progress pills.
   - Masonry grid: cards per group with horizontal scroller of thumbnails.
   - Detail drawer: zoomable preview, metadata grid, selection toggles.
   - Footer: action buttons (`Move selected`, `Skip`, `Mark all safe`).
2. Build components with Tailwind + Framer Motion:
   - Responsive (min 1280px width target, degrade gracefully on smaller screens).
   - Lightweight glassmorphism accents, Fluent-inspired shadows.
3. Integrate state management (TanStack Query or Zustand) to handle snapshot loading, selection state, and command calls.
4. Implement optimistic updates and skeleton loaders for fluid UX.

## Phase 7 — Selection & Move Flow

1. Track selections per group; surface counts in header/footer.
2. Provide keyboard shortcuts (arrow nav, space to toggle) and context menu actions.
3. Hook “Move selected” to background Rust command that reuses `move_duplicates` behaviour (with progress events).
4. Show progress modal with streaming updates; write post-move summary to disk.

## Phase 8 — Polish & QA

1. Add filtering controls (file type pill chips, confidence slider, size range).
2. Provide quick metadata badges (dimensions, size, modified time) on card faces.
3. Implement error handling UX (e.g., thumbnail failures, permission issues) with retry options.
4. Write end-to-end instructions in `docs/preview-workflow.md`.
5. Capture follow-up tasks for cross-platform support and advanced dedupe heuristics.

## Acceptance Criteria

- `task check` continues to pass across all phases, ensuring the Rust core remains buildable.
- `task preview` launches the Tauri app, loads cached scan data, and renders duplicate groups with smooth interaction.
- Users can visually inspect groups, toggle selections, and relocate chosen files through the UI, with observable progress feedback and resulting filesystem changes.
- Documentation reflects setup, development workflow, and troubleshooting steps for the preview feature.

