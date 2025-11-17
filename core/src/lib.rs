//! Core duplicate detection engine for Camden.
//!
//! This crate exposes high-level scanning, reporting, and file-move
//! operations used by both the CLI and future preview UI. The public API
//! focuses on data-transfer objects (`ScanSummary`, `DuplicateGroup`,
//! `DuplicateEntry`) that are serialisable for downstream consumers.

pub mod detector;
pub mod operations;
pub mod progress;
pub mod reporting;
pub mod scanner;
pub mod snapshot;
pub mod thumbnails;

pub use operations::{move_duplicates, MoveError, MoveStats};
pub use reporting::{print_duplicates, write_json};
pub use scanner::{
    count_entries, scan, DuplicateEntry, DuplicateGroup, ScanConfig, ScanSummary, ThreadingMode,
};
pub use snapshot::{
    create_snapshot, default_snapshot_path, read_snapshot, write_snapshot, ScanSnapshot,
    SnapshotError,
};
pub use thumbnails::{ThumbnailCache, ThumbnailError};
