use crate::scanner::ScanSummary;
use dirs::data_local_dir;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use time::{format_description::well_known::Rfc3339, OffsetDateTime};

const SNAPSHOT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanSnapshot {
    pub version: u32,
    pub generated_at: String,
    pub root: PathBuf,
    pub summary: ScanSummary,
}

impl ScanSnapshot {
    pub fn new(root: PathBuf, summary: ScanSummary) -> Self {
        Self {
            version: SNAPSHOT_VERSION,
            generated_at: OffsetDateTime::now_utc()
                .format(&Rfc3339)
                .unwrap_or_else(|_| String::from("unknown")),
            root,
            summary,
        }
    }
}

pub fn create_snapshot(root: &Path, summary: ScanSummary) -> ScanSnapshot {
    ScanSnapshot::new(root.to_path_buf(), summary)
}

pub fn write_snapshot<P: AsRef<Path>>(
    snapshot: &ScanSnapshot,
    path: P,
) -> Result<(), SnapshotError> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| SnapshotError::Io {
            source,
            path: parent.to_path_buf(),
        })?;
    }
    let file = File::create(path).map_err(|source| SnapshotError::Io {
        source,
        path: path.to_path_buf(),
    })?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, snapshot).map_err(SnapshotError::Serialization)
}

pub fn read_snapshot<P: AsRef<Path>>(path: P) -> Result<ScanSnapshot, SnapshotError> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|source| SnapshotError::Io {
        source,
        path: path.to_path_buf(),
    })?;
    let reader = BufReader::new(file);
    serde_json::from_reader(reader).map_err(SnapshotError::Serialization)
}

pub fn default_snapshot_path() -> Option<PathBuf> {
    let mut dir = data_local_dir()?;
    dir.push("Camden");
    dir.push("cache");
    dir.push("groups.json");
    Some(dir)
}

#[derive(Debug)]
pub enum SnapshotError {
    Io {
        source: std::io::Error,
        path: PathBuf,
    },
    Serialization(serde_json::Error),
}

impl Display for SnapshotError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { source, path } => write!(f, "io error for {}: {}", path.display(), source),
            Self::Serialization(error) => write!(f, "serialization error: {}", error),
        }
    }
}

impl Error for SnapshotError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Serialization(error) => Some(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scanner::{DuplicateEntry, DuplicateGroup, ScanSummary};
    use tempfile::tempdir;

    #[test]
    fn writes_and_reads_snapshot() {
        let dir = tempdir().unwrap();
        let output = dir.path().join("groups.json");
        let summary = ScanSummary {
            groups: vec![DuplicateGroup {
                fingerprint: 123,
                files: vec![DuplicateEntry {
                    path: PathBuf::from("a.jpg"),
                    size_bytes: 42,
                    dimensions: (100, 50),
                    modified: Some(String::from("now")),
                    captured_at: None,
                    dominant_color: [10, 20, 30],
                    confidence: 0.9,
                    thumbnail: None,
                    resolution_tier: crate::resolution::ResolutionTier::High,
                    moderation_tier: None,
                    tags: Vec::new(),
                }],
            }],
        };

        let snapshot = create_snapshot(Path::new("/tmp"), summary.clone());
        write_snapshot(&snapshot, &output).unwrap();
        let loaded = read_snapshot(&output).unwrap();
        assert_eq!(loaded.version, SNAPSHOT_VERSION);
        assert_eq!(loaded.root, PathBuf::from("/tmp"));
        assert_eq!(loaded.summary.groups.len(), 1);
        assert_eq!(loaded.summary.groups[0].files.len(), 1);
        assert_eq!(loaded.summary.groups[0].files[0].size_bytes, 42);
    }

    #[test]
    fn default_snapshot_path_points_to_cache() {
        let path = default_snapshot_path();
        if let Some(path) = path {
            assert!(path.ends_with(Path::new("Camden/cache/groups.json")));
        }
    }
}
