use crate::scanner::ScanSummary;
use serde::Serialize;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use time::OffsetDateTime;

#[derive(Serialize)]
struct IdenticalFiles {
    checksum: u64,
    files: Vec<FileDescriptor>,
}

#[derive(Serialize)]
struct FileDescriptor {
    path: String,
    size_bytes: u64,
    dimensions: (i32, i32),
    modified: Option<String>,
    captured_at: Option<String>,
    dominant_color: [u8; 3],
    confidence: f32,
    thumbnail: Option<String>,
}

#[derive(Debug)]
pub enum ReportingError {
    Io(std::io::Error),
    Serialization(serde_json::Error),
}

impl Display for ReportingError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "io error: {}", error),
            Self::Serialization(error) => write!(f, "serialization error: {}", error),
        }
    }
}

impl Error for ReportingError {}

pub fn print_duplicates(summary: &ScanSummary) {
    for group in summary.duplicate_groups() {
        println!("Identical files:");
        for entry in &group.files {
            println!("  {}", entry.path.display());
        }
        println!();
    }
}

pub fn write_json(summary: &ScanSummary, output_path: &Path) -> Result<(), ReportingError> {
    let identical_files: Vec<IdenticalFiles> = summary
        .duplicate_groups()
        .map(|group| IdenticalFiles {
            checksum: group.fingerprint,
            files: group
                .files
                .iter()
                .map(|entry| FileDescriptor {
                    path: entry.path.to_string_lossy().into_owned(),
                    size_bytes: entry.size_bytes,
                    dimensions: entry.dimensions,
                    modified: entry.modified.clone(),
                    captured_at: entry.captured_at.clone(),
                    dominant_color: entry.dominant_color,
                    confidence: entry.confidence,
                    thumbnail: entry
                        .thumbnail
                        .as_ref()
                        .map(|path| path.to_string_lossy().into_owned()),
                })
                .collect(),
        })
        .collect();

    let file = File::create(output_path).map_err(ReportingError::Io)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &identical_files).map_err(ReportingError::Serialization)
}

/// Classification entry for a single file.
#[derive(Serialize)]
pub struct ClassificationEntry {
    /// Relative path from root
    pub path: String,
    /// Moderation tier: Safe, Sensitive, Mature, Restricted
    pub moderation_tier: String,
    /// Up to 5 descriptive tags
    pub tags: Vec<String>,
}

/// Complete classification report for a directory scan.
#[derive(Serialize)]
pub struct ClassificationReport {
    /// Version of the report format
    pub version: u32,
    /// When the classification was performed
    pub created_at: String,
    /// Root directory that was scanned
    pub root: String,
    /// Number of files classified
    pub file_count: usize,
    /// Classification results
    pub files: Vec<ClassificationEntry>,
}

/// Write classification results to a single JSON file at the source root.
///
/// Creates `<root>/.camden-classifications.json` containing moderation tiers
/// and tags for all classified images.
pub fn write_classification_report(
    summary: &ScanSummary,
    root: &Path,
) -> Result<(), ReportingError> {
    let mut entries: Vec<ClassificationEntry> = Vec::new();

    for group in &summary.groups {
        for file in &group.files {
            // Only include files that have classification data
            if file.moderation_tier.is_some() || !file.tags.is_empty() {
                let rel_path = file
                    .path
                    .strip_prefix(root)
                    .unwrap_or(&file.path)
                    .to_string_lossy()
                    .into_owned();

                entries.push(ClassificationEntry {
                    path: rel_path,
                    moderation_tier: file.moderation_tier.clone().unwrap_or_default(),
                    tags: file.tags.clone(),
                });
            }
        }
    }

    // Don't write empty reports
    if entries.is_empty() {
        return Ok(());
    }

    let report = ClassificationReport {
        version: 1,
        created_at: OffsetDateTime::now_utc()
            .format(&time::format_description::well_known::Rfc3339)
            .unwrap_or_else(|_| "unknown".to_string()),
        root: root.to_string_lossy().into_owned(),
        file_count: entries.len(),
        files: entries,
    };

    let output_path = root.join(".camden-classifications.json");
    let file = File::create(&output_path).map_err(ReportingError::Io)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &report).map_err(ReportingError::Serialization)
}
