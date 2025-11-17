use crate::scanner::ScanSummary;
use serde::Serialize;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

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
                })
                .collect(),
        })
        .collect();

    let file = File::create(output_path).map_err(ReportingError::Io)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &identical_files).map_err(ReportingError::Serialization)
}
