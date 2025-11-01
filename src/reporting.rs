use serde::Serialize;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

#[derive(Serialize)]
struct IdenticalFiles {
    checksum: u64,
    files: Vec<String>,
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

pub fn print_duplicates(checksum_map: &HashMap<u64, Vec<PathBuf>>) {
    for (_, files) in checksum_map.iter().filter(|(_, files)| files.len() > 1) {
        println!("Identical files:");
        for file in files {
            println!("  {}", file.display());
        }
        println!();
    }
}

pub fn write_json(
    checksum_map: &HashMap<u64, Vec<PathBuf>>,
    output_path: &Path,
) -> Result<(), ReportingError> {
    let identical_files: Vec<IdenticalFiles> = checksum_map
        .iter()
        .filter(|(_, files)| files.len() > 1)
        .map(|(checksum, files)| IdenticalFiles {
            checksum: *checksum,
            files: files
                .iter()
                .map(|file| file.to_string_lossy().into_owned())
                .collect(),
        })
        .collect();

    let file = File::create(output_path).map_err(ReportingError::Io)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &identical_files).map_err(ReportingError::Serialization)
}
