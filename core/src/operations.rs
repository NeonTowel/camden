use crate::progress;
use crate::scanner::ScanSummary;
use indicatif::ProgressBar;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};

pub struct MoveStats {
    pub moved: usize,
}

#[derive(Debug)]
pub enum MoveError {
    Io {
        source: std::io::Error,
        path: PathBuf,
    },
    MissingFileName(PathBuf),
}

impl Display for MoveError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { source, path } => write!(f, "failed to move {}: {}", path.display(), source),
            Self::MissingFileName(path) => write!(f, "file name not found for {}", path.display()),
        }
    }
}

impl Error for MoveError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

pub fn move_duplicates(
    summary: &ScanSummary,
    target_directory: &Path,
) -> Result<MoveStats, MoveError> {
    let total = total_duplicates(summary) as u64;
    let progress_bar = ProgressBar::new(total);
    progress_bar.set_style(progress::default_style());

    let mut moved = 0;
    for group in summary.groups.iter().filter(|group| group.files.len() > 1) {
        for entry in group.files.iter().skip(1) {
            let source = &entry.path;
            let destination = resolve_destination(target_directory, source)?;
            fs::rename(source, &destination).map_err(|error| MoveError::Io {
                source: error,
                path: source.clone(),
            })?;
            moved += 1;
            progress_bar.inc(1);
            progress_bar.set_message(format!("Moving: {}", destination.display()));
        }
    }

    progress_bar.finish_with_message("File moving complete");
    Ok(MoveStats { moved })
}

fn total_duplicates(summary: &ScanSummary) -> usize {
    summary
        .duplicate_groups()
        .map(|group| group.files.len() - 1)
        .sum()
}

fn resolve_destination(target_directory: &Path, source: &Path) -> Result<PathBuf, MoveError> {
    let file_name = source
        .file_name()
        .ok_or_else(|| MoveError::MissingFileName(source.to_path_buf()))?;

    let mut candidate = target_directory.join(file_name);
    if !candidate.exists() {
        return Ok(candidate);
    }

    let stem = source
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map(|stem| stem.to_string())
        .unwrap_or_else(|| String::from("file"));
    let extension = source.extension().and_then(|ext| ext.to_str());
    let mut index = 1;

    loop {
        let mut name = format!("{} ({})", stem, index);
        if let Some(ext) = extension {
            name.push('.');
            name.push_str(ext);
        }
        candidate = target_directory.join(name);
        if !candidate.exists() {
            return Ok(candidate);
        }
        index += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scanner::{DuplicateEntry, DuplicateGroup, ScanSummary};
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn move_duplicates_moves_files_and_generates_unique_names() {
        let source_dir = tempdir().unwrap();
        let target_dir = tempdir().unwrap();
        let keep = source_dir.path().join("dup.jpg");
        fs::write(&keep, b"same").unwrap();
        let nested_dir = source_dir.path().join("nested");
        fs::create_dir_all(&nested_dir).unwrap();
        let nested = nested_dir.join("dup.jpg");
        fs::write(&nested, b"same").unwrap();
        let other_dir = source_dir.path().join("other");
        fs::create_dir_all(&other_dir).unwrap();
        let other = other_dir.join("dup.jpg");
        fs::write(&other, b"same").unwrap();
        let existing = target_dir.path().join("dup.jpg");
        fs::write(&existing, b"existing").unwrap();
        let summary = ScanSummary {
            groups: vec![DuplicateGroup {
                fingerprint: 1,
                files: vec![
                    sample_entry(keep.clone()),
                    sample_entry(nested.clone()),
                    sample_entry(other.clone()),
                ],
            }],
        };
        let stats = move_duplicates(&summary, target_dir.path()).unwrap();
        assert_eq!(stats.moved, 2);
        assert!(keep.exists());
        assert!(!nested.exists());
        assert!(!other.exists());
        let mut names: Vec<_> = fs::read_dir(target_dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                String::from("dup (1).jpg"),
                String::from("dup (2).jpg"),
                String::from("dup.jpg")
            ]
        );
    }

    fn sample_entry(path: PathBuf) -> DuplicateEntry {
        DuplicateEntry {
            path,
            size_bytes: 4,
            dimensions: (1, 1),
            modified: None,
            captured_at: None,
            dominant_color: [0, 0, 0],
            confidence: 1.0,
        }
    }
}
