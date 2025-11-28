//! GUID-based file renaming utilities.

use once_cell::sync::Lazy;
use regex::Regex;
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

static GUID_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
        .expect("invalid GUID regex")
});

/// Checks if the file stem (name without extension) matches the GUID pattern.
pub fn is_guid_named(path: &Path) -> bool {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .map(|stem| GUID_PATTERN.is_match(stem))
        .unwrap_or(false)
}

/// Renames a file to a GUID-based name if it doesn't already match the pattern.
/// Returns the new path if renamed, or the original path if already GUID-named.
pub fn ensure_guid_name(path: &Path) -> Result<PathBuf, RenameError> {
    if is_guid_named(path) {
        return Ok(path.to_path_buf());
    }

    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_lowercase())
        .unwrap_or_default();

    let new_name = if extension.is_empty() {
        Uuid::new_v4().to_string()
    } else {
        format!("{}.{}", Uuid::new_v4(), extension)
    };

    let new_path = path
        .parent()
        .map(|parent| parent.join(&new_name))
        .unwrap_or_else(|| PathBuf::from(&new_name));

    fs::rename(path, &new_path).map_err(|err| RenameError::IoError {
        source: path.to_path_buf(),
        target: new_path.clone(),
        error: err,
    })?;

    Ok(new_path)
}

#[derive(Debug)]
pub enum RenameError {
    IoError {
        source: PathBuf,
        target: PathBuf,
        error: std::io::Error,
    },
}

impl std::fmt::Display for RenameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenameError::IoError {
                source,
                target,
                error,
            } => {
                write!(
                    f,
                    "failed to rename {} to {}: {}",
                    source.display(),
                    target.display(),
                    error
                )
            }
        }
    }
}

impl std::error::Error for RenameError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use tempfile::tempdir;

    #[test]
    fn detects_guid_names() {
        let guid_path = Path::new("c3a1b2d4-5678-4abc-9def-0123456789ab.jpg");
        assert!(is_guid_named(guid_path));

        let normal_path = Path::new("IMG_1234.jpg");
        assert!(!is_guid_named(normal_path));

        let partial_guid = Path::new("c3a1b2d4-5678.jpg");
        assert!(!is_guid_named(partial_guid));
    }

    #[test]
    fn renames_non_guid_file() {
        let dir = tempdir().unwrap();
        let original = dir.path().join("photo.jpg");
        File::create(&original).unwrap();

        let result = ensure_guid_name(&original).unwrap();

        assert!(!original.exists());
        assert!(result.exists());
        assert!(is_guid_named(&result));
        assert_eq!(result.extension().unwrap(), "jpg");
    }

    #[test]
    fn preserves_guid_named_file() {
        let dir = tempdir().unwrap();
        let guid_name = dir.path().join("c3a1b2d4-5678-4abc-9def-0123456789ab.png");
        File::create(&guid_name).unwrap();

        let result = ensure_guid_name(&guid_name).unwrap();

        assert_eq!(result, guid_name);
        assert!(guid_name.exists());
    }
}
