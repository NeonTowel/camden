use crate::cli::ThreadingMode;
use indicatif::ProgressBar;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::hash::Hasher;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use twox_hash::XxHash64;
use walkdir::WalkDir;

pub fn count_entries(root: &Path) -> u64 {
    WalkDir::new(root).into_iter().count() as u64
}

pub fn scan(
    root: &Path,
    extensions: &[String],
    threading: ThreadingMode,
    progress_bar: &Arc<ProgressBar>,
) -> HashMap<u64, Vec<PathBuf>> {
    let checksum_map: Arc<Mutex<HashMap<u64, Vec<PathBuf>>>> = Arc::new(Mutex::new(HashMap::new()));

    if matches!(threading, ThreadingMode::Parallel) {
        WalkDir::new(root)
            .into_iter()
            .par_bridge()
            .for_each(|entry| process_entry(&entry, &checksum_map, progress_bar, extensions));
    } else {
        for entry in WalkDir::new(root) {
            process_entry(&entry, &checksum_map, progress_bar, extensions);
        }
    }

    match Arc::try_unwrap(checksum_map) {
        Ok(mutex) => match mutex.into_inner() {
            Ok(map) => map,
            Err(poisoned) => poisoned.into_inner(),
        },
        Err(shared) => match shared.lock() {
            Ok(mut guard) => std::mem::take(&mut *guard),
            Err(poisoned) => {
                let mut guard = poisoned.into_inner();
                std::mem::take(&mut *guard)
            }
        },
    }
}

fn process_entry(
    entry: &Result<walkdir::DirEntry, walkdir::Error>,
    checksum_map: &Arc<Mutex<HashMap<u64, Vec<PathBuf>>>>,
    progress_bar: &Arc<ProgressBar>,
    extensions: &[String],
) {
    if let Ok(entry) = entry {
        let path = entry.path();
        if path.is_file() && has_image_extension(path, extensions) {
            if let Ok(checksum) = compute_checksum(path) {
                if let Ok(mut map) = checksum_map.lock() {
                    map.entry(checksum).or_default().push(path.to_path_buf());
                }
            }
        }
        progress_bar.inc(1);
        progress_bar.set_message(format!("Scanning: {}", path.display()));
    }
}

fn has_image_extension(path: &Path, extensions: &[String]) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            let lower = ext.to_lowercase();
            extensions.iter().any(|candidate| candidate == &lower)
        })
        .unwrap_or(false)
}

fn compute_checksum(path: &Path) -> std::io::Result<u64> {
    let mut file = File::open(path)?;
    let mut hasher = XxHash64::default();
    let mut buffer = [0; 8192];

    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.write(&buffer[..count]);
    }

    Ok(hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::ThreadingMode;
    use indicatif::ProgressBar;
    use std::fs;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn write_file(path: &Path, data: &[u8]) {
        fs::write(path, data).unwrap();
    }

    fn scan_duplicates(mode: ThreadingMode) {
        let dir = tempdir().unwrap();
        let first = dir.path().join("a.jpg");
        let second = dir.path().join("b.jpg");
        let third = dir.path().join("c.png");
        write_file(&first, b"same");
        write_file(&second, b"same");
        write_file(&third, b"diff");
        let progress = Arc::new(ProgressBar::hidden());
        let map = scan(
            dir.path(),
            &[String::from("jpg"), String::from("png")],
            mode,
            &progress,
        );
        let duplicates: Vec<_> = map.values().filter(|files| files.len() > 1).collect();
        assert_eq!(duplicates.len(), 1);
        let files = duplicates.first().unwrap();
        assert!(files.contains(&first));
        assert!(files.contains(&second));
        assert!(map
            .values()
            .any(|items| items.len() == 1 && items[0] == third));
    }

    #[test]
    fn scan_detects_duplicates_parallel() {
        scan_duplicates(ThreadingMode::Parallel);
    }

    #[test]
    fn scan_detects_duplicates_sequential() {
        scan_duplicates(ThreadingMode::Sequential);
    }

    #[test]
    fn count_entries_includes_root_and_files() {
        let dir = tempdir().unwrap();
        write_file(&dir.path().join("a.jpg"), b"one");
        write_file(&dir.path().join("b.jpg"), b"two");
        write_file(&dir.path().join("c.jpg"), b"three");
        assert_eq!(count_entries(dir.path()), 4);
    }
}
