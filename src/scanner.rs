use crate::cli::ThreadingMode;
use crate::detector::{DuplicateDetector, ImageFeatures};
use indicatif::ProgressBar;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use walkdir::WalkDir;

const BUCKET_PREFIX_BITS: u32 = 48;

pub type DuplicateMap = FxHashMap<u64, Vec<PathBuf>>;

pub fn count_entries(root: &Path) -> u64 {
    WalkDir::new(root).into_iter().count() as u64
}

pub fn scan(
    root: &Path,
    extensions: &[String],
    threading: ThreadingMode,
    progress_bar: &Arc<ProgressBar>,
) -> DuplicateMap {
    let detector = DuplicateDetector::default();
    match threading {
        ThreadingMode::Parallel => scan_parallel(root, extensions, progress_bar, &detector),
        ThreadingMode::Sequential => scan_sequential(root, extensions, progress_bar, &detector),
    }
}

fn scan_parallel(
    root: &Path,
    extensions: &[String],
    progress_bar: &Arc<ProgressBar>,
    detector: &DuplicateDetector,
) -> DuplicateMap {
    let records = WalkDir::new(root)
        .into_iter()
        .par_bridge()
        .filter_map(|entry| handle_entry(entry, extensions, progress_bar, detector))
        .fold(Vec::new, |mut collection, record| {
            collection.push(record);
            collection
        })
        .reduce(Vec::new, |mut left, mut right| {
            left.append(&mut right);
            left
        });

    group_records(records, detector)
}

fn scan_sequential(
    root: &Path,
    extensions: &[String],
    progress_bar: &Arc<ProgressBar>,
    detector: &DuplicateDetector,
) -> DuplicateMap {
    let mut records = Vec::new();
    for entry in WalkDir::new(root) {
        if let Some(record) = handle_entry(entry, extensions, progress_bar, detector) {
            records.push(record);
        }
    }
    group_records(records, detector)
}

fn handle_entry(
    entry: Result<walkdir::DirEntry, walkdir::Error>,
    extensions: &[String],
    progress_bar: &Arc<ProgressBar>,
    detector: &DuplicateDetector,
) -> Option<ImageRecord> {
    progress_bar.inc(1);
    match entry {
        Ok(entry) => {
            let path = entry.path().to_path_buf();
            progress_bar.set_message(format!("Scanning: {}", path.display()));
            if path.is_file() && has_image_extension(&path, extensions) {
                match detector.analyze(&path) {
                    Ok(features) => {
                        return Some(ImageRecord { path, features });
                    }
                    Err(error) => {
                        progress_bar.set_message(format!("Error: {}", error));
                    }
                }
            }
        }
        Err(error) => {
            progress_bar.set_message(format!("Error: {}", error));
        }
    }
    None
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

fn group_records(records: Vec<ImageRecord>, detector: &DuplicateDetector) -> DuplicateMap {
    let mut groups = Vec::new();
    let mut buckets: FxHashMap<u16, Vec<usize>> = FxHashMap::default();
    for record in records {
        insert_record(&mut groups, &mut buckets, record, detector);
    }

    let mut map = DuplicateMap::default();
    for group in groups {
        let fingerprint = group.features.fingerprint;
        insert_group(&mut map, fingerprint, group.paths);
    }
    map
}

fn insert_record(
    groups: &mut Vec<DuplicateGroup>,
    buckets: &mut FxHashMap<u16, Vec<usize>>,
    record: ImageRecord,
    detector: &DuplicateDetector,
) {
    let bucket = bucket_id(record.features.fingerprint);
    if let Some(indices) = buckets.get(&bucket) {
        for &index in indices {
            if detector.is_similar(&groups[index].features, &record.features) {
                groups[index].paths.push(record.path);
                return;
            }
        }
    }

    for (index, group) in groups.iter_mut().enumerate() {
        if detector.is_similar(&group.features, &record.features) {
            group.paths.push(record.path);
            ensure_bucket_mapping(buckets, bucket, index);
            return;
        }
    }

    let index = groups.len();
    groups.push(DuplicateGroup {
        features: record.features,
        paths: vec![record.path],
    });
    ensure_bucket_mapping(buckets, bucket, index);
}

fn ensure_bucket_mapping(buckets: &mut FxHashMap<u16, Vec<usize>>, bucket: u16, index: usize) {
    let entry = buckets.entry(bucket).or_default();
    if !entry.contains(&index) {
        entry.push(index);
    }
}

fn bucket_id(fingerprint: u64) -> u16 {
    (fingerprint >> BUCKET_PREFIX_BITS) as u16
}

struct ImageRecord {
    path: PathBuf,
    features: ImageFeatures,
}

struct DuplicateGroup {
    features: ImageFeatures,
    paths: Vec<PathBuf>,
}

fn insert_group(map: &mut DuplicateMap, mut fingerprint: u64, paths: Vec<PathBuf>) {
    while map.contains_key(&fingerprint) {
        fingerprint = fingerprint.wrapping_add(1);
    }
    map.insert(fingerprint, paths);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::ThreadingMode;
    use indicatif::ProgressBar;
    use opencv::core::{self, Scalar};
    use opencv::imgcodecs;
    use opencv::prelude::*;
    use opencv::types::VectorOfi32;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn write_image(path: &Path, color: u8) {
        let mut image = core::Mat::new_rows_cols_with_default(
            64,
            64,
            core::CV_8UC3,
            Scalar::from((color as f64, color as f64, color as f64, 0.0)),
        )
        .unwrap();
        let params = VectorOfi32::new();
        imgcodecs::imwrite(path.to_string_lossy().as_ref(), &image, &params).unwrap();
    }

    fn scan_duplicates(mode: ThreadingMode) {
        let dir = tempdir().unwrap();
        let first = dir.path().join("a.jpg");
        let second = dir.path().join("b.jpg");
        let third = dir.path().join("c.png");
        write_image(&first, 64);
        write_image(&second, 64);
        write_image(&third, 200);
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
        write_image(&dir.path().join("a.jpg"), 0);
        write_image(&dir.path().join("b.jpg"), 128);
        write_image(&dir.path().join("c.jpg"), 255);
        assert_eq!(count_entries(dir.path()), 4);
    }
}
