use crate::detector::{DuplicateDetector, ImageAnalysis, ImageFeatures, ImageMetadata};
use crate::thumbnails::ThumbnailCache;
use indicatif::ProgressBar;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use walkdir::WalkDir;

const BUCKET_PREFIX_BITS: u32 = 48;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreadingMode {
    Parallel,
    Sequential,
}

/// Parameters that control how the scanning pipeline behaves.
#[derive(Clone, Debug)]
pub struct ScanConfig {
    pub extensions: Vec<String>,
    pub threading: ThreadingMode,
    pub thumbnail_cache_root: Option<PathBuf>,
}

impl ScanConfig {
    /// Builds a new configuration from the supplied extensions and threading mode.
    pub fn new(extensions: Vec<String>, threading: ThreadingMode) -> Self {
        Self {
            extensions,
            threading,
            thumbnail_cache_root: None,
        }
    }

    pub fn with_thumbnail_root(mut self, root: PathBuf) -> Self {
        self.thumbnail_cache_root = Some(root);
        self
    }
}

/// A file entry that belongs to a duplicate group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateEntry {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub dimensions: (i32, i32),
    pub modified: Option<String>,
    pub captured_at: Option<String>,
    pub dominant_color: [u8; 3],
    pub confidence: f32,
    pub thumbnail: Option<PathBuf>,
}

/// A cluster of visually identical images identified during a scan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateGroup {
    pub fingerprint: u64,
    pub files: Vec<DuplicateEntry>,
}

/// Complete summary for a scan, suitable for serialisation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanSummary {
    pub groups: Vec<DuplicateGroup>,
}

impl ScanSummary {
    /// Returns an iterator over groups that contain potential duplicates.
    pub fn duplicate_groups(&self) -> impl Iterator<Item = &DuplicateGroup> {
        self.groups.iter().filter(|group| group.files.len() > 1)
    }

    /// Indicates whether any groups were discovered.
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }
}

pub fn count_entries(root: &Path) -> u64 {
    WalkDir::new(root).into_iter().count() as u64
}

pub fn scan(root: &Path, config: &ScanConfig, progress_bar: &Arc<ProgressBar>) -> ScanSummary {
    let detector = DuplicateDetector::default();
    let cache = ThumbnailCache::new(config.thumbnail_cache_root.clone())
        .ok()
        .map(Arc::new);

    let groups = match config.threading {
        ThreadingMode::Parallel => scan_parallel(root, config, progress_bar, &detector, &cache),
        ThreadingMode::Sequential => scan_sequential(root, config, progress_bar, &detector, &cache),
    };

    ScanSummary { groups }
}

fn scan_parallel(
    root: &Path,
    config: &ScanConfig,
    progress_bar: &Arc<ProgressBar>,
    detector: &DuplicateDetector,
    cache: &Option<Arc<ThumbnailCache>>,
) -> Vec<DuplicateGroup> {
    let records = WalkDir::new(root)
        .into_iter()
        .par_bridge()
        .filter_map(|entry| handle_entry(entry, config, progress_bar, detector, cache.as_deref()))
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
    config: &ScanConfig,
    progress_bar: &Arc<ProgressBar>,
    detector: &DuplicateDetector,
    cache: &Option<Arc<ThumbnailCache>>,
) -> Vec<DuplicateGroup> {
    let mut records = Vec::new();
    for entry in WalkDir::new(root) {
        if let Some(record) = handle_entry(entry, config, progress_bar, detector, cache.as_deref())
        {
            records.push(record);
        }
    }
    group_records(records, detector)
}

fn handle_entry(
    entry: Result<walkdir::DirEntry, walkdir::Error>,
    config: &ScanConfig,
    progress_bar: &Arc<ProgressBar>,
    detector: &DuplicateDetector,
    cache: Option<&ThumbnailCache>,
) -> Option<ImageRecord> {
    progress_bar.inc(1);
    match entry {
        Ok(entry) => {
            let path = entry.path().to_path_buf();
            progress_bar.set_message(format!("Scanning: {}", path.display()));
            if path.is_file() && has_image_extension(&path, &config.extensions) {
                match detector.analyze(&path) {
                    Ok(mut analysis) => {
                        if let Some(cache) = cache {
                            match cache.ensure(&path, analysis.features.fingerprint) {
                                Ok(thumbnail) => {
                                    analysis.metadata.thumbnail = Some(thumbnail);
                                }
                                Err(error) => {
                                    progress_bar.set_message(format!("Thumbnail error: {}", error));
                                }
                            }
                        }
                        return Some(ImageRecord { path, analysis });
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

fn group_records(records: Vec<ImageRecord>, detector: &DuplicateDetector) -> Vec<DuplicateGroup> {
    let mut groups = Vec::new();
    let mut buckets: FxHashMap<u16, Vec<usize>> = FxHashMap::default();
    for record in records {
        insert_record(&mut groups, &mut buckets, record, detector);
    }

    groups
        .into_iter()
        .map(|state| DuplicateGroup {
            fingerprint: state.features.fingerprint,
            files: state
                .entries
                .into_iter()
                .map(|entry| {
                    let AnalyzedFile { path, metadata } = entry;
                    DuplicateEntry {
                        path,
                        size_bytes: metadata.size_bytes,
                        dimensions: metadata.dimensions,
                        modified: metadata.modified,
                        captured_at: metadata.captured_at,
                        dominant_color: metadata.dominant_color,
                        confidence: metadata.confidence,
                        thumbnail: metadata.thumbnail,
                    }
                })
                .collect(),
        })
        .collect()
}

fn insert_record(
    groups: &mut Vec<GroupState>,
    buckets: &mut FxHashMap<u16, Vec<usize>>,
    record: ImageRecord,
    detector: &DuplicateDetector,
) {
    let ImageRecord { path, analysis } = record;
    let ImageAnalysis { features, metadata } = analysis;
    let bucket = bucket_id(features.fingerprint);
    if let Some(indices) = buckets.get(&bucket) {
        for &index in indices {
            if detector.is_similar(&groups[index].features, &features) {
                groups[index].entries.push(AnalyzedFile {
                    path: path.clone(),
                    metadata: metadata.clone(),
                });
                return;
            }
        }
    }

    for (index, group) in groups.iter_mut().enumerate() {
        if detector.is_similar(&group.features, &features) {
            group.entries.push(AnalyzedFile {
                path: path.clone(),
                metadata: metadata.clone(),
            });
            ensure_bucket_mapping(buckets, bucket, index);
            return;
        }
    }

    let index = groups.len();
    groups.push(GroupState {
        features,
        entries: vec![AnalyzedFile { path, metadata }],
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
    analysis: ImageAnalysis,
}

struct GroupState {
    features: ImageFeatures,
    entries: Vec<AnalyzedFile>,
}

struct AnalyzedFile {
    path: PathBuf,
    metadata: ImageMetadata,
}

#[cfg(test)]
mod tests {
    use super::*;
    use indicatif::ProgressBar;
    use opencv::core::{self, Scalar, Vector};
    use opencv::imgcodecs;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn write_image(path: &Path, color: u8) {
        let image = core::Mat::new_rows_cols_with_default(
            64,
            64,
            core::CV_8UC3,
            Scalar::from((color as f64, color as f64, color as f64, 0.0)),
        )
        .unwrap();
        let params = Vector::<i32>::new();
        imgcodecs::imwrite(path.to_string_lossy().as_ref(), &image, &params).unwrap();
    }

    fn scan_duplicates(mode: ThreadingMode) {
        let dir = tempdir().unwrap();
        let thumb_dir = tempdir().unwrap();
        let first = dir.path().join("a.jpg");
        let second = dir.path().join("b.jpg");
        let third = dir.path().join("c.png");
        write_image(&first, 64);
        write_image(&second, 64);
        write_image(&third, 200);
        let progress = Arc::new(ProgressBar::hidden());
        let config = ScanConfig::new(vec![String::from("jpg"), String::from("png")], mode)
            .with_thumbnail_root(thumb_dir.path().to_path_buf());
        let summary = scan(dir.path(), &config, &progress);
        let duplicates: Vec<_> = summary.duplicate_groups().collect();
        assert_eq!(duplicates.len(), 1);
        let files = &duplicates[0].files;
        let paths: Vec<_> = files.iter().map(|entry| entry.path.clone()).collect();
        assert!(paths.contains(&first));
        assert!(paths.contains(&second));
        assert!(files.iter().all(|entry| entry.size_bytes > 0));
        assert!(files.iter().all(|entry| entry.dimensions == (64, 64)));
        assert!(files.iter().all(|entry| entry
            .thumbnail
            .as_ref()
            .map(|path| path.exists())
            .unwrap_or(false)));
        assert!(summary
            .groups
            .iter()
            .any(|group| group.files.len() == 1 && group.files[0].path == third));
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
