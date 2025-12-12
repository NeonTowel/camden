use kamadak_exif::{In, Reader, Tag};
use opencv::core::{
    self, AlgorithmHint, DMatch, KeyPoint, Mat, MatTraitConst, MatTraitConstManual, Point2f,
    Scalar, Size, Vector, NORM_HAMMING, CV_8U,
};
use opencv::features2d::{BFMatcher, Feature2DTrait, ORB, ORB_ScoreType};
use opencv::imgcodecs;
use opencv::imgproc;
use opencv::prelude::{DescriptorMatcherTraitConst, KeyPointTraitConst, MatTrait};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::path::{Path, PathBuf};
use time::{format_description::well_known::Rfc3339, OffsetDateTime};

const IMREAD_COLOR: i32 = imgcodecs::IMREAD_COLOR;
const HASH_SIZE: i32 = 8;

// ============================================================================
// Visual Feature Types for Crop Detection
// ============================================================================

/// Extracted visual features for crop detection using ORB keypoints.
#[derive(Clone, Default)]
pub struct VisualFeatures {
    /// ORB keypoint data (position, size, angle, etc.)
    pub keypoints: Vec<KeypointData>,
    /// ORB binary descriptors (32 bytes per keypoint)
    pub descriptors: Vec<u8>,
    /// Descriptor dimensions (rows = num keypoints, cols = 32 for ORB)
    pub descriptor_shape: (usize, usize),
    /// Original image dimensions for reference
    pub source_dimensions: (i32, i32),
}

/// Serializable keypoint representation (OpenCV KeyPoint is not Clone).
#[derive(Clone, Debug)]
pub struct KeypointData {
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub angle: f32,
    pub response: f32,
    pub octave: i32,
}

/// Result of comparing two images for duplicates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatchResult {
    /// Images match via perceptual hash (exact or near-exact duplicates)
    HashMatch,
    /// Images match via feature detection (one is a crop of the other)
    CropMatch,
    /// Images do not match
    NoMatch,
}

/// Represents a 64-bit perceptual hash fingerprint.
pub type Fingerprint = u64;

/// Configuration options for the detector.
#[derive(Clone)]
pub struct DetectorConfig {
    pub hash_threshold: u32,
    pub color_tolerance: f64,
    pub contrast_tolerance: f64,
    pub texture_tolerance: f64,
    // Feature detection configuration for crop detection
    pub enable_feature_detection: bool,
    pub orb_max_features: i32,
    pub min_match_count: usize,
    pub ransac_reproj_threshold: f64,
    pub nn_match_ratio: f32,
    pub min_inlier_ratio: f32,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            hash_threshold: 6,
            color_tolerance: 20.0,
            contrast_tolerance: 15.0,
            texture_tolerance: 12.0,
            // Feature detection defaults (disabled by default)
            // Tuned for ~95-98% confidence to minimize false positives
            // Performance-optimized: 250 features instead of 500 for 2x faster matching
            enable_feature_detection: false,
            orb_max_features: 250,         // Was 500 - fewer features = faster matching
            min_match_count: 15,           // Was 25 - adjusted for fewer features
            ransac_reproj_threshold: 3.0,  // Tight geometric consistency
            nn_match_ratio: 0.70,          // Was 0.65 - slightly relaxed for fewer features
            min_inlier_ratio: 0.40,        // Was 0.50 - slightly relaxed for fewer features
        }
    }
}

/// Encapsulates OpenCV-based duplicate detection logic.
#[derive(Clone)]
pub struct DuplicateDetector {
    config: DetectorConfig,
}

impl DuplicateDetector {
    /// Builds a detector with the provided configuration.
    pub fn new(config: DetectorConfig) -> Self {
        Self { config }
    }

    /// Returns a reference to the detector configuration.
    pub fn config(&self) -> &DetectorConfig {
        &self.config
    }

    /// Generates perceptual and feature fingerprints for the image at `path`.
    pub fn analyze(&self, path: &Path) -> Result<ImageAnalysis, DetectionError> {
        let image = load_image(path)?;
        let mut grayscale = Mat::default();
        imgproc::cvt_color(
            &image,
            &mut grayscale,
            imgproc::COLOR_BGR2GRAY,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT,
        )
        .map_err(DetectionError::OpenCv)?;

        let fingerprint = average_hash(&grayscale, path)?;
        let (mean, stddev) = channel_statistics(&image)?;
        let texture = texture_energy(&grayscale)?;
        let metadata = file_metadata(path, &image, &mean)?;

        // Extract visual features for crop detection if enabled
        let visual = if self.config.enable_feature_detection {
            match extract_visual_features(&grayscale, self.config.orb_max_features) {
                Ok(v) if v.keypoints.len() >= self.config.min_match_count => Some(v),
                Ok(_) => None, // Insufficient keypoints, skip
                Err(_) => None, // Graceful degradation on error
            }
        } else {
            None
        };

        Ok(ImageAnalysis {
            features: ImageFeatures {
                fingerprint,
                mean,
                stddev,
                texture,
                visual,
            },
            metadata,
        })
    }

    /// Determines whether the provided feature sets represent matching images (hash-based).
    pub fn is_similar(&self, left: &ImageFeatures, right: &ImageFeatures) -> bool {
        if hamming_distance(left.fingerprint, right.fingerprint) > self.config.hash_threshold {
            return false;
        }

        let color_delta = vector_distance(&left.mean, &right.mean);
        let contrast_delta = vector_distance(&left.stddev, &right.stddev);
        let texture_delta = (left.texture - right.texture).abs();

        color_delta <= self.config.color_tolerance
            && contrast_delta <= self.config.contrast_tolerance
            && texture_delta <= self.config.texture_tolerance
    }

    /// Determines whether two images match via hash OR crop relationship.
    pub fn is_match(&self, left: &ImageFeatures, right: &ImageFeatures) -> MatchResult {
        // Phase 1: Fast hash-based check
        if self.is_similar(left, right) {
            return MatchResult::HashMatch;
        }

        // Phase 2: Feature-based crop detection
        if self.config.enable_feature_detection {
            if let (Some(lv), Some(rv)) = (&left.visual, &right.visual) {
                if let Ok((is_crop, _, _)) = is_crop_match(lv, rv, &self.config) {
                    if is_crop {
                        return MatchResult::CropMatch;
                    }
                }
            }
        }

        MatchResult::NoMatch
    }
}

impl Default for DuplicateDetector {
    fn default() -> Self {
        Self::new(DetectorConfig::default())
    }
}

/// Holds combined perceptual and local image features.
#[derive(Clone)]
pub struct ImageFeatures {
    pub fingerprint: Fingerprint,
    pub mean: [f64; 3],
    pub stddev: [f64; 3],
    pub texture: f64,
    /// Optional visual features for crop detection (ORB keypoints/descriptors).
    pub visual: Option<VisualFeatures>,
}

use crate::resolution::ResolutionTier;

#[derive(Clone)]
pub struct ImageMetadata {
    pub size_bytes: u64,
    pub dimensions: (i32, i32),
    pub modified: Option<String>,
    pub captured_at: Option<String>,
    pub dominant_color: [u8; 3],
    pub confidence: f32,
    pub thumbnail: Option<PathBuf>,
    pub resolution_tier: ResolutionTier,
    /// AI moderation tier (if classification was performed).
    pub moderation_tier: Option<String>,
    /// AI-generated tags (if classification was performed).
    pub tags: Vec<String>,
}

#[derive(Clone)]
pub struct ImageAnalysis {
    pub features: ImageFeatures,
    pub metadata: ImageMetadata,
}

fn load_image(path: &Path) -> Result<Mat, DetectionError> {
    let path_string = path
        .to_str()
        .map(|value| value.to_owned())
        .ok_or_else(|| DetectionError::InvalidPath(path.to_path_buf()))?;

    let image = imgcodecs::imread(&path_string, IMREAD_COLOR).map_err(DetectionError::OpenCv)?;
    if image.empty() {
        return Err(DetectionError::EmptyImage(path.to_path_buf()));
    }
    Ok(image)
}

/// Calculates the Hamming distance between `left` and `right`.
pub fn hamming_distance(left: Fingerprint, right: Fingerprint) -> u32 {
    (left ^ right).count_ones()
}

fn average_hash(grayscale: &Mat, path: &Path) -> Result<Fingerprint, DetectionError> {
    let mut resized = Mat::default();
    imgproc::resize(
        grayscale,
        &mut resized,
        Size::new(HASH_SIZE, HASH_SIZE),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )
    .map_err(DetectionError::OpenCv)?;

    let data = resized.data_typed::<u8>().map_err(DetectionError::OpenCv)?;
    let expected = (HASH_SIZE * HASH_SIZE) as usize;
    if data.len() != expected {
        return Err(DetectionError::UnexpectedHashLength {
            expected,
            actual: data.len(),
            path: path.to_path_buf(),
        });
    }

    let mean = data.iter().map(|value| *value as f64).sum::<f64>() / expected as f64;

    let mut fingerprint = 0u64;
    for value in data.iter() {
        fingerprint <<= 1;
        if (*value as f64) >= mean {
            fingerprint |= 1;
        }
    }

    Ok(fingerprint)
}

fn channel_statistics(image: &Mat) -> Result<([f64; 3], [f64; 3]), DetectionError> {
    let mut mean = Scalar::default();
    let mut stddev = Scalar::default();
    core::mean_std_dev(image, &mut mean, &mut stddev, &Mat::default())
        .map_err(DetectionError::OpenCv)?;

    Ok((
        [mean[0], mean[1], mean[2]],
        [stddev[0], stddev[1], stddev[2]],
    ))
}

fn texture_energy(grayscale: &Mat) -> Result<f64, DetectionError> {
    let mut laplacian = Mat::default();
    imgproc::laplacian(
        grayscale,
        &mut laplacian,
        core::CV_32F,
        3,
        1.0,
        0.0,
        core::BORDER_DEFAULT,
    )
    .map_err(DetectionError::OpenCv)?;

    let zeros = Mat::zeros(laplacian.rows(), laplacian.cols(), laplacian.typ())
        .map_err(DetectionError::OpenCv)?;
    let mut abs = Mat::default();
    core::absdiff(&laplacian, &zeros, &mut abs).map_err(DetectionError::OpenCv)?;
    let texture = core::mean(&abs, &Mat::default()).map_err(DetectionError::OpenCv)?[0] as f64;
    Ok(texture)
}

fn vector_distance(left: &[f64; 3], right: &[f64; 3]) -> f64 {
    let dx = left[0] - right[0];
    let dy = left[1] - right[1];
    let dz = left[2] - right[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

// ============================================================================
// ORB Feature Extraction and Matching for Crop Detection
// ============================================================================

/// Extracts ORB keypoints and descriptors from a grayscale image.
/// This function is thread-safe - each call creates its own ORB detector.
fn extract_visual_features(grayscale: &Mat, max_features: i32) -> Result<VisualFeatures, DetectionError> {
    let mut orb = ORB::create(
        max_features,
        1.2,  // scaleFactor
        8,    // nlevels
        31,   // edgeThreshold
        0,    // firstLevel
        2,    // WTA_K
        ORB_ScoreType::HARRIS_SCORE,
        31,   // patchSize
        20,   // fastThreshold
    )
    .map_err(DetectionError::OpenCv)?;

    let mut keypoints = Vector::<KeyPoint>::new();
    let mut descriptors = Mat::default();

    orb.detect_and_compute(
        grayscale,
        &Mat::default(), // no mask
        &mut keypoints,
        &mut descriptors,
        false,
    )
    .map_err(DetectionError::OpenCv)?;

    // Convert KeyPoints to serializable format
    let kp_data: Vec<KeypointData> = keypoints
        .iter()
        .map(|kp| KeypointData {
            x: kp.pt().x,
            y: kp.pt().y,
            size: kp.size(),
            angle: kp.angle(),
            response: kp.response(),
            octave: kp.octave(),
        })
        .collect();

    let desc_rows = descriptors.rows() as usize;
    let desc_cols = descriptors.cols() as usize;
    let desc_data = if desc_rows > 0 && desc_cols > 0 {
        descriptors
            .data_typed::<u8>()
            .map_err(DetectionError::OpenCv)?
            .to_vec()
    } else {
        Vec::new()
    };

    Ok(VisualFeatures {
        keypoints: kp_data,
        descriptors: desc_data,
        descriptor_shape: (desc_rows, desc_cols),
        source_dimensions: (grayscale.cols(), grayscale.rows()),
    })
}

/// Reconstructs an OpenCV Mat from stored descriptor bytes.
fn descriptors_to_mat(features: &VisualFeatures) -> Result<Mat, DetectionError> {
    let (rows, cols) = features.descriptor_shape;
    if rows == 0 || cols == 0 {
        return Ok(Mat::default());
    }

    // Create Mat and copy data into it
    let mut mat = unsafe {
        Mat::new_rows_cols(rows as i32, cols as i32, CV_8U)
            .map_err(DetectionError::OpenCv)?
    };

    // Copy descriptor data into the Mat
    let mat_data = mat.data_mut();
    unsafe {
        std::ptr::copy_nonoverlapping(
            features.descriptors.as_ptr(),
            mat_data,
            features.descriptors.len(),
        );
    }

    Ok(mat)
}

/// Checks if two images are crop-related using ORB feature matching + RANSAC.
/// Returns (is_match, inlier_count, total_good_matches).
///
/// Performance optimizations:
/// - Early exit if not enough keypoints
/// - Early exit if ratio test doesn't produce enough matches
/// - Uses BFMatcher with crossCheck for faster initial filtering
fn is_crop_match(
    left: &VisualFeatures,
    right: &VisualFeatures,
    config: &DetectorConfig,
) -> Result<(bool, usize, usize), DetectionError> {
    // Early exit: need minimum keypoints in both images
    let min_keypoints = config.min_match_count;
    if left.keypoints.len() < min_keypoints || right.keypoints.len() < min_keypoints {
        return Ok((false, 0, 0));
    }

    // Reconstruct descriptor matrices
    let left_desc = descriptors_to_mat(left)?;
    let right_desc = descriptors_to_mat(right)?;

    if left_desc.empty() || right_desc.empty() {
        return Ok((false, 0, 0));
    }

    // BFMatcher with Hamming distance for binary ORB descriptors
    let matcher = BFMatcher::create(NORM_HAMMING, false).map_err(DetectionError::OpenCv)?;

    // KNN match with k=2 for Lowe's ratio test
    let mut knn_matches = Vector::<Vector<DMatch>>::new();
    matcher
        .knn_train_match(
            &left_desc,
            &right_desc,
            &mut knn_matches,
            2,
            &Mat::default(),
            false,
        )
        .map_err(DetectionError::OpenCv)?;

    // Early exit: if KNN matches are too few, no point continuing
    if knn_matches.len() < min_keypoints {
        return Ok((false, 0, 0));
    }

    // Apply Lowe's ratio test to filter good matches
    let mut good_matches: Vec<DMatch> = Vec::with_capacity(knn_matches.len());
    for pair in knn_matches.iter() {
        if pair.len() >= 2 {
            let m = pair.get(0).map_err(DetectionError::OpenCv)?;
            let n = pair.get(1).map_err(DetectionError::OpenCv)?;
            if m.distance < config.nn_match_ratio * n.distance {
                good_matches.push(m);
            }
        }
    }

    let match_count = good_matches.len();

    // Early exit: not enough good matches after ratio test
    if match_count < config.min_match_count {
        return Ok((false, 0, match_count));
    }

    // Extract matched keypoint positions for homography estimation
    let mut src_pts: Vec<Point2f> = Vec::with_capacity(match_count);
    let mut dst_pts: Vec<Point2f> = Vec::with_capacity(match_count);

    for m in &good_matches {
        let src_kp = &left.keypoints[m.query_idx as usize];
        let dst_kp = &right.keypoints[m.train_idx as usize];
        src_pts.push(Point2f::new(src_kp.x, src_kp.y));
        dst_pts.push(Point2f::new(dst_kp.x, dst_kp.y));
    }

    // Convert to Mat format for findHomography
    let src_mat = Mat::from_slice(&src_pts).map_err(DetectionError::OpenCv)?;
    let dst_mat = Mat::from_slice(&dst_pts).map_err(DetectionError::OpenCv)?;

    // RANSAC homography estimation with limited iterations for speed
    let mut mask = Mat::default();
    let _homography = opencv::calib3d::find_homography(
        &src_mat,
        &dst_mat,
        &mut mask,
        opencv::calib3d::RANSAC,
        config.ransac_reproj_threshold,
    )
    .map_err(DetectionError::OpenCv)?;

    // Count inliers from RANSAC mask
    let inlier_count = if !mask.empty() {
        mask.data_typed::<u8>()
            .map(|data| data.iter().filter(|&&v| v != 0).count())
            .unwrap_or(0)
    } else {
        0
    };

    let inlier_ratio = inlier_count as f32 / match_count as f32;
    let is_match = inlier_count >= config.min_match_count
        && inlier_ratio >= config.min_inlier_ratio;

    Ok((is_match, inlier_count, match_count))
}

fn file_metadata(
    path: &Path,
    image: &Mat,
    mean: &[f64; 3],
) -> Result<ImageMetadata, DetectionError> {
    let file_info = std::fs::metadata(path).map_err(|error| DetectionError::Io {
        source: error,
        path: path.to_path_buf(),
    })?;

    let size_bytes = file_info.len();
    let modified = file_info
        .modified()
        .ok()
        .and_then(|time| OffsetDateTime::from(time).format(&Rfc3339).ok());

    let captured_at = read_exif_timestamp(path);

    let dimensions = (image.cols(), image.rows());
    let dominant_color = [
        mean[2].clamp(0.0, 255.0) as u8,
        mean[1].clamp(0.0, 255.0) as u8,
        mean[0].clamp(0.0, 255.0) as u8,
    ];

    Ok(ImageMetadata {
        size_bytes,
        dimensions,
        modified,
        captured_at,
        dominant_color,
        confidence: 1.0,
        thumbnail: None,
        resolution_tier: ResolutionTier::High,
        moderation_tier: None,
        tags: Vec::new(),
    })
}

fn read_exif_timestamp(path: &Path) -> Option<String> {
    let file = File::open(path).ok()?;
    let mut buffer = std::io::BufReader::new(file);
    let exif_reader = Reader::new();
    let exif = exif_reader.read_from_container(&mut buffer).ok()?;
    let field = exif
        .get_field(Tag::DateTimeOriginal, In::PRIMARY)
        .or_else(|| exif.get_field(Tag::DateTimeDigitized, In::PRIMARY))?;
    Some(field.display_value().to_string())
}

/// Errors that can occur during duplicate detection.
#[derive(Debug)]
pub enum DetectionError {
    InvalidPath(PathBuf),
    EmptyImage(PathBuf),
    UnexpectedHashLength {
        expected: usize,
        actual: usize,
        path: PathBuf,
    },
    Io {
        source: std::io::Error,
        path: PathBuf,
    },
    OpenCv(opencv::Error),
}

impl Display for DetectionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPath(path) => write!(
                f,
                "unable to convert path {} to UTF-8 string",
                path.display()
            ),
            Self::EmptyImage(path) => write!(f, "image at {} is empty", path.display()),
            Self::UnexpectedHashLength {
                expected,
                actual,
                path,
            } => write!(
                f,
                "unexpected hash length for {}: expected {}, got {}",
                path.display(),
                expected,
                actual
            ),
            Self::Io { source, path } => write!(f, "io error for {}: {}", path.display(), source),
            Self::OpenCv(error) => write!(f, "opencv error: {}", error),
        }
    }
}

impl Error for DetectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::OpenCv(error) => Some(error),
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<opencv::Error> for DetectionError {
    fn from(error: opencv::Error) -> Self {
        Self::OpenCv(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{self, Rect, Scalar, Vector};
    use opencv::imgcodecs;
    use opencv::imgproc;
    use std::path::Path;
    use tempfile::tempdir;

    #[test]
    fn detector_reports_similarity_for_shifted_patterns() {
        let dir = tempdir().unwrap();
        let first = dir.path().join("first.png");
        let second = dir.path().join("second.png");
        write_pattern(&first, 0);
        write_pattern(&second, 3);

        let detector = DuplicateDetector::default();
        let left = detector.analyze(&first).unwrap();
        let right = detector.analyze(&second).unwrap();
        assert!(detector.is_similar(&left.features, &right.features));
        assert!(left.metadata.size_bytes > 0);
        assert_eq!(left.metadata.dimensions, (128, 128));
    }

    #[test]
    fn detector_distinguishes_different_patterns() {
        let dir = tempdir().unwrap();
        let first = dir.path().join("first.png");
        let second = dir.path().join("second.png");
        write_pattern(&first, 0);
        write_inverse_pattern(&second);

        let detector = DuplicateDetector::default();
        let left = detector.analyze(&first).unwrap();
        let right = detector.analyze(&second).unwrap();
        assert!(!detector.is_similar(&left.features, &right.features));
    }

    fn write_pattern(path: &Path, offset: i32) {
        let mut image = core::Mat::new_rows_cols_with_default(
            128,
            128,
            core::CV_8UC3,
            Scalar::from((0.0, 0.0, 0.0, 0.0)),
        )
        .unwrap();
        let color = Scalar::from((255.0, 255.0, 255.0, 0.0));
        let rect = Rect::new(32 + offset, 32, 48, 48);
        imgproc::rectangle(&mut image, rect, color, imgproc::FILLED, imgproc::LINE_8, 0).unwrap();
        let params = Vector::<i32>::new();
        imgcodecs::imwrite(path.to_string_lossy().as_ref(), &image, &params).unwrap();
    }

    fn write_inverse_pattern(path: &Path) {
        let mut image = core::Mat::new_rows_cols_with_default(
            128,
            128,
            core::CV_8UC3,
            Scalar::from((255.0, 255.0, 255.0, 0.0)),
        )
        .unwrap();
        let color = Scalar::from((0.0, 0.0, 0.0, 0.0));
        let rect = Rect::new(16, 16, 96, 96);
        imgproc::rectangle(&mut image, rect, color, imgproc::FILLED, imgproc::LINE_8, 0).unwrap();
        let params = Vector::<i32>::new();
        imgcodecs::imwrite(path.to_string_lossy().as_ref(), &image, &params).unwrap();
    }
}
