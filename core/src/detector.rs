use kamadak_exif::{In, Reader, Tag};
use opencv::core::{self, AlgorithmHint, Mat, MatTraitConst, MatTraitConstManual, Scalar, Size};
use opencv::imgcodecs;
use opencv::imgproc;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::path::{Path, PathBuf};
use time::{format_description::well_known::Rfc3339, OffsetDateTime};

const IMREAD_COLOR: i32 = imgcodecs::IMREAD_COLOR;
const HASH_SIZE: i32 = 8;

/// Represents a 64-bit perceptual hash fingerprint.
pub type Fingerprint = u64;

/// Configuration options for the detector.
#[derive(Clone)]
pub struct DetectorConfig {
    pub hash_threshold: u32,
    pub color_tolerance: f64,
    pub contrast_tolerance: f64,
    pub texture_tolerance: f64,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            hash_threshold: 6,
            color_tolerance: 20.0,
            contrast_tolerance: 15.0,
            texture_tolerance: 12.0,
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

        Ok(ImageAnalysis {
            features: ImageFeatures {
                fingerprint,
                mean,
                stddev,
                texture,
            },
            metadata,
        })
    }

    /// Determines whether the provided feature sets represent matching images.
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
}

#[derive(Clone)]
pub struct ImageMetadata {
    pub size_bytes: u64,
    pub dimensions: (i32, i32),
    pub modified: Option<String>,
    pub captured_at: Option<String>,
    pub dominant_color: [u8; 3],
    pub confidence: f32,
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
    use opencv::core::{self, Rect, Scalar};
    use opencv::imgcodecs;
    use opencv::imgproc;
    use opencv::prelude::*;
    use opencv::types::VectorOfi32;
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
        let params = VectorOfi32::new();
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
        let params = VectorOfi32::new();
        imgcodecs::imwrite(path.to_string_lossy().as_ref(), &image, &params).unwrap();
    }
}
