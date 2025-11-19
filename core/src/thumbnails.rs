use crate::detector::Fingerprint;
use dirs::data_local_dir;
use opencv::core::{Mat, Size, Vector};
use opencv::imgcodecs;
use opencv::imgproc;
use opencv::prelude::*;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_MAX_DIMENSION: i32 = 512;

#[derive(Clone, Debug)]
pub struct ThumbnailCache {
    root: PathBuf,
    max_dimension: i32,
}

impl ThumbnailCache {
    pub fn new(root: Option<PathBuf>) -> Result<Self, ThumbnailError> {
        let root = match root {
            Some(path) => path,
            None => default_cache_root().ok_or(ThumbnailError::CacheDirectoryUnavailable)?,
        };
        fs::create_dir_all(&root).map_err(|source| ThumbnailError::Io {
            source,
            path: root.clone(),
        })?;
        Ok(Self {
            root,
            max_dimension: DEFAULT_MAX_DIMENSION,
        })
    }

    pub fn with_max_dimension(mut self, max_dimension: i32) -> Self {
        self.max_dimension = max_dimension.max(1);
        self
    }

    pub fn ensure(
        &self,
        source: &Path,
        fingerprint: Fingerprint,
    ) -> Result<PathBuf, ThumbnailError> {
        let target = self.root.join(format!("{:016x}.png", fingerprint));
        let legacy = self.root.join(format!("{:016x}.webp", fingerprint));
        if legacy.exists() {
            let _ = fs::remove_file(&legacy);
        }
        if target.exists() {
            return Ok(target);
        }
        generate_thumbnail(source, &target, self.max_dimension)?;
        Ok(target)
    }

    pub fn clear(&self) -> Result<(), ThumbnailError> {
        if !self.root.exists() {
            return Ok(());
        }
        for entry in fs::read_dir(&self.root).map_err(|source| ThumbnailError::Io {
            source,
            path: self.root.clone(),
        })? {
            let path = entry
                .map_err(|source| ThumbnailError::Io {
                    source,
                    path: self.root.clone(),
                })?
                .path();
            if path.is_file() {
                fs::remove_file(&path).map_err(|source| ThumbnailError::Io { source, path })?;
            }
        }
        Ok(())
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

fn default_cache_root() -> Option<PathBuf> {
    data_local_dir().map(|mut dir| {
        dir.push("Camden");
        dir.push("thumbnails");
        dir
    })
}

fn generate_thumbnail(
    source: &Path,
    target: &Path,
    max_dimension: i32,
) -> Result<(), ThumbnailError> {
    let source_str = source
        .to_str()
        .ok_or_else(|| ThumbnailError::InvalidPath(source.to_path_buf()))?;
    let image =
        imgcodecs::imread(source_str, imgcodecs::IMREAD_COLOR).map_err(ThumbnailError::OpenCv)?;
    if image.empty() {
        return Err(ThumbnailError::EmptyImage(source.to_path_buf()));
    }

    let width = image.cols();
    let height = image.rows();
    let max_current = width.max(height).max(1);
    let mut resized = Mat::default();
    let thumb = if max_current <= max_dimension {
        &image
    } else {
        let scale = max_dimension as f64 / max_current as f64;
        let new_size = Size::new(
            ((width as f64) * scale).round() as i32,
            ((height as f64) * scale).round() as i32,
        );
        imgproc::resize(
            &image,
            &mut resized,
            new_size,
            0.0,
            0.0,
            imgproc::INTER_AREA,
        )
        .map_err(ThumbnailError::OpenCv)?;
        &resized
    };

    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent).map_err(|source| ThumbnailError::Io {
            source,
            path: parent.to_path_buf(),
        })?;
    }
    let target_str = target
        .to_str()
        .ok_or_else(|| ThumbnailError::InvalidPath(target.to_path_buf()))?;
    let params = Vector::new();
    imgcodecs::imwrite(target_str, thumb, &params).map_err(ThumbnailError::OpenCv)?;
    Ok(())
}

#[derive(Debug)]
pub enum ThumbnailError {
    CacheDirectoryUnavailable,
    InvalidPath(PathBuf),
    EmptyImage(PathBuf),
    Io {
        source: std::io::Error,
        path: PathBuf,
    },
    OpenCv(opencv::Error),
}

impl Display for ThumbnailError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CacheDirectoryUnavailable => {
                write!(f, "unable to determine thumbnail cache directory")
            }
            Self::InvalidPath(path) => write!(f, "path is not valid UTF-8: {}", path.display()),
            Self::EmptyImage(path) => write!(f, "image at {} is empty", path.display()),
            Self::Io { source, path } => write!(f, "io error for {}: {}", path.display(), source),
            Self::OpenCv(error) => write!(f, "opencv error: {}", error),
        }
    }
}

impl Error for ThumbnailError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::OpenCv(error) => Some(error),
            _ => None,
        }
    }
}

impl From<opencv::Error> for ThumbnailError {
    fn from(error: opencv::Error) -> Self {
        Self::OpenCv(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{self, Scalar};
    use tempfile::tempdir;

    #[test]
    fn ensure_creates_thumbnail_once() {
        let image_dir = tempdir().unwrap();
        let thumb_dir = tempdir().unwrap();
        let source_path = image_dir.path().join("image.png");
        let image = Mat::new_rows_cols_with_default(
            1024,
            768,
            core::CV_8UC3,
            Scalar::from((120.0, 20.0, 220.0, 0.0)),
        )
        .unwrap();
        let params = Vector::new();
        imgcodecs::imwrite(source_path.to_string_lossy().as_ref(), &image, &params).unwrap();

        let cache = ThumbnailCache::new(Some(thumb_dir.path().to_path_buf())).unwrap();
        let thumbnail = cache.ensure(&source_path, 0).unwrap();
        assert!(thumbnail.exists());
        let metadata = fs::metadata(&thumbnail).unwrap();
        assert!(metadata.len() > 0);

        let second = cache.ensure(&source_path, 0).unwrap();
        assert_eq!(thumbnail, second);
    }
}
