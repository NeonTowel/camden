//! AI-powered image classification for content moderation and tagging.
//!
//! This module provides ONNX-based inference for:
//! - Content moderation (NSFW detection with 5 classes)
//! - Automatic image tagging (ImageNet classification)
//!
//! # Configuration
//!
//! Models can be configured via a TOML file (`camden-classifier.toml`):
//!
//! ```toml
//! models_dir = ".vendor/models"
//! active_moderation = "gantman-nsfw"
//! active_tagging = "mobilenetv2"
//!
//! [models.gantman-nsfw]
//! name = "GantMan NSFW"
//! type = "moderation"
//! path = "nsfw-inception-v3.onnx"
//! ```
//!
//! # Runtime Initialization
//!
//! Before using any classifier, you must initialize the ONNX Runtime by calling
//! [`init_ort_runtime`] with the path to the `onnxruntime.dll`. This is required
//! because we use dynamic loading to avoid CRT conflicts with static OpenCV.
//!
//! ```no_run
//! use camden_core::classifier::init_ort_runtime;
//!
//! // Initialize once at application startup
//! init_ort_runtime(".vendor/onnxruntime/lib/onnxruntime.dll").unwrap();
//! ```

mod config;
mod moderation;
mod runtime;
mod tagging;

pub use config::{ClassifierConfig, ModelConfig, ModelInputSpec, ModelOutputSpec, ModelPreset, ModelType};
pub use moderation::{ModerationCategories, ModerationConfig, ModerationFlags, ModerationModelFormat, ModerationTier, NsfwClassifier};
pub use runtime::{ClassifierError, ModelPaths};
pub use tagging::{ImageTag, TagCategory, TaggingClassifier, TaggingConfig};

use std::path::Path;
use std::sync::OnceLock;

/// Global flag to track if ORT runtime has been initialized.
static ORT_INITIALIZED: OnceLock<()> = OnceLock::new();

/// Initialize the ONNX Runtime with the path to the dynamic library.
///
/// This must be called once before using any classifier. The function is
/// idempotent - subsequent calls after successful initialization are no-ops.
///
/// # Arguments
///
/// * `dylib_path` - Path to `onnxruntime.dll` (Windows) or `libonnxruntime.so` (Linux)
///
/// # Errors
///
/// Returns an error if the library cannot be loaded or initialized.
///
/// # Example
///
/// ```no_run
/// use camden_core::classifier::init_ort_runtime;
///
/// // From .vendor directory
/// init_ort_runtime(".vendor/onnxruntime/lib/onnxruntime.dll")?;
/// # Ok::<(), camden_core::classifier::ClassifierError>(())
/// ```
pub fn init_ort_runtime(dylib_path: impl AsRef<Path>) -> Result<(), ClassifierError> {
    let path = dylib_path.as_ref();
    
    if ORT_INITIALIZED.get().is_some() {
        return Ok(());
    }
    
    if !path.exists() {
        return Err(ClassifierError::Processing(format!(
            "ONNX Runtime library not found at: {}. Run 'task deps-onnxruntime' to download.",
            path.display()
        )));
    }
    
    let path_str = path.to_str().ok_or_else(|| {
        ClassifierError::Processing("ONNX Runtime path contains invalid UTF-8".to_string())
    })?;
    
    ort::init_from(path_str)
        .commit()
        .map_err(ClassifierError::Ort)?;
    
    let _ = ORT_INITIALIZED.set(());
    Ok(())
}

/// Get the default path to the ONNX Runtime library.
///
/// Returns the path relative to the workspace root (.vendor/onnxruntime/lib/onnxruntime.dll).
pub fn default_ort_dylib_path() -> std::path::PathBuf {
    #[cfg(windows)]
    {
        std::path::PathBuf::from(".vendor/onnxruntime/lib/onnxruntime.dll")
    }
    #[cfg(not(windows))]
    {
        std::path::PathBuf::from(".vendor/onnxruntime/lib/libonnxruntime.so")
    }
}

/// Combined classifier that runs both moderation and tagging models.
pub struct ImageClassifier {
    moderation: NsfwClassifier,
    tagging: TaggingClassifier,
    config: ClassifierConfig,
}

impl ImageClassifier {
    /// Create a new classifier with models from the specified paths.
    pub fn new(paths: &ModelPaths) -> Result<Self, ClassifierError> {
        let moderation = NsfwClassifier::new(&paths.nsfw_model)?;
        let tagging = TaggingClassifier::new(&paths.tagging_model)?;
        Ok(Self {
            moderation,
            tagging,
            config: ClassifierConfig::default(),
        })
    }

    /// Create a classifier using default model paths (.vendor/models/).
    pub fn with_default_paths() -> Result<Self, ClassifierError> {
        let paths = ModelPaths::default();
        Self::new(&paths)
    }
    
    /// Create a classifier from a configuration.
    pub fn from_config(config: ClassifierConfig) -> Result<Self, ClassifierError> {
        let moderation_path = config.active_moderation_path().ok_or_else(|| {
            ClassifierError::Processing("no active moderation model configured".to_string())
        })?;
        let tagging_path = config.active_tagging_path().ok_or_else(|| {
            ClassifierError::Processing("no active tagging model configured".to_string())
        })?;
        
        // Get model-specific configuration for moderation
        let moderation = if let Some(model_config) = config.active_moderation_model() {
            let mod_config = ModerationConfig::from_specs(
                &model_config.input,
                &model_config.output.labels,
                model_config.output.format.as_deref(),
            );
            NsfwClassifier::with_config(&moderation_path, mod_config)?
        } else {
            NsfwClassifier::new(&moderation_path)?
        };
        
        // Get model-specific configuration for tagging
        let tagging = if let Some(model_config) = config.active_tagging_model() {
            let tag_config = TaggingConfig::from_specs(&model_config.input);
            TaggingClassifier::with_config(&tagging_path, tag_config)?
        } else {
            TaggingClassifier::new(&tagging_path)?
        };
        
        Ok(Self {
            moderation,
            tagging,
            config,
        })
    }
    
    /// Create a classifier by loading config from file or using defaults.
    pub fn from_config_or_default() -> Result<Self, ClassifierError> {
        let config = ClassifierConfig::load_or_default();
        Self::from_config(config)
    }
    
    /// Get the current configuration.
    pub fn config(&self) -> &ClassifierConfig {
        &self.config
    }

    /// Analyze an image for moderation flags only.
    pub fn moderate(&mut self, image_path: &Path) -> Result<ModerationFlags, ClassifierError> {
        self.moderation.classify(image_path)
    }

    /// Generate tags for an image.
    pub fn tag(&mut self, image_path: &Path, max_tags: usize) -> Result<Vec<ImageTag>, ClassifierError> {
        self.tagging.classify(image_path, max_tags)
    }

    /// Run full classification (moderation + tagging).
    pub fn classify(&mut self, image_path: &Path) -> Result<ClassificationResult, ClassifierError> {
        let moderation = self.moderation.classify(image_path)?;
        let tags = self.tagging.classify(image_path, 10)?;
        Ok(ClassificationResult { moderation, tags })
    }
}

/// Complete classification result containing moderation and tags.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ClassificationResult {
    pub moderation: ModerationFlags,
    pub tags: Vec<ImageTag>,
}
