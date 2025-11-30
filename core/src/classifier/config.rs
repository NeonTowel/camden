//! Configuration for AI image classification.
//!
//! Supports loading model configurations from TOML files to allow
//! switching between different models without recompiling.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::runtime::ClassifierError;

/// Default configuration file name.
pub const DEFAULT_CONFIG_FILE: &str = "camden-classifier.toml";

/// Model type identifier.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    /// Content moderation (NSFW detection)
    Moderation,
    /// Image tagging/classification
    Tagging,
}

/// Input format requirements for a model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelInputSpec {
    /// Input image width in pixels
    pub width: u32,
    /// Input image height in pixels
    pub height: u32,
    /// Whether to apply ImageNet normalization (mean/std)
    #[serde(default)]
    pub normalize: bool,
    /// Input tensor layout (default: "NCHW")
    #[serde(default = "default_layout")]
    pub layout: String,
}

fn default_layout() -> String {
    "NCHW".to_string()
}

impl Default for ModelInputSpec {
    fn default() -> Self {
        Self {
            width: 224,
            height: 224,
            normalize: true,
            layout: default_layout(),
        }
    }
}

/// Output format specification for a model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelOutputSpec {
    /// Number of output classes
    pub num_classes: usize,
    /// Class labels (optional, can be loaded from file)
    #[serde(default)]
    pub labels: Vec<String>,
    /// Path to labels file (optional, one label per line)
    #[serde(default)]
    pub labels_file: Option<PathBuf>,
    /// Optional format hint for moderation models (e.g., "gantman", "nsfwjs")
    #[serde(default)]
    pub format: Option<String>,
}

impl Default for ModelOutputSpec {
    fn default() -> Self {
        Self {
            num_classes: 1000,
            labels: Vec::new(),
            labels_file: None,
            format: None,
        }
    }
}

/// Configuration for a single model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Display name for the model
    pub name: String,
    /// Model type (moderation or tagging)
    #[serde(rename = "type")]
    pub model_type: ModelType,
    /// Path to the ONNX model file (relative to models_dir or absolute)
    pub path: PathBuf,
    /// Input specification
    #[serde(default)]
    pub input: ModelInputSpec,
    /// Output specification
    #[serde(default)]
    pub output: ModelOutputSpec,
    /// Model description
    #[serde(default)]
    pub description: String,
    /// Whether this model is enabled
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

fn default_enabled() -> bool {
    true
}

/// Available model presets that can be downloaded.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelPreset {
    /// Preset identifier
    pub id: String,
    /// Display name
    pub name: String,
    /// Download URL
    pub url: String,
    /// Expected file size in bytes (for progress)
    pub size_bytes: u64,
    /// SHA256 checksum for verification
    #[serde(default)]
    pub sha256: Option<String>,
    /// Model configuration to use after download
    pub config: ModelConfig,
}

/// Root classifier configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClassifierConfig {
    /// Base directory for model files (default: .vendor/models)
    #[serde(default = "default_models_dir")]
    pub models_dir: PathBuf,
    
    /// Path to ONNX Runtime library (default: .vendor/onnxruntime/lib/onnxruntime.dll)
    #[serde(default = "default_ort_lib")]
    pub ort_library: PathBuf,
    
    /// Active moderation model ID (references a model in `models`)
    #[serde(default)]
    pub active_moderation: Option<String>,
    
    /// Active tagging model ID (references a model in `models`)
    #[serde(default)]
    pub active_tagging: Option<String>,
    
    /// Registered models by ID
    #[serde(default)]
    pub models: HashMap<String, ModelConfig>,
    
    /// Available model presets for download
    #[serde(default)]
    pub presets: Vec<ModelPreset>,
}

fn default_models_dir() -> PathBuf {
    PathBuf::from(".vendor/models")
}

fn default_ort_lib() -> PathBuf {
    #[cfg(windows)]
    {
        PathBuf::from(".vendor/onnxruntime/lib/onnxruntime.dll")
    }
    #[cfg(not(windows))]
    {
        PathBuf::from(".vendor/onnxruntime/lib/libonnxruntime.so")
    }
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        let mut models = HashMap::new();
        
        // Default moderation model: GantMan NSFW (5-class)
        models.insert(
            "gantman-nsfw".to_string(),
            ModelConfig {
                name: "GantMan NSFW".to_string(),
                model_type: ModelType::Moderation,
                path: PathBuf::from("nsfw-inception-v3.onnx"),
                input: ModelInputSpec {
                    width: 299,
                    height: 299,
                    normalize: false, // 0-1 range, no ImageNet normalization
                    layout: "NCHW".to_string(),
                },
                output: ModelOutputSpec {
                    num_classes: 5,
                    labels: vec![
                        "drawings".to_string(),
                        "hentai".to_string(),
                        "neutral".to_string(),
                        "porn".to_string(),
                        "sexy".to_string(),
                    ],
                    labels_file: None,
                    format: Some("gantman".to_string()),
                },
                description: "5-class NSFW detection (drawings, hentai, neutral, porn, sexy)".to_string(),
                enabled: true,
            },
        );
        
        // Default tagging model: MobileNetV2
        models.insert(
            "mobilenetv2".to_string(),
            ModelConfig {
                name: "MobileNetV2".to_string(),
                model_type: ModelType::Tagging,
                path: PathBuf::from("mobilenetv2-12.onnx"),
                input: ModelInputSpec {
                    width: 224,
                    height: 224,
                    normalize: true, // ImageNet normalization
                    layout: "NCHW".to_string(),
                },
                output: ModelOutputSpec {
                    num_classes: 1000,
                    labels: Vec::new(), // Use built-in ImageNet labels
                    labels_file: None,
                    format: None,
                },
                description: "ImageNet 1000-class classification, optimized for mobile".to_string(),
                enabled: true,
            },
        );
        
        Self {
            models_dir: default_models_dir(),
            ort_library: default_ort_lib(),
            active_moderation: Some("gantman-nsfw".to_string()),
            active_tagging: Some("mobilenetv2".to_string()),
            models,
            presets: Self::default_presets(),
        }
    }
}

impl ClassifierConfig {
    /// Load configuration from a TOML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ClassifierError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(ClassifierError::Processing(format!(
                "config file not found: {}",
                path.display()
            )));
        }
        
        let content = std::fs::read_to_string(path).map_err(|e| {
            ClassifierError::Processing(format!("failed to read config: {}", e))
        })?;
        
        toml::from_str(&content).map_err(|e| {
            ClassifierError::Processing(format!("invalid config TOML: {}", e))
        })
    }
    
    /// Load configuration from default location, falling back to defaults.
    pub fn load_or_default() -> Self {
        // Try loading from current directory
        if let Ok(config) = Self::load(DEFAULT_CONFIG_FILE) {
            return config;
        }
        
        // Try loading from .vendor directory
        let vendor_config = PathBuf::from(".vendor").join(DEFAULT_CONFIG_FILE);
        if let Ok(config) = Self::load(&vendor_config) {
            return config;
        }
        
        Self::default()
    }
    
    /// Save configuration to a TOML file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ClassifierError> {
        let content = toml::to_string_pretty(self).map_err(|e| {
            ClassifierError::Processing(format!("failed to serialize config: {}", e))
        })?;
        
        std::fs::write(path, content).map_err(|e| {
            ClassifierError::Processing(format!("failed to write config: {}", e))
        })
    }
    
    /// Get the full path to a model file.
    pub fn model_path(&self, model_id: &str) -> Option<PathBuf> {
        self.models.get(model_id).map(|m| {
            if m.path.is_absolute() {
                m.path.clone()
            } else {
                self.models_dir.join(&m.path)
            }
        })
    }
    
    /// Get the active moderation model configuration.
    pub fn active_moderation_model(&self) -> Option<&ModelConfig> {
        self.active_moderation
            .as_ref()
            .and_then(|id| self.models.get(id))
    }
    
    /// Get the active tagging model configuration.
    pub fn active_tagging_model(&self) -> Option<&ModelConfig> {
        self.active_tagging
            .as_ref()
            .and_then(|id| self.models.get(id))
    }
    
    /// Get the full path to the active moderation model.
    pub fn active_moderation_path(&self) -> Option<PathBuf> {
        self.active_moderation.as_ref().and_then(|id| self.model_path(id))
    }
    
    /// Get the full path to the active tagging model.
    pub fn active_tagging_path(&self) -> Option<PathBuf> {
        self.active_tagging.as_ref().and_then(|id| self.model_path(id))
    }
    
    /// List all registered models.
    pub fn list_models(&self) -> Vec<(&str, &ModelConfig)> {
        self.models.iter().map(|(k, v)| (k.as_str(), v)).collect()
    }
    
    /// Validate that all active models exist.
    pub fn validate(&self) -> Result<(), ClassifierError> {
        if let Some(path) = self.active_moderation_path() {
            if !path.exists() {
                return Err(ClassifierError::ModelNotFound(path));
            }
        }
        
        if let Some(path) = self.active_tagging_path() {
            if !path.exists() {
                return Err(ClassifierError::ModelNotFound(path));
            }
        }
        
        if !self.ort_library.exists() {
            return Err(ClassifierError::Processing(format!(
                "ONNX Runtime library not found: {}",
                self.ort_library.display()
            )));
        }
        
        Ok(())
    }
    
    /// Default model presets available for download.
    fn default_presets() -> Vec<ModelPreset> {
        vec![
            // Moderation models
            ModelPreset {
                id: "gantman-nsfw".to_string(),
                name: "GantMan NSFW (5-class)".to_string(),
                url: "https://github.com/iola1999/nsfw-detect-onnx/releases/download/v1.0.0/model.onnx".to_string(),
                size_bytes: 87_000_000,
                sha256: None,
                config: ModelConfig {
                    name: "GantMan NSFW".to_string(),
                    model_type: ModelType::Moderation,
                    path: PathBuf::from("nsfw-inception-v3.onnx"),
                    input: ModelInputSpec {
                        width: 299,
                        height: 299,
                        normalize: false,
                        layout: "NCHW".to_string(),
                    },
                    output: ModelOutputSpec {
                        num_classes: 5,
                        labels: vec![
                            "drawings".to_string(),
                            "hentai".to_string(),
                            "neutral".to_string(),
                            "porn".to_string(),
                            "sexy".to_string(),
                        ],
                        labels_file: None,
                        format: Some("gantman".to_string()),
                    },
                    description: "5-class NSFW detection".to_string(),
                    enabled: true,
                },
            },
            // Tagging models
            ModelPreset {
                id: "mobilenetv2".to_string(),
                name: "MobileNetV2 (ImageNet)".to_string(),
                url: "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx".to_string(),
                size_bytes: 14_000_000,
                sha256: None,
                config: ModelConfig {
                    name: "MobileNetV2".to_string(),
                    model_type: ModelType::Tagging,
                    path: PathBuf::from("mobilenetv2-12.onnx"),
                    input: ModelInputSpec {
                        width: 224,
                        height: 224,
                        normalize: true,
                        layout: "NCHW".to_string(),
                    },
                    output: ModelOutputSpec {
                        num_classes: 1000,
                        labels: Vec::new(),
                        labels_file: None,
                        format: None,
                    },
                    description: "ImageNet 1000-class, fast inference".to_string(),
                    enabled: true,
                },
            },
            ModelPreset {
                id: "efficientnet-lite4".to_string(),
                name: "EfficientNet-Lite4 (ImageNet)".to_string(),
                url: "https://github.com/onnx/models/raw/main/validated/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx".to_string(),
                size_bytes: 50_000_000,
                sha256: None,
                config: ModelConfig {
                    name: "EfficientNet-Lite4".to_string(),
                    model_type: ModelType::Tagging,
                    path: PathBuf::from("efficientnet-lite4-11.onnx"),
                    input: ModelInputSpec {
                        width: 300,
                        height: 300,
                        normalize: true,
                        layout: "NCHW".to_string(),
                    },
                    output: ModelOutputSpec {
                        num_classes: 1000,
                        labels: Vec::new(),
                        labels_file: None,
                        format: None,
                    },
                    description: "Higher accuracy ImageNet classification".to_string(),
                    enabled: true,
                },
            },
            ModelPreset {
                id: "resnet50".to_string(),
                name: "ResNet-50 (ImageNet)".to_string(),
                url: "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx".to_string(),
                size_bytes: 100_000_000,
                sha256: None,
                config: ModelConfig {
                    name: "ResNet-50".to_string(),
                    model_type: ModelType::Tagging,
                    path: PathBuf::from("resnet50-v2-7.onnx"),
                    input: ModelInputSpec {
                        width: 224,
                        height: 224,
                        normalize: true,
                        layout: "NCHW".to_string(),
                    },
                    output: ModelOutputSpec {
                        num_classes: 1000,
                        labels: Vec::new(),
                        labels_file: None,
                        format: None,
                    },
                    description: "Classic ResNet, good accuracy/speed balance".to_string(),
                    enabled: true,
                },
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = ClassifierConfig::default();
        assert!(config.active_moderation.is_some());
        assert!(config.active_tagging.is_some());
        assert!(!config.models.is_empty());
    }
    
    #[test]
    fn test_config_roundtrip() {
        let config = ClassifierConfig::default();
        let toml = toml::to_string_pretty(&config).unwrap();
        let parsed: ClassifierConfig = toml::from_str(&toml).unwrap();
        assert_eq!(parsed.active_moderation, config.active_moderation);
        assert_eq!(parsed.active_tagging, config.active_tagging);
    }
    
    #[test]
    fn test_model_path_resolution() {
        let config = ClassifierConfig::default();
        let path = config.model_path("mobilenetv2").unwrap();
        assert!(path.ends_with("mobilenetv2-12.onnx"));
    }
}
