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
    /// Optional custom normalization mean for the model.
    #[serde(default)]
    pub mean: Option<[f32; 3]>,
    /// Optional custom normalization std for the model.
    #[serde(default)]
    pub std: Option<[f32; 3]>,
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
            mean: None,
            std: None,
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
    /// Whether this is a multi-label classifier (sigmoid outputs, no softmax).
    /// WD taggers and similar models use multi-label classification where each
    /// tag has an independent probability. Set to true to skip softmax.
    #[serde(default)]
    pub multi_label: bool,
}

impl Default for ModelOutputSpec {
    fn default() -> Self {
        Self {
            num_classes: 1000,
            labels: Vec::new(),
            labels_file: None,
            format: None,
            multi_label: false,
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
///
/// # Simple Configuration
///
/// The TOML config only needs to specify which models to use:
///
/// ```toml
/// moderation_model = "gantman-nsfw"
/// tagging_model = "mobilenetv2"
/// ```
///
/// All model definitions are built into the code. Available models:
///
/// **Moderation models:**
/// - `gantman-nsfw` - 5-class NSFW (drawings, hentai, neutral, porn, sexy)
/// - `adamcodd-vit-nsfw` - ViT NSFW detector (nsfw/sfw)
/// - `falconsai-nsfw` - Fast NSFW detector (normal/nsfw)
/// - `spiele-nsfw` - 4-tier severity (neutral → restricted)
///
/// **Tagging models:**
/// - `mobilenetv2` - Fast ImageNet 1000-class (default)
/// - `convnextv2-large` - High-accuracy ImageNet
/// - `wd-vit-tagger-v3` - Danbooru-style multi-label tagger
/// - `wd-swinv2-tagger-v3` - Danbooru-style (SwinV2 architecture)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClassifierConfig {
    /// Base directory for model files (default: .vendor/models)
    #[serde(default = "default_models_dir")]
    pub models_dir: PathBuf,

    /// Path to ONNX Runtime library (default: .vendor/onnxruntime/lib/onnxruntime.dll)
    #[serde(default = "default_ort_lib")]
    pub ort_library: PathBuf,

    /// Moderation model to use (e.g., "gantman-nsfw", "adamcodd-vit-nsfw")
    #[serde(default, alias = "active_moderation")]
    pub moderation_model: Option<String>,

    /// Tagging model to use (e.g., "mobilenetv2", "wd-vit-tagger-v3")
    #[serde(default, alias = "active_tagging")]
    pub tagging_model: Option<String>,

    /// Custom model configurations (optional, overrides built-in presets)
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub models: HashMap<String, ModelConfig>,

    /// Available model presets for download (internal use)
    #[serde(skip)]
    pub presets: Vec<ModelPreset>,

    /// Built-in model registry (not serialized, populated at runtime)
    #[serde(skip)]
    builtin_models: HashMap<String, ModelConfig>,
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
        let builtin_models = Self::builtin_model_registry();

        Self {
            models_dir: default_models_dir(),
            ort_library: default_ort_lib(),
            moderation_model: Some("gantman-nsfw".to_string()),
            tagging_model: Some("mobilenetv2".to_string()),
            models: HashMap::new(),
            presets: Self::default_presets(),
            builtin_models,
        }
    }
}

impl ClassifierConfig {
    /// Create built-in model registry with all supported models.
    fn builtin_model_registry() -> HashMap<String, ModelConfig> {
        let mut models = HashMap::new();

        // =========================================================================
        // MODERATION MODELS
        // =========================================================================

        // GantMan NSFW (5-class) - Default
        models.insert(
            "gantman-nsfw".to_string(),
            ModelConfig {
                name: "GantMan NSFW".to_string(),
                model_type: ModelType::Moderation,
                path: PathBuf::from("nsfw-inception-v3.onnx"),
                input: ModelInputSpec {
                    width: 299,
                    height: 299,
                    normalize: false,
                    layout: "NHWC".to_string(),
                    mean: None,
                    std: None,
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
                    multi_label: false,
                },
                description: "5-class NSFW detection (drawings, hentai, neutral, porn, sexy)"
                    .to_string(),
                enabled: true,
            },
        );

        // AdamCodd ViT NSFW
        models.insert(
            "adamcodd-vit-nsfw".to_string(),
            ModelConfig {
                name: "AdamCodd ViT NSFW".to_string(),
                model_type: ModelType::Moderation,
                path: PathBuf::from("adamcodd-vit-base-nsfw.onnx"),
                input: ModelInputSpec {
                    width: 384,
                    height: 384,
                    normalize: true,
                    layout: "NCHW".to_string(),
                    mean: None,
                    std: None,
                },
                output: ModelOutputSpec {
                    num_classes: 2,
                    labels: vec!["nsfw".to_string(), "sfw".to_string()],
                    labels_file: None,
                    format: Some("nsfw_sfw".to_string()),
                    multi_label: false,
                },
                description: "AdamCodd ViT NSFW detector (nsfw/sfw), 384x384".to_string(),
                enabled: true,
            },
        );

        // NOTE: Falconsai NSFW removed - no ONNX version available on HuggingFace
        // (only PyTorch/Safetensors format)

        // spiele NSFW (4-tier severity)
        models.insert(
            "spiele-nsfw".to_string(),
            ModelConfig {
                name: "spiele NSFW Detector".to_string(),
                model_type: ModelType::Moderation,
                path: PathBuf::from("spiele-nsfw.onnx"),
                input: ModelInputSpec {
                    width: 448,
                    height: 448,
                    normalize: true,
                    layout: "NCHW".to_string(),
                    mean: Some([0.5, 0.5, 0.5]),
                    std: Some([0.5, 0.5, 0.5]),
                },
                output: ModelOutputSpec {
                    num_classes: 4,
                    labels: vec![
                        "neutral".to_string(),
                        "sensitive".to_string(),
                        "mature".to_string(),
                        "nsfw-restricted".to_string(),
                    ],
                    labels_file: None,
                    format: None,
                    multi_label: false,
                },
                description: "spiele ViT-based NSFW severity detector (neutral → restricted)"
                    .to_string(),
                enabled: true,
            },
        );

        // TaufiqDP MobileNetV4 NSFW
        models.insert(
            "taufiqdp-mobilenetv4".to_string(),
            ModelConfig {
                name: "TaufiqDP MobileNetV4 NSFW".to_string(),
                model_type: ModelType::Moderation,
                path: PathBuf::from(
                    "taufiqdp-mobilenetv4_conv_small.e2400_r224_in1k_nsfw_classifier.onnx",
                ),
                input: ModelInputSpec {
                    width: 224,
                    height: 224,
                    normalize: true,
                    layout: "NCHW".to_string(),
                    mean: None,
                    std: None,
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
                    multi_label: false,
                },
                description: "TaufiqDP MobileNetV4 NSFW (GantMan 5-class)".to_string(),
                enabled: true,
            },
        );

        // NSFWJS (deepghs conversion)
        models.insert(
            "nsfwjs".to_string(),
            ModelConfig {
                name: "NSFWJS".to_string(),
                model_type: ModelType::Moderation,
                path: PathBuf::from("nsfwjs.onnx"),
                input: ModelInputSpec {
                    width: 224,
                    height: 224,
                    normalize: false,
                    layout: "NHWC".to_string(),
                    mean: None,
                    std: None,
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
                    multi_label: false,
                },
                description: "NSFWJS model (GantMan 5-class), lightweight".to_string(),
                enabled: true,
            },
        );

        // ONNX Community NSFW (ViT-based)
        models.insert(
            "onnx-community-nsfw".to_string(),
            ModelConfig {
                name: "ONNX Community NSFW".to_string(),
                model_type: ModelType::Moderation,
                path: PathBuf::from("onnx-community-nsfw.onnx"),
                input: ModelInputSpec {
                    width: 224,
                    height: 224,
                    normalize: true,
                    layout: "NCHW".to_string(),
                    mean: None,
                    std: None,
                },
                output: ModelOutputSpec {
                    num_classes: 2,
                    labels: vec!["nsfw".to_string(), "sfw".to_string()],
                    labels_file: None,
                    format: Some("nsfw_sfw".to_string()),
                    multi_label: false,
                },
                description: "ONNX Community ViT NSFW detector (nsfw/sfw)".to_string(),
                enabled: true,
            },
        );

        // =========================================================================
        // TAGGING MODELS - ImageNet
        // =========================================================================

        // MobileNetV2 - Default
        models.insert(
            "mobilenetv2".to_string(),
            ModelConfig {
                name: "MobileNetV2".to_string(),
                model_type: ModelType::Tagging,
                path: PathBuf::from("mobilenetv2-12.onnx"),
                input: ModelInputSpec {
                    width: 224,
                    height: 224,
                    normalize: true,
                    layout: "NCHW".to_string(),
                    mean: None,
                    std: None,
                },
                output: ModelOutputSpec {
                    num_classes: 1000,
                    labels: Vec::new(),
                    labels_file: None,
                    format: None,
                    multi_label: false,
                },
                description: "ImageNet 1000-class classification, optimized for mobile".to_string(),
                enabled: true,
            },
        );

        // ConvNeXtV2 Large
        models.insert(
            "convnextv2-large".to_string(),
            ModelConfig {
                name: "ConvNeXtV2 Large".to_string(),
                model_type: ModelType::Tagging,
                path: PathBuf::from("convnextv2-large-1k-224.onnx"),
                input: ModelInputSpec {
                    width: 224,
                    height: 224,
                    normalize: true,
                    layout: "NCHW".to_string(),
                    mean: None,
                    std: None,
                },
                output: ModelOutputSpec {
                    num_classes: 1000,
                    labels: Vec::new(),
                    labels_file: None,
                    format: None,
                    multi_label: false,
                },
                description: "ConvNeXtV2 Large ImageNet 1000-class".to_string(),
                enabled: true,
            },
        );

        // EfficientNet-Lite4
        models.insert(
            "efficientnet-lite4".to_string(),
            ModelConfig {
                name: "EfficientNet-Lite4".to_string(),
                model_type: ModelType::Tagging,
                path: PathBuf::from("efficientnet-lite4-11.onnx"),
                input: ModelInputSpec {
                    width: 224,
                    height: 224,
                    normalize: true,
                    layout: "NCHW".to_string(),
                    mean: None,
                    std: None,
                },
                output: ModelOutputSpec {
                    num_classes: 1000,
                    labels: Vec::new(),
                    labels_file: None,
                    format: None,
                    multi_label: false,
                },
                description: "EfficientNet-Lite4 ImageNet 1000-class, good accuracy".to_string(),
                enabled: true,
            },
        );

        // ResNet-50
        models.insert(
            "resnet50".to_string(),
            ModelConfig {
                name: "ResNet-50".to_string(),
                model_type: ModelType::Tagging,
                path: PathBuf::from("resnet50-v2-7.onnx"),
                input: ModelInputSpec {
                    width: 224,
                    height: 224,
                    normalize: true,
                    layout: "NCHW".to_string(),
                    mean: None,
                    std: None,
                },
                output: ModelOutputSpec {
                    num_classes: 1000,
                    labels: Vec::new(),
                    labels_file: None,
                    format: None,
                    multi_label: false,
                },
                description: "ResNet-50 ImageNet 1000-class, classic architecture".to_string(),
                enabled: true,
            },
        );

        // ViT Base (ImageNet)
        models.insert(
            "vit-base-224".to_string(),
            ModelConfig {
                name: "ViT Base 224".to_string(),
                model_type: ModelType::Tagging,
                path: PathBuf::from("vit-base-224.onnx"),
                input: ModelInputSpec {
                    width: 224,
                    height: 224,
                    normalize: true,
                    layout: "NCHW".to_string(),
                    mean: None,
                    std: None,
                },
                output: ModelOutputSpec {
                    num_classes: 1000,
                    labels: Vec::new(),
                    labels_file: None,
                    format: None,
                    multi_label: false,
                },
                description: "Vision Transformer (ViT) Base ImageNet 1000-class".to_string(),
                enabled: true,
            },
        );

        // ViT Base Quantized (ImageNet)
        models.insert(
            "vit-base-224-q8".to_string(),
            ModelConfig {
                name: "ViT Base 224 (Quantized)".to_string(),
                model_type: ModelType::Tagging,
                path: PathBuf::from("vit-base-224-q8.onnx"),
                input: ModelInputSpec {
                    width: 224,
                    height: 224,
                    normalize: true,
                    layout: "NCHW".to_string(),
                    mean: None,
                    std: None,
                },
                output: ModelOutputSpec {
                    num_classes: 1000,
                    labels: Vec::new(),
                    labels_file: None,
                    format: None,
                    multi_label: false,
                },
                description: "Vision Transformer (ViT) Base ImageNet, int8 quantized".to_string(),
                enabled: true,
            },
        );

        // =========================================================================
        // TAGGING MODELS - WD Taggers (Danbooru-style)
        // =========================================================================

        let wd_label_file = PathBuf::from("wd-tags.csv");

        for (id, name) in [
            ("wd-vit-tagger-v3", "WD ViT Tagger V3"),
            ("wd-vit-large-tagger-v3", "WD ViT Large Tagger V3"),
            ("wd-swinv2-tagger-v3", "WD SwinV2 Tagger V3"),
            ("wd-eva02-large-tagger-v3", "WD EVA02 Large Tagger V3"),
            ("p1atdev-wd-swinv2-tagger-v3-hf", "p1atdev WD SwinV2 Tagger V3"),
        ] {
            models.insert(
                id.to_string(),
                ModelConfig {
                    name: name.to_string(),
                    model_type: ModelType::Tagging,
                    path: PathBuf::from(format!("{}.onnx", id)),
                    input: ModelInputSpec {
                        width: 448,
                        height: 448,
                        normalize: true,
                        layout: "NHWC".to_string(),  // WD models use NHWC (batch, height, width, channels)
                        mean: Some([0.5, 0.5, 0.5]),
                        std: Some([0.5, 0.5, 0.5]),
                    },
                    output: ModelOutputSpec {
                        num_classes: 10861,
                        labels: Vec::new(),
                        labels_file: Some(wd_label_file.clone()),
                        format: None,
                        multi_label: true, // WD taggers use sigmoid (multi-label), not softmax
                    },
                    description: format!("{name} (Danbooru-style tagger)"),
                    enabled: true,
                },
            );
        }

        // NOTE: Camie Tagger (70K+ tags) available but requires JSON label format
        // Add support for JSON labels to enable: https://huggingface.co/Camais03/camie-tagger

        models
    }

    /// Get a model config by ID, checking user overrides first, then built-in.
    pub fn get_model(&self, id: &str) -> Option<&ModelConfig> {
        self.models.get(id).or_else(|| self.builtin_models.get(id))
    }

    /// List all available models (built-in + user overrides).
    pub fn available_models(&self) -> Vec<(&str, &ModelConfig)> {
        let mut result: HashMap<&str, &ModelConfig> = self
            .builtin_models
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect();

        // User overrides take precedence
        for (k, v) in &self.models {
            result.insert(k.as_str(), v);
        }

        result.into_iter().collect()
    }

    /// List available moderation models.
    pub fn moderation_models(&self) -> Vec<(&str, &ModelConfig)> {
        self.available_models()
            .into_iter()
            .filter(|(_, c)| c.model_type == ModelType::Moderation)
            .collect()
    }

    /// List available tagging models.
    pub fn tagging_models(&self) -> Vec<(&str, &ModelConfig)> {
        self.available_models()
            .into_iter()
            .filter(|(_, c)| c.model_type == ModelType::Tagging)
            .collect()
    }

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
    
    /// Load configuration from default location, merging with built-in defaults.
    ///
    /// User config only needs to specify model names:
    /// ```toml
    /// moderation_model = "gantman-nsfw"
    /// tagging_model = "mobilenetv2"
    /// ```
    pub fn load_or_default() -> Self {
        let mut config = Self::default();

        // Try loading user config from current directory
        let user_config = Self::load(DEFAULT_CONFIG_FILE)
            .or_else(|_| Self::load(PathBuf::from(".vendor").join(DEFAULT_CONFIG_FILE)));

        if let Ok(user) = user_config {
            // Merge user config with defaults
            if user.models_dir != default_models_dir() {
                config.models_dir = user.models_dir;
            }
            if user.ort_library != default_ort_lib() {
                config.ort_library = user.ort_library;
            }
            if user.moderation_model.is_some() {
                config.moderation_model = user.moderation_model;
            }
            if user.tagging_model.is_some() {
                config.tagging_model = user.tagging_model;
            }
            // User model overrides
            for (id, model) in user.models {
                config.models.insert(id, model);
            }
        }

        config
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
        self.get_model(model_id).map(|m| {
            if m.path.is_absolute() {
                m.path.clone()
            } else {
                self.models_dir.join(&m.path)
            }
        })
    }

    /// Get the active moderation model configuration.
    pub fn active_moderation_model(&self) -> Option<&ModelConfig> {
        self.moderation_model
            .as_ref()
            .and_then(|id| self.get_model(id))
    }

    /// Get the active tagging model configuration.
    pub fn active_tagging_model(&self) -> Option<&ModelConfig> {
        self.tagging_model
            .as_ref()
            .and_then(|id| self.get_model(id))
    }

    /// Get the full path to the active moderation model.
    pub fn active_moderation_path(&self) -> Option<PathBuf> {
        self.moderation_model
            .as_ref()
            .and_then(|id| self.model_path(id))
    }

    /// Get the full path to the active tagging model.
    pub fn active_tagging_path(&self) -> Option<PathBuf> {
        self.tagging_model
            .as_ref()
            .and_then(|id| self.model_path(id))
    }

    /// List all registered models (deprecated, use available_models instead).
    pub fn list_models(&self) -> Vec<(&str, &ModelConfig)> {
        self.available_models()
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
                        layout: "NHWC".to_string(),
                        mean: None,
                        std: None,
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
                        multi_label: false,
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
                        mean: None,
                        std: None,
                    },
                    output: ModelOutputSpec {
                        num_classes: 1000,
                        labels: Vec::new(),
                        labels_file: None,
                        format: None,
                        multi_label: false,
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
                        mean: None,
                        std: None,
                    },
                    output: ModelOutputSpec {
                        num_classes: 1000,
                        labels: Vec::new(),
                        labels_file: None,
                        format: None,
                        multi_label: false,
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
                        mean: None,
                        std: None,
                    },
                    output: ModelOutputSpec {
                        num_classes: 1000,
                        labels: Vec::new(),
                        labels_file: None,
                        format: None,
                        multi_label: false,
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
        assert!(config.moderation_model.is_some());
        assert!(config.tagging_model.is_some());
        assert!(!config.builtin_models.is_empty());
    }

    #[test]
    fn test_config_roundtrip() {
        let config = ClassifierConfig::default();
        let toml = toml::to_string_pretty(&config).unwrap();
        let parsed: ClassifierConfig = toml::from_str(&toml).unwrap();
        assert_eq!(parsed.moderation_model, config.moderation_model);
        assert_eq!(parsed.tagging_model, config.tagging_model);
    }

    #[test]
    fn test_model_path_resolution() {
        let config = ClassifierConfig::default();
        let path = config.model_path("mobilenetv2").unwrap();
        assert!(path.ends_with("mobilenetv2-12.onnx"));
    }

    #[test]
    fn test_simple_toml_config() {
        let toml = r#"
            moderation_model = "adamcodd-vit-nsfw"
            tagging_model = "wd-vit-tagger-v3"
        "#;
        let parsed: ClassifierConfig = toml::from_str(toml).unwrap();
        assert_eq!(
            parsed.moderation_model,
            Some("adamcodd-vit-nsfw".to_string())
        );
        assert_eq!(parsed.tagging_model, Some("wd-vit-tagger-v3".to_string()));
    }

    #[test]
    fn test_builtin_models_available() {
        let config = ClassifierConfig::default();
        // Check moderation models
        assert!(config.get_model("gantman-nsfw").is_some());
        assert!(config.get_model("adamcodd-vit-nsfw").is_some());
        // Check tagging models
        assert!(config.get_model("mobilenetv2").is_some());
        assert!(config.get_model("wd-vit-tagger-v3").is_some());
    }
}
