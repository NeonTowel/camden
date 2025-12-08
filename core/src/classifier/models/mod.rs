//! Model registry for ONNX classification models.
//!
//! This module provides the built-in model registry with Phase 1 models:
//! - 3 moderation models (n4xtan, spiele, taufiqdp)
//! - 4 tagging models (wd-convnextv2, joytag, vit-face-expression, cl-tagger)

mod moderation;
mod tagging;

use super::config::ModelConfig;
use std::collections::HashMap;

/// Build the complete model registry with all Phase 1 models.
pub fn builtin_model_registry() -> HashMap<String, ModelConfig> {
    let mut models = HashMap::new();

    // Add all moderation models
    for (id, config) in moderation::moderation_models() {
        models.insert(id.to_string(), config);
    }

    // Add all tagging models
    for (id, config) in tagging::tagging_models() {
        models.insert(id.to_string(), config);
    }

    models
}

/// List all available moderation model IDs.
#[allow(dead_code)]
pub fn moderation_model_ids() -> Vec<&'static str> {
    moderation::moderation_models()
        .iter()
        .map(|(id, _)| *id)
        .collect()
}

/// List all available tagging model IDs.
#[allow(dead_code)]
pub fn tagging_model_ids() -> Vec<&'static str> {
    tagging::tagging_models()
        .iter()
        .map(|(id, _)| *id)
        .collect()
}
