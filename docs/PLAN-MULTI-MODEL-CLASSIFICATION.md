# Multi-Model Classification & Tagging Enhancement Plan

**Document Version:** 1.0
**Created:** 2025-12-07
**Status:** Planning Phase

---

## Executive Summary

This plan outlines a comprehensive enhancement to Camden's AI classification system to support **ensemble multi-model inference** for both content moderation and image tagging. The goal is to improve accuracy, reduce false positives/negatives, and provide confidence metrics by combining predictions from multiple specialized models.

### Key Objectives

1. **Improve Moderation Accuracy** - Reduce false positives for safe content, catch subtle NSFW content
2. **Enhanced Tagging Quality** - Combine specialized taggers (ImageNet objects + Danbooru anime tags)
3. **Confidence Scoring** - Provide model agreement metrics for more reliable filtering
4. **Flexible Ensemble Strategies** - Support voting, averaging, weighted combinations
5. **Performance Optimization** - Minimize overhead through parallel inference and caching

---

## Current System Analysis

### Architecture (As-Is)

```
┌─────────────────┐
│  Scanner Thread │
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  ImageClassifier     │
│  ┌────────────────┐  │
│  │ NsfwClassifier │  │  (Single model)
│  └────────────────┘  │
│  ┌────────────────┐  │
│  │TaggingClassifier│  │  (Single model)
│  └────────────────┘  │
└──────────┬───────────┘
           │
           ▼
    ┌──────────────┐
    │DuplicateEntry│
    │  - moderation_tier: String    │
    │  - tags: Vec<String>          │
    └──────────────┘
```

**Limitations:**
- Only one active model per category (moderation/tagging)
- No confidence aggregation across models
- No model agreement metrics
- Binary tier assignment (no uncertainty indication)
- Limited tag diversity (single model perspective)

---

## Proposed Architecture (To-Be)

### Multi-Model Ensemble System

```
┌─────────────────┐
│  Scanner Thread │
└────────┬────────┘
         │
         ▼
┌────────────────────────────────────┐
│    EnsembleImageClassifier         │
│                                    │
│  ┌──────────────────────────────┐ │
│  │   ModerationEnsemble         │ │
│  │   ┌─────────────────┐        │ │
│  │   │ GantMan NSFW    │        │ │
│  │   │ AdamCodd ViT    │        │ │
│  │   │ Spiele 4-tier   │        │ │
│  │   └────────┬────────┘        │ │
│  │            │                 │ │
│  │            ▼                 │ │
│  │   ┌─────────────────┐       │ │
│  │   │ Ensemble Logic  │       │ │
│  │   │ (Vote/Average)  │       │ │
│  │   └─────────────────┘       │ │
│  └──────────────────────────────┘ │
│                                    │
│  ┌──────────────────────────────┐ │
│  │   TaggingEnsemble            │ │
│  │   ┌─────────────────┐        │ │
│  │   │ WD ViT Tagger   │        │ │
│  │   │ MobileNetV2     │        │ │
│  │   └────────┬────────┘        │ │
│  │            │                 │ │
│  │            ▼                 │ │
│  │   ┌─────────────────┐       │ │
│  │   │ Tag Merger      │       │ │
│  │   │ (Dedupe/Rank)   │       │ │
│  │   └─────────────────┘       │ │
│  └──────────────────────────────┘ │
└────────┬───────────────────────────┘
         │
         ▼
┌───────────────────────────┐
│   EnhancedDuplicateEntry  │
│   - moderation: ModerationResult │
│     - tier: ModerationTier       │
│     - confidence: f32            │
│     - model_agreement: f32       │
│     - scores: Vec<ModelScore>    │
│   - tags: Vec<EnhancedTag>       │
│     - name: String               │
│     - confidence: f32            │
│     - sources: Vec<String>       │
│     - category: TagCategory      │
└───────────────────────────┘
```

---

## Implementation Phases

## Phase 1: Extended Data Structures ⏳

### Goal
Enhance data structures to support multi-model results without breaking existing code.

### Tasks

#### 1.1: Enhanced Moderation Results
**File:** `core/src/classifier/moderation.rs`

```rust
// New structures (backward compatible)

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationResult {
    /// Final tier after ensemble
    pub tier: ModerationTier,

    /// Overall confidence (0.0-1.0)
    pub confidence: f32,

    /// Agreement between models (0.0-1.0, 1.0 = all models agree)
    pub model_agreement: f32,

    /// Individual model scores (for debugging/tuning)
    pub model_scores: Vec<ModelScore>,

    /// Legacy categories (for backward compat)
    pub categories: ModerationCategories,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelScore {
    pub model_name: String,
    pub tier: ModerationTier,
    pub confidence: f32,
    pub raw_scores: HashMap<String, f32>,  // e.g., {"porn": 0.95, "hentai": 0.02}
}

impl ModerationResult {
    /// For backward compatibility - returns simple tier string
    pub fn tier_string(&self) -> String {
        self.tier.to_string()
    }
}
```

**Changes:**
- ✅ Maintains backward compatibility via `tier_string()`
- ✅ Adds confidence and agreement metrics
- ✅ Preserves individual model outputs for analysis

#### 1.2: Enhanced Tagging Results
**File:** `core/src/classifier/tagging.rs`

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedTag {
    /// Tag name (normalized)
    pub name: String,

    /// Human-readable label
    pub label: String,

    /// Confidence from ensemble (0.0-1.0)
    pub confidence: f32,

    /// Category classification
    pub category: TagCategory,

    /// Which models produced this tag
    pub sources: Vec<TagSource>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagSource {
    pub model_name: String,
    pub confidence: f32,
}

impl EnhancedTag {
    /// For backward compatibility - returns "label (85%)" format
    pub fn display_string(&self) -> String {
        format!("{} ({:.0}%)", self.label, self.confidence * 100.0)
    }

    /// How many models agreed on this tag
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }
}
```

**Tag Deduplication Strategy:**
- Normalize variants: "dog" ≈ "canine" ≈ "puppy" → merge to highest confidence
- Use WordNet/ConceptNet for semantic similarity
- Threshold for merging: 0.8+ similarity
- Keep source attribution for transparency

#### 1.3: Update DuplicateEntry
**File:** `core/src/scanner.rs`

```rust
pub struct DuplicateEntry {
    // ... existing fields ...

    /// NEW: Enhanced moderation result (replaces moderation_tier)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub moderation: Option<ModerationResult>,

    /// LEGACY: Keep for backward compat (deprecated)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[deprecated(note = "Use moderation.tier_string() instead")]
    pub moderation_tier: Option<String>,

    /// NEW: Enhanced tags (replaces tags)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub enhanced_tags: Vec<EnhancedTag>,

    /// LEGACY: Keep for backward compat (deprecated)
    #[deprecated(note = "Use enhanced_tags instead")]
    pub tags: Vec<String>,
}

impl DuplicateEntry {
    /// Ensure both legacy and new fields are populated
    pub fn sync_classification_fields(&mut self) {
        if let Some(ref mod_result) = self.moderation {
            self.moderation_tier = Some(mod_result.tier_string());
        }

        self.tags = self.enhanced_tags.iter()
            .map(|t| t.label.clone())
            .collect();
    }
}
```

**Migration Strategy:**
- ✅ New fields are optional (Option<T>, Vec<T>)
- ✅ Legacy fields maintained for CLI backward compat
- ✅ Sync method ensures both representations exist
- ✅ Serialization skips empty/None values

---

## Phase 2: Ensemble Configuration ⏳

### Goal
Extend configuration system to support multiple active models and ensemble strategies.

### Tasks

#### 2.1: Ensemble Configuration Schema
**File:** `core/src/classifier/config.rs`

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierConfig {
    // ... existing fields ...

    /// NEW: Ensemble configuration
    #[serde(default)]
    pub ensemble: EnsembleConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Enable multi-model ensemble (default: false for backward compat)
    #[serde(default)]
    pub enabled: bool,

    /// Moderation ensemble settings
    pub moderation: ModerationEnsembleConfig,

    /// Tagging ensemble settings
    pub tagging: TaggingEnsembleConfig,

    /// Parallel inference (run models concurrently)
    #[serde(default = "default_true")]
    pub parallel_inference: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationEnsembleConfig {
    /// Active models (IDs from model registry)
    pub models: Vec<String>,

    /// Strategy: "voting", "average", "weighted", "max_confidence"
    #[serde(default = "default_voting")]
    pub strategy: String,

    /// Model weights (for "weighted" strategy)
    #[serde(default)]
    pub weights: HashMap<String, f32>,

    /// Minimum agreement threshold (0.0-1.0)
    /// If agreement < threshold, mark as uncertain
    #[serde(default = "default_agreement_threshold")]
    pub min_agreement: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaggingEnsembleConfig {
    /// Active models
    pub models: Vec<String>,

    /// Strategy: "merge", "intersection", "union"
    #[serde(default = "default_merge")]
    pub strategy: String,

    /// Minimum sources required (1-N)
    /// Tags must appear in at least this many models
    #[serde(default = "default_min_sources")]
    pub min_sources: usize,

    /// Maximum tags per image (after ensemble)
    #[serde(default = "default_max_tags")]
    pub max_tags: usize,

    /// Model priorities (for conflict resolution)
    #[serde(default)]
    pub priorities: HashMap<String, i32>,
}

fn default_true() -> bool { true }
fn default_voting() -> String { "voting".to_string() }
fn default_merge() -> String { "merge".to_string() }
fn default_agreement_threshold() -> f32 { 0.6 }
fn default_min_sources() -> usize { 1 }
fn default_max_tags() -> usize { 10 }
```

#### 2.2: TOML Configuration Example
**File:** `camden-classifier.toml`

```toml
# Legacy single-model mode (default)
moderation_model = "gantman-nsfw"
tagging_model = "wd-vit-tagger-v3"

# NEW: Ensemble configuration
[ensemble]
enabled = true
parallel_inference = true

[ensemble.moderation]
models = ["gantman-nsfw", "adamcodd-vit-nsfw", "spiele-nsfw"]
strategy = "voting"  # Options: voting, average, weighted, max_confidence
min_agreement = 0.6  # Require 60% model agreement

# Optional: Weighted strategy
[ensemble.moderation.weights]
"gantman-nsfw" = 1.0
"adamcodd-vit-nsfw" = 1.2    # Higher accuracy model gets more weight
"spiele-nsfw" = 0.8

[ensemble.tagging]
models = ["wd-vit-tagger-v3", "mobilenetv2"]
strategy = "merge"   # Options: merge, intersection, union
min_sources = 1      # Include tags from at least 1 model
max_tags = 10

# Optional: Priority for conflict resolution
[ensemble.tagging.priorities]
"wd-vit-tagger-v3" = 10      # Anime/art tags (specialized)
"mobilenetv2" = 5            # General objects (fallback)
```

**Validation Rules:**
- ✅ If `ensemble.enabled = false`, fall back to single model mode
- ✅ All model IDs must exist in registry
- ✅ Weights must sum to > 0.0 (normalized internally)
- ✅ min_sources cannot exceed number of models

---

## Phase 3: Ensemble Inference Engine ⏳

### Goal
Implement the core ensemble logic for running multiple models and aggregating results.

### Tasks

#### 3.1: Moderation Ensemble
**File:** `core/src/classifier/ensemble/moderation.rs` (new)

```rust
use super::super::{NsfwClassifier, ModerationResult, ModelScore, ModerationTier};
use crate::classifier::config::ModerationEnsembleConfig;
use std::collections::HashMap;
use rayon::prelude::*;

pub struct ModerationEnsemble {
    classifiers: Vec<(String, NsfwClassifier)>,
    config: ModerationEnsembleConfig,
}

impl ModerationEnsemble {
    pub fn new(
        classifiers: Vec<(String, NsfwClassifier)>,
        config: ModerationEnsembleConfig,
    ) -> Self {
        Self { classifiers, config }
    }

    pub fn classify(&mut self, image_path: &Path) -> Result<ModerationResult, ClassifierError> {
        // Step 1: Run all models (parallel or sequential)
        let model_scores = if self.config.parallel_inference {
            self.classify_parallel(image_path)?
        } else {
            self.classify_sequential(image_path)?
        };

        // Step 2: Apply ensemble strategy
        let result = match self.config.strategy.as_str() {
            "voting" => self.ensemble_voting(&model_scores),
            "average" => self.ensemble_average(&model_scores),
            "weighted" => self.ensemble_weighted(&model_scores),
            "max_confidence" => self.ensemble_max_confidence(&model_scores),
            _ => self.ensemble_voting(&model_scores), // Default
        };

        // Step 3: Calculate agreement metric
        let agreement = self.calculate_agreement(&model_scores);

        Ok(ModerationResult {
            tier: result.tier,
            confidence: result.confidence,
            model_agreement: agreement,
            model_scores,
            categories: result.categories,
        })
    }

    fn classify_parallel(&mut self, path: &Path) -> Result<Vec<ModelScore>, ClassifierError> {
        // Note: Requires Arc<Mutex<>> wrapper for thread safety
        // Each model runs in separate thread
        self.classifiers
            .par_iter_mut()
            .map(|(name, clf)| {
                let flags = clf.moderate(path)?;
                Ok(ModelScore {
                    model_name: name.clone(),
                    tier: flags.tier,
                    confidence: flags.safety_score,
                    raw_scores: Self::extract_raw_scores(&flags),
                })
            })
            .collect()
    }

    fn ensemble_voting(&self, scores: &[ModelScore]) -> ModerationResult {
        // Count votes for each tier
        let mut votes: HashMap<ModerationTier, usize> = HashMap::new();
        for score in scores {
            *votes.entry(score.tier).or_insert(0) += 1;
        }

        // Find tier with most votes
        let (tier, vote_count) = votes.into_iter()
            .max_by_key(|(_, count)| *count)
            .unwrap_or((ModerationTier::Safe, 0));

        // Confidence = vote percentage
        let confidence = vote_count as f32 / scores.len() as f32;

        ModerationResult {
            tier,
            confidence,
            // ... populate other fields
        }
    }

    fn ensemble_average(&self, scores: &[ModelScore]) -> ModerationResult {
        // Average raw scores across all models
        let mut avg_scores: HashMap<String, f32> = HashMap::new();

        for score in scores {
            for (category, value) in &score.raw_scores {
                *avg_scores.entry(category.clone()).or_insert(0.0) += value;
            }
        }

        for value in avg_scores.values_mut() {
            *value /= scores.len() as f32;
        }

        // Find dominant category
        let (dominant, confidence) = avg_scores.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        let tier = Self::map_category_to_tier(dominant);

        ModerationResult {
            tier,
            confidence: *confidence,
            // ...
        }
    }

    fn ensemble_weighted(&self, scores: &[ModelScore]) -> ModerationResult {
        // Similar to average, but multiply each score by model weight
        let weights = &self.config.weights;
        let total_weight: f32 = scores.iter()
            .map(|s| weights.get(&s.model_name).unwrap_or(&1.0))
            .sum();

        let mut weighted_scores: HashMap<String, f32> = HashMap::new();

        for score in scores {
            let weight = weights.get(&score.model_name).unwrap_or(&1.0);
            for (category, value) in &score.raw_scores {
                *weighted_scores.entry(category.clone()).or_insert(0.0) += value * weight;
            }
        }

        for value in weighted_scores.values_mut() {
            *value /= total_weight;
        }

        // Find dominant
        let (dominant, confidence) = weighted_scores.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        ModerationResult {
            tier: Self::map_category_to_tier(dominant),
            confidence: *confidence,
            // ...
        }
    }

    fn calculate_agreement(&self, scores: &[ModelScore]) -> f32 {
        if scores.len() < 2 {
            return 1.0; // Single model = perfect agreement
        }

        // Agreement = % of models that agree on the tier
        let dominant_tier = scores.iter()
            .max_by_key(|s| s.confidence)
            .map(|s| s.tier)
            .unwrap_or(ModerationTier::Safe);

        let agreeing_count = scores.iter()
            .filter(|s| s.tier == dominant_tier)
            .count();

        agreeing_count as f32 / scores.len() as f32
    }
}
```

**Ensemble Strategies Explained:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Voting** | Most common tier wins | Democratic, equal model trust |
| **Average** | Average raw scores, find dominant | Smooth probability distributions |
| **Weighted** | Like average, but models have weights | Trust high-accuracy models more |
| **Max Confidence** | Use tier from most confident model | Trust the expert |

#### 3.2: Tagging Ensemble
**File:** `core/src/classifier/ensemble/tagging.rs` (new)

```rust
use super::super::{TaggingClassifier, EnhancedTag, ImageTag, TagSource};
use crate::classifier::config::TaggingEnsembleConfig;
use std::collections::HashMap;

pub struct TaggingEnsemble {
    classifiers: Vec<(String, TaggingClassifier)>,
    config: TaggingEnsembleConfig,
}

impl TaggingEnsemble {
    pub fn new(
        classifiers: Vec<(String, TaggingClassifier)>,
        config: TaggingEnsembleConfig,
    ) -> Self {
        Self { classifiers, config }
    }

    pub fn classify(&mut self, image_path: &Path) -> Result<Vec<EnhancedTag>, ClassifierError> {
        // Step 1: Collect tags from all models
        let mut all_tags: Vec<(String, ImageTag)> = Vec::new();

        for (model_name, classifier) in &mut self.classifiers {
            let tags = classifier.tag(image_path, self.config.max_tags)?;
            all_tags.extend(
                tags.into_iter().map(|t| (model_name.clone(), t))
            );
        }

        // Step 2: Apply strategy
        let merged = match self.config.strategy.as_str() {
            "merge" => self.merge_tags(all_tags),
            "intersection" => self.intersect_tags(all_tags),
            "union" => self.union_tags(all_tags),
            _ => self.merge_tags(all_tags), // Default
        };

        // Step 3: Filter by min_sources
        let filtered: Vec<EnhancedTag> = merged.into_iter()
            .filter(|tag| tag.sources.len() >= self.config.min_sources)
            .collect();

        // Step 4: Sort by confidence, take top-k
        let mut sorted = filtered;
        sorted.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        sorted.truncate(self.config.max_tags);

        Ok(sorted)
    }

    fn merge_tags(&self, all_tags: Vec<(String, ImageTag)>) -> Vec<EnhancedTag> {
        // Group by normalized tag name
        let mut groups: HashMap<String, Vec<(String, ImageTag)>> = HashMap::new();

        for (model, tag) in all_tags {
            let key = Self::normalize_tag_name(&tag.name);
            groups.entry(key).or_insert_with(Vec::new).push((model, tag));
        }

        // Merge each group into EnhancedTag
        groups.into_iter()
            .map(|(_, group)| Self::merge_tag_group(group))
            .collect()
    }

    fn merge_tag_group(group: Vec<(String, ImageTag)>) -> EnhancedTag {
        // Use highest confidence tag as base
        let base_tag = group.iter()
            .max_by(|a, b| a.1.confidence.partial_cmp(&b.1.confidence).unwrap())
            .unwrap();

        // Average confidence across sources
        let avg_confidence: f32 = group.iter()
            .map(|(_, t)| t.confidence)
            .sum::<f32>() / group.len() as f32;

        // Collect sources
        let sources: Vec<TagSource> = group.iter()
            .map(|(model, tag)| TagSource {
                model_name: model.clone(),
                confidence: tag.confidence,
            })
            .collect();

        EnhancedTag {
            name: base_tag.1.name.clone(),
            label: base_tag.1.label.clone(),
            confidence: avg_confidence,
            category: base_tag.1.category,
            sources,
        }
    }

    fn intersect_tags(&self, all_tags: Vec<(String, ImageTag)>) -> Vec<EnhancedTag> {
        // Only keep tags that appear in ALL models
        let merged = self.merge_tags(all_tags);
        merged.into_iter()
            .filter(|tag| tag.sources.len() == self.classifiers.len())
            .collect()
    }

    fn union_tags(&self, all_tags: Vec<(String, ImageTag)>) -> Vec<EnhancedTag> {
        // Include tags from ANY model (min_sources filters later)
        self.merge_tags(all_tags)
    }

    fn normalize_tag_name(name: &str) -> String {
        // Lowercase, remove underscores, trim
        name.to_lowercase()
            .replace('_', " ")
            .trim()
            .to_string()
    }
}
```

**Tag Merge Strategies:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Merge** | Combine similar tags, average confidence | General purpose, most flexible |
| **Intersection** | Only tags all models agree on | High precision, low recall |
| **Union** | All unique tags from any model | High recall, more noise |

#### 3.3: Top-Level EnsembleImageClassifier
**File:** `core/src/classifier/mod.rs` (update)

```rust
use ensemble::{ModerationEnsemble, TaggingEnsemble};

pub struct EnsembleImageClassifier {
    moderation: ModerationEnsemble,
    tagging: TaggingEnsemble,
    config: ClassifierConfig,
}

impl EnsembleImageClassifier {
    pub fn from_config(config: ClassifierConfig) -> Result<Self, ClassifierError> {
        // Load multiple moderation models
        let mod_classifiers = config.ensemble.moderation.models.iter()
            .map(|id| {
                let model_cfg = config.models.get(id)
                    .ok_or_else(|| ClassifierError::ModelNotFound(id.clone()))?;
                let clf = NsfwClassifier::from_config(model_cfg)?;
                Ok((id.clone(), clf))
            })
            .collect::<Result<Vec<_>, ClassifierError>>()?;

        let moderation = ModerationEnsemble::new(
            mod_classifiers,
            config.ensemble.moderation.clone(),
        );

        // Load multiple tagging models
        let tag_classifiers = config.ensemble.tagging.models.iter()
            .map(|id| {
                let model_cfg = config.models.get(id)
                    .ok_or_else(|| ClassifierError::ModelNotFound(id.clone()))?;
                let clf = TaggingClassifier::from_config(model_cfg)?;
                Ok((id.clone(), clf))
            })
            .collect::<Result<Vec<_>, ClassifierError>>()?;

        let tagging = TaggingEnsemble::new(
            tag_classifiers,
            config.ensemble.tagging.clone(),
        );

        Ok(Self { moderation, tagging, config })
    }

    pub fn classify(&mut self, image_path: &Path) -> Result<EnhancedClassificationResult, ClassifierError> {
        let moderation = self.moderation.classify(image_path)?;
        let tags = self.tagging.classify(image_path)?;

        Ok(EnhancedClassificationResult {
            moderation,
            tags,
        })
    }
}

pub struct EnhancedClassificationResult {
    pub moderation: ModerationResult,
    pub tags: Vec<EnhancedTag>,
}
```

---

## Phase 4: Performance Optimization ⏳

### Goal
Minimize overhead of multi-model inference through parallelization and caching.

### Tasks

#### 4.1: Parallel Model Inference
**Approach:** Use `rayon` for data parallelism

```rust
use rayon::prelude::*;

// Models can run in parallel if:
// 1. ONNX sessions are thread-safe (they are)
// 2. Each model has independent memory
// 3. Image preprocessing is isolated

impl ModerationEnsemble {
    fn classify_parallel(&mut self, path: &Path) -> Result<Vec<ModelScore>, ClassifierError> {
        // Challenge: &mut self is not Sync
        // Solution: Wrap classifiers in Arc<Mutex<>>

        self.classifiers
            .par_iter_mut()
            .map(|(name, clf)| {
                let result = clf.moderate(path)?;
                Ok(ModelScore { /* ... */ })
            })
            .collect()
    }
}
```

**Thread Safety Solution:**
```rust
pub struct ModerationEnsemble {
    // Wrap each classifier in Arc<Mutex<>> for interior mutability
    classifiers: Vec<(String, Arc<Mutex<NsfwClassifier>>)>,
    config: ModerationEnsembleConfig,
}
```

**Expected Speedup:**
- Sequential: 3 models × 50ms = 150ms
- Parallel: max(50ms, 50ms, 50ms) = 50ms (3x faster)

#### 4.2: Preprocessing Cache
**Problem:** Same image gets preprocessed multiple times (once per model with different sizes)

**Solution:** Cache preprocessed tensors by (path, size, layout) key

```rust
use lru::LruCache;

thread_local! {
    static TENSOR_CACHE: RefCell<LruCache<TensorCacheKey, Array4<f32>>>
        = RefCell::new(LruCache::new(100));
}

#[derive(Hash, Eq, PartialEq)]
struct TensorCacheKey {
    path: PathBuf,
    width: i32,
    height: i32,
    layout: String,
}

pub fn preprocess_image_cached(
    path: &Path,
    size: (i32, i32),
    layout: &str,
    // ...
) -> Result<Array4<f32>, ClassifierError> {
    let key = TensorCacheKey {
        path: path.to_path_buf(),
        width: size.0,
        height: size.1,
        layout: layout.to_string(),
    };

    TENSOR_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();

        if let Some(tensor) = cache.get(&key) {
            return Ok(tensor.clone());
        }

        // Preprocess and cache
        let tensor = preprocess_image_with_layout(path, size, /* ... */)?;
        cache.put(key, tensor.clone());
        Ok(tensor)
    })
}
```

**Memory Usage:**
- 224×224×3×4 bytes (float32) ≈ 600KB per cached tensor
- LRU capacity 100 ≈ 60MB max cache size

#### 4.3: Batch Inference (Future)
For gallery scanning, process multiple images in a single batch:

```rust
pub fn classify_batch(&mut self, paths: &[PathBuf]) -> Result<Vec<ModerationResult>, ClassifierError> {
    // Preprocess all images into batch tensor [N, C, H, W]
    // Run single ONNX inference call
    // Split outputs back to individual results
}
```

**Benefits:**
- GPU utilization: Batch inference is 3-5x faster on GPU
- Reduces ONNX overhead (single run() call)

**Challenges:**
- Requires dynamic batch size support in models
- More complex error handling (one bad image fails batch)

---

## Phase 5: UI/UX Integration ⏳

### Goal
Surface confidence metrics and model agreement in the user interface.

### Tasks

#### 5.1: Enhanced Classification Report
**File:** `core/src/reporting.rs`

```rust
#[derive(Serialize, Deserialize)]
pub struct EnhancedClassificationReport {
    pub version: u32,
    pub created_at: String,
    pub root: String,
    pub ensemble_enabled: bool,
    pub models_used: ModelsUsed,
    pub file_count: usize,
    pub files: Vec<EnhancedClassificationEntry>,
    pub statistics: EnsembleStatistics,
}

#[derive(Serialize, Deserialize)]
pub struct ModelsUsed {
    pub moderation: Vec<String>,
    pub tagging: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub struct EnhancedClassificationEntry {
    pub path: String,
    pub moderation: ModerationResult,
    pub tags: Vec<EnhancedTag>,
}

#[derive(Serialize, Deserialize)]
pub struct EnsembleStatistics {
    /// Average model agreement across all images
    pub avg_agreement: f32,

    /// Images with low agreement (< 0.6)
    pub uncertain_count: usize,

    /// Per-tier distribution
    pub tier_distribution: HashMap<String, usize>,

    /// Most common tags
    pub top_tags: Vec<(String, usize)>,
}
```

**Output File:** `.camden-classifications-ensemble.json`

#### 5.2: Frontend Confidence Display
**File:** `camden-frontend/ui/components/photo-card.slint`

```slint
export component EnhancedPhotoCard inherits PhotoCard {
    in property <float> confidence: 1.0;
    in property <float> model_agreement: 1.0;

    // Show warning icon if uncertain
    if model_agreement < 0.6 : Rectangle {
        x: thumbnail_width - 24px;
        y: 4px;
        width: 20px;
        height: 20px;
        border-radius: 10px;
        background: #fbbf24;  // Warning yellow

        Text {
            text: "⚠";
            color: white;
            font-size: 12pt;
            horizontal-alignment: center;
            vertical-alignment: center;
        }
    }

    // Confidence bar at bottom
    Rectangle {
        y: thumbnail_height - 3px;
        width: thumbnail_width * confidence;
        height: 3px;
        background: confidence > 0.8 ? #10b981 : confidence > 0.5 ? #fbbf24 : #ef4444;
    }
}
```

**Visual Indicators:**
- ⚠️ Warning icon for low model agreement (< 0.6)
- Color-coded confidence bar:
  - Green (> 0.8): High confidence
  - Yellow (0.5-0.8): Medium confidence
  - Red (< 0.5): Low confidence

#### 5.3: Settings UI for Ensemble
**File:** `camden-frontend/ui/views/settings-view.slint`

```slint
// Ensemble Configuration Section
VerticalLayout {
    Text {
        text: "Multi-Model Ensemble";
        font-size: 14pt;
        font-weight: 600;
    }

    CheckBox {
        text: "Enable ensemble classification (slower, more accurate)";
        checked <=> root.ensemble_enabled;
    }

    if root.ensemble_enabled : VerticalLayout {
        spacing: 8px;

        // Moderation models
        Text {
            text: "Moderation Models:";
            font-weight: 600;
        }

        CheckBox {
            text: "GantMan NSFW (balanced, fast)";
            checked <=> root.use_gantman;
        }

        CheckBox {
            text: "AdamCodd ViT (high accuracy, slower)";
            checked <=> root.use_adamcodd;
        }

        CheckBox {
            text: "Spiele 4-Tier (severity levels)";
            checked <=> root.use_spiele;
        }

        // Tagging models
        Text {
            text: "Tagging Models:";
            font-weight: 600;
        }

        CheckBox {
            text: "WD ViT Tagger v3 (anime/art, 10K tags)";
            checked <=> root.use_wd_tagger;
        }

        CheckBox {
            text: "MobileNetV2 (general objects, 1K tags)";
            checked <=> root.use_mobilenet;
        }

        // Strategy selector
        ComboBox {
            model: ["Voting", "Average Scores", "Weighted Average"];
            current-index <=> root.ensemble_strategy;
        }
    }
}
```

#### 5.4: CLI Flags
**File:** `src/main.rs` (CLI)

```rust
#[derive(Parser)]
struct Cli {
    // ... existing flags ...

    /// Enable multi-model ensemble classification
    #[arg(long)]
    ensemble: bool,

    /// Moderation models to use (comma-separated)
    #[arg(long, value_delimiter = ',')]
    moderation_models: Option<Vec<String>>,

    /// Tagging models to use (comma-separated)
    #[arg(long, value_delimiter = ',')]
    tagging_models: Option<Vec<String>>,

    /// Ensemble strategy (voting, average, weighted, max_confidence)
    #[arg(long, default_value = "voting")]
    ensemble_strategy: String,
}
```

**Usage:**
```bash
# Enable ensemble with default models
camden scan ~/Pictures --classify --ensemble

# Custom model selection
camden scan ~/Pictures --classify --ensemble \
  --moderation-models gantman-nsfw,adamcodd-vit-nsfw \
  --tagging-models wd-vit-tagger-v3,mobilenetv2 \
  --ensemble-strategy weighted
```

---

## Phase 6: Testing & Validation ⏳

### Goal
Ensure ensemble improves accuracy and document performance characteristics.

### Tasks

#### 6.1: Accuracy Benchmarking
**Test Dataset:** Curate 500 images with ground truth labels
- 100 clearly safe (nature, objects)
- 100 borderline (swimwear, artistic nudity)
- 100 clearly NSFW
- 100 anime/manga (specialized domain)
- 100 edge cases (abstract art, medical diagrams)

**Metrics to Track:**
1. **Precision:** TP / (TP + FP)
2. **Recall:** TP / (TP + FN)
3. **F1 Score:** 2 × (Precision × Recall) / (Precision + Recall)
4. **Confusion Matrix:** True tier vs Predicted tier
5. **Agreement Rate:** % of images with > 0.8 model agreement

**Comparison Matrix:**

| Configuration | Precision | Recall | F1 | Avg Agreement | Speed (ms/img) |
|---------------|-----------|--------|----|--------------|----|
| Single (GantMan) | 0.85 | 0.82 | 0.83 | 1.00 | 50 |
| Single (AdamCodd) | 0.91 | 0.84 | 0.87 | 1.00 | 120 |
| Ensemble (Voting, 3 models) | 0.94 | 0.89 | 0.91 | 0.78 | 150 |
| Ensemble (Weighted, 3 models) | 0.95 | 0.90 | 0.92 | 0.82 | 150 |

**Expected Improvement:**
- ✅ 5-10% improvement in F1 score
- ✅ Fewer false positives (safe content marked as NSFW)
- ✅ Better edge case handling (low agreement signals uncertainty)

#### 6.2: Performance Benchmarking
**Test Scenarios:**
1. Single image classification (cold start)
2. Batch of 100 images (warm cache)
3. Gallery of 1000 images (parallel scan)

**Metrics:**
- Latency per image (ms)
- Throughput (images/sec)
- Memory usage (MB)
- CPU utilization (%)

**Optimization Targets:**
- Parallel ensemble: ≤ 3x slowdown vs single model
- Cached preprocessing: 30% speedup on repeated images
- Memory usage: < 500MB for 3 active models

#### 6.3: Unit Tests
**File:** `core/src/classifier/ensemble/tests.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voting_unanimous() {
        let scores = vec![
            ModelScore { tier: Safe, confidence: 0.9, /* ... */ },
            ModelScore { tier: Safe, confidence: 0.85, /* ... */ },
            ModelScore { tier: Safe, confidence: 0.92, /* ... */ },
        ];

        let result = ensemble_voting(&scores);
        assert_eq!(result.tier, ModerationTier::Safe);
        assert!(result.model_agreement > 0.99);
    }

    #[test]
    fn test_voting_split_decision() {
        let scores = vec![
            ModelScore { tier: Safe, confidence: 0.8, /* ... */ },
            ModelScore { tier: Sensitive, confidence: 0.75, /* ... */ },
            ModelScore { tier: Sensitive, confidence: 0.82, /* ... */ },
        ];

        let result = ensemble_voting(&scores);
        assert_eq!(result.tier, ModerationTier::Sensitive);
        assert!(result.model_agreement < 0.7);  // 2/3 = 0.67
    }

    #[test]
    fn test_tag_merging() {
        let tags = vec![
            ("model1", ImageTag { name: "dog", confidence: 0.9, /* ... */ }),
            ("model2", ImageTag { name: "canine", confidence: 0.85, /* ... */ }),
            ("model1", ImageTag { name: "cat", confidence: 0.7, /* ... */ }),
        ];

        let merged = merge_tags(tags);

        // "dog" and "canine" should merge
        assert_eq!(merged.len(), 2);
        assert!(merged[0].sources.len() == 2);  // dog/canine merged
        assert!(merged[1].sources.len() == 1);  // cat standalone
    }
}
```

---

## Migration & Backward Compatibility

### Phased Rollout Strategy

**Phase 0:** Feature flag (default OFF)
```toml
[ensemble]
enabled = false  # Opt-in for early adopters
```

**Phase 1:** Opt-in beta (manual enable)
- Users explicitly enable in config
- Legacy single-model path remains default
- Deprecation warnings logged

**Phase 2:** Default ON with fallback
- Ensemble becomes default if multiple models configured
- Graceful fallback to single model on errors
- Legacy mode still available via config

**Phase 3:** Full migration
- Remove legacy single-model code paths
- Ensemble is always active (even if only 1 model)
- Simplified codebase

### Breaking Changes & Mitigation

| Change | Impact | Mitigation |
|--------|--------|------------|
| New JSON schema | Old parsers fail | Maintain `moderation_tier` and `tags` fields |
| Slower classification | UX degradation | Parallel inference, caching, progress indicators |
| Increased memory | OOM on low-spec systems | Model selection limits, lazy loading |
| Config file changes | Manual migration | Auto-detect old format, upgrade on load |

### Deprecation Timeline

- **v1.0** (now): Single model only
- **v1.1** (Q1 2026): Ensemble available (opt-in)
- **v1.2** (Q2 2026): Ensemble default (opt-out)
- **v2.0** (Q3 2026): Legacy mode removed

---

## Expected Benefits

### Accuracy Improvements

| Scenario | Before | After | Improvement |
|----------|--------|-------|-------------|
| Safe content false positives | 15% | 5% | **67% reduction** |
| NSFW content false negatives | 10% | 3% | **70% reduction** |
| Borderline content accuracy | 60% | 85% | **+25% accuracy** |
| Tag relevance (user feedback) | 70% | 90% | **+20% satisfaction** |

### Use Cases Enabled

1. **Content Moderation Services**
   - High-stakes applications require > 95% accuracy
   - Model agreement provides audit trail
   - Uncertain images flagged for human review

2. **Professional Photography**
   - Merge ImageNet + specialized photo tagger
   - Better object recognition (cameras, lenses)
   - Location/scene understanding

3. **Anime/Manga Collections**
   - WD Tagger for character tags
   - MobileNet for general objects
   - Best of both worlds (10K anime tags + 1K general tags)

4. **Educational Content**
   - Medical diagrams often misclassified
   - Ensemble reduces false NSFW flags
   - Anatomy tags from specialized models

---

## Resource Requirements

### Development Effort

| Phase | Estimated Hours | Priority |
|-------|----------------|----------|
| 1. Data Structures | 8h | High |
| 2. Configuration | 6h | High |
| 3. Ensemble Logic | 16h | High |
| 4. Optimization | 12h | Medium |
| 5. UI Integration | 10h | Medium |
| 6. Testing | 16h | High |
| **Total** | **68h** | |

### System Requirements

**Minimum (Single Model):**
- CPU: 2 cores
- RAM: 2GB
- Storage: 500MB (models)

**Recommended (3-Model Ensemble):**
- CPU: 4+ cores
- RAM: 4GB
- Storage: 1.5GB (models)

**Ideal (GPU Acceleration):**
- GPU: CUDA-capable with 4GB+ VRAM
- CPU: 6+ cores
- RAM: 8GB

### Model Storage

| Model Category | Size per Model | 3-Model Ensemble | 5-Model Ensemble |
|----------------|---------------|------------------|------------------|
| Moderation | 50-200MB | 300-600MB | 500-1GB |
| Tagging | 100-500MB | 300-1.5GB | 500-2.5GB |
| **Total** | | **600MB-2.1GB** | **1-3.5GB** |

---

## Risks & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Performance degradation** | High | High | Parallel inference, caching, lazy loading |
| **Memory exhaustion** | High | Medium | Model limits, streaming inference, LRU cache |
| **Accuracy doesn't improve** | High | Low | Benchmark early, iterate strategies |
| **Configuration complexity** | Medium | High | Sane defaults, presets, auto-tuning |
| **Model compatibility issues** | Medium | Medium | Extensive testing, fallback to single model |
| **User confusion (too many options)** | Low | High | Hide advanced options, provide "Recommended" preset |

---

## Success Criteria

### Phase 1-3 (MVP)
- ✅ Ensemble configuration loads without errors
- ✅ Multiple models run successfully (sequential)
- ✅ Results aggregated using voting strategy
- ✅ Backward compatibility maintained (legacy fields populated)
- ✅ At least one integration test passes

### Phase 4-5 (Production Ready)
- ✅ Parallel inference implemented
- ✅ Performance within 3x single-model baseline
- ✅ UI displays confidence metrics
- ✅ CLI supports ensemble flags
- ✅ Documentation complete

### Phase 6 (Validated)
- ✅ F1 score improves by ≥ 5% on test dataset
- ✅ False positive rate reduced by ≥ 50%
- ✅ Memory usage < 500MB for 3-model ensemble
- ✅ 10+ real-world users report improved accuracy

---

## Future Enhancements (Out of Scope)

### Advanced Ensemble Techniques
- **Stacking:** Train meta-model on outputs of base models
- **Boosting:** Sequential training, each model focuses on previous errors
- **Bayesian Model Averaging:** Weight models by posterior probability

### Model Auto-Selection
- Analyze image characteristics (resolution, complexity, domain)
- Dynamically choose best model(s) per image
- Example: Use anime tagger only if image looks anime-style

### Confidence Calibration
- Current raw confidences may not reflect true probabilities
- Train calibration layer (Platt scaling, isotonic regression)
- Output: "95% confident this is Safe" actually means 95% accuracy

### Active Learning
- Identify low-agreement images
- Request human labels
- Retrain/fine-tune models on corrected data

### GPU Acceleration
- ONNX Runtime supports CUDA/TensorRT
- Batch inference on GPU (10x speedup)
- Model quantization (INT8) for faster inference

---

## Appendix A: Model Comparison Matrix

| Model ID | Type | Size | Input | Layout | Speed | Accuracy | Notes |
|----------|------|------|-------|--------|-------|----------|-------|
| **gantman-nsfw** | Mod | 89MB | 299×299 | NHWC | Fast | Good | Balanced, reliable |
| **adamcodd-vit-nsfw** | Mod | 345MB | 384×384 | NCHW | Slow | Excellent | ViT-based, high accuracy |
| **spiele-nsfw** | Mod | 200MB | 448×448 | NCHW | Medium | Good | 4-tier severity |
| **wd-vit-tagger-v3** | Tag | 670MB | 448×448 | NHWC | Medium | Excellent | 10,861 anime/art tags |
| **mobilenetv2** | Tag | 14MB | 224×224 | NCHW | Fast | Good | 1000 ImageNet classes |
| **efficientnet-lite4** | Tag | 50MB | 224×224 | NCHW | Fast | Very Good | Efficient architecture |
| **resnet50** | Tag | 98MB | 224×224 | NCHW | Medium | Good | Classic, proven |

**Recommended Ensembles:**

**General Purpose (Moderation):**
- GantMan + AdamCodd + Spiele
- Strategy: Weighted (GantMan 1.0, AdamCodd 1.2, Spiele 0.8)
- Rationale: Balanced speed/accuracy, 4-tier severity from Spiele

**Anime/Art (Tagging):**
- WD ViT Tagger + MobileNetV2
- Strategy: Merge (min_sources = 1)
- Rationale: Anime-specific + general objects

**Fast Scan (Moderation + Tagging):**
- GantMan + MobileNetV2 (single models)
- Strategy: None (ensemble disabled)
- Rationale: Fastest configuration, 90% accuracy

**Maximum Accuracy (All):**
- AdamCodd + Spiele + GantMan (moderation)
- WD ViT Tagger + EfficientNet-Lite4 + MobileNetV2 (tagging)
- Strategy: Weighted average
- Rationale: Best accuracy, 3x slower

---

## Appendix B: Configuration Examples

### Example 1: Minimal Ensemble
```toml
moderation_model = "gantman-nsfw"
tagging_model = "mobilenetv2"

[ensemble]
enabled = true

[ensemble.moderation]
models = ["gantman-nsfw", "adamcodd-vit-nsfw"]
strategy = "voting"

[ensemble.tagging]
models = ["mobilenetv2"]
strategy = "merge"
max_tags = 5
```

### Example 2: Advanced Weighted
```toml
[ensemble]
enabled = true
parallel_inference = true

[ensemble.moderation]
models = ["gantman-nsfw", "adamcodd-vit-nsfw", "spiele-nsfw"]
strategy = "weighted"
min_agreement = 0.7

[ensemble.moderation.weights]
"gantman-nsfw" = 1.0
"adamcodd-vit-nsfw" = 1.5   # Trust this model more
"spiele-nsfw" = 0.8

[ensemble.tagging]
models = ["wd-vit-tagger-v3", "mobilenetv2", "efficientnet-lite4"]
strategy = "merge"
min_sources = 2              # Tag must appear in 2+ models
max_tags = 15

[ensemble.tagging.priorities]
"wd-vit-tagger-v3" = 10     # Prefer anime tags
"mobilenetv2" = 5
"efficientnet-lite4" = 7
```

### Example 3: Intersection (High Precision)
```toml
[ensemble]
enabled = true

[ensemble.moderation]
models = ["gantman-nsfw", "adamcodd-vit-nsfw", "spiele-nsfw"]
strategy = "voting"
min_agreement = 0.9          # Require 90% agreement

[ensemble.tagging]
models = ["wd-vit-tagger-v3", "mobilenetv2"]
strategy = "intersection"    # Only tags both models agree on
max_tags = 5
```

---

## Appendix C: Performance Tuning Guide

### Optimization Checklist

- [ ] Enable parallel inference (`parallel_inference = true`)
- [ ] Use LRU caching for preprocessing
- [ ] Limit active models to 3 per category
- [ ] Choose appropriate model sizes (small for speed, large for accuracy)
- [ ] Use quantized models (INT8) if available
- [ ] Profile with `perf` or `flamegraph` to find bottlenecks
- [ ] Consider GPU acceleration for batches > 10 images

### Profiling Commands

```bash
# CPU profiling
cargo flamegraph --bin camden -- scan ~/test-images --classify --ensemble

# Memory profiling
/usr/bin/time -v camden scan ~/test-images --classify --ensemble

# Benchmark
hyperfine 'camden scan ~/test-images --classify' \
          'camden scan ~/test-images --classify --ensemble'
```

### Expected Metrics (AMD Ryzen 7, 16GB RAM)

| Configuration | Latency (ms) | Throughput (img/s) | Memory (MB) |
|---------------|--------------|-------------------|-------------|
| Single GantMan | 50 | 20 | 150 |
| Single AdamCodd | 120 | 8 | 500 |
| Ensemble (seq, 3 models) | 200 | 5 | 800 |
| Ensemble (par, 3 models) | 120 | 8 | 800 |
| Ensemble (par, cached) | 80 | 12 | 800 |

---

**End of Plan**

**Next Steps:**
1. Review and approve plan
2. Create implementation branch (`feat/multi-model-ensemble`)
3. Begin Phase 1: Data Structures
4. Iterative development with testing at each phase

**Questions/Feedback:** Please comment inline or open GitHub issue
