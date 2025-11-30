# AI Image Classification & Tagging Plan

## Overview

Add pre-trained ML model inference to Camden for:
1. **Content moderation** — flag images with tiered safety classifications
2. **Automatic tagging** — generate descriptive labels from image content

Results stored with `camden\` prefix in metadata and displayed in the preview UI.

---

## Phase 1 — Model Selection & Infrastructure

### 1.1 ML Runtime Selection

**Recommended: ONNX Runtime**
- Cross-platform, Rust bindings (`ort` crate), GPU optional
- Supports most pre-trained vision models
- Can bundle models as assets or download on first run

```toml
# core/Cargo.toml additions
ort = { version = "2", default-features = false, features = ["download-binaries"] }
ndarray = "0.15"
```

### 1.2 Pre-trained Models

| Purpose | Model | Input | Output | Size |
|---------|-------|-------|--------|------|
| **Tagging** | MobileNetV2-12 | 224×224 | ImageNet 1000 classes | ~14MB |
| **Moderation** | GantMan NSFW (Inception V3) | 299×299 | 5 classes | ~85MB |

#### NSFW Model Classes → Moderation Tiers

The [GantMan nsfw_model](https://github.com/GantMan/nsfw_model) outputs 5 nuanced classes:

| Model Class | Description | → Tier |
|-------------|-------------|--------|
| `neutral` | Safe, neutral images | **Safe** |
| `drawings` | Safe illustrations, anime | **Safe** |
| `sexy` | Suggestive but not explicit | **Sensitive** |
| `hentai` | Explicit illustrations | **Mature** |
| `porn` | Explicit photographic content | **Restricted** |

**Tier assignment logic:**
```rust
fn classify_tier(scores: &NsfwScores) -> ModerationTier {
    let dominant = scores.max_class();
    match dominant {
        "neutral" | "drawings" => ModerationTier::Safe,
        "sexy" => ModerationTier::Sensitive,
        "hentai" => ModerationTier::Mature,
        "porn" => ModerationTier::Restricted,
    }
}
```

**Confidence thresholds** can refine edge cases (e.g., if `sexy` score is < 0.3, classify as Safe).

Models stored in `.vendor/models/`, consistent with OpenCV/vcpkg:
- `mobilenetv2-12.onnx` — Tagging
- `nsfw-inception-v3.onnx` — Moderation

Download via: `task deps-models`

### 1.3 New Module Structure

```
core/src/
├── classifier/
│   ├── mod.rs           # Public API
│   ├── runtime.rs       # ONNX session management
│   ├── models.rs        # Model registry & download
│   ├── moderation.rs    # Safety classification
│   └── tagging.rs       # Auto-tag generation
```

---

## Phase 2 — Moderation Flags (Content Safety)

### 2.1 Flag Taxonomy

```rust
/// Content safety flags with tiered severity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModerationFlags {
    /// Overall safety score 0.0 (safe) to 1.0 (unsafe)
    pub safety_score: f32,
    
    /// Tier assignment based on thresholds
    pub tier: ModerationTier,
    
    /// Individual category scores
    pub categories: ModerationCategories,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModerationTier {
    Safe,           // No concerning content detected
    Sensitive,      // Mild/suggestive (swimwear, artistic nudity)
    Mature,         // Adult content, violence
    Restricted,     // Explicit content requiring intervention
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModerationCategories {
    pub nsfw_score: f32,          // General NSFW probability
    pub violence_score: f32,      // Gore, weapons, fighting
    pub suggestive_score: f32,    // Provocative but not explicit
    pub hate_symbols_score: f32,  // Offensive symbols/gestures
    pub drugs_score: f32,         // Drug paraphernalia
    pub medical_gore_score: f32,  // Surgical/injury content
}
```

### 2.2 Tier Thresholds (Configurable)

```rust
pub struct ModerationConfig {
    pub sensitive_threshold: f32,   // default: 0.25
    pub mature_threshold: f32,      // default: 0.50
    pub restricted_threshold: f32,  // default: 0.75
    pub enabled_categories: HashSet<ModerationCategory>,
}
```

### 2.3 API Surface

```rust
pub struct ImageClassifier {
    runtime: OrtRuntime,
    moderation_model: Option<ModerationModel>,
    tagging_model: Option<TaggingModel>,
}

impl ImageClassifier {
    pub fn new(config: ClassifierConfig) -> Result<Self, ClassifierError>;
    
    /// Analyze image for moderation flags
    pub fn moderate(&self, image_data: &[u8]) -> Result<ModerationFlags, ClassifierError>;
    
    /// Generate descriptive tags
    pub fn generate_tags(&self, image_data: &[u8], max_tags: usize) -> Result<Vec<ImageTag>, ClassifierError>;
    
    /// Combined analysis (more efficient for both)
    pub fn analyze(&self, image_data: &[u8]) -> Result<ClassificationResult, ClassifierError>;
}
```

---

## Phase 3 — Automatic Tagging

### 3.1 Tag Structure

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageTag {
    /// Normalized tag name (lowercase, hyphenated)
    pub name: String,
    
    /// Human-readable label
    pub label: String,
    
    /// Model confidence 0.0-1.0
    pub confidence: f32,
    
    /// Tag category for organization
    pub category: TagCategory,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TagCategory {
    Object,      // dog, car, chair
    Scene,       // beach, mountain, office
    Activity,    // running, cooking, reading
    Style,       // portrait, landscape, macro
    Color,       // dominant colors
    Technical,   // blurry, overexposed, low-light
}
```

### 3.2 Tag Generation Pipeline

1. **Run MobileNet/EfficientNet** → Top-K ImageNet classes
2. **Map to semantic tags** → Group similar classes (e.g., "golden retriever" → "dog")
3. **Add scene context** → Places365 or scene classification head
4. **Technical analysis** → Blur detection, exposure (reuse OpenCV)
5. **Color extraction** → Already in `ImageMetadata.dominant_color`

### 3.3 Tag Filtering

```rust
pub struct TaggingConfig {
    pub min_confidence: f32,        // default: 0.3
    pub max_tags: usize,            // default: 10
    pub include_technical: bool,    // blur, exposure tags
    pub include_colors: bool,       // dominant color tags
    pub custom_blocklist: Vec<String>, // filter unwanted tags
}
```

---

## Phase 4 — Metadata Storage

### 4.1 Metadata Schema

Write to EXIF/IPTC/XMP using `camden\` prefix to avoid conflicts:

| Field | EXIF/XMP Path | Example |
|-------|---------------|---------|
| Tags | `Iptc.Application2.Keywords` | `camden\dog`, `camden\outdoor` |
| Safety Tier | `Xmp.camden.ModerationTier` | `Safe` |
| Safety Score | `Xmp.camden.SafetyScore` | `0.12` |
| NSFW Score | `Xmp.camden.NsfwScore` | `0.05` |
| Analyzed Date | `Xmp.camden.AnalyzedAt` | ISO 8601 timestamp |
| Model Version | `Xmp.camden.ModelVersion` | `mobilenet-v3-1.0` |

### 4.2 Crate Selection

```toml
# For metadata read/write
rexiv2 = "0.10"     # libexiv2 bindings (full EXIF/IPTC/XMP)
# OR
img-parts = "0.3"   # Pure Rust, JPEG/PNG only
kamadak-exif = "0.5" # Already in use (read-only)
```

**Recommendation**: `rexiv2` for full XMP support, or custom XMP sidecar files (`.xmp`) for non-destructive workflow.

### 4.3 Sidecar Alternative

For non-destructive workflows, write `<filename>.camden.json`:

```json
{
  "analyzed_at": "2025-11-29T12:00:00Z",
  "model_version": "mobilenet-v3-1.0",
  "moderation": {
    "tier": "Safe",
    "safety_score": 0.08,
    "categories": { "nsfw": 0.02, "violence": 0.01 }
  },
  "tags": [
    { "name": "dog", "confidence": 0.92, "category": "Object" },
    { "name": "outdoor", "confidence": 0.85, "category": "Scene" }
  ]
}
```

---

## Phase 5 — Integration with Core

### 5.1 Extend `ImageMetadata`

```rust
// detector.rs additions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageMetadata {
    // ... existing fields ...
    
    /// AI-generated moderation flags (optional)
    pub moderation: Option<ModerationFlags>,
    
    /// AI-generated tags (optional)
    pub tags: Vec<ImageTag>,
    
    /// Whether AI classification has been performed
    pub classified: bool,
}
```

### 5.2 Extend `ScanConfig`

```rust
pub struct ScanConfig {
    // ... existing fields ...
    
    /// Enable AI classification during scan
    pub enable_classification: bool,
    
    /// Classification configuration
    pub classification_config: Option<ClassifierConfig>,
}
```

### 5.3 Processing Pipeline

```
scan_with_progress()
    ├── walk directory
    ├── analyze images (perceptual hash, features)
    ├── IF enable_classification:
    │   ├── load classifier (lazy, first image)
    │   ├── moderate(image) → ModerationFlags
    │   ├── generate_tags(image) → Vec<ImageTag>
    │   └── attach to ImageMetadata
    ├── group duplicates
    └── return ScanSummary with enriched metadata
```

---

## Phase 6 — Preview UI Integration

### 6.1 Slint Data Structures

```slint
export struct ModerationData {
    tier: int,           // 0=Safe, 1=Sensitive, 2=Mature, 3=Restricted
    safety_score: float,
    nsfw_score: float,
}

export struct TagData {
    name: string,
    confidence: float,
    category: int,       // 0=Object, 1=Scene, etc.
}

export struct FileData {
    // ... existing fields ...
    moderation: ModerationData,
    tags: [TagData],
}
```

### 6.2 UI Components

```slint
// Moderation badge (color-coded by tier)
component ModerationBadge {
    in property <ModerationData> data;
    
    Rectangle {
        background: data.tier == 0 ? #27ae60   // Safe - green
                  : data.tier == 1 ? #f39c12   // Sensitive - orange
                  : data.tier == 2 ? #e74c3c   // Mature - red
                  : #8e44ad;                    // Restricted - purple
        // ...
    }
}

// Tag chips in detail view
component TagList {
    in property <[TagData]> tags;
    
    HorizontalLayout {
        for tag in tags : Rectangle {
            Text { text: tag.name; }
        }
    }
}
```

### 6.3 Filtering & Sorting

- Filter by moderation tier (hide Restricted, show only Safe, etc.)
- Filter by tag (show only images with "dog" tag)
- Sort by safety score

---

## Phase 7 — CLI Integration

### 7.1 New Commands

```bash
# Classify images without duplicate scanning
camden classify <path> [--output json|sidecar|metadata]

# Scan with classification enabled
camden scan <path> --classify

# Filter scan results by moderation tier
camden scan <path> --classify --max-tier sensitive
```

### 7.2 Output Formats

```bash
# JSON report with classifications
camden classify ./photos --output json > report.json

# Write sidecar files
camden classify ./photos --output sidecar

# Write to image metadata (destructive)
camden classify ./photos --output metadata --write
```

---

## Phase 8 — Testing & Validation

### 8.1 Test Assets

Create `tests/assets/classification/` with:
- Safe images (landscapes, objects)
- Edge cases (swimwear, art)
- Synthetic test patterns

### 8.2 Unit Tests

```rust
#[test]
fn classifier_detects_dog() {
    let classifier = ImageClassifier::new(Default::default()).unwrap();
    let result = classifier.generate_tags(DOG_IMAGE_BYTES, 5).unwrap();
    assert!(result.iter().any(|t| t.name.contains("dog")));
}

#[test]
fn moderation_flags_safe_image() {
    let classifier = ImageClassifier::new(Default::default()).unwrap();
    let flags = classifier.moderate(LANDSCAPE_BYTES).unwrap();
    assert_eq!(flags.tier, ModerationTier::Safe);
}
```

### 8.3 Benchmarks

- Classification throughput (images/sec)
- Memory usage with model loaded
- Cold start vs warm classification

---

## Implementation Order

| Step | Deliverable | Effort |
|------|-------------|--------|
| 1 | Add `ort` + ONNX runtime infrastructure | 1 day |
| 2 | Model download/cache system | 0.5 day |
| 3 | MobileNet tagging implementation | 1 day |
| 4 | NSFW moderation model integration | 1 day |
| 5 | Extend `ImageMetadata` + `ScanConfig` | 0.5 day |
| 6 | Sidecar JSON output | 0.5 day |
| 7 | CLI `classify` command | 0.5 day |
| 8 | Slint UI badges/tags/filters | 1 day |
| 9 | XMP/EXIF metadata writing (optional) | 1 day |
| 10 | Tests + documentation | 1 day |

**Total: ~8 days**

---

## Feature Flags

```toml
[features]
default = ["classification"]
classification = ["dep:ort", "dep:ndarray"]
classification-gpu = ["classification", "ort/cuda"]  # Optional GPU
metadata-write = ["dep:rexiv2"]  # XMP/EXIF writing
```

---

## Open Questions

1. **Model licensing** — Verify licenses for bundled models (MIT/Apache preferred)
2. **GPU support** — Worth the complexity for local usage?
3. **Sidecar vs embedded** — Default to sidecar (non-destructive) or ask user?
4. **Model updates** — How to handle model version upgrades?
5. **Privacy** — All processing local; document this clearly

---

## Diagram: Classification Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Image File  │────▶│ OpenCV Load  │────▶│ Resize to 224px │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                    ┌──────────────────────────────┼──────────────────────────────┐
                    │                              ▼                              │
                    │  ┌─────────────────┐   ┌─────────────────┐                 │
                    │  │ MobileNet ONNX  │   │   NSFW Model    │                 │
                    │  └────────┬────────┘   └────────┬────────┘                 │
                    │           │                     │                           │
                    │           ▼                     ▼                           │
                    │  ┌─────────────────┐   ┌─────────────────┐                 │
                    │  │ Top-K Classes   │   │ Safety Scores   │                 │
                    │  └────────┬────────┘   └────────┬────────┘                 │
                    │           │                     │                           │
                    │           ▼                     ▼                           │
                    │  ┌─────────────────┐   ┌─────────────────┐                 │
                    │  │ Semantic Tags   │   │ModerationFlags  │                 │
                    │  └────────┬────────┘   └────────┬────────┘                 │
                    │           │                     │                           │
                    └───────────┼─────────────────────┼───────────────────────────┘
                                │                     │
                                ▼                     ▼
                    ┌─────────────────────────────────────────┐
                    │           ImageMetadata                 │
                    │  { tags: [...], moderation: {...} }     │
                    └─────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
            ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
            │ Sidecar JSON│     │  XMP/EXIF   │     │  Preview UI │
            └─────────────┘     └─────────────┘     └─────────────┘
```
