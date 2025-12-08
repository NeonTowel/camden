# Multi-Model Ensemble Classification

Camden supports running multiple ONNX models simultaneously and aggregating their predictions for improved content moderation accuracy.

## Quick Start

1. **Download ensemble models**:
   ```bash
   task deps-models-hf
   ```

2. **Configure ensemble** in `camden-classifier.toml`:
   ```toml
   # Moderation ensemble (2-3 models for better accuracy)
   moderation_models = ["gantman-nsfw", "vladmandic-nudenet"]

   # Tagging ensemble (2 models for diverse tag coverage)
   tagging_models = ["mobilenetv2", "vit-base-224-q8"]
   ```

3. **Run scan** with classification enabled:
   ```bash
   task run CLI_ARGS="scan /path/to/photos --classify"
   # Or use the frontend
   task frontend
   ```

## Recommended Ensemble Configurations

### Lightweight Ensemble (2 models, ~100MB) ⭐ Recommended
Best balance of speed and accuracy for most use cases.

```toml
moderation_models = ["gantman-nsfw", "vladmandic-nudenet"]
```

**Models:**
- `gantman-nsfw` - 5-class baseline (drawings, hentai, neutral, porn, sexy) - 89MB
- `vladmandic-nudenet` - Body part detection (320×320) - 12MB

**Performance:** ~150ms per image

### Balanced Ensemble (3 models, ~445MB)
Higher accuracy with ViT-based model.

```toml
moderation_models = [
    "gantman-nsfw",
    "vladmandic-nudenet",
    "adamcodd-vit-nsfw"
]
```

**Additional model:**
- `adamcodd-vit-nsfw` - ViT-based detector (384×384, 344MB)

**Performance:** ~300ms per image

### High-Accuracy Ensemble (4 models, ~545MB)
Maximum accuracy for sensitive content filtering.

```toml
moderation_models = [
    "gantman-nsfw",
    "adamcodd-vit-nsfw",
    "vladmandic-nudenet",
    "spiele-nsfw"
]
```

**Additional model:**
- `spiele-nsfw` - 4-tier severity model (~100MB)

**Performance:** ~400ms per image

## How Ensemble Works

### Moderation Ensemble

The moderation ensemble uses **averaging** by default:
- Each model outputs probability scores for its classes
- Scores are normalized and averaged across all models
- Final moderation tier is determined from averaged scores

Example with 3 models:
```
Model 1: porn=0.8, neutral=0.2
Model 2: nsfw=0.7, safe=0.3
Model 3: nsfw=0.9, safe=0.1

Average NSFW score: (0.8 + 0.7 + 0.9) / 3 = 0.8
Result: Restricted tier (high confidence)
```

### Tagging Ensemble

The tagging ensemble uses **tag merging** with confidence averaging:
- Each model generates its top N tags
- Tags are collected from all models
- Duplicate tags (same name) have their confidence scores averaged
- Final top N tags are selected by averaged confidence

Example with 2 tagging models:
```
Model 1 (MobileNetV2): dog=0.9, puppy=0.7, animal=0.6
Model 2 (ViT): dog=0.85, canine=0.7, pet=0.65

Merged tags:
  dog: (0.9 + 0.85) / 2 = 0.875 (both models agree)
  puppy: 0.7 (only from Model 1)
  animal: 0.6 (only from Model 1)
  canine: 0.7 (only from Model 2)
  pet: 0.65 (only from Model 2)

Result: More comprehensive tags with validated confidence
```

### Benefits

**Moderation:**
1. **Reduced false positives** - Multiple models must agree on content tier
2. **Better generalization** - Different architectures catch different NSFW patterns
3. **Robust to edge cases** - One model's weakness covered by others

**Tagging:**
1. **Broader vocabulary** - Each model contributes unique tags
2. **Higher confidence** - Agreement between models boosts tag reliability
3. **Diverse perspectives** - CNN vs Transformer models see different features
4. **Better coverage** - Reduces missed tags from single-model blind spots

## Available Models

All models automatically downloaded by `task deps-models-hf`:

### Moderation Models

| Model ID | Architecture | Input Size | Classes | Size | Notes |
|----------|--------------|------------|---------|------|-------|
| `gantman-nsfw` | InceptionV3 | 299×299 | 5 | 89MB | Baseline, good for anime/drawings ⭐ |
| `vladmandic-nudenet` | Custom | 320×320 | 2 | 12MB | Body part detection, lightweight ⭐ |
| `adamcodd-vit-nsfw` | ViT | 384×384 | 2 | 344MB | High accuracy, slower |
| `spiele-nsfw` | ViT | 448×448 | 4 | ~100MB | 4-tier severity levels |
| `nsfwjs` | MobileNet | 224×224 | 5 | 10MB | Very lightweight |
| `onnx-community-nsfw` | ViT | 224×224 | 2 | 343MB | Community model |
| `taufiqdp-mobilenetv4` | MobileNetV4 | 224×224 | 5 | 10MB | Fast mobile inference |

⭐ = Recommended for ensembles

### Tagging Models (Supports Ensemble)

Tagging models can now be combined in ensembles. Tags are merged across models, with confidence scores averaged for duplicate detections.

| Model ID | Type | Classes | Notes |
|----------|------|---------|-------|
| `mobilenetv2` | ImageNet | 1000 | Default, fast, good for ensembles |
| `vit-base-224` | ImageNet | 1000 | Transformer-based, diverse predictions |
| `vit-base-224-q8` | ImageNet | 1000 | Quantized ViT (4x smaller) |
| `convnextv2-large` | ImageNet | 1000 | ConvNeXt architecture |
| `wd-vit-tagger-v3` | Danbooru | 10861 | Anime-style tagging |
| `wd-swinv2-tagger-v3` | Danbooru | 10861 | High accuracy anime tags |

**Recommended Tagging Ensembles:**
- **Fast**: `["mobilenetv2", "vit-base-224-q8"]` - 2 models, diverse architectures
- **Balanced**: `["mobilenetv2", "vit-base-224", "convnextv2-large"]` - 3 ImageNet models
- **Anime**: `["wd-vit-tagger-v3", "wd-swinv2-tagger-v3"]` - Best for anime/manga

## Performance Considerations

### Memory Usage

Ensemble mode loads all models into memory simultaneously:

**Moderation only:**
- 2 models: ~400MB RAM
- 3 models: ~600MB RAM
- 5 models: ~1.2GB RAM

**With tagging ensemble:**
- Add ~300MB per tagging model (ImageNet)
- Add ~500MB per WD tagger model (Danbooru)

**Example**: 3 moderation + 2 tagging = ~1.2GB RAM

### Speed vs Accuracy Trade-off

**Moderation Ensemble:**
```
Single model:  ~100ms/image, baseline accuracy
2-model:       ~150ms/image, +15% accuracy
3-model:       ~300ms/image, +25% accuracy (recommended)
5-model:       ~500ms/image, +30% accuracy
```

**Tagging Ensemble:**
```
Single model:  ~80ms/image, baseline coverage
2-model:       ~160ms/image, +30% tag diversity
3-model:       ~240ms/image, +45% tag coverage
```

### Batch Scanning

For large photo libraries (1000+ images):
- **Fast ensemble** (3 models): ~5 minutes per 1000 images
- **High-accuracy** (5 models): ~8 minutes per 1000 images

Use parallel scanning with `ThreadingMode::Parallel` (default) for best performance.

## Advanced Configuration

### Custom Aggregation (Future)

Currently only averaging is supported. Future versions will support:
- **MaxTier** - Most conservative (highest tier from any model)
- **Weighted** - Per-model weights based on validation performance
- **Voting** - Majority vote with tie-breaking

### Model-Specific Settings

Override input specs for custom models:

```toml
[models.custom-nsfw]
name = "Custom NSFW Model"
type = "moderation"
path = "custom-model.onnx"

[models.custom-nsfw.input]
width = 512
height = 512
normalize = true
layout = "NCHW"

[models.custom-nsfw.output]
num_classes = 2
labels = ["safe", "nsfw"]
format = "nsfw_sfw"
```

## Troubleshooting

### Models Not Found

```bash
# Download all models
task deps-models-hf

# Verify models exist
ls .vendor/models/*.onnx
```

### High Memory Usage

Reduce ensemble size or use quantized models:
```toml
moderation_models = ["nsfwjs", "erax-anti-nsfw"]  # Lighter models
```

### Slow Inference

Try smaller models or reduce ensemble size:
```toml
moderation_models = ["gantman-nsfw", "nsfwjs"]  # Fast combo
```

## See Also

- [AI-MODELS.md](./AI-MODELS.md) - Model selection guide
- [MODERATION-DEBUGGING.md](../MODERATION-DEBUGGING.md) - Troubleshooting classification
- [camden-classifier-ensemble.toml](../camden-classifier-ensemble.toml) - Example config
