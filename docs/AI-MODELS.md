# AI Models for Camden

Camden uses ONNX Runtime for AI-powered image classification and content moderation. This document explains the available models and how to add new ones.

## Quick Start

```bash
# Download default models (MobileNetV2 + GantMan NSFW)
task deps-models

# Download ONNX Runtime
task deps-onnxruntime

# (Recommended) Download pre-converted HuggingFace models - no Python required!
task deps-models-hf

# (Optional) Download extra models from ONNX Model Zoo
task deps-models-extra
```

## Available Models

### Default Models (task deps-models)

| Model | Type | Size | Input | Accuracy | Notes |
|-------|------|------|-------|----------|-------|
| **MobileNetV2** | Tagging | ~14MB | 224×224 | ~72% top-1 | Fast, ImageNet 1000 classes |
| **GantMan NSFW** | Moderation | ~85MB | 299×299 | Good | 5 classes: drawings, hentai, neutral, porn, sexy |

### HuggingFace Pre-Converted Models (task deps-models-hf) ⭐ Recommended

**No Python/conversion required - ready to use!**

| Model | Type | Size | Input | Notes |
|-------|------|------|-------|-------|
| **ONNX Community NSFW** | Moderation | ~343MB | 224×224 | ViT-based, 5-class, high accuracy |
| **NSFWJS** | Moderation | ~10MB | 224×224 | MobileNet, 5-class, fastest |
| **ViT-Base-224** | Tagging | ~347MB | 224×224 | Vision Transformer, highest accuracy |
| **ViT-Base-224 (Q8)** | Tagging | ~88MB | 224×224 | Quantized, faster inference |

### Extra Models (task deps-models-extra)

| Model | Type | Size | Input | Accuracy |
|-------|------|------|-------|----------|
| **EfficientNet-Lite4** | Tagging | ~50MB | 300×300 | ~80% top-1 |
| **ResNet-50** | Tagging | ~100MB | 224×224 | ~76% top-1 |

## Converting HuggingFace Models

Since the ONNX Model Zoo is being deprecated (July 2025), new models should be sourced from Hugging Face and converted to ONNX.

### Using the Conversion Script

```bash
# Install dependencies
pip install torch transformers onnx onnxruntime pillow

# List available presets
python scripts/convert_hf_to_onnx.py --list-presets

# Convert a model
python scripts/convert_hf_to_onnx.py \
  --model Falconsai/nsfw_image_detection \
  --output .vendor/models/falconsai-nsfw.onnx

# Or use a preset
python scripts/convert_hf_to_onnx.py \
  --model falconsai-nsfw \
  --output .vendor/models/falconsai-nsfw.onnx
```

### Recommended HuggingFace Models

#### NSFW/Moderation

| Model | HuggingFace ID | Classes | Accuracy |
|-------|----------------|---------|----------|
| **Falconsai NSFW** | `Falconsai/nsfw_image_detection` | normal, nsfw | 98% |
| **AdamCodd NSFW** | `AdamCodd/vit-base-nsfw-detector` | sfw, nsfw | 96.5% |
| **spiele NSFW** | `spiele/nsfw_image_detector-ONNX` | sfw, nsfw | >96% |
| **TaufiqDP MobileNetV4 NSFW** | `taufiqdp/mobilenetv4_conv_small.e2400_r224_in1k_nsfw_classifier` | sfw, nsfw | ~95% |

#### Image Classification

| Model | HuggingFace ID | Classes | Notes |
|-------|----------------|---------|-------|
| **ViT-Base** | `google/vit-base-patch16-224` | ImageNet 1K | Best accuracy |
| **ConvNeXt-Tiny** | `facebook/convnext-tiny-224` | ImageNet 1K | Good balance |
| **SwinV2** | `microsoft/swinv2-tiny-patch4-window8-256` | ImageNet 1K | Latest architecture |
| **WD ViT Tagger V3** | `SmilingWolf/wd-vit-tagger-v3` | Tagging | Light ViT, made for CLIP-style tags |
| **WD ViT Large Tagger V3** | `SmilingWolf/wd-vit-large-tagger-v3` | Tagging | Larger ViT variant, more labels |
| **WD SwinV2 Tagger V3** | `SmilingWolf/wd-swinv2-tagger-v3` | Tagging | Swin Transformer backbone |
| **WD EVA02 Large Tagger V3** | `SmilingWolf/wd-eva02-large-tagger-v3` | Tagging | EVA02 backbone, wide tag coverage |
| **WD SwinV2 Tagger V3 HF** | `p1atdev/wd-swinv2-tagger-v3-hf` | Tagging | HuggingFace-converted SwinV2 tagger |

### Pre-converted ONNX Models on HuggingFace

Some models are already converted to ONNX:
- `Xenova/vit-base-patch16-224` - Full and quantized versions
- `Qdrant/resnet50-onnx`
- Models from `onnx-community/*`

Browse: https://huggingface.co/models?library=onnx&pipeline_tag=image-classification

## Configuration

Models are configured in `camden-classifier.toml`:

```toml
# Select active models
active_moderation = "gantman-nsfw"
active_tagging = "mobilenetv2"

# Define a new model
[models.my-custom-model]
name = "My Custom Model"
type = "tagging"  # or "moderation"
path = "my-model.onnx"
enabled = true

[models.my-custom-model.input]
width = 224
height = 224
normalize = true  # ImageNet normalization
layout = "NCHW"

[models.my-custom-model.output]
num_classes = 1000
labels = []  # or ["class1", "class2", ...]
```

## Model Input/Output Specifications

### Input Format

All models expect:
- **Layout**: NCHW (batch, channels, height, width)
- **Channels**: 3 (RGB, not BGR)
- **Data type**: Float32
- **Range**: 0.0-1.0 (before normalization)

**ImageNet normalization** (when `normalize = true`):
```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

### Output Format

- **Tagging models**: Softmax probabilities for each class (1000 for ImageNet)
- **Moderation models**: Probabilities for safety categories

## Adding a New Model

1. **Convert or download** the ONNX model to `.vendor/models/`
2. **Add configuration** to `camden-classifier.toml`
3. **Update classifier code** if the model has a different output format

For moderation models with different output classes, you may need to modify `core/src/classifier/moderation.rs` to handle the new class mapping.

## Performance Tips

1. **Use quantized models** (Q8/INT8) for faster CPU inference
2. **Batch processing** is more efficient than single images
3. **Warm up** the model with a dummy inference before benchmarking
4. **GPU support** available with ONNX Runtime CUDA provider (requires different build)

## Troubleshooting

### Model not found
```
Error: model not found: .vendor/models/xxx.onnx
```
Run `task deps-models` or download the specific model.

### ONNX Runtime not initialized
```
Error: ONNX Runtime library not found
```
Run `task deps-onnxruntime` to download the runtime.

### Wrong input size
Check the model's expected input size in the config and ensure `preprocess_image()` resizes correctly.

### Low accuracy
- Verify normalization settings match the model's training
- Check if the model expects BGR vs RGB input
- Some models are trained on specific image types (photos vs. generated)
