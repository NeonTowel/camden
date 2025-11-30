# NSFWJS Moderation Debugging Guide

## Issue
When using NSFWJS model for moderation classification, results are empty while tagging works fine.

## Cause
Moderation inference is failing silently. Errors are caught and logged to stderr but not shown in the progress bar.

## How to Debug

### 1. Check stderr output during scan
```bash
task scan <path> 2>&1 | tee scan-output.txt
```

Look for lines starting with "Moderation classification failed for" in the output. These will show the actual error.

### 2. Common causes and solutions

#### Model file not found
- Ensure nsfwjs.onnx exists in `.vendor/models/`
- Run `task deps-models-hf` to download it

#### Tensor shape mismatch
- NSFWJS expects 224x224 input in **NHWC format** (not NCHW)
- Check that `[models.nsfwjs.input]` in camden-classifier.toml has:
  ```toml
  width = 224
  height = 224
  normalize = false
  layout = "NHWC"
  ```
- This is different from GantMan which uses NCHW layout

#### ONNX Runtime error
- Ensure ONNX Runtime library is available
- Run `task deps-onnxruntime` to download

#### Output extraction failure
- NSFWJS should output 5 probability scores
- Check that `[models.nsfwjs.output]` specifies:
  ```toml
  num_classes = 5
  labels = ["drawings", "hentai", "neutral", "porn", "sexy"]
  ```

## Verification

1. **Check configuration validity:**
   ```bash
   task check
   ```

2. **Run with verbose stderr:**
   ```bash
   task scan <path> 2>&1
   ```

3. **Check JSON output:**
   - Look for `.camden-classifications.json` in the root
   - Should contain moderation_tier and tags for each image

## Notes

- NSFWJS and GantMan5Class both use the same 5-class output order
- Format is automatically detected from class labels
- Optional `format` field in config can override detection if needed
