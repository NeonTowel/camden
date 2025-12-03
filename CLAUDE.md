# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

See **AGENTS.md** for complete guidelines on commands, architecture, code style, and workflow principles. This file supplements AGENTS.md with additional context.

## Build Commands

Always use `task` commands (not direct cargo) to load vcpkg/OpenCV environment variables:

| Command | Description |
|---------|-------------|
| `task check` | Analyze workspace with cargo check |
| `task build` | Debug build (static OpenCV via vcpkg) |
| `task release` | Optimized release build |
| `task test` | Run all tests |
| `task fmt` | Format with rustfmt |
| `task frontend` | Launch Slint GUI |
| `task run CLI_ARGS="--help"` | Run CLI with arguments |

### Dependency Setup
- `task deps` - Install dev dependencies (scoop, vcpkg)
- `task install-opencv-static` - Build OpenCV statically
- `task deps-onnxruntime` - Download ONNX Runtime
- `task deps-models-hf` - Download AI models from Hugging Face (no Python required)

## Project Structure

Three-crate Rust workspace:
- **camden** (src/) - CLI binary, command dispatch
- **camden-core** (core/) - Scanner, detector, operations, classifier, thumbnails
- **camden-frontend** - Slint GUI application

## AI Classification (Feature-Gated)

The `classification` feature (default-on) enables AI-powered image analysis:

- **core/src/classifier/** - ONNX Runtime inference for moderation and tagging
- **Models** stored in `.vendor/models/` (ONNX format)
- **Configuration** via `camden-classifier.toml` (see example file)

Default models:
- MobileNetV2-12 (tagging) - ImageNet 1000 classes
- GantMan NSFW (moderation) - 5-class content safety

Tensor layouts vary by model - check `layout` field in config (NCHW vs NHWC).

## Key Patterns

### CLI/Frontend Parity (Critical)
Features must work identically in both CLI and Slint frontend. Use shared `ScanConfig` builder from camden-core:
```rust
ScanConfig::new(extensions, threading_mode)
    .with_guid_rename(bool)
    .with_low_resolution_detection(bool)
    .with_classification(bool)
```

### Error Handling
- Use `Result<T, E>` with `?` propagation
- No `panic!` for recoverable errors
- `thiserror`/`anyhow` for error types

### Outputs
- `identical_files.json` - Duplicate scan results
- `.camden-classifications.json` - AI classification results

## Environment (Windows)

Static OpenCV build requires vcpkg environment. The `.env` file sets:
- `VCPKG_ROOT`, `OPENCV_DIR`, `OPENCV_LINK_PATHS`
- `VCPKGRS_TRIPLET=x64-windows-static-md`

Vendor directory (`.vendor/`) contains vcpkg, onnxruntime, and models.

## Documentation

- **AGENTS.md** - Developer guidelines and workflow
- **docs/AI-MODELS.md** - Model selection, configuration, conversion
- **docs/SETUP-WINDOWS.md** - Windows setup guide
- **MODERATION-DEBUGGING.md** - Troubleshooting content moderation
