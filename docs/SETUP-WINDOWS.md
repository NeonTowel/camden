# Windows Development Environment Setup

Complete guide for setting up Camden development on Windows, including AI classification dependencies.

---

## Prerequisites

### Required Software

| Tool | Version | Purpose |
|------|---------|---------|
| **Rust** | 1.75+ | Core language |
| **Git** | 2.40+ | Version control |
| **Visual Studio Build Tools** | 2022 | C++ compiler (MSVC) |
| **Scoop** | Latest | Package manager |
| **Task** | 3.x | Task runner |

### Hardware Recommendations

- **RAM**: 8GB minimum (16GB for GPU inference)
- **Disk**: 5GB free (models + vcpkg cache)
- **GPU** (optional): NVIDIA with CUDA 11.8+ for accelerated inference

---

## Step 1: Install Core Tools

### 1.1 Install Scoop (Package Manager)

Open PowerShell as Administrator:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression
```

### 1.2 Install Development Tools

```powershell
# Add extras bucket for additional packages
scoop bucket add extras

# Install required tools
scoop install git rust task llvm

# Verify installations
rustc --version
task --version
git --version
```

### 1.3 Install Visual Studio Build Tools

Download and install [Visual Studio Build Tools 2022](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

During installation, select:
- **"Desktop development with C++"** workload
- Individual components:
  - MSVC v143 (or latest)
  - Windows 11 SDK (or Windows 10 SDK)
  - C++ CMake tools

Alternatively via winget:

```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

---

## Step 2: Clone Repository

```powershell
git clone https://github.com/NeonTowel/camden.git
cd camden
```

---

## Step 3: Install OpenCV (Static Build)

Camden uses OpenCV for image processing. We build a static version via vcpkg for easy distribution.

### 3.1 Install vcpkg and OpenCV

```powershell
# This installs vcpkg to .vendor/vcpkg and builds static OpenCV
# Takes 15-30 minutes on first run
task install-opencv-static
```

This will:
1. Clone vcpkg to `.vendor/vcpkg`
2. Bootstrap vcpkg
3. Build `opencv:x64-windows-static` (~2GB disk space)

### 3.2 Verify OpenCV Installation

```powershell
# Check that vcpkg has opencv installed
.vendor\vcpkg\vcpkg list | Select-String opencv
```

Expected output:
```
opencv:x64-windows-static    4.x.x    ...
```

---

## Step 4: Install ONNX Runtime (AI Classification)

ONNX Runtime powers the AI classification and tagging features. We use **dynamic loading** 
to avoid CRT conflicts with static OpenCV — the DLL is loaded at runtime.

### 4.1 Download via Task (Recommended)

```powershell
task deps-onnxruntime
```

This downloads ONNX Runtime 1.21.0 (~50MB) to `.vendor/onnxruntime/`.

Contents after download:
```
.vendor/onnxruntime/
├── include/           # C API headers
├── lib/
│   ├── onnxruntime.dll    # Runtime library (ship with your app)
│   └── onnxruntime.lib    # Import library
└── LICENSE
```

### 4.2 Runtime Initialization

The ONNX Runtime DLL is loaded at runtime, not linked at compile time. Your code must 
initialize it before using classifiers:

```rust
use camden_core::classifier::{init_ort_runtime, ImageClassifier};

// Initialize once at application startup
init_ort_runtime(".vendor/onnxruntime/lib/onnxruntime.dll")?;

// Now classifiers work
let classifier = ImageClassifier::with_default_paths()?;
```

### 4.3 Distribution

When distributing your application, include `onnxruntime.dll` alongside the executable:

```
my-app/
├── camden.exe
├── onnxruntime.dll      # Required at runtime
└── models/
    ├── mobilenetv2-12.onnx
    └── nsfw-inception-v3.onnx
```

### 4.4 GPU Support (CUDA) - Optional

For NVIDIA GPU acceleration, download the GPU variant instead:

```powershell
$version = "1.21.0"
$url = "https://github.com/microsoft/onnxruntime/releases/download/v$version/onnxruntime-win-x64-gpu-$version.zip"
Invoke-WebRequest -Uri $url -OutFile onnxruntime-gpu.zip
Expand-Archive onnxruntime-gpu.zip -DestinationPath .vendor\onnxruntime-gpu
```

Requirements:
- [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads)
- [cuDNN 9.x](https://developer.nvidia.com/cudnn)

---

## Step 5: Download AI Models

Models are stored in `.vendor/models/` alongside OpenCV and vcpkg.

### 5.1 Download via Task

```powershell
task deps-models
```

This downloads:

| Model | Purpose | Size | Input |
|-------|---------|------|-------|
| `mobilenetv2-12.onnx` | Image tagging (ImageNet 1000 classes) | ~14MB | 224×224 |
| `nsfw-inception-v3.onnx` | Content moderation (5 classes) | ~85MB | 299×299 |

**NSFW Model Classes:**
- `neutral` → Safe
- `drawings` → Safe  
- `sexy` → Sensitive
- `hentai` → Mature
- `porn` → Restricted

### 5.2 Manual Download

Download models manually if behind a firewall:

```powershell
mkdir -p .vendor\models

# MobileNetV2 (tagging) - ~14MB
Invoke-WebRequest `
  -Uri "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx" `
  -OutFile ".vendor\models\mobilenetv2-12.onnx"

# GantMan NSFW Inception V3 (moderation) - ~85MB
Invoke-WebRequest `
  -Uri "https://github.com/iola1999/nsfw-detect-onnx/releases/download/v1.0.0/model.onnx" `
  -OutFile ".vendor\models\nsfw-inception-v3.onnx"
```

### 5.3 Verify Downloads

```powershell
Get-FileHash ".vendor\models\mobilenetv2-12.onnx" -Algorithm SHA256
Get-FileHash ".vendor\models\nsfw-inception-v3.onnx" -Algorithm SHA256
```

---

## Step 6: Build and Verify

### 6.1 Check Build

```powershell
task check
```

Expected: No errors.

### 6.2 Run Tests

```powershell
task test
```

### 6.3 Build Debug

```powershell
task build
```

### 6.4 Build Release

```powershell
task release
```

### 6.5 Run Frontend

```powershell
task frontend
```

---

## Step 7: IDE Setup (Optional)

### VS Code

Install extensions:
- **rust-analyzer** — Rust language support
- **Even Better TOML** — Cargo.toml syntax
- **Slint** — Slint UI language support

Create `.vscode/settings.json`:

```json
{
  "rust-analyzer.cargo.features": ["classification"],
  "rust-analyzer.check.command": "check",
  "terminal.integrated.env.windows": {
    "VCPKG_ROOT": "${workspaceFolder}\\.vendor\\vcpkg",
    "RUSTFLAGS": "-C target-feature=+crt-static"
  }
}
```

### RustRover / CLion

1. Open project folder
2. Set environment variables in Run Configuration:
   - `VCPKG_ROOT` = `<project>\.vendor\vcpkg`
   - `RUSTFLAGS` = `-C target-feature=+crt-static`

---

## Environment Variables Reference

| Variable | Value | Purpose |
|----------|-------|---------|
| `VCPKG_ROOT` | `.vendor\vcpkg` | OpenCV static build location |
| `RUSTFLAGS` | `-C target-feature=+crt-static` | Static CRT linking (for OpenCV) |
| `ORT_DYLIB_PATH` | `.vendor\onnxruntime\lib\onnxruntime.dll` | Runtime DLL path (alternative to code init) |

To persist environment variables:

```powershell
# Add to PowerShell profile
notepad $PROFILE

# Add these lines:
$env:VCPKG_ROOT = "C:\path\to\camden\.vendor\vcpkg"
$env:RUSTFLAGS = "-C target-feature=+crt-static"
```

---

## Troubleshooting

### OpenCV Build Fails

**Error**: `opencv2/core.hpp not found`

**Solution**: Ensure vcpkg built successfully:
```powershell
.vendor\vcpkg\vcpkg install opencv:x64-windows-static --recurse
```

### ONNX Runtime Not Found

**Error**: `ONNX Runtime library not found at: .vendor/onnxruntime/lib/onnxruntime.dll`

**Solution**: Download ONNX Runtime:
```powershell
task deps-onnxruntime
```

### ONNX Runtime DLL Load Failed

**Error**: `failed to load onnxruntime` or `0xc0000135`

**Solution**: Ensure VC++ Runtime 2019+ is installed:
```powershell
winget install Microsoft.VCRedist.2015+.x64
```

### CUDA Not Detected

**Error**: GPU inference not working

**Solution**:
1. Verify CUDA installation: `nvcc --version`
2. Check GPU: `nvidia-smi`
3. Ensure cuDNN is in PATH
4. Use GPU ONNX Runtime build

### Slow First Build

**Cause**: vcpkg compiling OpenCV from source (~15-30 min)

**Solution**: This is normal for first build. Subsequent builds use cache.

### Task Not Found

**Error**: `'task' is not recognized`

**Solution**:
```powershell
scoop install task
# Or restart terminal after install
```

---

## Directory Structure After Setup

```
camden/
├── .vendor/
│   ├── vcpkg/                    # vcpkg installation (static OpenCV)
│   │   ├── installed/
│   │   │   └── x64-windows-static/
│   │   │       ├── include/opencv2/
│   │   │       └── lib/
│   │   └── vcpkg.exe
│   ├── models/                   # AI classification models
│   │   ├── mobilenetv2-12.onnx
│   │   └── nsfw-inception-v3.onnx
│   └── onnxruntime/              # ONNX Runtime (dynamic loading)
│       ├── include/
│       └── lib/
│           ├── onnxruntime.dll   # Ship with your app
│           └── onnxruntime.lib
├── target/
│   ├── debug/
│   └── release/
├── core/
├── camden-frontend/
└── ...
```

Runtime data (thumbnails, cache) still in user profile:
```
%LOCALAPPDATA%\Camden\
├── thumbnails/
└── cache/
```

---

## Quick Reference Commands

```powershell
# Full setup from scratch
scoop install git rust task llvm
git clone https://github.com/NeonTowel/camden.git
cd camden
task install-opencv-static   # ~20 min first time
task build

# Daily development
task check      # Type check
task test       # Run tests
task build      # Debug build
task frontend   # Run GUI
task fmt        # Format code

# Clean rebuild
cargo clean
task build
```

---

## Next Steps

- [PLAN-AI-CLASSIFICATION.md](../PLAN-AI-CLASSIFICATION.md) — AI feature implementation plan
- [PLAN.md](../PLAN.md) — Overall project roadmap
