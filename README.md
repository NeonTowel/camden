# Camden: Fast Image Duplicate Finder

Camden is a high-performance, multi-threaded tool that recursively scans directories to identify duplicate image files using checksum comparisons.

## Features

- Recursive directory scanning
- Fast checksum computation using xxHash
- Optional multi-threading for improved performance
- Progress bar for real-time scanning feedback
- Supports common image formats (jpg, jpeg, png, gif, bmp)

## Building from WSL when MSVC is required

The native Rust/OpenCV build must run against the MSVC toolchain, but you can keep working from inside WSL and invoke the Windows build steps via the helper:

```bash
./scripts/build-windows.sh
```

The script converts the WSL path to a Windows literal path, sets `OPENCV_DIR` to the locally cached MSVC OpenCV install (`target/opencv/windows-msvc`), and runs `cargo build` through PowerShell inside Windows. Pass any `cargo` arguments (e.g., `--release`) directly to the script.

Keep your source tree on the shared `/mnt/c/...` mount so the Windows build operates on the same files.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
