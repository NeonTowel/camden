#!/usr/bin/env python3
"""
Convert Hugging Face Vision Transformer models to ONNX format.

This script uses the `optimum` library to export Hugging Face models
to ONNX format for use with Camden's classifier.

Usage:
    pip install optimum[exporters] onnx onnxruntime torch transformers pillow
    python scripts/convert_hf_to_onnx.py --model Falconsai/nsfw_image_detection --output .vendor/models/falconsai-nsfw.onnx
    python scripts/convert_hf_to_onnx.py --model google/vit-base-patch16-224 --output .vendor/models/vit-imagenet.onnx

For image classification models, the script will:
1. Load the model from Hugging Face
2. Export to ONNX with dynamic batch size
3. Optionally quantize for smaller size
"""

import argparse
import os
import sys
from pathlib import Path


def install_dependencies():
    """Install required dependencies if not present."""
    try:
        import torch
        import transformers
        import onnx
    except ImportError:
        print("Installing required dependencies...")
        os.system(f"{sys.executable} -m pip install torch transformers onnx onnxruntime pillow")


def export_with_torch(model_name: str, output_path: str, opset: int = 14):
    """Export model using torch.onnx.export (low-level, more control)."""
    import torch
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    
    print(f"Loading model: {model_name}")
    model = AutoModelForImageClassification.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model.eval()
    
    # Get input size from processor config
    if hasattr(processor, 'size'):
        if isinstance(processor.size, dict):
            height = processor.size.get('height', 224)
            width = processor.size.get('width', 224)
        else:
            height = width = processor.size
    else:
        height = width = 224
    
    print(f"Input size: {width}x{height}")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, height, width)
    
    # Export to ONNX
    print(f"Exporting to: {output_path}")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['pixel_values'],
        output_names=['logits'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    
    # Save model info
    info = {
        'model_name': model_name,
        'input_size': [height, width],
        'num_classes': model.config.num_labels,
        'id2label': model.config.id2label,
        'normalize': True,  # ViT models use ImageNet normalization
    }
    
    info_path = output_path.replace('.onnx', '_info.json')
    import json
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Model info saved to: {info_path}")
    print(f"Classes: {list(model.config.id2label.values())}")
    
    return output_path


def export_with_optimum(model_name: str, output_dir: str):
    """Export model using optimum library (high-level, recommended)."""
    try:
        from optimum.onnxruntime import ORTModelForImageClassification
    except ImportError:
        print("Installing optimum...")
        os.system(f"{sys.executable} -m pip install optimum[onnxruntime]")
        from optimum.onnxruntime import ORTModelForImageClassification
    
    print(f"Loading and converting: {model_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and export in one step
    model = ORTModelForImageClassification.from_pretrained(
        model_name,
        export=True
    )
    
    # Save the ONNX model
    model.save_pretrained(output_dir)
    print(f"Model saved to: {output_dir}")
    
    return output_dir


def export_with_cli(model_name: str, output_dir: str):
    """Export using optimum-cli command line."""
    import subprocess
    
    cmd = [
        sys.executable, "-m", "optimum.exporters.onnx",
        "--model", model_name,
        "--task", "image-classification",
        output_dir
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    return output_dir


def download_preconverted(model_name: str, output_path: str):
    """Download pre-converted ONNX model from Hugging Face."""
    from huggingface_hub import hf_hub_download
    
    # Common ONNX paths on HF
    onnx_paths = [
        "onnx/model.onnx",
        "model.onnx",
        "onnx/model_quantized.onnx",
    ]
    
    for onnx_path in onnx_paths:
        try:
            print(f"Trying to download: {model_name}/{onnx_path}")
            local_path = hf_hub_download(
                repo_id=model_name,
                filename=onnx_path,
                local_dir=os.path.dirname(output_path) or '.',
            )
            # Move to desired location
            import shutil
            shutil.move(local_path, output_path)
            print(f"Downloaded to: {output_path}")
            return output_path
        except Exception as e:
            continue
    
    raise ValueError(f"No ONNX model found for {model_name}")


# Pre-defined model configurations for Camden
KNOWN_MODELS = {
    # NSFW/Moderation models
    "falconsai-nsfw": {
        "hf_name": "Falconsai/nsfw_image_detection",
        "type": "moderation",
        "input_size": (224, 224),
        "normalize": True,
        "classes": ["normal", "nsfw"],
    },
    "adamcodd-nsfw": {
        "hf_name": "AdamCodd/vit-base-nsfw-detector",
        "type": "moderation", 
        "input_size": (384, 384),
        "normalize": True,
        "classes": ["nsfw", "sfw"],
    },
    # Image tagging models
    "vit-base-224": {
        "hf_name": "google/vit-base-patch16-224",
        "type": "tagging",
        "input_size": (224, 224),
        "normalize": True,
        "classes": "imagenet1k",
    },
    "vit-base-384": {
        "hf_name": "google/vit-base-patch16-384",
        "type": "tagging",
        "input_size": (384, 384),
        "normalize": True,
        "classes": "imagenet1k",
    },
    "convnext-tiny": {
        "hf_name": "facebook/convnext-tiny-224",
        "type": "tagging",
        "input_size": (224, 224),
        "normalize": True,
        "classes": "imagenet1k",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face models to ONNX for Camden"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Hugging Face model name or known preset (falconsai-nsfw, vit-base-224, etc.)"
    )
    parser.add_argument(
        "--output", "-o",
        default=".vendor/models/converted.onnx",
        help="Output path for ONNX model"
    )
    parser.add_argument(
        "--method",
        choices=["torch", "optimum", "download"],
        default="torch",
        help="Export method: torch (recommended), optimum, or download pre-converted"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)"
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List known model presets"
    )
    
    args = parser.parse_args()
    
    if args.list_presets:
        print("\nKnown model presets:")
        print("-" * 60)
        for name, config in KNOWN_MODELS.items():
            print(f"  {name}:")
            print(f"    HuggingFace: {config['hf_name']}")
            print(f"    Type: {config['type']}")
            print(f"    Input: {config['input_size']}")
            print()
        return
    
    # Resolve preset name to HF model name
    model_name = args.model
    if model_name in KNOWN_MODELS:
        model_name = KNOWN_MODELS[args.model]["hf_name"]
        print(f"Using preset '{args.model}' -> {model_name}")
    
    # Export
    if args.method == "torch":
        export_with_torch(model_name, args.output, args.opset)
    elif args.method == "optimum":
        export_with_optimum(model_name, os.path.dirname(args.output))
    elif args.method == "download":
        download_preconverted(model_name, args.output)
    
    print("\nDone! To use this model in Camden:")
    print(f"  1. Add model config to camden-classifier.toml")
    print(f"  2. Set active_moderation or active_tagging to your model ID")


if __name__ == "__main__":
    main()
