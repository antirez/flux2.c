#!/usr/bin/env python3
"""
Download FLUX.2-klein-4B model files from HuggingFace.

Usage:
    python download_model.py [--output-dir DIR] [--include-text-encoder]

Requirements:
    pip install huggingface_hub

This downloads the VAE and transformer safetensors files needed for inference.
The text encoder (~8GB) is optional - only needed if you plan to implement
the Qwen3 text encoder in the future.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Download FLUX.2-klein-4B model files from HuggingFace'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='./flux-klein-model',
        help='Output directory (default: ./flux-klein-model)'
    )
    parser.add_argument(
        '--include-text-encoder',
        action='store_true',
        help='Also download the text encoder (~8GB, not required for basic inference)'
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub not installed")
        print("Install with: pip install huggingface_hub")
        return 1

    output_dir = Path(args.output_dir)

    print("FLUX.2-klein-4B Model Downloader")
    print("================================")
    print()
    print(f"Repository: black-forest-labs/FLUX.2-klein-4B")
    print(f"Output dir: {output_dir}")
    print()

    # Files to download - only VAE and transformer are needed for inference
    patterns = [
        "vae/*.safetensors",
        "vae/*.json",
        "transformer/*.safetensors",
        "transformer/*.json",
    ]

    if args.include_text_encoder:
        patterns.extend([
            "text_encoder/*",
            "text_encoder_2/*",
            "tokenizer/*",
            "tokenizer_2/*",
        ])
        print("Including text encoder (this will download ~8GB extra)")
        print()

    print("Downloading files...")
    print("(This may take a while depending on your connection)")
    print()

    try:
        model_dir = snapshot_download(
            "black-forest-labs/FLUX.2-klein-4B",
            local_dir=str(output_dir),
            allow_patterns=patterns,
            ignore_patterns=["*.bin", "*.pt", "*.pth"],  # Skip pytorch format
        )
        print()
        print("Download complete!")
        print(f"Model saved to: {model_dir}")
        print()

        # Show file sizes
        vae_path = output_dir / "vae" / "diffusion_pytorch_model.safetensors"
        tf_path = output_dir / "transformer" / "diffusion_pytorch_model.safetensors"

        total_size = 0
        if vae_path.exists():
            vae_size = vae_path.stat().st_size
            total_size += vae_size
            print(f"  VAE:         {vae_size / 1024 / 1024:.1f} MB")
        if tf_path.exists():
            tf_size = tf_path.stat().st_size
            total_size += tf_size
            print(f"  Transformer: {tf_size / 1024 / 1024 / 1024:.2f} GB")

        if total_size > 0:
            print(f"  Total:       {total_size / 1024 / 1024 / 1024:.2f} GB")
        print()
        print("Usage:")
        print(f"  ./flux -d {output_dir} -p \"your prompt\" -o output.png")
        print()
        print("Note: Text-to-image requires text embeddings (use -e option)")
        print("      until the Qwen3 text encoder is implemented.")
        print()

    except Exception as e:
        print(f"Error downloading: {e}")
        print()
        print("If you need to authenticate, run:")
        print("  huggingface-cli login")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
