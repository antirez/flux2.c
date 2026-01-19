#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "huggingface_hub>=0.23",
# ]
# ///
"""
Download FLUX.2-klein-4B model files from Hugging Face using the Python API.

This script is self-contained: `uv` will provision the right Python + dependencies
from the inline PEP 723 metadata block above.

Usage:
  ./download_model.py --output-dir ./flux-klein-model
  uv run --script download_model.py --output-dir ./flux-klein-model

Auth:
  If the repo requires authentication, set `HF_TOKEN` in your environment.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Download FLUX.2-klein-4B model files from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./flux-klein-model",
        help="Output directory (default: ./flux-klein-model)",
    )
    parser.add_argument(
        "--repo-id",
        default="black-forest-labs/FLUX.2-klein-4B",
        help="Hugging Face repo id (default: black-forest-labs/FLUX.2-klein-4B)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional git revision/branch/tag to download (default: latest)",
    )
    args = parser.parse_args()

    # Hugging Face Python API (not CLI).
    from huggingface_hub import snapshot_download

    output_dir = Path(args.output_dir)

    print("FLUX.2-klein-4B Model Downloader")
    print("================================")
    print()
    print(f"Repository: {args.repo_id}")
    print(f"Output dir: {output_dir}")
    print()

    # Files to download - VAE, transformer, and text encoder
    patterns = [
        "vae/*.safetensors",
        "vae/*.json",
        "transformer/*.safetensors",
        "transformer/*.json",
        "text_encoder/*",
        "text_encoder_2/*",
        "tokenizer/*",
        "tokenizer_2/*",
    ]

    print("Downloading files (~16GB total)...")
    print("(This may take a while depending on your connection)")
    print()

    try:
        model_dir = snapshot_download(
            repo_id=args.repo_id,
            revision=args.revision,
            local_dir=str(output_dir),
            allow_patterns=patterns,
            ignore_patterns=["*.bin", "*.pt", "*.pth"],  # Skip pytorch format
            resume_download=True,
        )
        print()
        print("Download complete!")
        print(f"Model saved to: {model_dir}")
        print()

        # Show file sizes
        vae_path = output_dir / "vae" / "diffusion_pytorch_model.safetensors"
        tf_path = output_dir / "transformer" / "diffusion_pytorch_model.safetensors"
        te_path = output_dir / "text_encoder_2"

        total_size = 0
        if vae_path.exists():
            vae_size = vae_path.stat().st_size
            total_size += vae_size
            print(f"  VAE:          {vae_size / 1024 / 1024:.1f} MB")
        if tf_path.exists():
            tf_size = tf_path.stat().st_size
            total_size += tf_size
            print(f"  Transformer:  {tf_size / 1024 / 1024 / 1024:.2f} GB")
        if te_path.exists():
            te_size = sum(f.stat().st_size for f in te_path.rglob("*") if f.is_file())
            total_size += te_size
            print(f"  Text encoder: {te_size / 1024 / 1024 / 1024:.2f} GB")

        if total_size > 0:
            print(f"  Total:        {total_size / 1024 / 1024 / 1024:.2f} GB")
        print()
        print("Usage:")
        print(f'  ./flux -d {output_dir} -p "your prompt" -o output.png')
        print()

    except Exception as e:
        print(f"Error downloading: {e}")
        print()
        print("If you need to authenticate, set an access token:")
        print("  export HF_TOKEN=...  (or HUGGINGFACE_HUB_TOKEN=...)")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
