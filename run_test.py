#!/usr/bin/env python3
"""
FLUX test runner - verifies inference correctness against reference images.
Usage: python3 run_test.py [--flux-binary PATH]
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# Test cases: (name, prompt, seed, steps, width, height, reference_file, max_diff)
# Optional: "input" for img2img tests
TESTS = [
    {
        "name": "64x64 quick test (2 steps)",
        "prompt": "A fluffy orange cat sitting on a windowsill",
        "seed": 42,
        "steps": 2,
        "width": 64,
        "height": 64,
        "reference": "test_vectors/reference_2step_64x64_seed42.png",
        "max_diff": 35,  # BLAS and MPS have different numerical precision
    },
    {
        "name": "512x512 full test (4 steps)",
        "prompt": "A red apple on a wooden table",
        "seed": 123,
        "steps": 4,
        "width": 512,
        "height": 512,
        "reference": "test_vectors/reference_4step_512x512_seed123.png",
        "max_diff": 18,
    },
    {
        "name": "256x256 img2img test (4 steps)",
        "prompt": "A colorful oil painting of a cat",
        "seed": 456,
        "steps": 4,
        "width": 256,
        "height": 256,
        "input": "test_vectors/img2img_input_256x256.png",
        "reference": "test_vectors/reference_img2img_256x256_seed456.png",
        "max_diff": 10,
    },
]


def run_test(flux_binary: str, test: dict, model_dir: str) -> tuple[bool, str]:
    """Run a single test case. Returns (passed, message)."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        output_path = f.name

    cmd = [
        flux_binary,
        "-d", model_dir,
        "-p", test["prompt"],
        "--seed", str(test["seed"]),
        "--steps", str(test["steps"]),
        "-W", str(test["width"]),
        "-H", str(test["height"]),
        "-o", output_path,
    ]

    # Add input image for img2img tests
    if "input" in test:
        cmd.extend(["-i", test["input"]])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return False, f"flux exited with code {result.returncode}: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, "timeout (300s)"
    except FileNotFoundError:
        return False, f"binary not found: {flux_binary}"

    # Compare images
    try:
        ref = np.array(Image.open(test["reference"]))
        out = np.array(Image.open(output_path))
    except Exception as e:
        return False, f"failed to load images: {e}"

    if ref.shape != out.shape:
        return False, f"shape mismatch: ref={ref.shape}, out={out.shape}"

    diff = np.abs(ref.astype(float) - out.astype(float))
    max_diff = diff.max()
    mean_diff = diff.mean()

    if max_diff <= test["max_diff"]:
        return True, f"max_diff={max_diff:.1f}, mean={mean_diff:.4f}"
    else:
        return False, f"max_diff={max_diff:.1f} > {test['max_diff']} (mean={mean_diff:.4f})"


def main():
    parser = argparse.ArgumentParser(description="Run FLUX inference tests")
    parser.add_argument("--flux-binary", default="./flux", help="Path to flux binary")
    parser.add_argument("--model-dir", default="flux-klein-model", help="Path to model")
    parser.add_argument("--quick", action="store_true", help="Run only the quick 64x64 test")
    args = parser.parse_args()

    tests_to_run = TESTS[:1] if args.quick else TESTS

    print(f"Running {len(tests_to_run)} test(s)...\n")

    passed = 0
    failed = 0

    for i, test in enumerate(tests_to_run, 1):
        print(f"[{i}/{len(tests_to_run)}] {test['name']}...")
        ok, msg = run_test(args.flux_binary, test, args.model_dir)

        if ok:
            print(f"    PASS: {msg}")
            passed += 1
        else:
            print(f"    FAIL: {msg}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
