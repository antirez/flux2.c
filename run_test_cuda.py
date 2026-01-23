#!/usr/bin/env python3
"""
FLUX CUDA test runner - verifies CUDA inference correctness against reference images.

Uses the same reference images as run_test.py but with slightly higher tolerance
because GPU floating-point operations differ from CPU BLAS due to:
- TF32 tensor core precision (19-bit mantissa vs 23-bit float32)
- Different operation ordering in fused kernels
- cuBLAS internal algorithms vs OpenBLAS

Usage: python3 run_test_cuda.py [--flux-binary PATH]
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# Test cases: same as run_test.py but with higher tolerance for GPU precision differences
# Empirically determined: CUDA produces max_diff of 5-9 vs BLAS references
# due to TF32 precision and different floating-point operation ordering
TESTS = [
    {
        "name": "64x64 quick test (2 steps)",
        "prompt": "A fluffy orange cat sitting on a windowsill",
        "seed": 42,
        "steps": 2,
        "width": 64,
        "height": 64,
        "reference": "test_vectors/reference_2step_64x64_seed42.png",
        "max_diff": 10,  # CUDA: observed max_diff=5, allow headroom
    },
    {
        "name": "512x512 full test (4 steps)",
        "prompt": "A red apple on a wooden table",
        "seed": 123,
        "steps": 4,
        "width": 512,
        "height": 512,
        "reference": "test_vectors/reference_4step_512x512_seed123.png",
        "max_diff": 10,  # CUDA: observed max_diff=9, allow headroom
    },
]


def check_cuda_binary(flux_binary: str) -> tuple[bool, str]:
    """Verify the binary is built with CUDA support."""
    try:
        result = subprocess.run(
            [flux_binary, "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        # Just check it runs, CUDA detection happens at runtime
        return True, "binary found"
    except FileNotFoundError:
        return False, f"binary not found: {flux_binary}"
    except Exception as e:
        return False, str(e)


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

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return False, f"flux exited with code {result.returncode}: {result.stderr}"

        # Verify CUDA was actually used
        if "CUDA:" not in result.stdout and "CUDA:" not in result.stderr:
            return False, "CUDA not detected in output - is the binary built with 'make cuda'?"

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
    parser = argparse.ArgumentParser(description="Run FLUX CUDA inference tests")
    parser.add_argument("--flux-binary", default="./flux", help="Path to flux binary")
    parser.add_argument("--model-dir", default="flux-klein-model", help="Path to model")
    parser.add_argument("--quick", action="store_true", help="Run only the quick 64x64 test")
    args = parser.parse_args()

    # Check binary exists
    ok, msg = check_cuda_binary(args.flux_binary)
    if not ok:
        print(f"Error: {msg}")
        print("Build with: make cuda")
        return 1

    tests_to_run = TESTS[:1] if args.quick else TESTS

    print(f"Running {len(tests_to_run)} CUDA test(s)...\n")

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
