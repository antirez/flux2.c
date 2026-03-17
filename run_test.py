#!/usr/bin/env python3
"""
Iris test runner - verifies inference correctness against reference images.
Usage: python3 run_test.py [--binary PATH] [--full]
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Test cases: uses mean_diff threshold to allow bf16 precision differences
# while still catching actual bugs. Observed mean_diff values:
#   - 64x64: ~3.4, 512x512: ~1.7, img2img: ~6-17 (varies due to GPU non-determinism)
# Threshold of 20 accounts for GPU floating-point non-determinism while
# still catching catastrophic failures (wrong image would have mean_diff > 50).
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
        "mean_diff_threshold": 20,
    },
    {
        "name": "512x512 full test (4 steps)",
        "prompt": "A red apple on a wooden table",
        "seed": 123,
        "steps": 4,
        "width": 512,
        "height": 512,
        "reference": "test_vectors/reference_4step_512x512_seed123.png",
        "mean_diff_threshold": 20,
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
        "mean_diff_threshold": 20,
    },
]

# Full-only tests: these are slow and require visual inspection.
FULL_TESTS = [
    {
        "name": "1024x1024 img2img with attention budget shrinking (4 steps)",
        "prompt": "A blue sports car parked on a rainy city street at night",
        "seed": 99,
        "steps": 4,
        "width": 1024,
        "height": 1024,
        "expect_stderr": "reference image resized",
        "visual_check": "a blue sports car on a rainy city street at night, "
                        "output is 1024x1024",
    },
]

# Optional Z-Image smoke test.
# This runs only if a Z-Image model directory is auto-detected (or provided).
ZIMAGE_SMOKE_TEST = {
    "name": "Z-Image smoke test (2 steps, 64x64)",
    "prompt": "A simple geometric logo",
    "seed": 7,
    "steps": 2,
    "width": 64,
    "height": 64,
}


def is_zimage_model_dir(model_dir: Path) -> bool:
    """Return True if model_dir looks like a Z-Image model directory."""
    if not model_dir.is_dir():
        return False

    model_index = model_dir / "model_index.json"
    if not model_index.exists():
        return False

    try:
        text = model_index.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False

    return ("ZImagePipeline" in text) or ("Z-Image" in text)


def detect_zimage_model_dir(explicit_dir: Optional[str]) -> Optional[Path]:
    """Find a Z-Image model directory."""
    if explicit_dir:
        p = Path(explicit_dir)
        return p if is_zimage_model_dir(p) else None

    # Common local names first.
    for p in (Path("zimage-turbo"), Path("zimage"), Path("Z-Image-Turbo")):
        if is_zimage_model_dir(p):
            return p

    # Fallback: scan direct subdirectories.
    for p in sorted(Path(".").iterdir()):
        if p.is_dir() and is_zimage_model_dir(p):
            return p

    return None


def run_test(binary: str, test: dict, model_dir: str) -> tuple[bool, str]:
    """Run a single test case. Returns (passed, message)."""
    if "output" in test:
        output_path = test["output"]
    else:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

    cmd = [
        binary,
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if result.returncode != 0:
            return False, f"process exited with code {result.returncode}: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, "timeout (900s)"
    except FileNotFoundError:
        return False, f"binary not found: {binary}"

    # If the test has no reference image, it's a visual-check-only test.
    if "reference" not in test:
        # Verify output was created and has expected dimensions.
        try:
            out = Image.open(output_path)
        except Exception as e:
            return False, f"failed to load output: {e}"
        if out.width != test["width"] or out.height != test["height"]:
            return False, (f"wrong output size: {out.width}x{out.height}, "
                           f"expected {test['width']}x{test['height']}")

        # Check expected stderr substring (e.g. the resize note).
        if "expect_stderr" in test:
            if test["expect_stderr"] not in result.stderr:
                return False, (f"expected '{test['expect_stderr']}' in "
                               f"stderr but not found")

        return True, f"output saved to {output_path}"

    # Compare images against reference
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

    threshold = test["mean_diff_threshold"]
    if mean_diff <= threshold:
        return True, f"mean_diff={mean_diff:.2f}, max_diff={max_diff:.0f}"
    else:
        return False, f"mean_diff={mean_diff:.2f} > {threshold} (max={max_diff:.0f})"


def main():
    parser = argparse.ArgumentParser(description="Run Iris inference tests")
    parser.add_argument("--flux-binary", default="./iris", help="Path to iris binary")
    parser.add_argument("--model-dir", default="flux-klein-4b", help="Path to model")
    parser.add_argument("--zimage-model-dir", default=None,
                        help="Optional Z-Image model dir (auto-detected if omitted)")
    parser.add_argument("--quick", action="store_true", help="Run only the quick 64x64 test")
    parser.add_argument("--full", action="store_true",
                        help="Also run slow tests that require visual inspection")
    args = parser.parse_args()

    if args.quick:
        tests_to_run = TESTS[:1]
    else:
        tests_to_run = list(TESTS)
    full_tests_to_run = list(FULL_TESTS) if args.full else []

    # Optional zImage coverage: run only in non-quick mode.
    scheduled_tests: list[tuple[dict, str]] = [(t, args.model_dir) for t in tests_to_run]
    zimage_dir = detect_zimage_model_dir(args.zimage_model_dir)
    if not args.quick:
        if zimage_dir:
            print(f"Detected Z-Image model dir: {zimage_dir}")
            scheduled_tests.append((ZIMAGE_SMOKE_TEST, str(zimage_dir)))
        elif args.zimage_model_dir:
            print(f"Warning: --zimage-model-dir '{args.zimage_model_dir}' is not a valid Z-Image model dir")
            print("Skipping optional Z-Image smoke test.")
        else:
            print("No Z-Image model dir detected; skipping optional Z-Image smoke test.")

    total = len(scheduled_tests) + len(full_tests_to_run)
    print(f"Running {total} test(s)...\n")

    passed = 0
    failed = 0
    visual_checks = []

    for i, (test, model_dir) in enumerate(scheduled_tests, 1):
        print(f"[{i}/{total}] {test['name']}...")
        ok, msg = run_test(args.flux_binary, test, model_dir)

        if ok:
            print(f"    PASS: {msg}")
            passed += 1
        else:
            print(f"    FAIL: {msg}")
            failed += 1

    for j, test in enumerate(full_tests_to_run, len(scheduled_tests) + 1):
        print(f"[{j}/{total}] {test['name']}...")

        # Step 1: Generate a reference image to use as img2img input.
        ref_path = "/tmp/iris_test_ref_1024.png"
        print(f"    Step 1: Generating 1024x1024 reference image...")
        ref_cmd = [
            args.flux_binary, "-d", args.model_dir,
            "-p", "A red sports car parked on a sunny city street",
            "--seed", "42", "--steps", "4",
            "-W", "1024", "-H", "1024", "-o", ref_path,
        ]
        try:
            r = subprocess.run(ref_cmd, capture_output=True, text=True,
                               timeout=300)
            if r.returncode != 0:
                print(f"    FAIL: could not generate reference: {r.stderr}")
                failed += 1
                continue
        except Exception as e:
            print(f"    FAIL: {e}")
            failed += 1
            continue
        print(f"    Step 1: Done ({ref_path})")

        # Step 2: Run img2img with the reference — this should trigger
        # the attention budget shrinking and print a resize note.
        output_path = "/tmp/iris_test_img2img_1024.png"
        print(f"    Step 2: Running img2img with attention budget "
              f"shrinking (reference should be auto-resized)...")
        test_with_input = dict(test)
        test_with_input["input"] = ref_path
        test_with_input["output"] = output_path
        ok, msg = run_test(args.flux_binary, test_with_input, args.model_dir)

        if ok:
            print(f"    Step 2: Done ({output_path})")
            print(f"    PASS: {msg}")
            passed += 1
            if "visual_check" in test:
                visual_checks.append((test["name"], output_path,
                                      test["visual_check"]))
        else:
            print(f"    FAIL: {msg}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")

    if visual_checks:
        print("\n--- Visual verification needed ---")
        for name, out_file, description in visual_checks:
            print(f"  {name}:")
            print(f"    Open: {out_file}")
            print(f"    Check: {description}")

    if not args.full and not args.quick:
        print("\nTo run a more complete (and slow) test, run with --full")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
