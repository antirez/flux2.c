# Test Vectors

Reference images for verifying that optimizations don't break inference.

## reference_2step_64x64_seed42.png (Primary Test)

Uses 2 denoising steps to catch bugs in multi-step state handling (e.g., caching issues between steps).

Generated with:
```bash
./flux -d flux-klein-model -p "A fluffy orange cat sitting on a windowsill" --seed 42 --steps 2 -o test_vectors/reference_2step_64x64_seed42.png -W 64 -H 64
```

Parameters:
- Model: flux-klein-model
- Prompt: "A fluffy orange cat sitting on a windowsill"
- Seed: 42
- Steps: 2
- Size: 64x64 pixels

## reference_1step_64x64_seed42.png (Legacy)

Single-step test, kept for backward compatibility.

Generated with:
```bash
./flux -d flux-klein-model -p "A fluffy orange cat sitting on a windowsill" --seed 42 --steps 1 -o test_vectors/reference_1step_64x64_seed42.png -W 64 -H 64
```

## Verification

Run `make test` from the project root, or manually:
```bash
./flux -d flux-klein-model -p "A fluffy orange cat sitting on a windowsill" --seed 42 --steps 2 -o /tmp/test_output.png -W 64 -H 64

python3 -c "
import numpy as np
from PIL import Image
ref = np.array(Image.open('test_vectors/reference_2step_64x64_seed42.png'))
test = np.array(Image.open('/tmp/test_output.png'))
diff = np.abs(ref.astype(float) - test.astype(float))
print(f'Max diff: {diff.max()}, Mean diff: {diff.mean():.4f}')
print('PASS' if diff.max() < 2 else 'FAIL')
"
```
