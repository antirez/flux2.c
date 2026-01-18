# FLUX.2-klein-4B Pure C Implementation

A dependency-free C implementation of the FLUX.2-klein-4B image generation model. Runs inference using only the C standard library and BLAS (Apple Accelerate on macOS, OpenBLAS on Linux).

On Apple Silicon Macs, Metal GPU acceleration is automatically enabled for faster matrix operations.

## Building

```bash
make
```

To check build configuration:
```bash
make info
```

## Downloading the Model

```bash
pip install huggingface_hub
python download_model.py
```

This downloads the VAE and transformer (~8GB) to `./flux-klein-model`.

## Usage

```bash
./flux -d flux-klein-model -e embeddings.bin -o output.png
```

Options:
- `-d PATH` - Model directory (required)
- `-p TEXT` - Text prompt (requires text encoder, see limitations)
- `-e PATH` - Pre-computed text embeddings file
- `-o PATH` - Output image path
- `-W N` - Width (default: 1024)
- `-H N` - Height (default: 1024)
- `-s N` - Sampling steps (default: 4)
- `-S N` - Random seed
- `-v` - Verbose output with progress

## Current Limitations

**No text encoder**: The Qwen3 text encoder (~8GB) is not yet implemented. To generate images, you must provide pre-computed text embeddings via the `-e` option. These can be generated using the Python diffusers library:

```python
from diffusers import FluxPipeline
import numpy as np

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B")
prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
    "your prompt here", max_sequence_length=512
)
# Save prompt_embeds as binary file for use with -e option
```

## License

MIT
