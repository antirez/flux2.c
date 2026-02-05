#!/bin/bash
set -e

# Default: distilled model. Use --base for undistilled base model.
REPO="FLUX.2-klein-4B"
OUT="./flux-klein-model"

if [ "$1" = "--base" ]; then
    REPO="FLUX.2-klein-base-4B"
    OUT="./flux-klein-base-model"
    echo "Downloading base (undistilled) model..."
else
    echo "Downloading distilled model... (use --base for base model)"
fi

BASE="https://huggingface.co/black-forest-labs/$REPO/resolve/main"

mkdir -p "$OUT"/{text_encoder,tokenizer,transformer,vae}

# model_index.json (needed for autodetection)
curl -L -o "$OUT/model_index.json" "$BASE/model_index.json"

# text_encoder (Qwen3 - ~8GB total)
curl -L -o "$OUT/text_encoder/config.json" "$BASE/text_encoder/config.json"
curl -L -o "$OUT/text_encoder/generation_config.json" "$BASE/text_encoder/generation_config.json"
curl -L -o "$OUT/text_encoder/model.safetensors.index.json" "$BASE/text_encoder/model.safetensors.index.json"
curl -L -o "$OUT/text_encoder/model-00001-of-00002.safetensors" "$BASE/text_encoder/model-00001-of-00002.safetensors"
curl -L -o "$OUT/text_encoder/model-00002-of-00002.safetensors" "$BASE/text_encoder/model-00002-of-00002.safetensors"

# tokenizer
curl -L -o "$OUT/tokenizer/added_tokens.json" "$BASE/tokenizer/added_tokens.json"
curl -L -o "$OUT/tokenizer/chat_template.jinja" "$BASE/tokenizer/chat_template.jinja"
curl -L -o "$OUT/tokenizer/merges.txt" "$BASE/tokenizer/merges.txt"
curl -L -o "$OUT/tokenizer/special_tokens_map.json" "$BASE/tokenizer/special_tokens_map.json"
curl -L -o "$OUT/tokenizer/tokenizer.json" "$BASE/tokenizer/tokenizer.json"
curl -L -o "$OUT/tokenizer/tokenizer_config.json" "$BASE/tokenizer/tokenizer_config.json"
curl -L -o "$OUT/tokenizer/vocab.json" "$BASE/tokenizer/vocab.json"

# transformer (~7.75 GB)
curl -L -o "$OUT/transformer/config.json" "$BASE/transformer/config.json"
curl -L -o "$OUT/transformer/diffusion_pytorch_model.safetensors" "$BASE/transformer/diffusion_pytorch_model.safetensors"

# vae (~168 MB)
curl -L -o "$OUT/vae/config.json" "$BASE/vae/config.json"
curl -L -o "$OUT/vae/diffusion_pytorch_model.safetensors" "$BASE/vae/diffusion_pytorch_model.safetensors"

echo "Done. Total ~16GB -> $OUT"
