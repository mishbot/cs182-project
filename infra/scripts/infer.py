#!/usr/bin/env python3
"""Load an archived Qwen chat model from S3 and print its generated response."""

import argparse
import sys

from model_artifacts import download_model, parse_s3_uri


def generate(model_dir, prompt, max_new_tokens, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    selected = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if selected == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; run Docker with --gpus all or select --device cpu.")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        torch_dtype=torch.float16 if selected == "cuda" else torch.float32,
    ).to(selected).eval()
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(selected)
    input_length = inputs["input_ids"].shape[1]
    if input_length + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Prompt plus --max-new-tokens exceeds the model context window.")
    with torch.inference_mode():
        result = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(result[0, input_length:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-s3-uri", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--cache-dir", default="/cache/models")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if not 1 <= args.max_new_tokens <= 4096:
        parser.error("--max-new-tokens must be between 1 and 4096")
    try:
        parse_s3_uri(args.model_s3_uri)
    except ValueError as error:
        parser.error(str(error))
    try:
        import boto3

        model_dir = download_model(boto3.client("s3"), args.model_s3_uri, args.cache_dir)
        print(generate(model_dir, args.prompt, args.max_new_tokens, args.device))
    except Exception as error:
        print(f"Inference failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
