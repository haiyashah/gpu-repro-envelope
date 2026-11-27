#!/usr/bin/env python3
"""H100 precision audit for a pretrained causal LM on WikiText-2.

Compares FP32, TF32, BF16, and BF16 with margin-gated FP32 lm_head
recomputation. Reports held-out next-token NLL/perplexity, greedy continuation
agreement, fallback rate, and wall-clock throughput. This is a replication /
platform extension of prior precision-sensitivity work, not a novelty claim.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL = "Qwen/Qwen2.5-0.5B"


def environment() -> dict:
    props = torch.cuda.get_device_properties(0)
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        driver = None
    import datasets, transformers
    return {"gpu": props.name, "compute_capability": f"{props.major}.{props.minor}",
            "driver": driver, "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": transformers.__version__, "datasets": datasets.__version__}


def examples(tokenizer, count: int, max_tokens: int) -> list[list[int]]:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    result = []
    for row in ds:
        text = row["text"].strip()
        if len(text) < 80:
            continue
        ids = tokenizer(text, add_special_tokens=False, truncation=True,
                        max_length=max_tokens).input_ids
        if len(ids) >= 16:
            result.append(ids)
        if len(result) == count:
            break
    if len(result) != count:
        raise RuntimeError(f"Could only build {len(result)} usable text examples (requested {count})")
    return result


def set_mode(mode: str) -> None:
    enabled = mode == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def copy_model(reference, dtype: torch.dtype):
    model = copy.deepcopy(reference).to(dtype=dtype).eval()
    return model


def nll(model, samples: list[list[int]], mode: str, batch_size: int = 4) -> dict:
    set_mode(mode)
    total_nll = 0.0
    total_tokens = 0
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        width = max(map(len, batch))
        ids = torch.full((len(batch), width), model.config.eos_token_id,
                         dtype=torch.long, device="cuda")
        mask = torch.zeros_like(ids)
        for i, seq in enumerate(batch):
            ids[i, :len(seq)] = torch.tensor(seq, device="cuda")
            mask[i, :len(seq)] = 1
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            logits = logits[:, :-1, :].float().contiguous()
            labels = ids[:, 1:].contiguous()
            valid = mask[:, 1:].reshape(-1).bool()
            losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                     labels.reshape(-1), reduction="none")
            total_nll += losses[valid].sum().item()
            total_tokens += int(valid.sum().item())
        del ids, mask, logits, labels, valid, losses
    mean_nll = total_nll / total_tokens
    return {"tokens": total_tokens, "mean_nll": mean_nll,
            "perplexity": float(torch.exp(torch.tensor(mean_nll)))}


def patch_selective_head(model, reference, threshold: float) -> Counter:
    body_head = model.lm_head
    ref_head = reference.lm_head
    low_weight, low_bias = body_head.weight, body_head.bias
    full_weight = ref_head.weight.detach()
    full_bias = ref_head.bias.detach() if ref_head.bias is not None else None
    counts = Counter()

    def selective_forward(hidden):
        logits = F.linear(hidden, low_weight, low_bias).float()
        top = logits.topk(2, dim=-1).values
        margin = top[..., 0] - top[..., 1]
        selected = margin <= threshold
        counts["positions"] += selected.numel()
        counts["fallbacks"] += int(selected.sum().item())
        if selected.any():
            mask = selected.reshape(-1)
            hidden_flat = hidden.reshape(-1, hidden.shape[-1])
            logits_flat = logits.reshape(-1, logits.shape[-1])
            logits_flat[mask] = F.linear(hidden_flat[mask].float(), full_weight, full_bias)
        return logits

    body_head.forward = selective_forward
    return counts


def generate(model, tokenizer, prompts: list[list[int]], mode: str,
             new_tokens: int, threshold: float | None = None, reference=None) -> dict:
    set_mode(mode)
    counts = Counter()
    if threshold is not None:
        if reference is None:
            raise ValueError("Selective fallback requires FP32 reference weights")
        counts = patch_selective_head(model, reference, threshold)
    sequences = []
    token_total = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for seq in prompts:
            ids = torch.tensor([seq], dtype=torch.long, device="cuda")
            generated = model.generate(
                input_ids=ids, max_new_tokens=new_tokens, do_sample=False,
                use_cache=True, eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )[0, len(seq):].tolist()
            sequences.append(generated)
            token_total += len(generated)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {"sequences": sequences, "elapsed_seconds": elapsed,
            "generated_tokens": token_total, "tokens_per_second": token_total / elapsed,
            "fallbacks": int(counts["fallbacks"]), "head_positions": int(counts["positions"]),
            "fallback_fraction": counts["fallbacks"] / counts["positions"] if counts["positions"] else None}


def agreement(reference: list[list[int]], candidate: list[list[int]]) -> dict:
    prompt_divergence = 0
    token_matches = 0
    token_denominator = 0
    first_divergence = []
    for left, right in zip(reference, candidate):
        common = min(len(left), len(right))
        matches = sum(left[i] == right[i] for i in range(common))
        token_matches += matches
        token_denominator += max(len(left), len(right))
        diff = next((i for i in range(common) if left[i] != right[i]), None)
        if diff is None and len(left) != len(right):
            diff = common
        if diff is not None:
            prompt_divergence += 1
            first_divergence.append(diff)
    first_divergence.sort()
    return {"prompts": len(reference), "prompts_diverged": prompt_divergence,
            "exact_continuation_agreement": (len(reference) - prompt_divergence) / len(reference),
            "token_agreement": token_matches / token_denominator if token_denominator else 1.0,
            "median_first_divergence_token": first_divergence[len(first_divergence) // 2]
            if first_divergence else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--examples", type=int, default=96)
    parser.add_argument("--prompts", type=int, default=24)
    parser.add_argument("--context-tokens", type=int, default=192)
    parser.add_argument("--new-tokens", type=int, default=48)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    parser.add_argument("--require-h100", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.prompts > args.examples:
        raise SystemExit("--prompts cannot exceed --examples")
    torch.use_deterministic_algorithms(True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("The selected tokenizer has no EOS token for padding")
    text = examples(tokenizer, args.examples, args.context_tokens)
    prompts = text[:args.prompts]
    hardware = environment()
    if args.require_h100 and "H100" not in hardware["gpu"]:
        raise SystemExit(f"Expected H100, found {hardware['gpu']}")
    reference = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="eager"
    ).to("cuda").eval()
    if reference.config.eos_token_id is None:
        reference.config.eos_token_id = tokenizer.eos_token_id

    output = {"model": args.model, "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:test",
              "hardware": hardware, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "procedure": {"examples": args.examples, "prompts": args.prompts,
                            "context_tokens_max": args.context_tokens,
                            "new_tokens_max": args.new_tokens, "thresholds": args.thresholds,
                            "attention": "eager", "decoding": "greedy",
                            "deterministic_algorithms": True,
                            "precision_note": "FP32 checkpoint values are the reference; the same values are cast to BF16 for BF16 arms.",
                            "fallback_note": "BF16 body/head; recompute full vocabulary lm_head in FP32 where BF16 top-two logit margin is <= threshold."},
              "next_token_metrics": {}, "generation": {}}
    output["next_token_metrics"]["fp32"] = nll(reference, text, "fp32")
    tf32 = copy_model(reference, torch.float32)
    output["next_token_metrics"]["tf32"] = nll(tf32, text, "tf32")
    bf16 = copy_model(reference, torch.bfloat16)
    output["next_token_metrics"]["bf16"] = nll(bf16, text, "bf16")

    ref_gen = generate(reference, tokenizer, prompts, "fp32", args.new_tokens)
    output["generation"]["fp32"] = {k: v for k, v in ref_gen.items() if k != "sequences"}
    tf32_gen = generate(tf32, tokenizer, prompts, "tf32", args.new_tokens)
    output["generation"]["tf32"] = {**{k: v for k, v in tf32_gen.items() if k != "sequences"},
                                     **agreement(ref_gen["sequences"], tf32_gen["sequences"])}
    bf16_gen = generate(bf16, tokenizer, prompts, "bf16", args.new_tokens)
    output["generation"]["bf16"] = {**{k: v for k, v in bf16_gen.items() if k != "sequences"},
                                     **agreement(ref_gen["sequences"], bf16_gen["sequences"])}
    for threshold in args.thresholds:
        selective_model = copy_model(reference, torch.bfloat16)
        result = generate(selective_model, tokenizer, prompts, "bf16", args.new_tokens,
                          threshold=threshold, reference=reference)
        key = f"bf16_selective_head_tau_{threshold:g}"
        output["generation"][key] = {"threshold": threshold,
                                     **{k: v for k, v in result.items() if k != "sequences"},
                                     **agreement(ref_gen["sequences"], result["sequences"])}
        del selective_model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "gpu": hardware["gpu"],
                      "next_token_metrics": output["next_token_metrics"],
                      "generation": output["generation"]}, indent=2))
    del reference, tf32, bf16
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
