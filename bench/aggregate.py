#!/usr/bin/env python3
"""Summarize all matched run pairs in a directory (same seed/data/init only)."""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

import torch


def load_tensors(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/aggregate.json"))
    args = parser.parse_args()
    records = []
    for meta_path in sorted(args.runs_dir.glob("*.json")):
        if meta_path == args.output:
            continue
        meta = json.loads(meta_path.read_text())
        tensor_path = meta_path.with_name(meta["weights_file"])
        if not tensor_path.exists():
            continue
        records.append((meta_path, meta, load_tensors(tensor_path)))

    pairs = []
    for (pa, a, wa), (pb, b, wb) in itertools.combinations(records, 2):
        if a["procedure"]["seed"] != b["procedure"]["seed"]:
            continue
        if wa["data_sha256"] != wb["data_sha256"] or wa["initial_weights_sha256"] != wb["initial_weights_sha256"]:
            continue
        ka, kb = wa["checkpoints"], wb["checkpoints"]
        common = sorted(set(ka) & set(kb), key=int)
        if not common:
            continue
        final = common[-1]
        xa, xb = ka[final]["weights"].double(), kb[final]["weights"].double()
        la, lb = ka[final]["logits"].double(), kb[final]["logits"].double()
        ld = la - lb
        same_gpu = a["environment"]["gpu"] == b["environment"]["gpu"]
        same_stack = all(a["environment"].get(k) == b["environment"].get(k)
                         for k in ("torch", "torch_cuda", "cudnn"))
        gpu_pair = sorted([a["environment"]["gpu"], b["environment"]["gpu"]])
        pairs.append({
            "seed": a["procedure"]["seed"], "baseline_run": pa.name, "comparison_run": pb.name,
            "gpu_pair": gpu_pair,
            "pair_kind": "same_gpu" if same_gpu else "cross_gpu",
            "same_software_stack": same_stack,
            "step": int(final),
            "weights_l2_delta": (xa - xb).norm().item(),
            "weights_relative_l2_delta": ((xa - xb).norm() / xa.norm().clamp_min(1e-30)).item(),
            "logits_rms_delta": ld.square().mean().sqrt().item(),
            "prediction_disagreement": (la.argmax(1) != lb.argmax(1)).double().mean().item(),
            "accuracy_abs_delta": abs(a["metrics"][-1]["test_accuracy"] - b["metrics"][-1]["test_accuracy"]),
        })

    summaries = []
    keys = sorted({(p["pair_kind"], p["same_software_stack"], tuple(p["gpu_pair"])) for p in pairs})
    for kind, same_stack, gpu_pair in keys:
        selected = [p for p in pairs if (p["pair_kind"], p["same_software_stack"], tuple(p["gpu_pair"])) == (kind, same_stack, gpu_pair)]
        row = {"pair_kind": kind, "same_software_stack": same_stack,
               "gpu_pair": list(gpu_pair), "n_pairs": len(selected)}
        for metric in ("weights_l2_delta", "weights_relative_l2_delta", "logits_rms_delta",
                       "prediction_disagreement", "accuracy_abs_delta"):
            values = [p[metric] for p in selected]
            row[metric] = {"median": statistics.median(values), "max": max(values)}
        summaries.append(row)

    result = {"run_count": len(records), "matched_pair_count": len(pairs),
              "warning": "Descriptive pilot summaries only; pairs are not statistically independent.",
              "pair_summaries": summaries, "pairs": pairs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"run_count": len(records), "matched_pair_count": len(pairs),
                      "pair_summaries": summaries, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
