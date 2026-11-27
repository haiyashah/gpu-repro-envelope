#!/usr/bin/env python3
"""Summarize paired precision-mode runs and boundary flips."""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

import torch


def load_tensor_file(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def median(values):
    return statistics.median(values) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/precision-frontier-summary.json"))
    args = parser.parse_args()
    records = {}
    for meta_path in sorted(args.runs_dir.glob("*.json")):
        if meta_path == args.output:
            continue
        meta = json.loads(meta_path.read_text())
        tensors = load_tensor_file(meta_path.with_name(meta["weights_file"]))
        key = (meta["procedure"]["seed"], meta["procedure"]["mode"])
        records[key] = (meta_path.name, meta, tensors)

    pairs, by_mode = [], {"tf32": [], "bf16": []}
    bin_edges = [0.0, 0.005, 0.01, 0.025, 0.05, 0.1, float("inf")]
    bin_rows = {m: [{"lower": bin_edges[i], "upper": bin_edges[i + 1],
                     "examples": 0, "flips": 0} for i in range(len(bin_edges) - 1)]
                for m in by_mode}
    final_by_seed = {}
    for seed in sorted({k[0] for k in records}):
        if (seed, "fp32") not in records:
            continue
        _, ref_meta, ref_tensors = records[(seed, "fp32")]
        ref_cp = ref_tensors["checkpoints"]
        for mode in by_mode:
            if (seed, mode) not in records:
                continue
            run_name, meta, tensors = records[(seed, mode)]
            if (tensors["data_sha256"] != ref_tensors["data_sha256"] or
                    tensors["initial_weights_sha256"] != ref_tensors["initial_weights_sha256"]):
                raise SystemExit(f"unmatched inputs for seed {seed}, mode {mode}")
            common = sorted(set(ref_cp) & set(tensors["checkpoints"]), key=int)
            first_weight_drift = next((int(k) for k in common if not torch.equal(
                ref_cp[k]["weights"], tensors["checkpoints"][k]["weights"])), None)
            for checkpoint in common:
                a, b = ref_cp[checkpoint], tensors["checkpoints"][checkpoint]
                normal_flip = a["test_logits"].argmax(1) != b["test_logits"].argmax(1)
                stress_flip = a["stress_logits"].argmax(1) != b["stress_logits"].argmax(1)
                pair = {
                    "seed": seed, "mode": mode, "step": int(checkpoint),
                    "weights_l2_delta": (a["weights"].double() - b["weights"].double()).norm().item(),
                    "logits_rms_delta": (a["test_logits"].double() - b["test_logits"].double()).square().mean().sqrt().item(),
                    "test_prediction_flip_count": int(normal_flip.sum()),
                    "test_prediction_flip_rate": normal_flip.float().mean().item(),
                    "stress_prediction_flip_count": int(stress_flip.sum()),
                    "stress_prediction_flip_rate": stress_flip.float().mean().item(),
                    "fp32_test_accuracy": next(x["test_accuracy"] for x in ref_meta["metrics"] if x["step"] == int(checkpoint)),
                    "mode_test_accuracy": next(x["test_accuracy"] for x in meta["metrics"] if x["step"] == int(checkpoint)),
                    "fp32_stress_accuracy": next(x["stress_accuracy"] for x in ref_meta["metrics"] if x["step"] == int(checkpoint)),
                    "mode_stress_accuracy": next(x["stress_accuracy"] for x in meta["metrics"] if x["step"] == int(checkpoint)),
                    "first_weight_drift_checkpoint": first_weight_drift,
                }
                pairs.append(pair)
                if int(checkpoint) == max(map(int, common)):
                    by_mode[mode].append(pair)
                    final_by_seed.setdefault(str(seed), {})[mode] = pair
                    # Flips versus strict-FP32 reference predictions, grouped by FP32 logit margin.
                    margin = (a["stress_logits"][:, 0] - a["stress_logits"][:, 1]).abs()
                    indices = [((margin >= lo) & (margin < hi)) for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]
                    for row, mask in zip(bin_rows[mode], indices):
                        row["examples"] += int(mask.sum())
                        row["flips"] += int(stress_flip[mask].sum())

    summaries = {}
    for mode, selected in by_mode.items():
        summaries[mode] = {
            "seed_count": len(selected),
            "median_weight_l2_delta": median([p["weights_l2_delta"] for p in selected]),
            "median_test_logit_rms_delta": median([p["logits_rms_delta"] for p in selected]),
            "median_test_prediction_flip_rate": median([p["test_prediction_flip_rate"] for p in selected]),
            "median_stress_prediction_flip_rate": median([p["stress_prediction_flip_rate"] for p in selected]),
            "total_stress_prediction_flips": sum(p["stress_prediction_flip_count"] for p in selected),
            "seed_level_final": selected,
            "stress_flip_rate_by_fp32_margin": [
                {**row, "flip_rate": row["flips"] / row["examples"] if row["examples"] else None}
                for row in bin_rows[mode]
            ],
        }
    result = {
        "design": "paired H100 training across strict FP32, TF32, and BF16 autocast",
        "seed_count": len(final_by_seed), "pair_count_all_checkpoints": len(pairs),
        "warning": "One GPU model and one synthetic task. Margin bins are descriptive, not independent statistical samples.",
        "summaries": summaries, "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"seed_count": result["seed_count"], "summaries": {
        k: {m: v for m, v in val.items() if m != "seed_level_final" and m != "stress_flip_rate_by_fp32_margin"}
        for k, val in summaries.items()
    }, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
