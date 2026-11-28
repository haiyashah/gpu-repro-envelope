#!/usr/bin/env python3
"""Compare two train_path.py trials and their saved checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_weights(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        return torch.load(path, map_location="cpu")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline_json", type=Path)
    p.add_argument("comparison_json", type=Path)
    p.add_argument("--baseline-weights", type=Path)
    p.add_argument("--comparison-weights", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    a = json.loads(args.baseline_json.read_text())
    b = json.loads(args.comparison_json.read_text())
    ap = args.baseline_weights or args.baseline_json.with_name(a["weights_file"])
    bp = args.comparison_weights or args.comparison_json.with_name(b["weights_file"])
    wa, wb = load_weights(ap), load_weights(bp)
    if wa["data_sha256"] != wb["data_sha256"]:
        raise SystemExit("dataset hashes differ; these runs are not a valid paired comparison")
    if wa["initial_weights_sha256"] != wb["initial_weights_sha256"]:
        raise SystemExit("initial weight hashes differ; these runs are not a valid paired comparison")
    ca, cb = wa["checkpoints"], wb["checkpoints"]
    common = sorted(set(ca) & set(cb), key=int)
    rows = []
    first_nonidentical = None
    for key in common:
        x, y = ca[key]["weights"].double(), cb[key]["weights"].double()
        lx, ly = ca[key]["logits"].double(), cb[key]["logits"].double()
        diff = x - y
        ldiff = lx - ly
        if first_nonidentical is None and torch.count_nonzero(diff).item():
            first_nonidentical = int(key)
        rows.append({
            "step": int(key),
            "weights_bitwise_equal": not bool(torch.count_nonzero(diff).item()),
            "weights_l2_delta": diff.norm().item(),
            "weights_relative_l2_delta": (diff.norm() / x.norm().clamp_min(1e-30)).item(),
            "weights_max_abs_delta": diff.abs().max().item(),
            "logits_rms_delta": ldiff.square().mean().sqrt().item(),
            "prediction_disagreement": (lx.argmax(1) != ly.argmax(1)).double().mean().item(),
            "baseline_accuracy": next(m["test_accuracy"] for m in a["metrics"] if m["step"] == int(key)),
            "comparison_accuracy": next(m["test_accuracy"] for m in b["metrics"] if m["step"] == int(key)),
        })
    result = {
        "baseline": {"gpu": a["environment"]["gpu"], "driver": a["environment"]["driver"],
                     "torch": a["environment"]["torch"], "torch_cuda": a["environment"]["torch_cuda"]},
        "comparison": {"gpu": b["environment"]["gpu"], "driver": b["environment"]["driver"],
                       "torch": b["environment"]["torch"], "torch_cuda": b["environment"]["torch_cuda"]},
        "same_data": True,
        "same_initial_weights": True,
        "first_nonidentical_checkpoint": first_nonidentical,
        "checkpoints": rows,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
