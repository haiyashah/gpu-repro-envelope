#!/usr/bin/env python3
"""Paired numerical-mode training trial for the Precision Frontier study.

Each seed has fixed CPU-generated training, ordinary test, and radial stress
inputs. Run the same seed under strict FP32, TF32 matmul, and BF16 autocast.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn


def digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def environment() -> dict:
    p = torch.cuda.get_device_properties(0)
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        driver = None
    return {
        "gpu": p.name, "uuid": str(getattr(p, "uuid", None)),
        "compute_capability": f"{p.major}.{p.minor}", "driver": driver,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def make_data(seed: int, n_train: int = 4096, n_test: int = 2048):
    g = torch.Generator(device="cpu").manual_seed(seed + 100_003)

    def split(n: int):
        labels = torch.randint(0, 2, (n,), generator=g)
        angle = torch.rand(n, generator=g) * (2 * torch.pi)
        radius = torch.where(labels == 0, 0.55, 1.25) + 0.12 * torch.randn(n, generator=g)
        xy = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=1)
        xy += 0.035 * torch.randn((n, 2), generator=g)
        return xy.float(), labels.long()

    xtr, ytr = split(n_train)
    xte, yte = split(n_test)
    # A fixed radial decision stress set, densely sampling the class boundary.
    angles = torch.arange(128, dtype=torch.float32) * (2 * torch.pi / 128)
    radii = torch.linspace(0.65, 1.05, 401)
    rr, aa = torch.meshgrid(radii, angles, indexing="ij")
    stress = torch.stack((rr.flatten() * aa.flatten().cos(),
                          rr.flatten() * aa.flatten().sin()), dim=1)
    stress_labels = (rr.flatten() >= 0.9).long()
    return xtr, ytr, xte, yte, stress, stress_labels


class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.layers(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=("fp32", "tf32", "bf16"), required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; this experiment requires an NVIDIA GPU")

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = args.mode == "tf32"
    torch.backends.cudnn.allow_tf32 = args.mode == "tf32"

    data = make_data(args.seed)
    xtr_cpu, ytr_cpu, xte_cpu, yte_cpu, xstress_cpu, ystress_cpu = data
    data_hash = hashlib.sha256(b"".join(
        x.contiguous().numpy().tobytes() for x in data
    )).hexdigest()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 9_000_001)
        model = SmallMLP()
    initial = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()
    initial_hash = digest(initial)

    device = torch.device("cuda:0")
    model = model.to(device)
    xtr, ytr, xte, yte, xstress, ystress = (
        x.to(device) for x in (xtr_cpu, ytr_cpu, xte_cpu, yte_cpu, xstress_cpu, ystress_cpu)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    order_gen = torch.Generator(device="cpu").manual_seed(args.seed + 20_000_033)
    batches = []
    while len(batches) < args.steps:
        perm = torch.randperm(len(xtr_cpu), generator=order_gen)
        batches.extend(perm[i:i + 128] for i in range(0, len(xtr_cpu), 128))
    batches = batches[:args.steps]

    checkpoints, metrics = {}, []
    start = time.perf_counter()
    for step in range(args.steps + 1):
        if step == 0 or step % args.checkpoint_every == 0 or step == args.steps:
            torch.cuda.synchronize()
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.mode == "bf16"):
                    test_logits = model(xte)
                    stress_logits = model(xstress)
                test_logits = test_logits.float()
                stress_logits = stress_logits.float()
                accuracy = (test_logits.argmax(1) == yte).float().mean().item()
                loss = loss_fn(test_logits, yte).item()
                stress_accuracy = (stress_logits.argmax(1) == ystress).float().mean().item()
                flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).cpu().clone()
                checkpoints[str(step)] = {
                    "weights": flat,
                    "test_logits": test_logits.cpu().clone(),
                    "stress_logits": stress_logits.cpu().clone(),
                }
                metrics.append({
                    "step": step, "test_loss": loss, "test_accuracy": accuracy,
                    "stress_accuracy": stress_accuracy, "weights_sha256": digest(flat),
                })
        if step == args.steps:
            break
        idx = batches[step].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.mode == "bf16"):
            logits = model(xtr.index_select(0, idx))
        loss = loss_fn(logits.float(), ytr.index_select(0, idx))
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tensor_path = args.output.with_suffix(".pt")
    torch.save({"checkpoints": checkpoints, "data_sha256": data_hash,
                "initial_weights_sha256": initial_hash}, tensor_path)
    record = {
        "schema_version": 2, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "host": {"platform": platform.platform(), "python": sys.version},
        "environment": environment(),
        "procedure": {
            "task": "synthetic_concentric_rings", "model": "MLP(2,64,64,2)-GELU",
            "optimizer": "AdamW", "learning_rate": 0.003, "weight_decay": 0.0001,
            "mode": args.mode, "seed": args.seed, "steps": args.steps,
            "batch_size": 128, "checkpoint_every": args.checkpoint_every,
            "train_examples": len(xtr_cpu), "test_examples": len(xte_cpu),
            "stress_examples": len(xstress_cpu), "stress_boundary_radius": 0.9,
            "deterministic_algorithms": True,
        },
        "inputs": {"dataset_sha256": data_hash, "initial_weights_sha256": initial_hash},
        "metrics": metrics, "elapsed_seconds": elapsed, "weights_file": tensor_path.name,
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "weights": str(tensor_path),
                      "gpu": record["environment"]["gpu"], "mode": args.mode,
                      "elapsed_seconds": elapsed, "final": metrics[-1]}, indent=2))


if __name__ == "__main__":
    main()
