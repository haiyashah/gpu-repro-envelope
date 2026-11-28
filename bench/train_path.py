#!/usr/bin/env python3
"""Run a controlled FP32 training trial and save comparable checkpoints."""
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

# Must be set before CUDA initializes. It makes cuBLAS select a reproducible
# workspace mode where supported; deterministic algorithms are also enabled below.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn


def sha256_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def gpu_fingerprint() -> dict:
    p = torch.cuda.get_device_properties(0)
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        driver = None
    return {
        "gpu": p.name,
        "uuid": str(getattr(p, "uuid", None)),
        "compute_capability": f"{p.major}.{p.minor}",
        "driver": driver,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def make_data(seed: int, n_train: int, n_test: int) -> tuple[torch.Tensor, ...]:
    """CPU-only concentric-ring data, so all devices receive identical bytes."""
    g = torch.Generator(device="cpu").manual_seed(seed + 100_003)
    def split(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        labels = torch.randint(0, 2, (n,), generator=g)
        angle = torch.rand(n, generator=g) * (2 * torch.pi)
        radius = torch.where(labels == 0, 0.55, 1.25) + 0.12 * torch.randn(n, generator=g)
        xy = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=1)
        xy += 0.035 * torch.randn((n, 2), generator=g)
        return xy.float(), labels.long()
    return (*split(n_train), *split(n_test))


class SmallMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--output", type=Path, default=Path("runs/trial.json"))
    parser.add_argument("--weights", type=Path, default=None)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.checkpoint_every < 1:
        parser.error("steps, batch-size, and checkpoint-every must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; this experiment requires an NVIDIA GPU")

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    xtr, ytr, xte, yte = make_data(args.seed, 4096, 2048)
    data_hash = hashlib.sha256(b"".join(x.contiguous().numpy().tobytes() for x in (xtr, ytr, xte, yte))).hexdigest()
    # Initialize on CPU so the starting parameter bytes are identical on every GPU.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 9_000_001)
        model = SmallMLP()
    init_flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()
    init_hash = sha256_tensor(init_flat)
    device = torch.device("cuda:0")
    model = model.to(device)
    xtr, ytr, xte, yte = (v.to(device) for v in (xtr, ytr, xte, yte))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Precompute every minibatch index on CPU. The ordering is fixed across devices.
    order_gen = torch.Generator(device="cpu").manual_seed(args.seed + 20_000_033)
    batches: list[torch.Tensor] = []
    while len(batches) < args.steps:
        permutation = torch.randperm(len(xtr), generator=order_gen)
        batches.extend(permutation[i:i + args.batch_size] for i in range(0, len(xtr), args.batch_size))
    batches = batches[:args.steps]

    checkpoints: dict[str, dict[str, torch.Tensor]] = {}
    metrics: list[dict] = []
    start = time.perf_counter()
    for step in range(args.steps + 1):
        if step == 0 or step % args.checkpoint_every == 0 or step == args.steps:
            torch.cuda.synchronize()
            with torch.no_grad():
                logits = model(xte)
                accuracy = (logits.argmax(dim=1) == yte).float().mean().item()
                loss = loss_fn(logits, yte).item()
                flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).cpu().clone()
                cp_name = str(step)
                checkpoints[cp_name] = {"weights": flat, "logits": logits.detach().cpu().clone()}
                metrics.append({"step": step, "test_loss": loss, "test_accuracy": accuracy,
                                "weights_sha256": sha256_tensor(flat)})
        if step == args.steps:
            break
        idx = batches[step].to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(xtr.index_select(0, idx))
        loss = loss_fn(out, ytr.index_select(0, idx))
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    output = args.output
    weights_path = args.weights or output.with_suffix(".pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"checkpoints": checkpoints, "data_sha256": data_hash, "initial_weights_sha256": init_hash}, weights_path)
    record = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "host": {"platform": platform.platform(), "python": sys.version},
        "environment": gpu_fingerprint(),
        "procedure": {"task": "synthetic_concentric_rings", "model": "MLP(2,64,64,2)-GELU",
                      "optimizer": "AdamW", "learning_rate": 0.003, "weight_decay": 0.0001,
                      "dtype": "float32", "seed": args.seed, "steps": args.steps,
                      "batch_size": args.batch_size, "checkpoint_every": args.checkpoint_every,
                      "train_examples": len(xtr), "test_examples": len(xte),
                      "deterministic_algorithms": True, "tf32": False},
        "inputs": {"dataset_sha256": data_hash, "initial_weights_sha256": init_hash},
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "weights_file": weights_path.name,
    }
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"output": str(output), "weights": str(weights_path),
                      "environment": record["environment"], "final": metrics[-1]}, indent=2))


if __name__ == "__main__":
    main()
