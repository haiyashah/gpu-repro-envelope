#!/usr/bin/env python3
"""Measure CPU cost of content-bound media IDs for synthetic payload sizes."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mm_cache_guard import derive_media_cache_id


def benchmark(sizes: list[int], repeats: int) -> dict:
    secret = b"benchmark-secret-32-bytes-long!!!"
    rows = []
    for size in sizes:
        payload = bytes((i * 131 + 17) & 0xFF for i in range(size))
        samples_ms = []
        for _ in range(repeats):
            start = time.perf_counter()
            derive_media_cache_id(
                secret=secret,
                tenant_id="tenant-bench",
                model_revision="model-r1",
                processor_revision="processor-r1",
                modality="image",
                payload=payload,
                client_media_id="same-client-id",
            )
            samples_ms.append((time.perf_counter() - start) * 1000)
        median_ms = statistics.median(samples_ms)
        rows.append({
            "payload_bytes": size,
            "median_ms": round(median_ms, 4),
            "p95_ms": round(sorted(samples_ms)[min(len(samples_ms) - 1, int(0.95 * len(samples_ms)))], 4),
            "effective_mib_per_second": round((size / (1024 * 1024)) / (median_ms / 1000), 2),
            "repeats": repeats,
        })
    return {"algorithm": "HMAC-SHA256, one in-memory payload pass", "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1024, 65536, 1048576, 8388608])
    parser.add_argument("--repeats", type=int, default=25)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.sizes, args.repeats), indent=2))


if __name__ == "__main__":
    main()
