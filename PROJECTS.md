# Project index

This repository is a collection of experiments, not one finished breakthrough. Each entry states what was actually run and what remains unverified.

## GPU precision and reproducibility

| Project | What is present | Evidence boundary |
|---|---|---|
| H100 precision frontier | FP32, TF32, and BF16 training comparison on a synthetic classification task | 12 seeds, one H100, constructed boundary-heavy test slice. Descriptive pilot only. |
| Qwen precision fallback | Selective FP32 vocabulary-head reranking pilot on Qwen2.5-0.5B | One H100, short prompts, 12 continuations, one short timing run. Token agreement improved in the sample; exact continuation agreement did not, and no speedup was shown. |
| A40 repeatability | Three same-seed deterministic FP32 training runs | One seed, one GPU, one synthetic task. No cross-GPU comparison. Raw checkpoints are not in this repository. |
| Grammar-conditioned precision | Experimental protocol comparing full-vocabulary and grammar-legal token margins | Protocol only. No result has been run. |

## Inference cache correctness

| Project | What is present | Evidence boundary |
|---|---|---|
| KV page ownership | CPU state-space model and fail-closed reference allocator | Bounded model only. No GPU, RDMA, or serving-engine integration. The general risk and related fixes are already public. |
| Cache partition conformance | Framework-neutral test harness for cache identity dimensions | CPU tests and synthetic fixtures. It is not wired into a live serving engine. |
| Multimodal cache identity guard | HMAC-based tenant prefix salts and content-bound media IDs | Nine focused CPU tests; same-length, different-content collision is rejected. The 8 MiB hashing microbenchmark measured 4.93 to 5.15 ms across two runs. No vLLM gateway integration or GPU serving test. |
| HiCache L3 prefetch reservation | CPU reference model for protecting completed prefetches until admission | The model is illustrative and overlaps an existing SGLang fix under review. It is not a new upstream fix. |

## Prefetch policy

The prefetch timing frontier is a synthetic CPU simulation that compares reactive loading, median-timed prefetch, and a cost-aware timing rule under three synthetic tool-wait distributions. It does not estimate production speedups. See [the report](experiments/prefetch_frontier/REPORT.md) and [the simulation](experiments/prefetch_frontier/prefetch_timing_frontier.py).

## Reproduction status

- The repository's 35 CPU unit tests pass with `python3 -m unittest discover -s tests -v`.
- The H100 summary tables are present, but raw run logs and large checkpoint tensors are not included.
- The A40 raw JSON and checkpoint files remain outside this repository.
- The multimodal identity guard and cache models have no live engine integration.
- No result here establishes company profitability, production latency gains, or a first-in-literature discovery.
