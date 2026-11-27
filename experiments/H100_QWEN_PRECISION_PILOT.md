# H100 Qwen precision and selective-head pilot

## Question

How much do reduced-precision inference modes change next-token scores and short greedy continuations for a small pretrained language model? Can recomputing only low-margin vocabulary-head positions in FP32 recover any agreement?

This is a small reproducibility and platform-extension pilot. It is not a claim of a new phenomenon, a production speedup, or an unreported literature gap.

## Setup

- Model: `Qwen/Qwen2.5-0.5B`, loaded once as an FP32 reference; reduced-precision arms use copies of the same checkpoint cast to BF16.
- Text: 48 usable examples from the test split of `Salesforce/wikitext`, configuration `wikitext-2-raw-v1`.
- Scoring: next-token NLL and perplexity across 4,682 valid next-token labels, with up to 128 tokens per example.
- Generation: 12 prompts, greedy decoding, up to 32 new tokens per prompt. The actual generated total was 384 tokens per arm.
- Modes: FP32, TF32 matmul, BF16, and BF16 with FP32 recomputation of the vocabulary `lm_head` when the BF16 top-two logit margin was at most `tau`.
- Thresholds: `tau ∈ {0.02, 0.05, 0.1}` in raw logit units. The transformer body remains BF16 in selective runs; only selected vocabulary-head rows are recomputed.
- Runtime: NVIDIA H100 80GB HBM3 (compute capability 9.0), driver 580.196.07, CUDA 13.2, PyTorch 2.15.0+cu132, Transformers 5.24.1, Datasets 5.2.0. Attention implementation was eager. Deterministic algorithms were enabled.
- Run source SHA-256: `384a0b3fe16edbe6eaba076eb169b6945e846f9fe3ad67678d04f021f0d73e46`.

## Results

| Mode | Mean NLL | Perplexity | Greedy token agreement vs FP32 | Exact prompts vs FP32 | Measured decode rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 3.243501 | 25.6233 | reference | reference | 69.1 tok/s |
| TF32 | 3.243550 | 25.6245 | 100.0% | 12/12 | 71.9 tok/s |
| BF16 | 3.249405 | 25.7750 | 62.24% | 4/12 | 58.7 tok/s |
| BF16 + selective FP32 head, `tau=0.02` | scored in a separate pass | scored in a separate pass | 74.48% | 5/12 | 57.5 tok/s |
| BF16 + selective FP32 head, `tau=0.05` | scored in a separate pass | scored in a separate pass | 74.48% | 5/12 | 58.4 tok/s |
| BF16 + selective FP32 head, `tau=0.1` | scored in a separate pass | scored in a separate pass | 74.48% | 5/12 | 58.3 tok/s |

At `tau=0.02`, the fallback ran for 17 of 384 generated positions (4.43%). At `tau=0.1`, it ran for 23 of 384 (5.99%). All three thresholds happened to produce the same continuations on these 12 prompts. Selective recomputation gained 12.24 percentage points of token agreement over BF16 in this sample, but exact continuation agreement remained only 5/12 and measured decoding was slower than the plain BF16 arm.

TF32 and FP32 were close on next-token loss and produced the same short continuations for these prompts. The generation throughput numbers are noisy measurements from one short, serial, eager-attention run; they are not a throughput benchmark and should not be compared as a performance claim.

## Interpretation

This run demonstrates a concrete engineering tradeoff worth measuring: a small fraction of vocabulary-head recomputations improved token-level agreement on this sample, while failing to restore most whole-prompt continuations or produce a speed benefit. Once an autoregressive sequence diverges, later tokens are conditioned on different prefixes, so continuation agreement is a stricter and path-dependent metric. The experiment does not isolate that effect from numerical differences in the BF16 transformer body.

The practical follow-up is to test across more prompts and longer contexts, repeat the run, report uncertainty, compare faster decoding kernels and batch sizes, and measure exact task quality. A selective scheme would need a reliable calibration criterion and end-to-end latency savings before it could be recommended.

## Limits and reproducibility

This was 48 short WikiText examples and 12 short prompts on one H100, using one 0.5B model and one software stack. It is an exploratory pilot, not evidence that the result generalizes to other models, serving stacks, GPUs, or production requests. The BF16 perplexity difference is descriptive for this sample. The threshold is in raw logit units and may not transfer across models or tokenizers. The decode loop was serial, with no warmup protocol or repeated timing trials, so tokens/sec is especially provisional.

Re-run using the script at [`../llm_precision_audit.py`](../llm_precision_audit.py). Example command:

```bash
python llm_precision_audit.py \
  --output runs/h100_qwen_precision.json \
  --examples 48 --prompts 12 --context-tokens 128 --new-tokens 32 \
  --thresholds 0.02 0.05 0.1 --require-h100
```

The compact result record is [`H100_QWEN_PRECISION_RESULTS.json`](H100_QWEN_PRECISION_RESULTS.json). It contains the observed aggregate metrics and procedure, not the original per-token logits or full raw run file.
