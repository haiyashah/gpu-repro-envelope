# Same Seed, Different Trajectory?

## A controlled test of cross-GPU training reproducibility

The surprising question is not whether two GPUs return bit-identical floating-point results. They often will not. It is whether tiny, deterministic arithmetic differences change the *learning trajectory* enough to alter the trained model's decisions.

This repository tests that claim with fixed data, initialization, minibatch order, optimizer, and deterministic PyTorch settings. The current experiments measure repeated runs and numerical-mode changes on individual GPUs. A cross-architecture comparison remains unmeasured, so the project does not yet estimate cross-GPU trajectory drift.

The scope is deliberately small: a synthetic nonlinear classification task and a compact MLP. The experiments isolate repeatability and numerical-mode effects from data-loader randomness, dropout, and distributed communication.

## Current status

This repository now contains two separate single-GPU pilots, not a cross-GPU result.

- On an A40, three fresh processes trained the same seed-7 FP32 workload for 500 steps. Their final parameter SHA-256 matched, and each reached 99.3164% test accuracy and 0.0149633 test loss. See [the A40 pilot record](PILOT_RESULTS.md).
- On an H100, 12 matched seeds were trained for 500 steps in FP32, TF32, and BF16. Mean ordinary test accuracy was 99.62% in all modes, while disagreements on an intentionally boundary-dense stress set were concentrated among low-FP32-margin examples. See the [H100 result report](H100_PRECISION_FRONTIER_REPORT.md), [machine-readable summary](H100_PRECISION_FRONTIER_SUMMARY.json), and [fixed protocol](experiments/PRECISION_FRONTIER_PROTOCOL.md).

The H100 result is descriptive for one synthetic task. Its stress set deliberately oversamples a constructed boundary, so its disagreement rates are not production error estimates. Existing work studies numerical sensitivity and precision-related output divergence; this repository does not claim a new phenomenon or an established literature gap.

## Capture an environment

```bash
python src/fingerprint.py capture env.json
```

The fingerprint records GPU/driver data from `nvidia-smi`, CUDA compiler information, selected package versions, container tooling, and a short allowlist of CUDA-related environment variables. It intentionally does not dump the full process environment, which can contain credentials.

Compare captures:

```bash
python src/fingerprint.py diff baseline.json replay.json
```

## Run a training trial

```bash
python bench/train_path.py --steps 500 --seed 7 --output runs/gpu-seed7-r1.json
python bench/aggregate.py runs --output runs/aggregate.json
```

The training script writes a JSON record and a `.pt` checkpoint file with environment details and checkpoint metrics. Each run has a content hash for its fixed CPU-generated dataset and initial weights. It does not download data. The aggregate script compares every matched pair and keeps same-GPU controls separate from cross-GPU pairs.

## Research question

When algorithm-level nondeterminism is disabled, does moving a fixed training procedure across GPU architectures produce trajectory differences larger than same-device repeat noise, and do those differences change held-out decisions?

### Pilot protocol

1. Run at least three same-device repetitions per seed to estimate execution noise.
2. Repeat the same seeds on each available GPU architecture, preserving software versions and the JSON dataset/initialization hashes.
3. Enable deterministic algorithms, disable TF32, and fix minibatch order. Record any unsupported deterministic operation as a failed condition.
4. Compare parameter distance and logit distance at checkpoints, along with held-out accuracy and prediction agreement.
5. Run a small seed sweep. Compare cross-device differences with the empirical distribution from same-device repeats; do not infer general behavior from one seed.
6. Publish all raw records, environment fingerprints, commands, exclusions, and analysis code.

### Falsifiable outcomes

- The effect is weak in this setup if cross-device trajectory/decision differences remain within same-device repeat variation across the seed sweep.
- The effect is supported in this setup if cross-device differences exceed same-device variation consistently, especially if a stable training recipe changes held-out decisions.

## Scope limits

A small pilot can establish reproducible examples and estimate feasibility. It cannot establish how often GPU training results change in the wider literature. Divergent weights alone do not mean meaningful model behavior changed; prediction agreement and task metrics are reported separately.
  


## Additional pilot: pretrained LLM inference (Nov 27, 2026)

A separate H100 run measured Qwen2.5-0.5B on 48 WikiText examples and 12 short greedy prompts. Selective FP32 vocabulary-head recomputation at 17/384 token positions increased token agreement with FP32 from 62.24% to 74.48%, but exact continuation agreement remained 5/12 and the short timing showed no speedup. This is an exploratory result, not a production claim. See the [pilot report](experiments/H100_QWEN_PRECISION_PILOT.md), [aggregate results](experiments/H100_QWEN_PRECISION_RESULTS.json), and [benchmark script](llm_precision_audit.py).
