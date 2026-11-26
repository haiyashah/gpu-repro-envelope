# Same Seed, Different Trajectory?

## A controlled test of cross-GPU training reproducibility

The question is whether arithmetic differences across GPU architectures can change a training trajectory enough to change the model's decisions. This repository begins with the required control: repeated runs on one device.

The current pilot uses fixed data, initialization, minibatch order, optimizer, deterministic PyTorch settings, and FP32. Three independent NVIDIA A40 runs of the same seed-7 workload each trained for 500 steps. Their final parameter SHA-256 hashes matched exactly. Each reached 99.3164% accuracy (2,034/2,048) and 0.0149633 test loss.

This is a single-GPU pilot, not a cross-GPU finding. It shows bitwise repeatability in this one setup across three fresh processes. It does not show that A40 and H100 training diverge, and it does not establish a general property of GPU training.

The source code and experiment protocol are included in [gpu-repro-envelope-source.zip](gpu-repro-envelope-source.zip). Raw run records and checkpoints are not yet included in this repository.

## Why this control matters

A one-run-versus-one-run cross-device comparison confounds architecture effects with ordinary run-to-run variation. Same-device repeats give a baseline distribution. The next experiment is to hold software, data, initialization, and training order fixed while changing the GPU, then compare parameter distance, logit RMS, prediction disagreement, and task metrics.

## Reproduce

Unzip the source bundle, then run:

~~~bash
python src/fingerprint.py capture env.json
python bench/train_path.py --steps 500 --seed 7 --output runs/gpu-seed7-r1.json
python bench/aggregate.py runs --output runs/aggregate.json
~~~

The training script writes a JSON record and a .pt checkpoint with environment details and checkpoint metrics. It uses a fixed CPU-generated synthetic dataset and does not download data. Deterministic algorithms are enabled and TF32 is disabled.

## Research question

When algorithm-level nondeterminism is disabled, does moving a fixed training procedure across GPU architectures produce trajectory differences larger than same-device repeat variation, and do those differences change held-out decisions?

## Next steps

1. Run at least three same-device repetitions per seed.
2. Repeat the same seeds on another GPU architecture while preserving software versions and data/initialization hashes.
3. Record deterministic-operation failures rather than silently changing settings.
4. Compare weight distance and logit distance at checkpoints, alongside held-out accuracy and prediction agreement.
5. Run a seed sweep and compare cross-device differences with same-device variation.
6. Publish raw records, environment fingerprints, commands, exclusions, and analysis code.

A small pilot can establish feasibility, not general behavior. Divergent weights alone do not imply a meaningful change in model behavior.
