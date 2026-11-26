# A40 pilot: same-device repeatability

## Result

Three fresh processes ran the same seed-7 FP32 training workload for 500 steps on one NVIDIA A40. The runs used the same fixed CPU-generated synthetic dataset, initialization, minibatch order, deterministic PyTorch algorithms, and disabled TF32.

| Run | Steps | Test accuracy | Test loss | Final parameter SHA-256 |
|---|---:|---:|---:|---|
| seed7-r1 | 500 | 0.9931640625 | 0.014963345602154732 | 736cdf084fb631d8d53b312b6149eaef7a00e586f9f381ca707e277f22ab7eb8 |
| seed7-r2 | 500 | 0.9931640625 | 0.014963345602154732 | 736cdf084fb631d8d53b312b6149eaef7a00e586f9f381ca707e277f22ab7eb8 |
| seed7-r3 | 500 | 0.9931640625 | 0.014963345602154732 | 736cdf084fb631d8d53b312b6149eaef7a00e586f9f381ca707e277f22ab7eb8 |

The 2,048-example test set produced 2,034 correct classifications (99.3164%). The recorded final parameter hashes were identical across the three processes.

## Interpretation

This is evidence of bitwise repeatability for this one model, seed, device, software stack, and procedure. It does not test another GPU architecture, another seed, another model, mixed precision, distributed training, or a production workload. It is not evidence that cross-GPU training changes model behavior.

## Environment and method

The reported A40 environment used PyTorch 2.8.0 and CUDA 12.8.1 with NVIDIA driver 595.112.07. The run fixed the seed and all generated inputs, enabled deterministic algorithms, disabled TF32, and used FP32. The synthetic data were generated locally by the benchmark; the experiment downloaded no dataset.

The benchmark source and planned cross-device protocol are in [gpu-repro-envelope-source.zip](gpu-repro-envelope-source.zip). Raw JSON run records and checkpoints are not included in this repository yet.

## Next experiment

Repeat the same set of seeds on a second GPU architecture under the same software stack and compare matched checkpoints. Report parameter distance, logit RMS, prediction disagreement, and task metrics against the same-device repeat baseline.
