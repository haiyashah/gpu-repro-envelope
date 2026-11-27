# Precision Frontier: H100 Paired Pilot

## Result

On one NVIDIA H100 PCIe, 12 matched seeds trained for 500 steps in strict FP32, TF32, and BF16 autocast. The benchmark held generated data, initial weights, and minibatch order fixed within each seed. It saved per-checkpoint weights and logits, then compared each reduced-precision run with its seed-matched FP32 run.

The ordinary held-out set showed almost no visible change: mean accuracy was 99.62% in all three modes, and the median held-out prediction-flip rate was zero for both TF32 and BF16. TF32 produced 118 prediction flips across the 12 seed-specific radial stress sets; BF16 produced 2,241. These counts are aggregated across stress points and seeds, so the 615,936 point comparisons are not independent observations.

The strongest pattern appeared after grouping the stress examples by the strict-FP32 absolute logit margin:

| FP32 margin | TF32 disagreements | BF16 disagreements |
| --- | ---: | ---: |
| 0 to 0.005 | 104 / 406 (25.62%) | 206 / 406 (50.74%) |
| 0.005 to 0.01 | 12 / 338 (3.55%) | 167 / 338 (49.41%) |
| 0.01 to 0.025 | 2 / 1,083 (0.18%) | 435 / 1,083 (40.17%) |
| 0.025 to 0.05 | 0 / 1,907 | 574 / 1,907 (30.10%) |
| 0.05 to 0.1 | 0 / 3,684 | 571 / 3,684 (15.50%) |
| 0.1 and above | 0 / 608,518 | 288 / 608,518 (0.0473%) |

The median per-seed stress-set disagreement was 0.0185% for TF32 and 0.314% for BF16. The comparison also showed median parameter L2 deltas of 0.00555 for TF32 and 0.0341 for BF16, relative to the paired FP32 checkpoints.

## What this suggests

Aggregate accuracy can hide where numerical-mode changes affect decisions. In this controlled task, disagreements were concentrated among examples whose FP32 logits were close. A practical precision-change review should therefore report disagreement by reference-margin band, in addition to loss and aggregate accuracy, and should include a deliberately difficult boundary slice.

This is a measurement and audit pattern, not a demonstrated production mitigation. A follow-up on real workloads should test whether the same pattern persists, compare precision modes across multiple GPUs, and evaluate calibration or selective review on naturally occurring low-margin examples.

## Scope and limits

- This is one H100 model, one synthetic concentric-rings task, and 12 training seeds. It does not establish behavior on language models, production traffic, other GPUs, or naturally occurring data.
- The radial stress set deliberately oversamples the class boundary at radius 0.9. Its disagreement rates are diagnostic for this constructed boundary and must not be read as expected real-world error rates.
- The seed is the experimental unit. Stress points within a seed are correlated and are not independent samples for confidence intervals.
- Existing research already studies precision-related output changes, numerical sensitivity, and low-margin effects. This report makes no first-discovery or literature-gap claim.
- The H100 run compared training modes. It did not compare same-mode repeated runs on the H100, so it does not estimate the H100's own run-to-run nondeterminism.

## Reproduce

The scripts are in [`bench/precision_frontier.py`](bench/precision_frontier.py) and [`bench/analyze_precision_frontier.py`](bench/analyze_precision_frontier.py). The fixed protocol and literature boundary are in [`experiments/PRECISION_FRONTIER_PROTOCOL.md`](experiments/PRECISION_FRONTIER_PROTOCOL.md). Run the seed/mode loop from the repository root, then pass the output folder to the analyzer. Each JSON record includes the GPU, software versions, input hashes, and measured metrics; the paired `.pt` files contain checkpoints and logits used by the analyzer.

The compact machine-readable summary is in [`H100_PRECISION_FRONTIER_SUMMARY.json`](H100_PRECISION_FRONTIER_SUMMARY.json). The full per-run logs and larger per-checkpoint `.pt` tensors were generated on the H100 but are not included in this repository snapshot; the result report and summary preserve the aggregate evidence, while the public scripts let others rerun the experiment.
