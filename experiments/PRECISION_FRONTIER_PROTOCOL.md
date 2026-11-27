# Precision Frontier: preregistered pilot protocol

## Question

On one H100, how do strict FP32, TF32 matrix multiplication, and BF16 autocast change the learned trajectory under otherwise fixed training? Do prediction changes concentrate among examples with small strict-FP32 logit margins, including on a deliberately dense decision-boundary stress set?

This is a controlled single-GPU numerical-mode experiment. It does **not** test cross-GPU behavior, LLM pretraining, or a production workload. The active H100 literature already studies small perturbations, optimizer-state sensitivity, precision effects, and margin-dependent output changes. This pilot is a transparent paired measurement and an auditable prototype for a precision-change review, not a claim that these phenomena are new.

## Fixed protocol

- One NVIDIA H100 80 GB instance, with the actual GPU, driver, PyTorch, CUDA, and cuDNN versions recorded in every run.
- Twelve independent CPU seeds; for each seed, one run in each of strict FP32, TF32 matmul, and BF16 autocast.
- A CPU-generated concentric-rings classification task; identical dataset bytes, initial weights, and precomputed minibatch order for all three modes within each seed.
- `MLP(2,64,64,2)-GELU`, AdamW, batch size 128, 500 steps, fixed learning rate and weight decay.
- PyTorch deterministic algorithms enabled, cuDNN benchmarking disabled, and cuBLAS workspace configuration recorded.
- One ordinary held-out set plus a deterministic radial stress set spanning the class boundary at radius 0.9. The stress-set labels are defined by that threshold. It is diagnostic, not a natural-data generalization set.
- Save per-run metadata, input hashes, model checkpoints and logits every 100 steps. Compare only runs with matching seed, input hash, and initial-weight hash.

## Outcomes

Primary: final prediction disagreement versus strict FP32, reported separately on ordinary test examples and the fixed boundary stress set.

Secondary: first checkpoint with parameter drift, parameter L2 distance, logit RMS distance, task accuracy, and precision-mode run time. For the stress set, report disagreement by strict-FP32 absolute logit-margin bins. Seeds, not individual stress points, are the experimental units; summaries are descriptive.

## Decision boundary

If prediction changes are rare and confined to small-margin examples, the practical output is a reproducible audit recipe: report decision disagreement and margin-conditioned risk alongside accuracy and loss when changing numerical modes. If the modes change predictions broadly, this pilot identifies a stronger sensitivity signal for follow-up on a real workload. If they do not change predictions, the negative result remains useful only for this task and setup.

## Cost and stopping

The selected instance was listed at $3/hour and did not support stop/restart. The fixed sweep was run within the existing $30 credit balance, with no payment method connected. The instance is deleted after the run.

## Literature boundary

Relevant prior work includes the 2026 preprint [Hidden Sensitivity in LLM Pre-training: A Dynamic System View](https://arxiv.org/abs/2611.28830), which perturbs weights and Adam state and reports low-margin prediction changes; the 2026 preprint [Greedy Decoding Is Not Precision-Invariant](https://arxiv.org/abs/2609.26621), which studies precision-dependent LLM output divergence and margin-triggered recomputation; and the [PyTorch numerical-accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html), which describe precision-related numerical tradeoffs. This experiment does not establish a literature gap. It is a small controlled measurement that needs broader review and real-workload validation.
