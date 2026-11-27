# LinkedIn draft

**A precision change can leave benchmark accuracy untouched while changing exactly the decisions closest to the boundary.**

I tested this on an NVIDIA H100 with 12 matched training seeds. For each seed, I held the dataset, initial weights, and minibatch order fixed, then trained the same small classifier in strict FP32, TF32, and BF16 for 500 steps.

On the ordinary held-out set, mean accuracy was 99.62% in every mode. Median prediction disagreement with FP32 was zero.

Then I evaluated a deliberately boundary-dense stress set. BF16 disagreed with the paired FP32 model on 206 of 406 examples whose FP32 logit margin was below 0.005. For margins above 0.1, it disagreed on 288 of 608,518 examples. TF32 showed a smaller effect: 104 of 406 flips in the lowest-margin band, and none above 0.025.

The useful engineering point is not that BF16 is unsafe. This is one synthetic task, and the stress set intentionally over-samples a constructed boundary. These rates are not production error estimates.

The point is that aggregate accuracy can hide *where* numerical changes move decisions. For a precision change, I would now report prediction disagreement by reference-margin band and include a boundary-focused evaluation slice alongside accuracy and loss.

I built a small reproducible benchmark and an analyzer that checks matched seeds, input hashes, parameter drift, logit drift, and margin-conditioned flips. The experiment protocol and limitations are public here:

https://github.com/haiyashah/gpu-repro-envelope

Next I want to run the same audit on a real model and compare GPU architectures. The current result is a pilot, not a claim that this behavior generalizes.

#MachineLearningSystems #GPUComputing #AIInfrastructure #Reproducibility
