# LinkedIn draft: A tiny precision fallback, measured end to end

Can a model spend FP32 compute only where BF16 is least certain?

I ran a small H100 pilot to find out.

I compared FP32, TF32, BF16, and a simple selective scheme on Qwen2.5-0.5B. The selective version kept the transformer body in BF16, then recomputed the vocabulary head in FP32 when the BF16 gap between the top two logits was at most a threshold.

On 12 short WikiText prompts, plain BF16 matched the FP32 greedy output token by token 62.24% of the time. With a threshold of 0.02, the FP32 head ran at 17 of 384 generated positions. Token agreement rose to 74.48%.

But the result has a catch: only 5 of 12 full continuations exactly matched FP32, and this implementation was slower than plain BF16 in the short timing run. A few corrected decisions do not undo the downstream effects of an already-diverged autoregressive prefix. The BF16 body also remained unchanged.

That is the useful finding for me: *a precision fallback can improve local token agreement without recovering sequence-level agreement or improving latency.* We need to measure both. A promising per-token metric is not enough to claim a serving optimization.

This is an exploratory pilot, not a production result. It used one 0.5B model, 48 WikiText examples for scoring, 12 prompts for generation, one H100, and one short timing per mode. The throughput numbers are not robust benchmarks, and the threshold may not transfer to another model.

I published the code, aggregate measurements, and limitations here:
https://github.com/haiyashah/gpu-repro-envelope

Next step: repeat across more prompts and models, longer sequences, batched serving, and warmed-up timing trials. Then test whether a calibrated fallback can improve task quality without adding end-to-end latency.

#LLMInference #AIInfrastructure #GPUComputing #MachineLearningSystems
