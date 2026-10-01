---
title: "One launch: a fused AdamW kernel for GPT-2 in CUDA"
collection: publications
date: 2026-10-02
permalink: /research/fused-adamw-cuda/
excerpt: "One CUDA kernel that reads p, g, m, v once and writes p, m, v once: on a Tesla T4 it updates GPT-2 small's 124M parameters in 14.7 ms, 3.5x faster per value than the flat-buffer AdamW and 5–10% faster than PyTorch's fused AdamW timed the same way, within 1e-5 of PyTorch in every test."
read_time: true
tags:
  - optimization
  - adamw
  - gpu
  - gpt-2
  - cuda
---

**Abstract.** The [previous post](/research/flat-buffers-adamw/) ended with a gap: a flat-buffer
AdamW in C# had removed the per-parameter loop, but it was memory-bound at 23 passes over memory per
step, and PyTorch's fused kernel, at 7 passes, stayed 1.9x faster. Closing that gap requires fusing
the operations themselves. I write AdamW, with gradient clipping and weight decay, as one CUDA
kernel that reads each parameter, gradient and moment once and writes each once, over one buffer
that holds the whole model. Every test against PyTorch's own fused AdamW agrees within $$10^{-5}$$
of the largest value (at most $$2.1 \times 10^{-6}$$ observed). On a Tesla T4, on the same
8,949,504 values as the previous post, **one update takes 1.043 ms: 3.5x faster than the flat-buffer
AdamW (3.64 ms)** and **1.10x faster than PyTorch's fused AdamW timed the same way (1.147 ms)**. At
GPT-2 small's full 124,439,808 values it takes 14.65 ms against PyTorch's 15.39 ms, sustaining an
estimated 238–240 GB/s, about 75% of the T4's peak.

* TOC
{:toc}

## 1. Introduction

I am pre-training a GPT-2 small [3] from scratch. The project began in C# on TorchSharp, and is now
being rebuilt in C and CUDA in the style of Karpathy's llm.c [6]: one GPU allocation per kind of
state, hand-written kernels, and tests against PyTorch for every piece. The code is open source:
[github.com/ducbachsong/gpt2-small](https://github.com/ducbachsong/gpt2-small/tree/c-cuda-trainer)
(branch `c-cuda-trainer`; every link in this post is pinned to commit
[`f336dc8`](https://github.com/ducbachsong/gpt2-small/tree/f336dc8)).

The optimiser is the first piece. The previous post [4] measured why a fused kernel matters: once the
per-parameter loop is gone, AdamW is limited by the bytes it moves, and only fusion moves fewer.
The C# branch had a fused kernel too, compiled at run time through NVRTC, but it was never timed
on a GPU. This post builds, tests and times one.

**Contributions.**

1. An AdamW kernel that does clipping, decoupled weight decay, both moments, bias correction and the
   update in one pass, four values per thread (Section 3.1).
2. A layout for every piece of optimiser state in one GPU allocation, split into named groups, with
   weight decay chosen by position rather than by mask (Sections 3.2–3.3), and the alignment rule
   the layout must keep, found the hard way (Section 4.3).
3. Tests against PyTorch's fused AdamW (`at::_fused_adamw_`, the operation behind Python's
   `fused=True`), plus the C# trainer's hand-worked cases (Section 5).
4. Timings at two sizes on a Tesla T4, one of them identical to the previous post's benchmark, with
   a memory-traffic model that accounts for both (Section 6).

## 2. Background

### 2.1 AdamW, with clipping

For a parameter $$p$$ with gradient $$g$$ at step $$t$$, learning rate $$\eta$$, weight decay
$$\lambda$$ and clipping threshold $$c$$, the step is [1, 2]

$$
\begin{aligned}
g &\leftarrow g \cdot \min\!\left(1, \frac{c}{\lVert g \rVert + 10^{-6}}\right) && \text{clipping by the global norm} \\
p &\leftarrow p \,(1 - \eta \lambda) && \text{decoupled weight decay (matrices only)} \\
m &\leftarrow \beta_1 m + (1 - \beta_1)\, g && \text{first moment} \\
v &\leftarrow \beta_2 v + (1 - \beta_2)\, g^2 && \text{second moment} \\
p &\leftarrow p - \frac{\eta}{1 - \beta_1^t} \cdot \frac{m}{\sqrt{v} / \sqrt{1 - \beta_2^t} + \epsilon} && \text{bias-corrected Adam step}
\end{aligned}
$$

Every line is element-wise except one number: $$\lVert g \rVert$$, the norm of every gradient in
the model. Everything else about value $$i$$ depends only on $$p_i, g_i, m_i, v_i$$ and a few
scalars that are the same for all values.

### 2.2 What fusion removes

In the previous post's flat-buffer AdamW, each line above was one or more tensor operations, each a
kernel that reads whole buffers from GPU memory and writes whole buffers back:

| implementation | kernels per step | passes over memory | bytes per value |
|---|---:|---:|---:|
| flat buffers, C# (previous post) | 10 | 23 (13 reads, 10 writes) | 92 |
| **one fused kernel (this post)** | **1** | **7 (4 reads, 3 writes)** | **28** |

*Table 1. One pass is one read or one write of a whole buffer of fp32 values. The flat-buffer
step also zeroed the gradients, one of its 10 writes; this kernel leaves them for the trainer.*

The previous post measured the flat-buffer step at an estimated 226 GB/s, already about 70% of the
T4's peak. It was not slow because of launches; it moved 23 / 7 ≈ 3.3x more bytes than a fused
kernel needs. Fusion should therefore win by roughly that factor, and no more.

## 3. Method

### 3.1 The kernel

Each thread updates four consecutive values. It loads $$p, g, m, v$$ for them as four `float4`s
(16 bytes each, which must start on a multiple of 16 bytes [7]), runs the five lines of Section 2.1 in registers, and stores $$p, m, v$$ back:
4 reads and 3 writes of 16 bytes, the 7 passes of Table 1. The gradients are only read, so they
are unchanged after the step.

The scalars that are the same for every value are worked out once on the CPU before the launch, in
single precision: the step size $$\eta / (1 - \beta_1^t)$$, the reciprocal $$1 / \sqrt{1 - \beta_2^t}$$
and the decay factor $$1 - \eta\lambda$$. The clip factor needs $$\lVert g \rVert$$, which is on the
GPU; every thread reads it (one float, served from cache) and computes the factor itself, so the CPU
never waits for the norm.

The norm itself cannot be computed inside the same kernel in one pass: every value's update needs
the sum over all 124M squared gradients, and the blocks of an ordinary launch cannot wait for one
another. A norm kernel runs first, and passes its result as that one float. In the tests of this
post the norm is computed on the CPU; the trainer will use a kernel.

### 3.2 One allocation, in groups

Every piece of state lives in one `TensorBuffer`: one `cudaMalloc`, all zeros at the start, with
each tensor a view of its part. The buffer records which tensors were added together as a
*group*, and gives each group back as one flat tensor, so one launch covers a whole group:

```
buffer   [ params: K tensors | grads: K | m: K | v: K | grad_norm: 1 value ]
           group 0             group 1    group 2 group 3 group 4
```

The model reads `param(k)` and backward writes `grad(k)`; both are tensor $$k$$ of their group,
with its own shape. The update calls the kernel once with the start of each group and the group's
length: 124,439,808 for GPT-2 small, which with four values per thread and 256 threads per block
is 121,523 blocks.

### 3.3 Weight decay by position

Weight decay applies to matrices (rank ≥ 2) and not to biases or LayerNorm parameters. As in the
previous post, the matrices come first in every group, so the decayed values are one range,
$$[0, \text{decayed\_count})$$, and the kernel decides per thread with one comparison, no mask. Every
tensor size in GPT-2 is a multiple of 4, so a thread's four values are always on the same side of
the border. The optimiser computes `decayed_count` from the shapes when it is built, and stops with
an error if a matrix comes after a 1-D tensor, since the kernel would silently decay the wrong
values.

## 4. Implementation

The code is CUDA C++17, built with `nvcc -O3 -arch=sm_75`. Comments are abridged in the listings;
the linked files have them in full.

**Listing 1.** The kernel —
[`llmc/kernels/adamw.cuh`](https://github.com/ducbachsong/gpt2-small/blob/f336dc8/llmc/kernels/adamw.cuh)

```cpp
__device__ __forceinline__ float adamw_value(float p, float g, float* m, float* v, float clip,
                                             float decay_factor, float beta1, float beta2, float eps,
                                             float step_size, float inverse_sqrt_bias_correction2) {
    g = g * clip;
    p = p * decay_factor;
    *m = beta1 * *m + (1.0f - beta1) * g;
    *v = beta2 * *v + (1.0f - beta2) * g * g;
    float denominator = sqrtf(*v) * inverse_sqrt_bias_correction2 + eps;
    return p - step_size * *m / denominator;
}

__global__ void adamw_kernel(float* params, const float* grads, float* m_memory, float* v_memory,
                             size_t n, size_t decayed_count, const float* grad_norm, float max_norm,
                             float weight_decay_factor, float beta1, float beta2, float eps,
                             float step_size, float inverse_sqrt_bias_correction2) {
    size_t group = (size_t)blockIdx.x * blockDim.x + threadIdx.x;   // values 4 group .. 4 group + 3
    if (group * 4 >= n) return;
    float clip = 1.0f;
    if (max_norm > 0.0f) {
        clip = max_norm / (*grad_norm + 1e-6f);
        if (clip > 1.0f) clip = 1.0f;
    }
    float decay_factor = group * 4 < decayed_count ? weight_decay_factor : 1.0f;
    float4 p = reinterpret_cast<float4*>(params)[group];
    float4 g = reinterpret_cast<const float4*>(grads)[group];
    float4 m = reinterpret_cast<float4*>(m_memory)[group];
    float4 v = reinterpret_cast<float4*>(v_memory)[group];
    p.x = adamw_value(p.x, g.x, &m.x, &v.x, clip, decay_factor, beta1, beta2, eps, step_size,
                      inverse_sqrt_bias_correction2);
    // ... the same for p.y, p.z, p.w
    reinterpret_cast<float4*>(params)[group] = p;
    reinterpret_cast<float4*>(m_memory)[group] = m;
    reinterpret_cast<float4*>(v_memory)[group] = v;
}
```

**Listing 2.** The optimiser —
[`llmc/optimizer.cuh`](https://github.com/ducbachsong/gpt2-small/blob/f336dc8/llmc/optimizer.cuh)

```cpp
inline void Optimizer::step() {
    steps_taken_++;                 // t: counted here, so no algorithm can forget it
    update(steps_taken_);
}

inline void AdamW::update(int t) {
    // Each group as one flat block: all K tensors, one after the other.
    const Tensor& all_params = buffer_.group(params_group_index_);
    const Tensor& all_grads = buffer_.group(grads_group_index_);
    const Tensor& all_m = buffer_.group(m_group_index_);
    const Tensor& all_v = buffer_.group(v_group_index_);
    adamw_update(all_params.data(), all_grads.data(), all_m.data(), all_v.data(), all_params.numel(),
                 decayed_count_, grad_norm().data(), config_.max_norm, config_.learning_rate,
                 config_.beta1, config_.beta2, config_.eps, config_.weight_decay, t);
}
```

### 4.1 Optimizer and algorithm

`Optimizer` holds what every algorithm needs: the buffer, the params and grads groups, the norm and
the step count. `AdamW` builds on it and adds only its own state (the m and v groups) and its own
`update(t)`. Another algorithm, SGD with momentum or Lion, would be one more class of the same
shape, adding its groups before the single allocation.

### 4.2 The buffer's groups

[`llmc/tensor_buffer.cuh`](https://github.com/ducbachsong/gpt2-small/blob/f336dc8/llmc/tensor_buffer.cuh)
gained the group bookkeeping: `add_many(shapes)` adds tensors as a group and returns its index,
`group(g)` is the group as one flat tensor, `tensor(g, k)` is tensor $$k$$ of group $$g$$ and stops
the program if $$k$$ is past the group's end (where it would silently be the next group's tensor).
Keeping the bookkeeping in the buffer removed every index calculation from the optimiser.

### 4.3 An alignment bug

The first version of the layout put `grad_norm` third, right after the gradients, because that is
where it belongs logically. The first run on the GPU stopped with
`CUDA error: misaligned address`, reported inside a PyTorch copy several calls later, far from the
cause.

![Where grad_norm goes in the buffer](/images/fused-adamw-cuda/fig1-layout.svg)
*Figure 1. A `float4` load must start on a multiple of 16 bytes. With `grad_norm`, a single float,
in the middle of the buffer, every group after it starts 4 bytes off: m at byte 36 and v at byte 52
in the smallest test. Moved to the end, every group the kernel reads starts on 16 bytes.*

The fix has three parts: `Optimizer::allocate()` adds `grad_norm` after the algorithm's own groups,
so it is always last; `adamw_update` checks that its four pointers start on 16 bytes and stops with
a message that names them; and a test checks the alignment of m and v directly. GPU errors are
reported at the next call that waits for the GPU, so a check at the point of launch is worth more
than it costs.

## 5. Correctness evaluation

### 5.1 The reference

The answer key is PyTorch's fused AdamW on the same GPU. libtorch's C++ `torch::optim::AdamW` has no
`fused` option, so the test calls the operation that Python's `fused=True` [5] calls, the way
`torch/optim/adam.py` does: per param group, add 1 to the step counts (`at::_foreach_add_`), then
`at::_fused_adamw_`. PyTorch's side is written out in each test: its tensors, its m, v and step
counts for each param group (matrices with weight decay, the rest without), `clip_grad_norm_` when
clipping is on, and the gradients assigned to `.grad`, the C++ form of Python's `w.grad = g`.

### 5.2 Criterion

Each comparison runs both optimisers on the same starting values and the same gradients and, after
every step, requires the largest difference to be at most $$10^{-5}$$ of the largest value. Each
also checks that PyTorch's values moved by more than $$10^{-3}$$ from where they started: two sides
that never moved would agree perfectly and prove nothing.

The previous post required bit-for-bit equality, and this one does not. That was possible because
the flat-buffer AdamW issued the same operations as PyTorch, in the same order. A fused kernel
cannot: here the bias corrections are computed once on the CPU in single precision, and the
compiler fuses multiply-adds, so the last bits differ from any other implementation's. The question
becomes how large the difference is and whether it grows.

| test | what it establishes | largest difference |
|---|---|---:|
| one step, 2x2 weight | a single update | equal at 6 digits |
| 10 steps, 2x2 weight, a new gradient each step | $$m$$, $$v$$ and bias correction over time | $$5.1 \times 10^{-8}$$ |
| 256x256 random weight, 100 steps, every 7th gradient $$\times 10^{-6}$$ | rounding at scale; $$\epsilon$$ with tiny gradients | $$2.1 \times 10^{-6}$$ |
| bias, one step | no decay on 1-D tensors (PyTorch's group with decay 0) | equal at 6 digits |
| 2-D, 3-D and 1-D tensors, 20 steps | the decay border, two param groups | $$4.5 \times 10^{-7}$$ |
| clipping, gradient norm 5.48 against a limit of 1 | the clip factor | $$6.0 \times 10^{-8}$$ |
| clipping, gradient norm 0.5 against a limit of 1 | no clipping below the limit | $$6.0 \times 10^{-8}$$ |
| limit 0, gradient norm 5.48 | clipping off even with a norm present | $$6.0 \times 10^{-8}$$ |
| a 1-layer GPT-2, 16 tensors, 4,464 values, 20 steps, clipped every other step | the whole model at once | $$3.8 \times 10^{-7}$$ |
| after the 20 timed updates (Section 6) | 8.9M and 124.4M values | $$7.7 \times 10^{-7}$$, $$1.1 \times 10^{-6}$$ |

*Table 2. Comparisons with PyTorch's fused AdamW, in
[`tests/test_adamw.cu`](https://github.com/ducbachsong/gpt2-small/blob/f336dc8/tests/test_adamw.cu).
Differences are relative to the largest value; the limit is $$10^{-5}$$.*

The largest difference comes from the 100-step test, and it grows slowly with the number of
steps, from $$5 \times 10^{-8}$$ after the first to $$2.1 \times 10^{-6}$$ after the hundredth: rounding
that accumulates, not an error in the update.

### 5.3 Rules worked out by hand

Eight more tests check behaviour against numbers worked out on paper, carried over from the C#
trainer's tests where they still apply:

- At $$t = 1$$, $$\hat m = g$$ and $$\hat v = g^2$$, so every value moves by $$\pm\eta$$ whatever
  the size of its gradient (gradients 1000 and 0.001 both move a weight by 0.05).
- With zero gradients, a 2x2 weight shrinks by $$1 - 0.5 \times 0.1 = 0.95$$ and a bias does not
  move.
- A parameter the loss did not use only decays ($$\times 0.99$$).
- The shape decides decay: a $$\{1, 4\}$$ matrix with one row decays; an 8-value bias does not.
- Inside one tensor, exactly the values before `decayed_count` decay.
- A launch over one tensor leaves its neighbour's $$p$$, $$m$$ and $$v$$ untouched, though the
  launch covers 1,024 values and the tensor has 8.
- The step leaves the gradients as they were (the C# step zeroed them).
- The layout: 5 groups, 17 tensors and 97 values for four small parameters; m and v on 16 bytes;
  the step count goes 0, then 2 after two steps.

All tests pass on a Colab Tesla T4, together with the `Tensor` and `TensorBuffer` tests, which now
include four for groups.

## 6. Performance evaluation

### 6.1 Setup

**Workloads.** Two sizes: **8,949,504 values**, the previous post's benchmark (GPT-2's 148
tensors at width 128), and **124,439,808 values**, GPT-2 small at full width. The update does not
depend on layers, only on the number of values and on the two param groups, so each size is one
matrix and one bias with that total: $$69{,}917 \times 128 + 128$$ and $$162{,}030 \times 768 + 768$$.
Settings: $$\beta_1 = 0.9$$, $$\beta_2 = 0.95$$, $$\epsilon = 10^{-8}$$, $$\lambda = 0.1$$,
$$\eta = 10^{-3}$$, clipping at 1.

**Protocol.** Only the update is timed, on the GPU's own clock: a CUDA event before it and one
after, and the time between them. The gradients are set once beforehand (PyTorch's clipped by
`clip_grad_norm_`, ours with their norm) and reused for all 20 updates. For mine the timed region
is one `adamw_update` launch; for PyTorch it is `_foreach_add_` plus `_fused_adamw_` for each of
the two groups. Step 1, which loads each side's kernels, is shown but left out of the mean of steps
2–20. After the 20 updates the two sides are compared as in Section 5.

| | |
|---|---|
| hardware | Google Colab, NVIDIA Tesla T4 (16 GB, 320 GB/s peak [8]) |
| software | CUDA C++17, `nvcc -O3 -arch=sm_75`; libtorch from Colab's preinstalled PyTorch (Python 3.13) |

*Table 3. Environment.*

### 6.2 Results

![AdamW update time on a Tesla T4](/images/fused-adamw-cuda/fig2-benchmark-t4.svg)
*Figure 2. Update time for 8,949,504 values. Solid bars: this post, GPU clock. Hatched bars: the
previous post, CPU clock, 148 tensors.*

| AdamW | 8.9M values | 124.4M values | clock |
|---|---:|---:|---|
| **my AdamW, one kernel** | **1.043 ms** | **14.654 ms** | GPU |
| PyTorch fused (C++) | 1.147 ms | 15.394 ms | GPU |
| PyTorch `fused=True` (Python), previous post | 1.95 ms | — | CPU |
| flat buffers (C#), previous post | 3.64 ms | — | CPU |

*Table 4. Mean of steps 2–20. Per-step times are in the log (Appendix A).*

At the previous post's size, one kernel is **3.5x faster than the flat-buffer AdamW**, close to the
3.3x that Table 1 predicts from bytes alone. Against PyTorch's fused AdamW timed the same way it is
**1.10x faster** at 8.9M values and **1.05x** at 124.4M values. The steps are steady: over steps
2–20 mine stays within 1.039–1.048 ms at the smaller size.

### 6.3 Analysis: the traffic model again

| AdamW | values | MB moved | time | effective bandwidth | of peak |
|---|---:|---:|---:|---:|---:|
| **my AdamW** | 8.9M | 251 | 1.043 ms | **240 GB/s** | 75% |
| **my AdamW** | 124.4M | 3,484 | 14.654 ms | **238 GB/s** | 74% |
| PyTorch fused (C++) | 124.4M | 3,484 | 15.394 ms | 226 GB/s | 71% |
| PyTorch fused (C++) | 8.9M | 251 | 1.147 ms | 219 GB/s | 68% |
| PyTorch fused (Python), previous post | 8.9M | 251 | 1.95 ms | 129 GB/s | 40% |
| flat buffers (C#), previous post | 8.9M | 823 | 3.64 ms | 226 GB/s | 71% |

*Table 5. 7 passes of $$4N$$ bytes for the fused kernels, 23 for the flat buffers. Bytes are
modelled from the passes, not measured with a profiler.*

![Estimated bandwidth on a Tesla T4](/images/fused-adamw-cuda/fig3-bandwidth-t4.svg)
*Figure 3. Estimated bandwidth: bytes moved per update divided by the measured time.*

Three observations follow:

1. **The speedup is the bytes.** The flat-buffer step and this kernel run at nearly the same
   bandwidth (226 and 240 GB/s); the kernel is faster because it moves 3.3x fewer bytes. This
   confirms the previous post's conclusion from the other side.
2. **Fixed costs are small, and smaller here than in PyTorch.** My kernel reaches the same bandwidth
   at 1 ms as at 15 ms, so its fixed cost (one launch) is negligible even at the smaller size.
   PyTorch's timed region has four launches, which cost a larger share at 1 ms (1.10x) than at
   15 ms (1.05x).
3. **The old Python fused number was mostly measurement.** The same fused operation, timed on the
   CPU clock from Python with 148 tensors, took 1.95 ms; timed on the GPU clock with 2 tensors it
   takes 1.147 ms. The fair comparison between the two kernels is therefore 1.10x, not the 1.87x
   that the two headline numbers would suggest.

## 7. Discussion

Fusion did what the traffic model said it would: 23 passes became 7, and the time fell by about the
same factor. The remaining question is how close 75% of peak is to what the T4 can do. A
memory-bound kernel rarely reaches the datasheet figure, and this one already does slightly better
than PyTorch's own fused kernel on the same values, so further gains in the kernel itself are
likely small.

The bigger cost is now outside it. The trainer needs the gradient norm before every update, which
is one more read of the gradients: 1 pass on top of 7, about 14% more traffic, roughly 2 ms per
step at full size if it runs at the same bandwidth. Nothing in this post times that kernel yet.

The design choices that made the kernel simple (one allocation, matrices first, groups that are
whole flat tensors) each come with a rule the caller must keep: matrices first, sizes a multiple of
4, and nothing of odd length in the middle of the buffer. Section 4.3 is what happens when one of
them is broken quietly; each rule is now checked where it can be, with an error that names it.

## 8. Threats to validity

- **Two tensors, not 148.** Each timing uses one matrix and one bias. My kernel does exactly the same
  work either way, but PyTorch's fused operation processes its tensor list in chunks, and 148 small
  tensors may need more launches than 2 large ones. The PyTorch figures here may be slightly
  optimistic for the real model.
- **Clipping in the timed region.** My kernel applies the clip factor inside the timed update;
  PyTorch's gradients were clipped by `clip_grad_norm_` before its timed region. This favours
  PyTorch slightly.
- **Sample size.** Each figure is the mean of 19 steps from one run in one Colab session. At the
  larger size my times drift from 14.38 to 14.82 ms over the 20 steps, most likely the GPU's clock
  changing with temperature; PyTorch's do not show the same drift, and they ran interleaved with
  mine.
- **Versions.** The PyTorch, CUDA and driver versions of the Colab session were not recorded.
- **Different clocks across posts.** The previous post's figures were taken on the CPU's clock and
  include host-side time; the comparisons across posts in Table 4 are reported as such, and the
  like-for-like comparison is the one within this post.
- **Traffic model.** As before, bytes are modelled from passes over memory; caching and the two
  kernels' actual access patterns can differ.

## 9. Future work

- **The norm kernel**, and timing the optimiser's whole work per step: norm plus update.
- **The real 148 tensors** in the timing run, to measure PyTorch's chunking rather than avoid it.
- **End to end.** The share of a full training step (forward, backward, optimiser) the update takes,
  once the rest of the C/CUDA trainer exists, to know how much the optimiser matters to tokens per
  second.

## 10. Conclusion

Writing AdamW as one CUDA kernel over one buffer turns 23 passes over memory into 7. On a Tesla T4
it updates the previous post's 8.9M values in 1.043 ms, 3.5x faster than the flat-buffer AdamW, and
GPT-2 small's 124M values in 14.65 ms, at about 75% of the GPU's peak bandwidth and slightly ahead of
PyTorch's own fused AdamW timed the same way. Every result agrees with PyTorch within $$10^{-5}$$.
The gap the previous post left open is closed; what remains is the gradient norm, one pass that the
update cannot absorb.

## Reproducibility

Code at commit [`f336dc8`](https://github.com/ducbachsong/gpt2-small/tree/f336dc8). On Google Colab
with a T4 (PyTorch, and with it libtorch, are preinstalled):

```bash
git clone https://github.com/ducbachsong/gpt2-small.git
cd gpt2-small && git checkout f336dc8

# Tensor and TensorBuffer
nvcc -O3 -std=c++17 -arch=sm_75 tests/test_tensor.cu -o test_tensor && ./test_tensor
nvcc -O3 -std=c++17 -arch=sm_75 tests/test_tensor_buffer.cu -o test_tensor_buffer && ./test_tensor_buffer

# AdamW against PyTorch's fused AdamW, then the two timing runs
T=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
ABI=$(python -c "import torch; print(int(torch.compiled_with_cxx11_abi()))")
nvcc -O3 -std=c++17 -arch=sm_75 -D_GLIBCXX_USE_CXX11_ABI=$ABI -Xcompiler -Wno-stringop-overflow \
    tests/test_adamw.cu -o test_adamw \
    -I$T/include -I$T/include/torch/csrc/api/include \
    -L$T/lib -Xlinker --no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
    -Xlinker -rpath=$T/lib
./test_adamw
```

`--no-as-needed` keeps `libtorch_cuda` linked although no call names it: PyTorch's GPU operations
are registered from inside it. On a GPU other than the T4, change `sm_75` (L4: `sm_89`,
A100: `sm_80`).

## References

1. D. P. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015.
   [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
2. I. Loshchilov, F. Hutter. *Decoupled Weight Decay Regularization.* ICLR 2019.
   [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
3. A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever. *Language Models are Unsupervised
   Multitask Learners.* OpenAI, 2019.
4. D. B. Song. *Flat buffers: a fast AdamW for GPT-2 in C#.* 2026.
   [/research/flat-buffers-adamw/](/research/flat-buffers-adamw/)
5. PyTorch documentation, `torch.optim.AdamW` (the `fused` option).
   [pytorch.org](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
6. A. Karpathy. *llm.c: LLM training in simple, raw C/CUDA.*
   [github.com/karpathy/llm.c](https://github.com/karpathy/llm.c)
7. NVIDIA. *CUDA C++ Programming Guide* (vector types and their alignment; events).
   [docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
8. NVIDIA. *Tesla T4 Tensor Core GPU* (datasheet: 320 GB/s memory bandwidth).
   [nvidia.com](https://www.nvidia.com/en-us/data-center/tesla-t4/)

## Appendix A. Raw logs

Index: [README.txt](/files/fused-adamw-cuda/README.txt).

| log | what |
|---|---|
| [test-adamw-t4-timing.log](/files/fused-adamw-cuda/test-adamw-t4-timing.log) | both timing runs on a Tesla T4, every step, and the check that ends each |

The full output of the run (every check of the three test files, about a thousand lines) is not
kept as a file; its largest differences from PyTorch are in Table 2.
