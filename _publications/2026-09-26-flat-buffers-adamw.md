---
title: "Flat buffers: a fast AdamW for GPT-2 in C#"
collection: publications
date: 2026-09-26
permalink: /research/flat-buffers-adamw/
redirect_from:
  - /blog/2026/09/26/flat-buffers-adamw/
excerpt: "One contiguous buffer per state and 10 element-wise operations per step: an AdamW for GPT-2 small in TorchSharp that matches PyTorch bit for bit and runs 4.3x faster than TorchSharp's own AdamW on a Tesla T4."
read_time: true
tags:
  - optimization
  - adamw
  - gpu
  - gpt-2
  - torchsharp
---

**Abstract.** TorchSharp, the .NET binding of LibTorch, ships only a per-parameter AdamW: for
GPT-2 small it launches about 1,500 GPU kernels per optimiser step, most of them on tensors of a
few hundred values. I place every parameter, gradient and optimiser moment of the model in one
contiguous buffer each, keep the model's own parameters as views into those buffers, and write
AdamW as 10 element-wise operations over the whole model. The result matches PyTorch's
`torch.optim.AdamW` bit for bit in every test, and on a Tesla T4 it takes **3.64 ms per step
against 15.75 ms for TorchSharp's AdamW (4.3x)**, level with PyTorch's `foreach` mode (4.09 ms).
PyTorch's fused kernel remains 1.9x faster (1.95 ms). A memory-traffic model explains all five
results: ours sustains an estimated 226 GB/s, about 70% of the T4's peak, so the remaining gap is
bytes moved, not launches.

* TOC
{:toc}

## 1. Introduction

I am pre-training a GPT-2 small [3] from scratch in C#: the model, the optimiser and the training
loop are written by hand on TorchSharp [6], while a Python data pipeline feeds token batches into
the same process. The project is open source:
[github.com/ducbachsong/gpt2-small](https://github.com/ducbachsong/gpt2-small/tree/csharp-trainer)
(branch `csharp-trainer`; every link in this post is pinned to commit
[`0c486d3`](https://github.com/ducbachsong/gpt2-small/tree/0c486d3)).

The model was the easy part. The optimiser was not: TorchSharp's `torch.optim.AdamW` loops over the
parameters, and neither of PyTorch's faster modes, `foreach` and `fused` [4], is available from C#.
This post describes how I closed most of that gap without writing a CUDA kernel.

**Contributions.**

1. A flat-buffer layout for a TorchSharp model that keeps the module API unchanged: layers keep
   their own parameter tensors, which become views into shared buffers (Section 3).
2. An AdamW step over those buffers in 10 element-wise operations, with weight decay restricted to
   matrices by buffer ordering rather than masks, and gradient clipping without a host sync
   (Sections 3–4).
3. A correctness evaluation against PyTorch's own AdamW that requires bit-for-bit equality, not
   closeness (Section 5).
4. A performance evaluation against TorchSharp's AdamW and PyTorch's three AdamW modes on the same
   shapes, with a memory-traffic model that accounts for every result (Section 6).

The idea itself is not new. Karpathy's llm.c [7] allocates all GPT-2 parameters in one block, and
PyTorch FSDP [8] flattens parameters for communication. What this post adds is the construction for
TorchSharp's module system, the verification, and the measurements.

## 2. Background

### 2.1 AdamW

For a parameter $$p$$ with gradient $$g$$ at step $$t$$, learning rate $$\eta$$ and weight decay
$$\lambda$$, AdamW [1, 2] updates

$$
\begin{aligned}
p &\leftarrow p - \eta \lambda p && \text{decoupled weight decay} \\
m &\leftarrow \beta_1 m + (1 - \beta_1)\, g && \text{first moment} \\
v &\leftarrow \beta_2 v + (1 - \beta_2)\, g^2 && \text{second moment} \\
p &\leftarrow p - \eta \, \frac{m / (1 - \beta_1^t)}{\sqrt{v / (1 - \beta_2^t)} + \epsilon} && \text{bias-corrected Adam step}
\end{aligned}
$$

Every line is element-wise: the update of one value never reads another. This is what makes it
possible to run the optimiser over any arrangement of the values, including one big buffer.

### 2.2 What an optimiser step costs on a GPU

Two costs dominate an element-wise optimiser on a GPU:

- **Launch overhead.** Each tensor operation is a kernel launch with a roughly fixed cost of a few
  microseconds (plus, in TorchSharp, a C# → C++ call), independent of the tensor's size.
- **Memory traffic.** Each operation reads and writes whole tensors. For a large model the step is
  memory-bound: its time is the bytes moved divided by the achievable bandwidth.

GPT-2 small has **148 parameter tensors**: 50 matrices (embeddings, attention, MLP) and 98 vectors
(biases, LayerNorm). A per-parameter loop of ~10 operations therefore launches ~1,500 kernels per
step, most of them on 768-value vectors that finish long before the next launch arrives.

### 2.3 Existing approaches

PyTorch offers three implementations of `torch.optim.AdamW` [4]:

| mode | how it runs the step | launches per step |
|---|---|---:|
| for-loop (`foreach=False`) | every operation, once per tensor | ~1,500 |
| `foreach=True` | every operation, once over the list of tensors (multi-tensor apply [5]) | ~10–30 |
| `fused=True` | the whole update in one kernel | ~1 |

TorchSharp 0.107 exposes only the first. Its AdamW is written in C# as a loop, and LibTorch's fused
kernel is not bound. (I checked the TorchSharp 0.107.0 assembly: it has no `fused` or `foreach`
option.)

## 3. Method

![Per-parameter loop against flat buffer](/images/flat-buffers-adamw/fig1-loop-vs-flat.svg)
*Figure 1. (a) A per-parameter loop runs every operation once per tensor. (b) With one flat buffer,
every operation runs once over the whole model.*

### 3.1 Layout

When the model is built, `FlatParameters` allocates two 1-D buffers of equal length, `Parameters`
and `Gradients`, and assigns each of the model's parameters a segment (Figure 2). AdamW allocates
its moments $$m$$ and $$v$$ in the same layout, so every AdamW operation is element-wise over four
aligned buffers.

![Flat buffer layout](/images/flat-buffers-adamw/fig2-layout.svg)
*Figure 2. Each parameter becomes a view into a segment of `Parameters`; its gradient is the
same segment of `Gradients`. Matrices come first, so weight decay is one operation on
`[0, DecayedCount)`. Segments start on multiples of 64 values.*

Three rules define the layout:

1. **Matrices first.** Weight decay applies to tensors of rank ≥ 2 (embeddings and weight matrices)
   and not to biases and LayerNorm parameters, following GPT-2 practice. Sorting the rank-≥ 2
   tensors to the front makes the decayed set one contiguous range, `[0, DecayedCount)`, so decay
   is a single `mul_` with no mask and no loop.
2. **Aligned segments.** Each segment starts at a multiple of 64 values (256 bytes) for aligned,
   vectorised access. The padding is zero-initialised and remains zero: its gradient is zero, so
   Adam never moves it and it contributes nothing to the gradient norm.
3. **Views, not copies.** Each parameter's values are copied into its segment once, then the
   parameter is re-pointed at that segment with `set_`. The layers still hold their own tensors,
   with their own shapes, and do not know the memory is shared.

### 3.2 Gradients as views

Each parameter's `.grad` is set to its segment of `Gradients`. Autograd accumulates into an
existing `.grad` in place, so `backward()` writes every gradient directly into the flat buffer
with no copy and no gather. This invariant is essential and fragile: if anything ever replaced a
`.grad` with a new tensor, that gradient would silently stop reaching the optimiser. Section 5
tests it through a real backward pass.

### 3.3 The step

With the buffers in place, one AdamW step over the whole model is:

| # | operation | on | reads | writes |
|---:|---|---|---:|---:|
| 1 | `decayed.mul_(1 - ηλ)` | first `DecayedCount` values of $$p$$ | 1 | 1 |
| 2 | `m.mul_(β₁)` | $$m$$ | 1 | 1 |
| 3 | `m.add_(g, 1 - β₁)` | $$m, g$$ | 2 | 1 |
| 4 | `v.mul_(β₂)` | $$v$$ | 1 | 1 |
| 5 | `v.addcmul_(g, g, 1 - β₂)` | $$v, g$$ | 2 | 1 |
| 6 | `d = v.sqrt()` | $$v$$ | 1 | 1 |
| 7 | `d.div_(√(1 - β₂ᵗ))` | $$d$$ | 1 | 1 |
| 8 | `d.add_(ε)` | $$d$$ | 1 | 1 |
| 9 | `p.addcdiv_(m, d, -η / (1 - β₁ᵗ))` | $$p, m, d$$ | 3 | 1 |
| 10 | `g.zero_()` | $$g$$ | 0 | 1 |
| | **total** | | **13** | **10** |

*Table 1. The step as issued. Each "read" or "write" is one pass over a buffer of $$N$$ values;
the step moves 23 passes, $$23 \times 4N$$ bytes.*

Operation 10 leaves the gradients at zero for the next `backward()`, so the training loop needs no
separate `zero_grad`. When gradient clipping is on, an 11th operation scales $$g$$ first.

### 3.4 Clipping without a host sync

Clipping by the global norm needs $$\lVert g \rVert$$ over the whole model, which with the flat
buffer is one `norm()` of `Gradients`. The clip factor
$$\min(1, c / (\lVert g \rVert + 10^{-6}))$$ is computed as a 0-d tensor on the GPU and applied
inside the next step, so the CPU never waits for the GPU to report a number. (TorchSharp's
`clip_grad_norm_` returns a `double`, which would force a sync every step.)

## 4. Implementation

The code is C# on TorchSharp 0.107.0 (LibTorch 2.10.0). Comments are abridged in the listings
below; the linked files have them in full.

**Listing 1.** Building the layout —
[`src/FlatParameters.cs`](https://github.com/ducbachsong/gpt2-small/blob/0c486d3/src/FlatParameters.cs)

```csharp
public FlatParameters(IEnumerable<Tensor> parameters)
{
    // 1. Matrices (weight decay) first, then 1-D tensors (no decay).
    var decayedParameters = parameters.Where(p => p.dim() >= 2).ToArray();
    var notDecayedParameters = parameters.Where(p => p.dim() < 2).ToArray();
    var parametersInBufferOrder = decayedParameters.Concat(notDecayedParameters).ToArray();
    if (parametersInBufferOrder.Length == 0) throw new ArgumentException("no parameters");

    // 2. Where each segment starts: sizes rounded up to a multiple of 64.
    var segmentStarts = new long[parametersInBufferOrder.Length];
    long totalLength = 0;
    for (int i = 0; i < parametersInBufferOrder.Length; i++)
    {
        if (i == decayedParameters.Length) DecayedCount = totalLength;
        segmentStarts[i] = totalLength;
        totalLength += RoundUpToMultiple(parametersInBufferOrder[i].numel(), SegmentAlignment);
    }
    if (notDecayedParameters.Length == 0) DecayedCount = totalLength;

    // 3. The two buffers, zeros (so the padding is 0), on the parameters' device and dtype.
    var device = parametersInBufferOrder[0].device;
    var dtype = parametersInBufferOrder[0].dtype;
    using var _ = no_grad();
    Parameters = zeros(totalLength, dtype: dtype, device: device).DetachFromDisposeScope();
    Gradients = zeros(totalLength, dtype: dtype, device: device).DetachFromDisposeScope();

    // 4. Copy each parameter in, make it a view of its segment, and point its .grad
    //    at the same segment of Gradients.
    for (int i = 0; i < parametersInBufferOrder.Length; i++)
    {
        var parameter = parametersInBufferOrder[i];
        long count = parameter.numel();
        using var slot = Parameters.narrow(0, segmentStarts[i], count).view(parameter.shape);
        slot.copy_(parameter);
        parameter.set_(slot);
        parameter.grad = Gradients.narrow(0, segmentStarts[i], count).view(parameter.shape);
    }
}
```

The model calls this as the last line of its constructor, after it has initialised its weights
and moved to its device: moving a flattened model would break the views.

**Listing 2.** The step and clipping —
[`src/AdamW.cs`](https://github.com/ducbachsong/gpt2-small/blob/0c486d3/src/AdamW.cs)

```csharp
public Tensor ClipGradientNorm(double maxNorm)
{
    using var _ = no_grad();
    var norm = flat.Gradients.norm();                     // the padding is 0: it adds nothing
    clipScale = (norm + 1e-6).reciprocal_().mul_(maxNorm).clamp_max_(1.0);
    return norm;                                          // stays on the GPU
}

public void Step(double learningRate)
{
    using var _ = no_grad();
    stepCount++;
    double biasCorrection1 = 1 - Math.Pow(config.Beta1, stepCount);
    double biasCorrection2 = 1 - Math.Pow(config.Beta2, stepCount);
    double weightDecayFactor = 1 - learningRate * config.WeightDecay;
    double stepSize = learningRate / biasCorrection1;

    var gradients = flat.Gradients;
    if (clipScale is not null) gradients.mul_(clipScale);                                    // g = g * clip
    if (flat.DecayedCount > 0)
        using (var decayed = flat.Parameters.narrow(0, 0, flat.DecayedCount))
            decayed.mul_(weightDecayFactor);                                                 // p = p - lr wd p
    firstMoment.mul_(config.Beta1).add_(gradients, alpha: 1 - config.Beta1);                 // m
    secondMoment.mul_(config.Beta2).addcmul_(gradients, gradients, value: 1 - config.Beta2); // v
    using var denominator = secondMoment.sqrt().div_(Math.Sqrt(biasCorrection2)).add_(config.Eps);
    flat.Parameters.addcdiv_(firstMoment, denominator, value: -stepSize);                    // p = p - lr m̂ / (...)
    gradients.zero_();                                                                       // g = 0
    clipScale = null;
}
```

**Order of operations matters for exactness.** $$\sqrt{\hat v}$$ is computed as
$$\sqrt{v} / \sqrt{1 - \beta_2^t}$$ and $$\eta \hat m$$ as $$(\eta / (1 - \beta_1^t))\, m$$, the order
PyTorch uses. Equivalent rearrangements, such as $$\sqrt{v / (1 - \beta_2^t)}$$, are equal on paper
but differ in the last bit of floating-point results, which would break the bit-for-bit comparison
in Section 5.

## 5. Correctness evaluation

A faster optimiser that is slightly wrong is worse than a slow one, so the tests hold ours to the
strictest standard available: **bit-for-bit equality with PyTorch's own AdamW**, through
TorchSharp's `torch.optim.AdamW`. Each comparison test trains two copies of the same starting
weights, ours and the reference's, on identical gradients, and compares every value.

Gradients are produced by a real `backward()` on a synthetic loss
$$L = \sum_i \langle p_i, g_i \rangle$$, whose gradient with respect to each $$p_i$$ is exactly the
chosen $$g_i$$. This exercises the same autograd path as training, including the in-place
accumulation into the flat `Gradients` buffer.

| test | what it establishes | criterion |
|---|---|---|
| one step, 2x2 weight | a single update | bit for bit |
| 10 steps, 2x2 weight, a separate gradient per entry and step | $$m$$, $$v$$ and bias correction over time | bit for bit, after every step |
| 256x256 random weight, 100 steps, every 7th gradient scaled by $$10^{-6}$$ | rounding at scale; $$\epsilon$$ with tiny gradients | bit for bit |
| mixed 1-D/2-D/3-D shapes in mixed order, 20 steps | segment offsets and alignment | bit for bit |
| bias, one step (reference with decay off) | no decay on 1-D tensors | bit for bit |
| backward through a 2-layer GPT-2, 2 steps | every `.grad` is a view into the flat buffer | $$\lVert g \rVert$$ from `.grad`s = $$\lVert g \rVert$$ of the buffer, within $$10^{-5}$$ relative |
| weight decay with rank ≥ 2 and 1-D tensors interleaved | decay follows shape, not order | matrices × $$(1 - \eta\lambda)$$, vectors unchanged, exact |
| gradient accumulation over two `backward()` calls | views accumulate like ordinary `.grad` tensors | equal to a plain PyTorch parameter |
| clipping above and below `maxNorm` | the clip factor and its application | within $$10^{-5}$$ |

*Table 2. Correctness tests in
[`src/test/AdamWTests.cs`](https://github.com/ducbachsong/gpt2-small/blob/0c486d3/src/test/AdamWTests.cs).*

Why bit for bit: the 256x256 test is large enough that a mathematically equivalent but rearranged
formula (for example folding $$\epsilon$$ into the step size) changes the last bit of about 12% of
the values. Exact equality therefore certifies the same operations in the same order, not merely
the same mathematics.

All 30 tests pass (AdamW, flat buffers, learning-rate schedule and the data feed):
[test-run-cpu.log](/files/flat-buffers-adamw/test-run-cpu.log).

## 6. Performance evaluation

### 6.1 Setup

**Workload.** The benchmark builds a real `Gpt2` with GPT-2's full set of 148 parameter tensors
(12 layers, 50,304-token padded vocabulary, 1,024 positions), narrowed to width 128 so the test
stays light: 8,949,504 fp32 values, 35.8 MB per buffer. At this width every tensor size is a
multiple of 64, so the buffer has no padding. Ours receives the tensors already flattened; the
references receive plain copies with the same values and settings
($$\beta_1 = 0.9$$, $$\beta_2 = 0.95$$, $$\epsilon = 10^{-8}$$, $$\lambda = 0.1$$, $$\eta = 10^{-3}$$).

**Protocol.** Only the optimiser is timed. Each step: produce gradients with `backward()`
(untimed), synchronise the device, start the clock, run the step, synchronise, stop the clock.
Three warm-up steps are discarded, and the reported time is the mean of the next 20. The
references are timed doing `step()` followed by `zero_grad()`, since ours zeroes the gradients
inside its step. The Python script applies the same protocol to PyTorch's three modes on exactly
the same shapes.

| | GPU run | CPU run |
|---|---|---|
| hardware | Google Colab, NVIDIA Tesla T4 (16 GB, 320 GB/s peak [9]) | Intel Core i7-13620H laptop |
| C# | .NET 8.0.425, TorchSharp 0.107.0, LibTorch 2.10.0 CUDA | .NET 8.0.425, TorchSharp 0.107.0, LibTorch 2.10.0 CPU |
| Python | PyTorch 2.11.0+cu128 | PyTorch 2.12.0+cpu |

*Table 3. Environment.*

### 6.2 Results

![AdamW step time on a Tesla T4](/images/flat-buffers-adamw/fig3-benchmark-t4.svg)
*Figure 3. Mean AdamW step time on a Tesla T4, 148 tensors, 8.9M values.*

| AdamW | T4 (ms / step) | vs ours | CPU (ms / step) |
|---|---:|---:|---:|
| PyTorch `fused=True` (Python) | 1.95 | 0.54x | n/a (GPU only) |
| **ours, flat buffers (C#)** | **3.64** | **1.00x** | **18.3–20.9** |
| PyTorch `foreach=True` (Python) | 4.09 | 1.12x | 23.0–23.1 |
| PyTorch for-loop (Python) | 15.47 | 4.25x | 24.2–24.5 |
| TorchSharp built-in (C#) | 15.75 | 4.33x | 40.9–53.1 |

*Table 4. Step time; "vs ours" is the T4 time relative to ours. CPU ranges span the runs listed in
Appendix A.*

On the T4, ours is **4.3x faster than TorchSharp's AdamW**, the only AdamW available from C#, and
**12% faster than PyTorch's `foreach`**. PyTorch's fused kernel is **1.9x faster than ours**. On the
CPU, where launches are cheap, all loop-free variants converge and ours leads by a smaller margin.

### 6.3 Analysis: a memory-traffic model

Counting kernel launches and passes over memory per step (Table 1 for ours; the published
implementations for PyTorch's modes [4]) accounts for every result:

| AdamW | launches / step | passes over memory | MB moved | T4 time | effective bandwidth |
|---|---:|---:|---:|---:|---:|
| for-loop / TorchSharp | ~1,500 | 23 | 823 | 15.5–15.8 ms | 53 GB/s |
| PyTorch `foreach` | ~10–30 | ~20 | 716 | 4.09 ms | 175 GB/s |
| **ours** | **10** | **23** | **823** | **3.64 ms** | **226 GB/s** |
| PyTorch `fused` | 1 | 7 | 251 | 1.95 ms | 129 GB/s |

*Table 5. One pass = 35.8 MB. Bytes are modelled from the operation sequence, not measured with a
profiler.*

![Estimated memory bandwidth on a Tesla T4](/images/flat-buffers-adamw/fig4-bandwidth-t4.svg)
*Figure 4. Estimated bandwidth: bytes moved per step divided by measured step time.*

Three observations follow:

1. **The loops are launch-bound.** They move the same 823 MB as ours but take 4.3x longer, reaching
   only 53 GB/s: the GPU mostly waits for the next of ~1,500 small launches.
2. **Ours is memory-bound.** At an estimated 226 GB/s it uses about 70% of the T4's 320 GB/s peak.
   Removing the loop has taken the launch cost out of the step; what remains is the cost of the
   bytes, and no rearrangement of launches can reduce it further.
3. **Fused wins on bytes, not bandwidth.** The fused kernel uses the bandwidth less fully
   (129 GB/s) but reads $$p, g, m, v$$ once and writes $$p, m, v$$ once: 7 passes instead of 23,
   3.3x fewer bytes, and so 1.9x less time. `foreach` sits between the two: few launches like ours,
   slightly fewer passes (it updates $$m$$ with one `lerp_`), lower achieved bandwidth.

## 7. Discussion

The flat buffer does exactly one thing: it removes the per-parameter loop. In C#, where
TorchSharp offers neither `foreach` nor `fused`, that is a 4.3x speedup obtained with ordinary
tensor operations, no native code, and a result that is bit-for-bit identical to PyTorch.

It cannot, by construction, match a fused kernel. Once launches are gone the step is bounded by
memory traffic (Section 6.3), and the flat buffer does not change the number of passes, only the
number of launches. Beating `fused` requires moving fewer bytes, which means fusing the
operations themselves.

The flat buffer also has costs in the code: the model must be flattened last and never moved
afterwards; every `.grad` must remain a view; and weight decay by shape must be kept consistent
with the reference when comparing (PyTorch's AdamW decays every tensor it is given).

## 8. Threats to validity

- **Width.** The benchmark uses width 128, not GPT-2's 768 (124M values). At full width each pass
  moves 14x more data; both launch-bound and memory-bound terms change, and the ratios should be
  re-measured there.
- **Sample size.** Each figure is the mean of 20 steps from one run. Run-to-run variation is
  visible: the first GPU run of the C# benchmark reported 4.37 ms for ours and 36.35 ms for
  TorchSharp's AdamW, but its GPU model was not recorded, so it is excluded and kept in
  Appendix A. The reported T4 figures come from one Colab session in which the GPU was confirmed
  by `nvidia-smi`.
- **Traffic model.** Bytes are counted from the operation sequence and assume each operation reads
  and writes whole buffers; caching and PyTorch's internal kernel choices can differ. The
  bandwidth figures are estimates, not profiler measurements.
- **Cross-language timing.** Ours and TorchSharp's are timed in C#, PyTorch's modes in Python.
  Both synchronise the device around the timed region, but host-side overheads differ between
  the two runtimes.

## 9. Future work

- **Fewer passes.** Merging operations, for example updating $$m$$ with one `lerp_` and folding the
  denominator's three operations into fewer, would cut several of the 23 passes. It gives up
  bit-for-bit equality with PyTorch, so the tests would move to a tolerance.
- **A fused kernel.** Binding LibTorch's fused AdamW, or a small custom CUDA kernel over the flat
  buffers, would reach the 7-pass regime.
- **Full width and end-to-end.** Measure at width 768, and measure the share of a full training
  step (forward, backward, optimiser) the optimiser takes, to know how much any of this matters to
  tokens per second.

## 10. Conclusion

Placing a TorchSharp model's parameters, gradients and optimiser state in flat buffers, while
keeping the parameters as views, turns AdamW from ~1,500 kernel launches per step into 10. On a
Tesla T4 this is 4.3x faster than TorchSharp's AdamW and on par with PyTorch's `foreach`, with
results identical to PyTorch's AdamW to the last bit. The remaining gap to PyTorch's fused kernel is
memory traffic, which the traffic model quantifies and which only operation fusion can close.

## Reproducibility

Code at commit [`0c486d3`](https://github.com/ducbachsong/gpt2-small/tree/0c486d3). On a machine
with an NVIDIA GPU and the .NET 8 SDK (on Colab: install .NET 8 with `dotnet-install.sh`; PyTorch is
preinstalled):

```bash
git clone https://github.com/ducbachsong/gpt2-small.git
cd gpt2-small && git checkout 0c486d3

# correctness: all tests on the CUDA build of LibTorch (the first build downloads ~2 GB)
dotnet test Gpt2Trainer.sln -p:TorchBackend=cuda-linux --filter "Category!=AdamW-Benchmark"

# ours against TorchSharp's AdamW
dotnet test Gpt2Trainer.sln -p:TorchBackend=cuda-linux --filter "Category=AdamW-Benchmark" \
    --logger "console;verbosity=detailed"

# PyTorch's for-loop, foreach and fused AdamW on the same shapes
python src/test/adamw-benchmark-torch.py
```

Without `-p:TorchBackend=cuda-linux` the tests build against the CPU LibTorch and the benchmark
runs on the CPU.

## References

1. D. P. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015.
   [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
2. I. Loshchilov, F. Hutter. *Decoupled Weight Decay Regularization.* ICLR 2019.
   [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
3. A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever. *Language Models are Unsupervised
   Multitask Learners.* OpenAI, 2019.
4. PyTorch documentation, `torch.optim.AdamW` (the `foreach` and `fused` options).
   [pytorch.org](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
5. NVIDIA Apex, multi-tensor apply and fused optimisers.
   [github.com/NVIDIA/apex](https://github.com/NVIDIA/apex)
6. .NET Foundation. *TorchSharp.* [github.com/dotnet/TorchSharp](https://github.com/dotnet/TorchSharp)
7. A. Karpathy. *llm.c: LLM training in simple, raw C/CUDA.*
   [github.com/karpathy/llm.c](https://github.com/karpathy/llm.c)
8. Y. Zhao et al. *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel.* VLDB 2023.
   [arXiv:2304.11277](https://arxiv.org/abs/2304.11277)
9. NVIDIA. *Tesla T4 Tensor Core GPU* (datasheet: 320 GB/s memory bandwidth).
   [nvidia.com](https://www.nvidia.com/en-us/data-center/tesla-t4/)

## Appendix A. Raw logs

Unedited output of every run behind the numbers above (local paths replaced by `<temp>`; terminal
control characters removed from the Colab output). Index: [README.txt](/files/flat-buffers-adamw/README.txt).

| log | what |
|---|---|
| [test-run-cpu.log](/files/flat-buffers-adamw/test-run-cpu.log) | all 30 tests, CPU |
| [benchmark-csharp-t4.log](/files/flat-buffers-adamw/benchmark-csharp-t4.log) | C# benchmark, Tesla T4 (reported) |
| [benchmark-pytorch-t4.log](/files/flat-buffers-adamw/benchmark-pytorch-t4.log) | PyTorch's three modes, same T4 session (reported) |
| [benchmark-csharp-gpu-first-run.log](/files/flat-buffers-adamw/benchmark-csharp-gpu-first-run.log) | first GPU run, GPU model not recorded (excluded) |
| [benchmark-csharp-cpu.log](/files/flat-buffers-adamw/benchmark-csharp-cpu.log) | C# benchmark, CPU: ours 18.71 ms, TorchSharp 40.85 ms |
| [benchmark-pytorch-cpu.log](/files/flat-buffers-adamw/benchmark-pytorch-cpu.log) | PyTorch for-loop and foreach, CPU |

Two earlier CPU runs of the C# benchmark, not kept as log files: ours 20.93 ms / TorchSharp
53.07 ms, and ours 18.26 ms / TorchSharp 49.68 ms.
