---
title: "How many threads? Launch sizes for GPT-2's encoder kernels on a T4"
collection: publications
date: 2026-10-07
permalink: /research/encoder-launch-sizes/
excerpt: "GPT-2's encoder as three CUDA kernels, each timed on a Tesla T4 with 32 launch sizes, from 40 blocks of 128 threads to 4,096 blocks of 1,024. Every launch gives the same result. Below the 40,960 threads a T4 holds at once, the time goes up, by up to 3.7x for the kernel that loads one float at a time; from there to about a million threads it is flat within run-to-run noise; at four million, empty blocks cost up to 32%. A bytes-in-flight model explains why the three kernels need different numbers of threads."
read_time: true
tags:
  - cuda
  - gpu
  - gpt-2
  - performance
  - occupancy
---

**Abstract.** A CUDA kernel is launched with two numbers: how many blocks, and how many threads in
each. The [note on threads, warps and blocks](/posts/gpu-threads-blocks-registers/) explained what
these numbers mean but measured nothing. This entry measures them. I take the three kernels of
GPT-2's encoder from my C/CUDA trainer (the forward pass, and the backward passes into the token
table and into the position table), launch each with 8 block counts times 4 block sizes, from
5,120 to 4,194,304 threads in all, and time them on a Tesla T4 at GPT-2 small's sizes. **All 32
launches give the same result**, checked against the usual launch, because every kernel walks its
data with a grid-stride loop. **The time has three regions.** With fewer threads than the 40,960 a
T4 runs at once, the GPU is partly empty and the kernels slow down: the backward into the token
table, which loads one float at a time, takes **3.7x as long** at 5,120 threads; the forward, which
loads 16 bytes at a time, 1.7x; the backward into the position table, which loads four 16-byte
values at once, only 1.09x. From 40,960 threads to about a million, every kernel is flat within
the run-to-run noise of a few percent, and the "fastest" launch in each column is noise. At four
million threads, where most blocks have nothing to do, the forward takes 17% longer and the
position backward 32%. A simple model, the bytes each thread keeps waiting on memory, predicts the
first region for all three kernels. The launch the code uses, 512 blocks of 1,024 threads, sits in
the flat region, so it stays.

* TOC
{:toc}

## 1. Introduction

I am pre-training a GPT-2 small [1] from scratch, rebuilding the trainer in C and CUDA in the style
of Karpathy's llm.c [2], with a test against PyTorch for every piece. The code is open source:
[github.com/ducbachsong/gpt2-small](https://github.com/ducbachsong/gpt2-small/tree/c-cuda-trainer)
(branch `c-cuda-trainer`; every link in this entry is pinned to commit
[`a1f1b7b`](https://github.com/ducbachsong/gpt2-small/tree/a1f1b7b)).

The previous entries built the optimiser's update [3] and the gradient norm [4]; both launch 512
blocks of 1,024 threads, a choice I made once and never measured. The encoder, the model's first
layer, uses the same launch. While writing it I asked two questions that the
[note](/posts/gpu-threads-blocks-registers/) [5] answers in principle but not with numbers: would
a different number of blocks or threads make the kernels faster, and how far can the numbers move
before something goes wrong? This entry answers both for the encoder's three kernels.

**Contributions.**

1. A test that launches each kernel directly with any `<<<blocks, threads>>>`, times it, and checks
   that its result equals the usual launch's (Section 3).
2. Timings of the three kernels for 32 launch sizes on a Tesla T4 (Section 5).
3. An explanation of the three regions of the results, with a bytes-in-flight model that predicts
   why the kernel with the narrowest loads needs the most threads (Section 6).

## 2. Background

### 2.1 The encoder and its three kernels

The encoder turns each token into a row of `embed_dim` numbers: its token's row of the table `wte`
plus its position's row of the table `wpe`.

$$
\text{encoded}[b][t] = \text{wte}[\text{token\_ids}[b][t]] + \text{wpe}[t]
$$

At GPT-2 small's sizes there are `batch_size` = 4 rows of text, `seq_len` = 1,024 tokens per row and
`embed_dim` = 768, so 4,096 tokens and 3,145,728 numbers. `wte` has 50,304 rows (the 50,257 tokens,
padded) and `wpe` 1,024. The code is in
[`llmc/kernels/encoder.cuh`](https://github.com/ducbachsong/gpt2-small/blob/a1f1b7b/llmc/kernels/encoder.cuh);
it has three kernels:

| kernel | one job | jobs | loads per job |
|---|---|---:|---|
| **forward** | one `float4` (4 numbers) of `encoded` | 786,432 | the token id; then 16 bytes of `wte`; 16 bytes of `wpe` |
| **wte backward** | one number of `encoded_grad`, added onto its token's row of `wte_grad` | 3,145,728 | the token id; 4 bytes of `encoded_grad`; then one `atomicAdd` |
| **wpe backward** | one `float4` of `wpe_grad`: the 4 rows of the batch at that position, added | 196,608 | four 16-byte values of `encoded_grad`; 16 bytes of `wpe_grad` |

*Table 1. The three kernels. A token can appear many times in a batch, so several threads may add
onto the same row of `wte_grad`; `atomicAdd` keeps those additions from overwriting each other, and
on a T4 it adds one `float` at a time (an `atomicAdd` on a `float4` needs compute capability 9.0
[6]). The position rows have no such conflict: one thread owns each `float4` of `wpe_grad` and adds
the 4 rows itself.*

All three do almost no arithmetic. Their time is the time to move their bytes, so they are
*memory-bound*, and that is what makes the launch size matter (Section 2.4).

### 2.2 What a T4 holds at once

A launch asks for blocks; the GPU runs them on its SMs, and only so many fit at a time. A block
stays on one SM until all its threads finish; the blocks that do not fit wait and start as others
end. For a Tesla T4, compute capability 7.5 [6, 7]:

| per SM | limit |
|---|---:|
| threads | 1,024 (32 warps) |
| blocks | 16 |
| registers | 65,536 |

The T4 has **40 SMs**, so at most **40 × 1,024 = 40,960 threads run at once**. The three kernels use
20, 16 and 38 registers per thread (the test prints them), at most 38 × 1,024 = 38,912 per SM, so
registers never limit them, and every block size here fills an SM:

| threads per block | blocks per SM | blocks at once on the T4 |
|---:|---:|---:|
| 128 | 8 | 320 |
| 256 | 4 | 160 |
| 512 | 2 | 80 |
| 1,024 | 1 | 40 |

*Table 2. The block counts 40, 80, 160 and 320 of the experiment are 1, 2, 4 and 8 blocks per SM: with
128 threads per block, 320 blocks fill the T4 exactly; with 1,024, 40 do.*

### 2.3 Any launch gives the same result

Each kernel walks its jobs with a *grid-stride loop*: thread $$n$$ does jobs $$n$$,
$$n + N$$, $$n + 2N$$, …, where $$N$$ is the number of threads launched, until it passes the last job.

```cpp
for (size_t index = thread_number; index < total_float4s; index += (size_t)gridDim.x * blockDim.x) {
    ...   // one job
}
```

With 4 threads and 10 jobs, thread 0 does 0, 4, 8; thread 1 does 1, 5, 9; thread 2 does 2, 6;
thread 3 does 3, 7. Every job is done exactly once whatever $$N$$ is, so the launch size can change
the time but not the result. Section 4 checks this.

### 2.4 Why the number of threads changes the time

A load from GPU memory takes roughly half a microsecond to come back. A thread can ask for a value
and go on; it stops only when it uses the value. To keep memory streaming at its 320 GB/s [8], the
GPU needs enough bytes asked for and not yet back, *in flight*, to cover that wait. By Little's law
[9], as in the [gradient norm entry](/research/grad-norm-cuda/) [4]:

$$
\text{bytes in flight needed} \approx 320\ \text{GB/s} \times 0.5\ \mu\text{s} \approx 160\ \text{KB}.
$$

The half microsecond is an estimate, so the 160 KB is too. Bytes in flight come from threads: each
thread keeps a few bytes waiting, and more resident threads keep more waiting. Below some number of
threads the GPU cannot ask for enough, memory sits partly idle, and the kernel is slower. Above
40,960 threads, the extra threads cannot run anyway; their blocks wait their turn.

## 3. Method

The usual entry points, `encoder_forward` and `encoder_backward`, always launch `ENCODER_BLOCKS` ×
`ENCODER_THREADS` = 512 × 1,024. The test
[`time_each_kernel_with_other_block_and_thread_counts`](https://github.com/ducbachsong/gpt2-small/blob/a1f1b7b/tests/test_encoder.cu)
launches the three kernels directly instead, with any size:

```cpp
encoder_forward_kernel<<<blocks, threads>>>((float4 *)encoded.data(), token_ids.data_ptr<int>(),
                                            (const float4 *)wte.data(), (const float4 *)wpe.data(),
                                            batch_size, seq_len, embed_dim);
```

**Workload.** GPT-2 small's sizes: 4 × 1,024 random token ids from 0 to 50,256, `wte` and `wpe` and
`encoded_grad` filled with random normal values.

**Launch sizes.** 8 block counts × 4 block sizes:

| blocks | 40, 80, 160, 320, 512, 1,024, 2,048, 4,096 |
|---|---|
| threads per block | 128, 256, 512, 1,024 |

From 40 × 128 = 5,120 threads, an eighth of what the T4 holds, to 4,096 × 1,024 = 4,194,304, about a
hundred times more.

**Protocol.** For each launch size, each kernel runs 20 times, each run timed on the GPU's clock
with a CUDA event before and after it; the first run is left out and the time is the mean of runs
2–20. Then the result is checked (Section 4). The backward kernels add onto their gradients, so
during timing the gradients grow; that changes the values, not the work.

| | |
|---|---|
| hardware | Google Colab, NVIDIA Tesla T4 (16 GB, 320 GB/s peak [8], 40 SMs [7]) |
| software | CUDA C++17, `nvcc -O3`; libtorch from Colab's preinstalled PyTorch, for the checks |

*Table 3. Environment.*

## 4. Correctness

Before the timing, the test runs `encoder_forward` and `encoder_backward` once with the usual launch
and keeps their results on the GPU as PyTorch tensors. After timing each launch size, it sets
`encoded`, `wte_grad` and `wpe_grad` to 0, runs the three kernels once with that size, and compares:

| result | criterion | why |
|---|---|---|
| `encoded` | equal to the last bit | each number is one addition of the same two values |
| `wpe_grad` | equal to the last bit | each thread adds b = 0, 1, 2, 3 in that order, whatever the launch |
| `wte_grad` | within $$10^{-5}$$ of the largest value | `atomicAdd`s land in whatever order threads arrive; floating-point addition is not associative |

*Table 4. The check of each launch size.*

**All 32 launch sizes pass.** The rest of the encoder's test suite also passes: the kernels against
PyTorch's `embedding` with autograd (equal for `encoded`, within $$10^{-6}$$ for the gradients),
seven tests of numbers worked out by hand, and a check that each kernel fits a block of 1,024
threads without spilling. The full log is in [Appendix A](#appendix-a-raw-log).

## 5. Results

![Encoder kernel time against threads launched](/images/encoder-launch-sizes/fig1-time-vs-threads.svg)
*Figure 1. All 96 timings (3 kernels × 32 launches) against the threads launched. The rings mark the
launch the code uses, 512 × 1,024.*

| blocks | threads | all threads | forward | wte backward | wpe backward |
|---:|---:|---:|---:|---:|---:|
| 40 | 128 | 5,120 | **0.266** | **0.600** | 0.085 |
| 40 | 256 | 10,240 | 0.165 | **0.332** | 0.082 |
| 40 | 512 | 20,480 | 0.157 | 0.202 | 0.080 |
| 40 | 1,024 | 40,960 | 0.161 | 0.184 | 0.082 |
| 80 | 128 | 10,240 | 0.167 | **0.336** | 0.081 |
| 80 | 256 | 20,480 | 0.157 | 0.200 | 0.081 |
| 80 | 512 | 40,960 | 0.161 | 0.177 | 0.081 |
| 80 | 1,024 | 81,920 | 0.160 | 0.177 | 0.081 |
| 160 | 128 | 20,480 | 0.158 | 0.202 | 0.079 |
| 160 | 256 | 40,960 | 0.160 | 0.171 | 0.080 |
| 160 | 512 | 81,920 | 0.158 | 0.171 | 0.079 |
| 160 | 1,024 | 163,840 | 0.155 | 0.168 | 0.078 |
| 320 | 128 | 40,960 | 0.160 | 0.174 | 0.080 |
| 320 | 256 | 81,920 | 0.161 | 0.171 | 0.081 |
| 320 | 512 | 163,840 | 0.159 | 0.166 | 0.079 |
| 320 | 1,024 | 327,680 | 0.156 | 0.165 | 0.078 |
| 512 | 128 | 65,536 | 0.152 | 0.172 | 0.078 |
| 512 | 256 | 131,072 | 0.157 | 0.168 | 0.078 |
| 512 | 512 | 262,144 | 0.159 | 0.165 | 0.078 |
| **512** | **1,024** | **524,288** | **0.159** | **0.161** | **0.078** |
| 1,024 | 128 | 131,072 | 0.157 | 0.169 | 0.079 |
| 1,024 | 256 | 262,144 | 0.160 | 0.163 | 0.078 |
| 1,024 | 512 | 524,288 | 0.159 | 0.161 | 0.078 |
| 1,024 | 1,024 | 1,048,576 | 0.161 | 0.158 | 0.080 |
| 2,048 | 128 | 262,144 | 0.161 | 0.166 | 0.078 |
| 2,048 | 256 | 524,288 | 0.159 | 0.161 | 0.078 |
| 2,048 | 512 | 1,048,576 | 0.157 | 0.157 | 0.081 |
| 2,048 | 1,024 | 2,097,152 | 0.169 | 0.157 | 0.088 |
| 4,096 | 128 | 524,288 | 0.159 | 0.161 | 0.078 |
| 4,096 | 256 | 1,048,576 | 0.159 | 0.156 | 0.081 |
| 4,096 | 512 | 2,097,152 | 0.162 | 0.155 | 0.087 |
| 4,096 | 1,024 | 4,194,304 | **0.186** | 0.162 | **0.103** |

*Table 5. Time in ms, mean of runs 2–20, for every launch size. In bold: the launch the code uses,
and the slowest cells. Every row's result equals the usual launch's.*

The test also names the fastest launch of each kernel: 512 × 128 for the forward (0.152 ms), 4,096 ×
512 for the wte backward (0.155 ms), 160 × 1,024 for the wpe backward (0.078 ms). Section 6.3
explains why these names mean little.

**Noise.** The same run also timed the usual launch through `encoder_forward` and
`encoder_backward` in two other tests, 20 steps each. The forward took 0.165 ms in both (single steps
from 0.161 to 0.171; 0.159 in Table 5, the difference likely being the shape check that
`encoder_forward` does on the CPU between the two events); the backward (both backward kernels)
0.243 and 0.257 ms. In the second of
those tests PyTorch, which was timed alongside and did not change, also took 6% longer, so the
difference was the machine, not the code. Differences of a few percent between launches are within
this noise.

## 6. Analysis

The results fall into three regions (Figure 1): too few threads, enough, and far too many.

### 6.1 Too few threads: not enough bytes in flight

![Slow-down below the T4's capacity](/images/encoder-launch-sizes/fig2-slowdown.svg)
*Figure 2. Each kernel's time while the T4 is not full, divided by its time at 512 × 1,024.*

| threads launched | share of the T4 | forward | wte backward | wpe backward |
|---:|---:|---:|---:|---:|
| 5,120 | 1/8 | 1.67x | **3.73x** | 1.09x |
| 10,240 | 1/4 | 1.04x | **2.07x** | 1.04x |
| 20,480 | 1/2 | 0.99x | 1.25x | 1.03x |
| 40,960 | all | 1.01x | 1.10x | 1.04x |

*Table 6. Slow-down against 512 × 1,024; where several launches have the same thread count (Table
5), their mean.*

The three kernels react very differently to the same number of threads. The model of Section 2.4
explains why: what matters is not threads but the bytes those threads keep in flight, and the three
kernels keep very different amounts per thread. Looking at one job of each (Table 1):

- **wte backward**: a 4-byte token id and a 4-byte number of `encoded_grad`. The `atomicAdd`'s result
  is never used, so the thread does not wait for it (it most likely compiles to a reduction
  instruction that sends the addition and goes on). About **8 bytes** in flight per thread.
- **forward**: the token id, and 16 bytes of `wpe`; when the id is back, 16 bytes of `wte`. The id
  is shared by the 192 neighbouring threads of the same token and mostly comes from the cache. About
  **16–32 bytes** per thread, and two waits per job instead of one.
- **wpe backward**: four 16-byte loads of `encoded_grad` (b = 0 to 3) that do not depend on each
  other, and 16 bytes of `wpe_grad`. If the compiler asks for all of them before adding, up to **80
  bytes** per thread. Its 38 registers, against 16 and 20 for the others, are what keeping several
  `float4`s at once would need, but I have not read the compiled code to confirm it.

Dividing the 160 KB needed by these:

| kernel | bytes in flight per thread | threads to keep 160 KB in flight | slow-down at 5,120 threads, predicted | measured |
|---|---:|---:|---:|---:|
| wte backward | ≈ 8 | ≈ 20,000 | 160 / 41 ≈ 3.9x | 3.73x |
| forward | ≈ 16–32 | ≈ 5,000–10,000 | 1 to 2x | 1.67x |
| wpe backward | up to ≈ 80 | ≈ 2,000 | ≈ 1x | 1.09x |

*Table 7. The model, with the estimated latency of Section 2.4. For the wte backward it also
predicts 160 / 82 ≈ 2.0x at 10,240 threads (measured 2.07x) and about 1x at 20,480 (measured
1.25x).*

The model gets the order of the kernels right, and for the wte backward even the size of the
slow-down at 5,120 and 10,240 threads. It is still an estimate: the latency is assumed, and the
bytes per thread come from reading the source, not from a profiler. The wte backward's residual
slow-down at 20,480 and 40,960 threads (1.25x and 1.10x) suggests that 8 bytes per thread is an
upper estimate, or that the 160 KB is too low for loads that miss the cache.

**The same number of threads, in blocks of different sizes.** At exactly 40,960 threads the wte
backward takes 0.184 ms as 40 blocks of 1,024, and 0.171–0.177 ms as 80, 160 or 320 smaller blocks.
A block holds its SM until its slowest warp finishes; with one block of 1,024 per SM, an SM whose
block ends early has nothing else to run, while with several smaller blocks per SM the end is
spread out. The [note](/posts/gpu-threads-blocks-registers/) [5] makes this argument for smaller
blocks; here it is worth 4–7%, close to the noise, and only for the kernel that is short of bytes
in flight.

### 6.2 Enough threads: the flat region

From 40,960 threads to about a million, all three kernels are flat: the forward between 0.152 and
0.161 ms, the wte backward between 0.155 and 0.177 ms (falling slowly with more threads), the wpe
backward between 0.078 and 0.081 ms. In this region the GPU is full and memory is busy; extra blocks
only wait their turn.

How busy? Counting the bytes each kernel must move at 512 × 1,024:

| kernel | bytes moved | time | bandwidth | of 320 GB/s |
|---|---|---:|---:|---:|
| wpe backward | 12.6 MB of `encoded_grad` read; 3.1 MB of `wpe_grad` read and written | 0.078 ms | 242 GB/s | 76% |
| forward | 12.6 MB of `wte` rows read; 12.6 MB of `encoded` written; `wpe`: 3.1 MB once, or 12.6 MB if read again for each row of the batch | 0.159 ms | 178 or 237 GB/s | 56 or 74% |
| wte backward | 12.6 MB of `encoded_grad` read; ≈ 3,934 rows of `wte_grad` (12.1 MB) read and written | 0.161 ms | ≈ 229 GB/s | 71% |

*Table 8. Bandwidth at the launch the code uses. The 3,934 rows are the expected number of different
ids among 4,096 drawn from 50,257 (some ids repeat); the wte backward's count assumes each touched row
goes to and from memory once.*

The wpe backward, which does no arithmetic worth the name, reaches 76% of the peak. If the forward
read `wpe` from memory only once it would be at 56%, far below; if it reads `wpe` again for each of the 4
rows of the batch, it is at 74%, with the other two. The second is the likelier: `wpe` is 3.1 MB, the
T4's L2 cache is 4 MB [10], and 25 MB of `wte` and `encoded` stream through the cache in between.
The test's earlier comment assumed the cache and gave a floor of 0.08 ms; the floor with `wpe`
read four times is 37.7 MB / 320 GB/s = 0.12 ms, and the comment now says so. All three kernels then
run at about three quarters of the peak, a little below the gradient norm's 87% [4], which only
reads.

### 6.3 The "fastest" launch is noise

Within the flat region, the fastest launch beats 512 × 1,024 by 4% for the forward, 4% for the
wte backward and 0% for the wpe backward. The noise between two tests of the same launch in the same
run was 6% for the backward (Section 5). The winners are therefore not meaningful: another run would
likely name others. The one trend that looks real is the wte backward's slow fall from 0.177 to
0.155 ms as threads grow from 81,920 to 2 million, consistent with smaller tails at the end of the
launch; it is under 10% and would need repeated runs to confirm.

### 6.4 Far too many threads: empty blocks

At 4,096 × 1,024 the forward takes 0.186 ms (+17%) and the wpe backward 0.103 ms (+32%), while the wte
backward does not change. The reason is how many blocks have nothing to do:

| kernel | jobs | blocks of 1,024 with work | empty blocks | extra time | per empty block on one SM |
|---|---:|---:|---:|---:|---:|
| forward | 786,432 | 768 | 3,328 | 0.027 ms | ≈ 0.3 µs |
| wpe backward | 196,608 | 192 | 3,904 | 0.025 ms | ≈ 0.3 µs |
| wte backward | 3,145,728 | 3,072 | 1,024 | ≈ 0 | — |

*Table 9. 4,096 blocks of 1,024. "Per empty block on one SM": the extra time divided by the empty
blocks each of the 40 SMs runs (83 and 98), one at a time. An estimate from one run.*

An empty block still has to be started on an SM, compute its first index, find it past the end, and
finish, before the next block can take its place. At one block of 1,024 per SM, that is a few
tenths of a microsecond each, and 80 to 100 of them in a row per SM add up to 0.025 ms. The 2,048 ×
1,024 launch shows the same at half the size (forward +6%, wpe backward +13%). For the wte backward
three quarters of the 4,096 blocks have work, so the cost is hidden.

## 7. Discussion

**What to launch.** For these kernels on a T4, any launch from about 40,960 threads to about a
million gives the same time within noise, and the launch the code uses, 512 × 1,024 = 524,288
threads, is in the middle of that range. It stays. The rule the results support:

1. **Launch at least as many threads as the GPU holds at once** (40,960 on a T4), more for kernels
   that keep few bytes in flight per thread. Below that, the time grows fast.
2. **Launch several times that, not a hundred times.** A few times the capacity leaves room for
   GPUs with more SMs: an A100 holds 108 × 2,048 = 221,184 threads [6], and 524,288 is still 2.4
   times that. A hundred times the capacity mostly adds empty blocks.
3. **Within that range, the block size matters little** for kernels like these, whose threads never
   work together. It matters more for kernels that use shared memory or `__syncthreads`, like the
   gradient norm, where the block size sets how many partial sums there are [4].

**Bytes, not threads.** The clearest result is that the three kernels need different numbers of
threads, and that the number depends on how many bytes each thread keeps waiting: the same point as
Volkov's, that a kernel whose threads each have more independent loads needs fewer threads to hide
the wait [11]. This points to the
better fix for the wte backward than more threads: let each thread load a whole `float4` of
`encoded_grad` and then do four `atomicAdd`s. That keeps 16 bytes in flight instead of 4, as the
forward does, and should make it as insensitive to the launch size as the forward; the atomic
additions themselves stay one float at a time.

**The TensorBuffer.** The same run also timed the forward and backward with `wte` and `wpe` in one
TensorBuffer, the trainer's single allocation for all weights, instead of a Tensor each. The times
matched (0.165 ms for the forward both ways). A kernel only sees pointers, so where the memory came
from cannot change its speed; the buffer's value is in passes over all the weights at once, as in
the gradient norm [4], not in any one kernel.

## 8. Threats to validity

- **One run.** Every number comes from one Colab run, each the mean of 19 timings. Section 5 shows
  6% between two tests of the same launch in that run, so differences below that are not results.
- **The model is an estimate.** The 0.5 µs latency is assumed, and the bytes in flight per thread
  are read from the source code. I did not use a profiler (Nsight Compute) or read the compiled
  instructions, so the claims that the `atomicAdd` does not wait and that the wpe backward issues
  its four loads together are likely, not shown.
- **The cache.** That the forward reads `wpe` from memory four times is inferred from its bandwidth
  compared with the wpe backward's, not measured.
- **Build.** The test was built without an `-arch` flag (Reproducibility), so the driver compiled the
  kernels for the T4 from PTX when the program started. The earlier entries used `-arch=native`;
  the generated code may differ slightly, and the register counts (20, 16, 38) are those of the
  driver's compilation.
- **One GPU, one size.** Only the T4 and GPT-2 small's batch of 4 × 1,024 tokens were tested. With
  more SMs, the region of too few threads moves right; with a smaller batch, empty blocks start at
  fewer threads.
- **Versions.** The PyTorch, CUDA and driver versions of the Colab session were not recorded.

## 9. Future work

- **Load a `float4` in the wte backward**, then four `atomicAdd`s, and repeat this experiment: the
  model predicts its slow-down at 5,120 threads should fall from 3.7x towards the forward's 1.7x.
- **Measure what the model assumes**: the bytes in flight and the cache hit rate for `wpe`, with
  Nsight Compute.
- **Repeat on a GPU with more SMs** (an A100 or L4 on Colab), to see the region of too few threads
  move with the capacity.
- **Choose the block count from the GPU** with `cudaGetDeviceProperties` (SM count × threads per SM ×
  a small factor) instead of a fixed 512, if another GPU shows 512 to be too few.

## 10. Conclusion

GPT-2's encoder is three memory-bound CUDA kernels. On a Tesla T4, every one of 32 launch sizes, from
5,120 to 4,194,304 threads, gives the same result, and the time has three regions. Below the 40,960
threads the T4 holds at once, the kernels slow down by as much as their threads fall short of
keeping about 160 KB in flight: 3.7x for the kernel that loads 4 bytes at a time, 1.7x for the one
that loads 16, 1.09x for the one that loads four 16-byte values at once. From there to about a
million threads, every launch is equal within noise, at about three quarters of the T4's peak
bandwidth. Past that, empty blocks add up to a third. The launch the code uses, 512 blocks of 1,024
threads, is in the flat middle and stays.

## Reproducibility

Code at commit [`a1f1b7b`](https://github.com/ducbachsong/gpt2-small/tree/a1f1b7b). On Google Colab
with a T4 (PyTorch, and with it libtorch, are preinstalled):

```bash
git clone https://github.com/ducbachsong/gpt2-small.git
cd gpt2-small && git checkout a1f1b7b

TORCH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
nvcc -O3 -std=c++17 tests/test_encoder.cu -o test_encoder \
    -D_GLIBCXX_USE_CXX11_ABI=$ABI \
    -I$TORCH/include -I$TORCH/include/torch/csrc/api/include \
    -L$TORCH/lib -Xlinker --no-as-needed -ltorch -ltorch_cuda -ltorch_cpu -lc10_cuda -lc10 \
    -Xlinker -rpath=$TORCH/lib
./test_encoder
```

This is the command of the run in this entry. Adding `-arch=native` builds for the GPU in the
machine instead of compiling from PTX at start-up.

## References

1. A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever. *Language Models are Unsupervised
   Multitask Learners.* OpenAI, 2019.
2. A. Karpathy. *llm.c: LLM training in simple, raw C/CUDA.*
   [github.com/karpathy/llm.c](https://github.com/karpathy/llm.c)
3. D. B. Song. *One launch: a fused AdamW kernel for GPT-2 in CUDA.* 2026.
   [/research/fused-adamw-cuda/](/research/fused-adamw-cuda/)
4. D. B. Song. *Adding 124 million squares: a one-kernel gradient norm for GPT-2 in CUDA.* 2026.
   [/research/grad-norm-cuda/](/research/grad-norm-cuda/)
5. D. B. Song. *How a CUDA kernel uses threads, warps, blocks and registers.* 2026.
   [/posts/gpu-threads-blocks-registers/](/posts/gpu-threads-blocks-registers/)
6. NVIDIA. *CUDA C++ Programming Guide* (compute capabilities and their limits; `atomicAdd` on
   `float2` and `float4` from compute capability 9.x).
   [docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-programming-guide/)
7. Z. Jia, M. Maggioni, J. Smith, D. P. Scarpazza. *Dissecting the NVidia Turing T4 GPU via
   Microbenchmarking.* 2019. [arXiv:1903.07486](https://arxiv.org/abs/1903.07486)
8. NVIDIA. *Tesla T4 Tensor Core GPU* (datasheet: 320 GB/s memory bandwidth).
   [nvidia.com](https://www.nvidia.com/en-us/data-center/tesla-t4/)
9. J. D. C. Little. *A Proof for the Queuing Formula: L = λW.* Operations Research 9(3), 1961.
10. NVIDIA. *NVIDIA Turing GPU Architecture* (whitepaper: the TU104's 4 MB L2 cache). 2018.
11. V. Volkov. *Better Performance at Lower Occupancy.* GPU Technology Conference, 2010.

## Appendix A. Raw log

Index: [README.txt](/files/encoder-launch-sizes/README.txt).

| log | what |
|---|---|
| [test-encoder-t4.log](/files/encoder-launch-sizes/test-encoder-t4.log) | every check of `tests/test_encoder.cu`, the two 20-step timing runs (a Tensor each, one TensorBuffer), and the 32 launch sizes |
