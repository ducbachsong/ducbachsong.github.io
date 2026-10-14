---
title: "How many threads? Launch sizes for GPT-2's encoder and LayerNorm kernels on a T4"
collection: publications
date: 2026-10-07
modified: 2026-10-14
version: 4
permalink: /research/encoder-launch-sizes/
excerpt: "GPT-2's encoder as three CUDA kernels, timed on a Tesla T4 with 32 launch sizes each, in 16 runs. Every launch gives the same result. With fewer threads than the 40,960 a T4 holds at once, the backward into the token table, which loaded one float at a time, slowed down 2.2-2.7x; the others hardly at all. A bytes-in-flight model predicted the order, and a test that changed only the load width confirmed the cause: with one float4 per load the slow-down fell to 1.02-1.06x, so the kernel now loads float4s. Part 2 (version 4) does the same for GPT-2's LayerNorm kernels and finds the opposite: both are fastest with far fewer warps than a T4 holds, most likely because each row is read again from the L1 cache. PyTorch's own launch rule stays inside the cache; my kernels can beat it with a launch chosen for this shape and a backward fused into one kernel."
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
these numbers mean but measured nothing. This entry measures them, on the three kernels of GPT-2's
encoder from my C/CUDA trainer, at GPT-2 small's sizes on a Tesla T4: each kernel with 8 block
counts times 4 block sizes, from 5,120 to 4,194,304 threads, in three batches of 6, 5 and 5 runs.
**Every launch gives the same result**, because every kernel walks its data with a grid-stride loop.
**With enough threads, from the 40,960 a T4 holds at once to about a million, the time is flat**;
far beyond that, blocks with nothing to do cost up to 40%. **Below 40,960 threads, the kernels
differ**: the backward into the token table, which loaded 4 bytes at a time, took 2.2–2.7x as long
at 5,120 threads, while the backward into the position table, which loads four 16-byte values at
once, hardly slowed down. A simple model, the bytes each thread keeps waiting on memory, predicted
that order. Then I tested it: the same kernel with only its load widened to 16 bytes slowed down by
**1.02–1.06x instead of 2.2–2.7x** in all ten runs, so the cause is confirmed and the kernel now
loads float4s. Two findings came along the way. The first runs varied by up to 1.6x between runs
because the GPU's clock was still rising after seconds of idling: measurements need a warm-up.
And one launch, **768 blocks of 128 threads, runs the forward 12% faster** than the 512 × 1,024 the
code uses, in all 11 runs that tried it, for a reason I have not found yet.

**Part 2: LayerNorm.** The same test on GPT-2's two LayerNorm kernels, which read each row of 768
numbers two or three times, gives the opposite picture. All 40 launches give the same result, but
**both kernels are fastest with far fewer warps than a T4 holds**: the forward 30% and the backward
17% faster than with the launches the code uses. Most likely the re-reads of a row come from the SM's
L1 cache only while the rows an SM works on fit in it. PyTorch, which works out its launch from the
tensor's shape, gives each row a block of 4 warps and so stays inside the cache without knowing the
GPU; at my usual launch its forward was as fast as mine or faster. My backward was 1.09–1.35x faster
than PyTorch's, most likely because it is one kernel that moves about half the bytes, not because of
its launch.

*Version 4, October 14, 2026: adds part 2, LayerNorm ([Section 11](#11-part-2-layernorm)). What each
version changed is listed under [Versions](#versions).*

* TOC
{:toc}

## 1. Introduction

I am pre-training a GPT-2 small [1] from scratch, rebuilding the trainer in C and CUDA in the style
of Karpathy's llm.c [2], with a test against PyTorch for every piece. The code is open source:
[github.com/ducbachsong/gpt2-small](https://github.com/ducbachsong/gpt2-small/tree/c-cuda-trainer)
(branch `c-cuda-trainer`). Two commits matter here: [`a1f1b7b`](https://github.com/ducbachsong/gpt2-small/tree/a1f1b7b),
the encoder as first measured, and [`e84ce51`](https://github.com/ducbachsong/gpt2-small/tree/e84ce51),
after the changes this entry led to. Links to the code go to `e84ce51`.

The previous entries built the optimiser's update [3] and the gradient norm [4]; both launch 512
blocks of 1,024 threads, a choice I made once and never measured. The encoder, the model's first
layer, uses the same launch. I asked two questions that the [note](/posts/gpu-threads-blocks-registers/)
[5] answers in principle but not with numbers: would a different number of blocks or threads make
the kernels faster, and how far can the numbers move before something goes wrong? The first answer
raised a third question, why the kernels react so differently, and most of this entry is about
answering that one properly: with a prediction made before the run, not an explanation fitted after.
Part 2 (Section 11), added a week later, asks the same questions of the next kernels of the trainer,
LayerNorm's, and a fourth: PyTorch works out its own launch, so why can a kernel with a launch of its
own be faster?

**Contributions.**

1. A test that launches each kernel directly with any `<<<blocks, threads>>>`, times it, and checks
   that its result equals the usual launch's (Section 3).
2. Timings of the three kernels for 32 launch sizes on a Tesla T4, in 16 runs (Section 5).
3. The compiled code of each kernel, read for what each thread waits on (Section 6.1).
4. A bytes-in-flight explanation of why the kernels need different numbers of threads, and a
   controlled test that confirms it: only the load width changes, and the slow-down goes away
   (Sections 6.2–6.3). The encoder's kernel was changed as a result.
5. Two lessons about measuring: a GPU needs a warm-up before short kernels are timed (Section 6.4),
   and launches should be compared inside one run, not across runs (Section 6.6).
6. Part 2: the same test for GPT-2's two LayerNorm kernels at 40 launch sizes, where fewer warps is
   faster, and a comparison with PyTorch's kernels and the launch PyTorch works out itself
   (Section 11).

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
[`llmc/kernels/encoder.cuh`](https://github.com/ducbachsong/gpt2-small/blob/e84ce51/llmc/kernels/encoder.cuh);
it has three kernels:

| kernel | one job | jobs | loads per job |
|---|---|---:|---|
| **forward** | one `float4` (4 numbers) of `encoded` | 786,432 | the token id; then 16 bytes of `wte` and 16 bytes of `wpe` |
| **wte backward**, as first written | one number of `encoded_grad`, added onto its token's row of `wte_grad` | 3,145,728 | the token id and 4 bytes of `encoded_grad`; then one `atomicAdd` |
| **wpe backward** | one `float4` of `wpe_grad`: the 4 rows of the batch at that position, added | 196,608 | four 16-byte values of `encoded_grad`; 16 bytes of `wpe_grad` |

*Table 1. The three kernels as first measured (commit `a1f1b7b`). A token can appear many times in a
batch, so several threads may add onto the same row of `wte_grad`; `atomicAdd` keeps those additions
from overwriting each other. A T4 can only `atomicAdd` one `float` at a time (an `atomicAdd` on a
`float4` needs compute capability 9.0 [6]), which is why the wte backward was written one number at
a time. The position rows have no such conflict: one thread owns each `float4` of `wpe_grad`.*

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

The T4 has **40 SMs**, so at most **40 × 1,024 = 40,960 threads run at once**. The kernels use 16 to
38 registers per thread (the test prints them), at most 38 × 1,024 = 38,912 per SM, so registers
never limit them, and every block size here fills an SM:

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

A load from GPU memory takes on the order of half a microsecond to come back. A thread can ask for a
value and go on; it stops only when it uses the value. To keep memory streaming at its 320 GB/s [8],
the GPU needs enough bytes asked for and not yet back, *in flight*, to cover that wait. By Little's
law [9], as in the [gradient norm entry](/research/grad-norm-cuda/) [4]:

$$
\text{bytes in flight needed} \approx \text{bandwidth} \times \text{latency} \approx 320\ \text{GB/s} \times 0.5\ \mu\text{s} \approx 160\ \text{KB}.
$$

The half microsecond is an assumption, so the 160 KB is a first estimate; Section 6.3 measures a
better one. Bytes in flight come from threads: each thread keeps a few bytes waiting, and more
resident threads keep more waiting. Below some number of threads the GPU cannot ask for enough,
memory sits partly idle, and the kernel is slower. Above 40,960 threads, the extra threads cannot
run anyway; their blocks wait their turn.

## 3. Method

The usual entry points, `encoder_forward` and `encoder_backward`, always launch `ENCODER_BLOCKS` ×
`ENCODER_THREADS` = 512 × 1,024. The tests in
[`tests/test_encoder.cu`](https://github.com/ducbachsong/gpt2-small/blob/e84ce51/tests/test_encoder.cu)
launch the kernels directly instead, with any size:

```cpp
encoder_forward_kernel<<<blocks, threads>>>((float4 *)encoded.data(), token_ids.data_ptr<int>(),
                                            (const float4 *)wte.data(), (const float4 *)wpe.data(),
                                            batch_size, seq_len, embed_dim);
```

**Workload.** GPT-2 small's sizes: 4 × 1,024 random token ids from 0 to 50,256, `wte`, `wpe` and
`encoded_grad` filled with random normal values.

**Launch sizes.** 8 block counts × 4 block sizes, from 40 × 128 = 5,120 threads, an eighth of what
the T4 holds, to 4,096 × 1,024 = 4,194,304, about a hundred times more:

| blocks | 40, 80, 160, 320, 512, 1,024, 2,048, 4,096 |
|---|---|
| threads per block | 128, 256, 512, 1,024 |

**Timing.** For each launch size, each kernel runs 20 times, each run timed on the GPU's clock with
a CUDA event before and after it; the first run is left out and the time is the mean of runs 2–20.
Then the result is checked (Section 4).

**Tests and batches.** The work went in three steps, each a batch of runs on Colab, each run the
whole test program:

| batch | runs | code | what it adds |
|---|---:|---|---|
| A | 6 | `a1f1b7b` | part 4: the 3 kernels × 32 launch sizes; no warm-up |
| B | 5 | `a1f1b7b` + a first part 5 | part 5: the wte backward with float loads against a copy with float4 loads, at 32 launch sizes; the forward with 256–1,024 blocks of 128 threads; the GPU clock logged |
| C | 5 | `e84ce51` | the float4 wte backward in `encoder.cuh`; a one-second warm-up before every timing loop; the GPU clock logged |

*Table 3. The batches. In part 5, the two kernels compared, or the launch and its reference, are timed
one right after the other, so both see the same state of the GPU.*

| | |
|---|---|
| hardware | Google Colab, NVIDIA Tesla T4 (16 GB, 320 GB/s peak [8], 40 SMs [7]) |
| software | CUDA C++17, `nvcc -O3`, which on Colab builds machine code for the T4 (sm_75) directly; libtorch from Colab's preinstalled PyTorch, for the checks |

*Table 4. Environment.*

## 4. Correctness

Before timing, each test runs the usual launch once and keeps its results on the GPU as PyTorch
tensors. After timing each launch size, it sets the outputs to 0, runs that launch once, and
compares:

| result | criterion | why |
|---|---|---|
| `encoded` | equal to the last bit | each number is one addition of the same two values |
| `wpe_grad` | equal to the last bit | each thread adds b = 0, 1, 2, 3 in that order, whatever the launch |
| `wte_grad` | within $$10^{-5}$$ of the largest value | `atomicAdd`s land in whatever order threads arrive; floating-point addition is not associative |

*Table 5. The check of each launch.*

**Every launch passed in every run**: 32 in part 4 and, in batches B and C, 32 for each wte backward
and 13 for the forward sweep. The rest of the encoder's test suite also passes in every run: the
kernels against PyTorch's `embedding` with autograd (equal for `encoded`, within $$10^{-6}$$ for the
gradients), seven tests of numbers worked out by hand, and a check that each kernel fits a block of
1,024 threads without spilling. The logs are in [Appendix A](#appendix-a-raw-logs).

## 5. Results

### 5.1 Batch A: three regions

![Encoder kernel time against threads launched, batch A](/images/encoder-launch-sizes/fig1-time-vs-threads.svg)
*Figure 1. Batch A: all 96 timings (3 kernels × 32 launches) against the threads launched; the dot is
the median of 6 runs, the bar runs from the fastest to the slowest run. The rings mark 512 × 1,024.*

| threads launched | forward | wte backward (float loads) | wpe backward |
|---:|---:|---:|---:|
| 5,120 (40 × 128) | 1.17–1.80x | **2.53–4.02x** | 1.04–1.14x |
| 10,240 | 0.96–1.10x | **1.45–2.24x** | 1.01–1.06x |
| 20,480 | 0.96–1.00x | 1.09–1.31x | 1.00–1.05x |
| 40,960 | 0.98–1.03x | 1.07–1.11x | 1.01–1.06x |
| **512 × 1,024**, in ms | **0.158–0.164** | **0.161–0.164** | **0.077–0.080** |
| 4,096 × 1,024, in ms | 0.186–0.187 | 0.162–0.164 | 0.103–0.104 |

*Table 6. Batch A, 6 runs. The first rows are the time divided by the same run's time at 512 × 1,024
(several launches with the same thread count: their mean); the lowest and highest of the 6 runs.*

Three regions show. **Too few threads**: below 40,960 the kernels slow down, the wte backward by far
the most. **Enough**: from 40,960 to about a million threads every kernel is flat. **Far too many**:
at four million threads the forward and the wpe backward slow down again. The full table of every
launch is in the logs.

### 5.2 How much to trust a difference

The flat region repeats well between runs: no launch there moved by more than 6.4% across the 6
runs. The region of too few threads does not: the wte backward at 5,120 threads ranged from 2.5x to
4.0x, which Section 6.4 traces to the GPU's clock.

Between launches of the same run, smaller differences turned out to be real. Batch A's test names
the fastest launch of each kernel, and for two kernels it named the same one every time:

| kernel | launch | faster than 512 × 1,024 in | by |
|---|---|---:|---|
| forward | 512 × 128 | 6 of 6 runs | 3.8–6.1% |
| wte backward (float loads) | 4,096 × 512 | 6 of 6 runs | 3.1–4.9% |
| wpe backward | a different one each run | — | 0–1%, ties |

*Table 7. Batch A. A difference that has the same sign in every run is real even when whole runs move
by more than its size: the launches of one run are timed seconds apart, the runs minutes apart.*

## 6. Analysis

### 6.1 What each thread waits for: the compiled code

The machine code of the test program shows what each kernel's loop asks memory for
(`cuobjdump -sass`, [Appendix A](#appendix-a-raw-logs)). `LDG.E.128` is a 16-byte load, `LDG.E` a
4-byte one, and `RED` an atomic addition whose result is not used:

| kernel | one job, as compiled | bytes in flight per thread |
|---|---|---:|
| wte backward, float loads | `LDG.E` token id + `LDG.E` 4 bytes, issued together; then one `RED` | ≈ 8 |
| forward | `LDG.E` token id, alone; when it is back, `LDG.E.128` of `wte` + `LDG.E.128` of `wpe` together | 4, then 32 |
| wpe backward | 4 × `LDG.E.128` back to back (the loop over b, unrolled by 4) | 64 |
| wte backward, float4 loads (Section 6.3) | `LDG.E` token id + `LDG.E.128` 16 bytes, together; then 4 × `RED` | ≈ 20 |

*Table 8. From the compiled program (sm_75). `RED` is sent and not waited for, so an `atomicAdd`
whose result is unused does not hold the thread up.*

### 6.2 Too few threads: the bytes-in-flight explanation

Section 2.4 says what matters is not the number of threads but the bytes those threads keep in
flight. Table 8 shows the three kernels keep very different amounts per thread, and the order
matches the order of their slow-downs in batch A:

| kernel | bytes per thread | slow-down at 5,120 threads (batch A) |
|---|---:|---:|
| wte backward, float loads | 8 | 2.53–4.02x |
| forward | 4, then 32 | 1.17–1.80x |
| wpe backward | 64 | 1.04–1.14x |

*Table 9. The fewer bytes each thread keeps waiting, the more threads the kernel needs.*

This is an explanation fitted to numbers already measured, which proves little: many explanations
fit three points. It does make a prediction that could fail. If the bytes in flight are the cause,
then widening only the wte backward's load, with nothing else changed, should make its slow-down
small, like the wpe backward's. If they are not the cause, the slow-down should stay.

### 6.3 Testing it: the same kernel with a float4 load

The test: the wte backward loads one `float4` of `encoded_grad` per job and then does four `float`
`atomicAdd`s, which a T4 supports. The additions are the same and land in the same places; only the
load is 16 bytes instead of 4, about 20 bytes in flight per thread instead of 8 (Table 8). Part 5
times both versions at all 32 launch sizes, one right after the other, and checks both against the
usual result. The prediction was written into the test before it ran.

![Slow-down below the T4's capacity, batch C](/images/encoder-launch-sizes/fig2-slowdown.svg)
*Figure 2. Batch C, with the warm-up of Section 6.4. Both wte backwards are from part 5 (timed side by
side), the forward and the wpe backward from part 4.*

| slow-down against 512 × 1,024 | 5,120 threads | 10,240 | 20,480 | 40,960 |
|---|---:|---:|---:|---:|
| wte backward, float loads, batch B | 2.18–2.21x | 1.24–1.28x | | |
| wte backward, **float4 loads**, batch B | **1.01–1.03x** | 0.99–1.02x | | |
| wte backward, float loads, batch C | 2.53–2.68x | 1.36–1.42x | 1.07–1.10x | 1.06–1.09x |
| wte backward, **float4 loads**, batch C | **1.02–1.06x** | 1.01–1.04x | 1.01–1.04x | 1.01–1.03x |

*Table 10. Part 5, 10 runs. Batch B printed only the first two columns. One run of batch C is left
out of the float4 rows: its reference at 512 × 1,024 came out at 0.188 ms instead of 0.163–0.167,
which made all its ratios fall below 1 (the log shows it).*

**The prediction held in all 10 runs.** With the wider load, the slow-down at 5,120 threads fell from
2.2–2.7x to 1.02–1.06x. Nothing else changed, so the bytes in flight are the cause, not just a story
that fits. At 512 × 1,024 both versions take the same time (0.161–0.166 against 0.163–0.167 ms), so
**`encoder.cuh` now uses the float4 version** (commit `e84ce51`).

**What the float4 load changes, and what it does not.** The float4 version does not add four numbers
at once. A T4 has no `atomicAdd` on a `float4` (Table 1), and the compiled code shows four separate
`RED`s per job, one per float, at `[R4]`, `[R4+0x4]`, `[R4+0x8]` and `[R4+0xc]` ([Appendix
A](#appendix-a-raw-logs)). Over the whole kernel both versions send the same 3,145,728 `RED`s
(3,145,728 jobs × 1 against 786,432 jobs × 4), the L2 cache carries out every one of them, and two
that land on the same number of `wte_grad` are applied one after the other in either version. The
bytes read do not change either: both read the same 12.6 MB of `encoded_grad`. What changes is how
much each thread asks for before it has to wait: 16 bytes of `encoded_grad` in one load instead of 4,
about 20 bytes in flight instead of 8 (Table 8). So the same 5,120 threads keep 2.5 times as many
bytes on their way, and memory stays busy with fewer threads. With enough threads, memory is busy
either way, which is why the two take the same time at 512 × 1,024. The forward is no different on
the adding side: a T4 has no instruction that adds two `float4`s, so its `make_float4(a.x + b.x, …)`
is four float additions as well, and what it gains from `float4` is also the load.

**The model's size was wrong, its direction right.** With the first estimate of 160 KB needed, the
model predicted 3.9x for float loads at 5,120 threads (160 KB / (5,120 × 8 B) = 160 / 41) and about
1.6x for float4 loads (160 / 102). The measured 2.2–2.7x and 1.02–1.06x both fit a need of about
**90–110 KB** instead: 41 KB is less than half of that, 102 KB is about all of it. That figure is
fitted to these two measurements, not measured on its own; it corresponds to a latency of about a
third of a microsecond at the T4's peak bandwidth.

### 6.4 The first measurements were unreliable: the GPU clock

Batch A's slow-downs below 40,960 threads changed a lot from run to run (2.5–4.0x for the wte
backward), while in part 5 of batch B the same kernel, at the same launch, in the same programs,
gave 2.18–2.21x every time. The difference is when they were measured. Just before part 4's loop,
the test spends seconds making 38 million random numbers on the CPU while the GPU idles, and an idle
GPU lowers its clock (the logs show 300 MHz). The first launches of part 4 are exactly the ones with
the fewest threads, and they ran while the clock was still rising. Part 5 ran right after part 4,
on a GPU that was already busy.

Two observations support this. In batch B, where `nvidia-smi` logged the clock every 200 ms, the
runs' part 4 times at 40 × 128 sort exactly by their median SM clock: 675 MHz gave 0.658 ms, 795 gave
0.607, 810 gave 0.543, 990 gave 0.477 and 1,110 gave 0.444. An earlier logging attempt did not show
this order, and a median over a whole run is a coarse measure of the clock during a 12 ms
measurement, so this is support, not proof. The stronger evidence is the fix: batch C keeps the GPU
busy for one second before every timing loop, and then

| part 4, slow-down at 5,120 threads | batch A (no warm-up) | batch C (warm-up) |
|---|---:|---:|
| forward | 1.17–1.80x | **1.09–1.14x** |
| wpe backward | 1.04–1.14x | **1.00–1.04x** |
| wte backward | 2.53–4.02x (float loads) | **1.02–1.04x** (float4 loads) |

*Table 11. Part 4 before and after the warm-up. The SM clock's median over each run rose from 675–1,110
MHz (batch B) to 1,110–1,222 MHz (batch C).*

With the warm-up the starved region is small and steady for every kernel: the forward's large and
variable slow-down in batch A was mostly the clock. The kernels that keep enough bytes in flight
need far fewer threads than 40,960. Why starved launches feel the clock and full ones do not is
consistent with the model: a starved kernel waits on the latency of each load, which depends partly
on the chip's clock, while a full one is limited by the memory's bandwidth, which does not.

The float-load slow-down itself still differed between batches B and C (2.2x against 2.5–2.7x), while
staying steady within each. I do not know why; both are far from the float4 version's.

### 6.5 Enough threads: the flat region

From 40,960 threads to about a million, all three kernels are flat. Counting the bytes each kernel
must move at 512 × 1,024 shows how busy memory is:

| kernel | bytes moved | time | bandwidth | of 320 GB/s |
|---|---|---:|---:|---:|
| wpe backward | 12.6 MB of `encoded_grad` read; 3.1 MB of `wpe_grad` read and written | 0.078 ms | 242 GB/s | 76% |
| forward | 12.6 MB of `wte` rows read; 12.6 MB of `encoded` written; `wpe`: 3.1 MB once, or 12.6 MB if read again for each row of the batch | 0.161 ms | 176 or 234 GB/s | 55 or 73% |
| wte backward | 12.6 MB of `encoded_grad` read; ≈ 3,934 rows of `wte_grad` (12.1 MB) read and written | 0.163 ms | ≈ 226 GB/s | 71% |

*Table 12. Bandwidth at 512 × 1,024, medians of batch C. The 3,934 rows are the expected number of
different ids among 4,096 drawn from 50,257; the wte backward's count assumes each touched row goes
to and from memory once.*

The wpe backward, which does no arithmetic worth the name, reaches 76% of the peak. If the forward
read `wpe` from memory only once it would be at 55%; if it reads `wpe` again for each of the 4 rows
of the batch, it is at 73%, with the other two. The second is the likelier: `wpe` is 3.1 MB, the T4's
L2 cache is 4 MB [10], and 25 MB of `wte` and `encoded` stream through the cache in between. All
three kernels then run at about three quarters of the peak, a little below the gradient norm's 87%
[4], which only reads.

### 6.6 Small differences that repeat, and one that is large

Inside the flat region, Table 7's differences of 4–6% repeat in every run of batch A, and batch C
repeats them: the forward is fastest at 512 × 128 in all 5 runs (0.153–0.155 ms against 0.160–0.161).
For the old float-load wte backward, more threads helped a little (0.155–0.156 ms at 4,096 × 256
against 0.161–0.166 at 512 × 1,024); the float4 version, with 4 times fewer jobs, does not gain
from it.

The forward's 512 × 128 was the starting point of a finer test: blocks of 128 threads, from 256 to
1,024 blocks in steps of 64, each timed right after 512 × 1,024.

![The forward with blocks of 128 threads](/images/encoder-launch-sizes/fig3-sweep.svg)
*Figure 3. Batch C, 5 runs: each launch's time divided by the time of 512 × 1,024 measured just before
it.*

| blocks × 128 | 256 | 384 | 512 | 640 | 704 | **768** | 832 | 896 | 1,024 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| against 512 × 1,024 | 1.00–1.01 | 0.98–0.99 | 0.95–0.96 | 1.00 | 0.96–0.97 | **0.877–0.892** | 0.90–0.91 | 0.98 | 0.98–0.99 |

*Table 13. Batch C, 5 runs (the full sweep is in the logs). Batch B's run 1 and an earlier execution of
batch C's code gave 768 × 128 the same 0.878–0.885.*

It is not a smooth dip. Most launches are within 2–5% of 512 × 1,024, but **768 × 128 is 12% faster
in all 11 runs that tried it**: 0.140–0.144 ms, which would be about 265 GB/s, 83% of the peak, if
`wpe` is read four times. The even split of the work does not explain it: 384 × 128 gives every
thread exactly 16 jobs and 768 × 128 exactly 8, but batch A's three launches with exactly 3 jobs per
thread were not faster.

One idea fits part of the pattern. With $$N$$ threads, a thread's jobs are $$N$$ float4s apart, and one
row of the batch is 196,608 float4s, so a thread comes back to the same position of `wpe` every
196,608 / $$N$$ jobs when that is a whole number: every 2nd job at 768 × 128, every 3rd at 512 × 128,
every 4th at 384 × 128 and every 6th at 256 × 128. Their speed-ups shrink in the same order (12%,
4.5%, 2%, 0%), as if a `wpe` row read soon again were still in the SM's own small L1 cache. But
832 × 128 does not come back to the same position at all and is 9–10% faster too, so this is a lead,
not an explanation. Nsight Compute's cache hit rates would test it.

### 6.7 Far too many threads: empty blocks

At 4,096 × 1,024 the forward takes 13–19% longer than at 512 × 1,024 and the wpe backward 30–41%
longer (batches A and C). The reason is how many blocks have nothing to do:

| kernel | jobs | blocks of 1,024 with work | empty blocks | extra time | per empty block on one SM |
|---|---:|---:|---:|---:|---:|
| forward | 786,432 | 768 | 3,328 | ≈ 0.027 ms | ≈ 0.3 µs |
| wpe backward | 196,608 | 192 | 3,904 | ≈ 0.025–0.032 ms | ≈ 0.3 µs |
| wte backward, float loads | 3,145,728 | 3,072 | 1,024 | ≈ 0 | — |
| wte backward, float4 loads | 786,432 | 768 | 3,328 | ≈ 0.027 ms | ≈ 0.3 µs |

*Table 14. 4,096 blocks of 1,024, batch A and C. "Per empty block on one SM": the extra time divided by
the empty blocks each of the 40 SMs runs, one at a time. An estimate.*

An empty block still has to be started on an SM, compute its first index, find it past the end, and
finish, before the next block can take its place: a few tenths of a microsecond each, and 80 to 100
of them in a row per SM add up. The float4 wte backward, with a quarter of the jobs, now pays this
cost like the forward (0.189–0.192 ms against 0.163–0.164 for the float-load version at this
launch). Launches this large are not used.

## 7. Discussion

**What to launch.** For these kernels on a T4, any launch from about 40,960 threads to about a million
is within a few percent, and the launch the code uses, 512 × 1,024 = 524,288 threads, is in the
middle of that range. The rule the results support:

1. **Launch at least as many threads as the GPU holds at once** (40,960 on a T4). Below that, how much
   the time grows depends on the kernel (rule 3).
2. **Launch several times that, not a hundred times.** A few times the capacity leaves room for
   GPUs with more SMs: an A100 holds 108 × 2,048 = 221,184 threads [6], and 524,288 is still 2.4
   times that. A hundred times the capacity mostly adds empty blocks.
3. **Load wide.** The kernel that loaded 4 bytes at a time needed several times more threads than
   the ones that load 16; with a 16-byte load it stopped needing them. It is the same point as
   Volkov's: a kernel whose threads each keep more loads waiting needs fewer threads to hide the wait
   [11]. On a T4 this works even around an `atomicAdd` that can only add one float, because the
   additions are not waited for.

**What changed in the code.** The wte backward now loads `float4`s (commit `e84ce51`): the same time
at 512 × 1,024, the same result within $$10^{-5}$$, and almost no slow-down with few threads. The
launch stays at 512 × 1,024. The forward could be 12% faster at 768 × 128, but that is about 0.02 ms
per step, and I would rather understand why before tuning to one GPU's particular launch.

**Measuring.** Two lessons outlast these kernels. A GPU that has been idle needs a warm-up before
short kernels are timed, or the first measurements depend on the clock instead of the code. And a
difference between two launches is best judged from the two measured side by side in one run,
repeated over runs, not from separate runs: whole runs moved by up to 6% while differences of 4% kept
their sign in every run.

**The TensorBuffer.** The same program also times the forward and backward with `wte` and `wpe` in
one TensorBuffer, the trainer's single allocation for all weights, against a Tensor each. The times
match within the run-to-run noise. A kernel only sees pointers, so where the memory came from cannot
change its speed; the buffer's value is in passes over all the weights at once, as in the gradient
norm [4], not in any one kernel.

## 8. Threats to validity

- **One GPU, one size.** Only the T4 and GPT-2 small's batch of 4 × 1,024 tokens were tested. With
  more SMs, the region of too few threads moves right; with a smaller batch, empty blocks start at
  fewer threads.
- **Colab machines.** The 16 runs come from several Colab sessions, which may not have had the same
  T4. Batches differ in places (the float-load slow-down: 2.2x in B, 2.5–2.7x in C) while each batch is
  steady.
- **The fitted need.** The 90–110 KB in flight is fitted to the float and float4 measurements, and the
  bytes per thread are read from the compiled code, not measured with a profiler. The test of Section
  6.3 confirms the cause; it does not measure the latency.
- **The clock.** The clock was sampled every 200 ms, much coarser than the measurements; the evidence
  that it caused batch A's variation is the order in batch B and the effect of the warm-up, not a
  clock reading during each launch.
- **The cache.** That the forward reads `wpe` from memory four times is inferred from its bandwidth.
  The L1 idea for 768 × 128 is untested.
- **One odd measurement.** In batch C's run 5 the float4 wte backward at 512 × 1,024 took 0.188 ms, in
  no other run more than 0.167; that run is left out of Table 10's float4 rows.
- **The 20-step tests** that time `encoder_forward` and `encoder_backward` have no warm-up; their
  backward varied from 0.245 to 0.282 ms across batch C's runs, while part 4's two backward kernels at
  the same launch added up to 0.240–0.245 ms.
- **Software versions.** The PyTorch, CUDA and driver versions of the Colab sessions were not recorded.

## 9. Future work

- **Find out why 768 × 128 is fast**: Nsight Compute's L1 and L2 hit rates for `wpe` at 512 × 1,024,
  768 × 128 and 832 × 128, and the same sweep with `seq_len` = 512, which halves the row of the batch.
- **Repeat on a GPU with more SMs** (an L4 or A100 on Colab), to see the region of too few threads
  move with the capacity, and whether 768 × 128 stays special.
- **Add the warm-up to the 20-step tests**, so every timing in the program is taken on a busy GPU.
- **Choose the block count from the GPU** with `cudaGetDeviceProperties` (SM count × threads per SM ×
  a small factor) instead of a fixed 512, if another GPU shows 512 to be too few.

## 10. Conclusion

GPT-2's encoder is three memory-bound CUDA kernels. On a Tesla T4, every one of 32 launch sizes, from
5,120 to 4,194,304 threads, gives the same result in all 16 runs. From the 40,960 threads a T4 holds
at once to about a million, every launch is within a few percent, at about three quarters of the
T4's peak bandwidth; far beyond, empty blocks add up to 40%. Below 40,960, what decides the time
is not the number of threads but the bytes they keep waiting on memory: the kernel that loaded 4
bytes at a time took 2.2–2.7x as long at 5,120 threads, and the same kernel loading 16 bytes took
1.02–1.06x. That test confirmed the explanation and changed the code. Along the way, the GPU's clock
turned out to need a warm-up before measuring, and one launch, 768 × 128, runs the forward 12% faster
for a reason still to be found. The launch the code uses, 512 blocks of 1,024 threads, stays.

## 11. Part 2: LayerNorm

*Added in version 4, October 14, 2026.*

Part 1 ended with a rule: launch at least as many threads as the GPU holds, and a few times more does
no harm. LayerNorm, the next kernels of the trainer, breaks it. This part runs the same test on them,
explains the result as far as one run allows, and compares them with PyTorch's kernels, which choose
their own launch from the tensor's shape.

### 11.1 The two kernels

LayerNorm moves each row of 768 numbers, one token, to mean 0 and standard deviation 1, then scales
and shifts each column [13]:

$$
\text{norm} = (x - \text{mean}) \cdot \text{rstd}, \qquad \text{rstd} = \frac{1}{\sqrt{\text{variance} + 10^{-5}}}, \qquad \text{output} = \text{norm} \cdot \text{weight} + \text{bias}
$$

GPT-2 small runs 25 of them in every forward pass. The code is in
[`llmc/kernels/layer_norm.cuh`](https://github.com/ducbachsong/gpt2-small/blob/c730f66/llmc/kernels/layer_norm.cuh)
(commit [`c730f66`](https://github.com/ducbachsong/gpt2-small/tree/c730f66)); the backward's formula
is derived step by step in the [note on LayerNorm's backward](/posts/layer-norm-backward/) [13]. Both
kernels give each row to one warp of 32 threads, which walks it a `float4` at a time and adds up a
sum across its threads with five shuffles; no other warp and no shared memory is involved:

| kernel | launch in the code | passes over a row | bytes to and from GPU memory |
|---|---|---|---|
| **forward** | 512 × 256: 4,096 warps, one row each | 3: the mean, the variance around it, the output | `input` read, `output` written: 25.2 MB |
| **backward** | 128 × 512: 2,048 warps, two rows each | 2: two sums over the row, then the three gradients | `input` and `output_grad` read, `input_grad` read and written (it adds onto it): 50.3 MB |

*Table 15. The two kernels at GPT-2 small's sizes, 4 × 1,024 = 4,096 rows of 768; a tensor of that
shape is 12.6 MB, one row of it 3 KB. The backward also sums `weight_grad` and `bias_grad` over all
rows: each block adds its rows into a copy in shared memory, then adds the copy into GPU memory once.*

Every pass after the first reads the row again. Whether that costs a trip to GPU memory depends on
whether the row is still in the SM's L1 cache, and that is where the launch comes in. Both kernels
use 64 registers per thread (the test prints it), so an SM holds at most 65,536 / 64 = 1,024 of
their threads, 32 warps: the same limit as in Table 2.

### 11.2 Method

Part 4 of
[`tests/test_layer_norm.cu`](https://github.com/ducbachsong/gpt2-small/blob/c730f66/tests/test_layer_norm.cu)
is Section 3's test for these kernels: it launches each kernel directly with any
`<<<blocks, threads>>>`, keeps the GPU busy for one second first (Section 6.4), times runs 2–20 and
checks the result. The sizes:

| blocks | 40, 80, 128, 256, 512, 1,024, 2,048, 4,096 |
|---|---|
| threads per block | 64, 128, 256, 512, 1,024 |

That is from 80 warps (40 × 64) to 131,072 (4,096 × 1,024). Blocks of 64 threads are in because one
warp per row makes small blocks natural; 128 and 512 blocks are the two launches the code uses. Part
1 of the same test times 20 forwards and backwards of mine against PyTorch's `layer_norm` with
autograd, at the same sizes, without a warm-up.

| run | test | what it adds |
|---|---|---|
| D1 | before `c730f66` | the 20-step timing; then stops at a test of numbers worked out by hand whose bias was typed as 1 instead of 0 (the test's error: the kernel gave norm + 1, as asked) |
| D2 | `c730f66` without part 4 | every check and the 20-step timing |
| D3 | `c730f66` | everything, with part 4 |

*Table 16. Batch D, three runs on Colab. The kernels are the same in all three. D1 averaged its 20
steps from step 2; its numbers here are recomputed from its log for steps 3–20, as in D2 and D3
(Table 23 says why step 2 is left out).*

**The prediction.** Before D3 ran, the test's comment predicted what part 1 had found for the
encoder: with fewer warps than the 1,280 a T4 holds, each warp takes many rows one after another, so
the time should go up. It did not.

### 11.3 Correctness

All 40 launches gave the same result as the usual ones: `output`, `mean`, `rstd` and `input_grad`
equal to the last bit, as they should be, because a row is always one warp's and its threads add in
the same order whatever the launch; `weight_grad` and `bias_grad` within $$10^{-5}$$, because their
`atomicAdd`s land in any order (the criteria of Table 5). The rest of the test passed in D2 and D3:
both kernels against PyTorch within $$3 \times 10^{-7}$$, and within $$3 \times 10^{-6}$$ after 20
backwards added up at GPT-2 small's size; and eight tests of numbers worked out by hand, among them
4,096 rows adding into one `bias_grad` without losing anything.

### 11.4 Results: fewer warps is faster

![LayerNorm kernel time against warps launched](/images/encoder-launch-sizes/fig4-layernorm-warps.svg)
*Figure 4. Run D3: the 80 timings (2 kernels × 40 launches) against the warps launched. The rings
mark the launches the code uses; the dashed lines mark where an SM's rows of `input` fill 64 KB, and
where the T4 is full.*

| warps launched | rows per SM | forward | backward |
|---:|---:|---:|---:|
| 80 | 2 | 0.159 | 0.962 |
| 160 | 4 | 0.117–0.118 | 0.480–0.507 |
| 256 | 6.4 | 0.126 | 0.368 |
| 320 | 8 | **0.117**–0.118 | 0.286–0.299 |
| 512 | 12.8 | 0.119–0.120 | **0.275** |
| 640 | 16 | 0.125–0.126 | 0.284–0.288 |
| 1,024 | 25.6 | 0.151–0.152 | 0.313–0.345 |
| 1,280 | 32 | 0.168–0.170 | 0.318–0.320 |
| 2,048 (the backward's launch) | 32 | 0.154–0.156 | 0.318–0.333 |
| 4,096 (the forward's launch) | 32 | 0.166–0.172 | 0.314–0.345 |
| 8,192–131,072 | 32 | 0.166–0.193 | 0.328–0.405 |

*Table 17. Run D3, in ms. Launches with the same number of warps took about the same time and are
grouped; the full table is in the log. "Rows per SM": the rows the SMs work on at once, the warps
launched spread over 40 SMs, at most 32.*

| | the code's launch | fastest launch | faster by | bandwidth |
|---|---|---|---:|---|
| forward | 512 × 256: 0.167 ms | 40 × 256: 0.117 ms | 30% | 151 → 215 GB/s, 47 → 67% of the peak |
| backward | 128 × 512: 0.333 ms | 256 × 64 or 128 × 128: 0.275 ms | 17% | 151 → 183 GB/s, 47 → 57% |

*Table 18. Bandwidth: the bytes of Table 15 divided by the time.*

**Up to 16 rows per SM, the forward is flat at 0.117–0.126 ms; from 25.6 it takes 0.151 ms or
more.** With very few rows per SM both kernels slow down, as part 1's did: the forward only at 2
(0.159 ms), the backward already at 4–6.4 (0.368–0.507 ms) and by 3.5x at 2. Far beyond the T4's
capacity the backward slows down again as blocks grow in number, to 0.357–0.405 ms at 4,096 blocks:
every block adds its 2 × 768 sums into GPU memory, even a block with no row, 6.3 million
`atomicAdd`s at 4,096 blocks against 196,608 at 128. One thing I cannot explain: with 32 rows per SM
in every case, the forward took 0.154–0.156 ms at 2,048 warps but 0.166–0.172 ms at 1,280 and 4,096.
One run cannot say whether that repeats.

### 11.5 Why: the rows an SM works on, and its L1 cache

A Turing SM has 96 KB of on-chip memory, split between shared memory and the L1 cache as either
32 KB + 64 KB or 64 KB + 32 KB [14]. So an SM keeps at most 64 KB of what it reads in L1. A row of
`input` is 3 KB:

| rows per SM | the forward's rows of `input` | fit in 64 KB? | forward |
|---:|---:|---|---:|
| 8 | 24 KB | yes | 0.117–0.118 ms |
| 16 | 48 KB | yes | 0.125–0.126 ms |
| 25.6 | 77 KB | no | 0.151–0.152 ms |
| 32 | 96 KB | no | 0.154–0.193 ms |

*Table 19. 64 KB holds 21 rows of 3 KB; the forward's time steps up between 16 and 25.6 rows.*

The forward reads every row three times. While the rows an SM is working on fit in its L1, passes 2
and 3 find them there, and GPU memory sees each byte about once: 25.2 MB in 0.117 ms is 215 GB/s,
67% of the peak, close to part 1's kernels, which read each byte once. When the rows do not fit,
they are pushed out of L1 before their second and third pass, and those passes go to the L2 cache or
to GPU memory. One warp per row at the usual launch is 32 rows per SM, 96 KB: outside.

This also says why the encoder behaved the other way. Its kernels read each byte once; with nothing
to read again, nothing is lost when the cache overflows, and more threads only mean more loads in
flight. LayerNorm reads each row again, so more rows at once means fewer of them still cached when
they are needed.

The backward fits only roughly. Its rows are 6 KB, a row of `input` and one of `output_grad`, so
64 KB holds 10.7; its best times are at 8–16 rows per SM (0.275–0.299 ms), around and past that
point, and beyond it the time rises more slowly than the forward's. It does more work per row than
the forward, two `atomicAdd`s per value into shared memory and more arithmetic, so it may need more
rows in flight to keep memory busy; that is a guess.

What would show the explanation wrong: L1 hit rates that do not drop between 16 and 25.6 rows per
SM, or a forward that reads each row once, from registers, and still runs faster with fewer warps.
Neither has been measured yet (Section 11.9).

### 11.6 PyTorch works out its own launch

A caller of `torch.layer_norm` never gives a launch; PyTorch's C++ code works one out from the
tensor's shape just before it launches [12]. For GPT-2 small's 4,096 rows of 768, in every release
from 2.12.0 to 2.14.1 (the code is the same there):

| | PyTorch's rule | for 4,096 rows of 768 | rows per SM |
|---|---|---|---:|
| forward | one block per row; 4 warps per block (`num_threads()` = 4 × 32) | 4,096 × 128 | at most 8 |
| backward: `input_grad` | one block per row, 128 threads | 4,096 × 128 | at most 8 |
| backward: `weight_grad`, `bias_grad` | a second kernel; for 256 rows or more, blocks of 32 × 32 threads, one per 32 columns, each walking down every row | 24 × 1,024 | — |

*Table 20. From `aten/src/ATen/native/cuda/layer_norm_kernel.cu` [12]. "At most 8": an SM holds
1,024 threads, 8 blocks of 128, fewer if the kernel needs more than 64 registers per thread. The
weight and bias kernel uses smaller tiles for fewer rows, and splits the rows over blocks when there
are very many rows and few columns. PyTorch's main branch, since October 13, 2026, also launches
fewer warps for rows of 256 numbers or fewer; 768 is not one.*

Compared with my kernels:

| | mine (`c730f66`) | PyTorch |
|---|---|---|
| a row is worked on by | one warp | one block of 4 warps |
| a row's sums are added | with 5 shuffles | with 5 shuffles, then across the 4 warps in shared memory, with `__syncthreads()` |
| rows per SM at the usual launch | 32 | at most 8 |
| forward: passes over a row | 3 | 2: the mean and variance in one pass, with Welford's method [15] |
| backward: `input_grad` | two passes: two sums over the row, then `input_grad` | the same |
| backward: `weight_grad`, `bias_grad` | in the same kernel | a second kernel, which reads `output_grad` and `input` again |
| backward: adding onto earlier gradients | in the kernel: `input_grad` is read and written | a separate addition onto `.grad` (below) |

*Table 21. What differs.*

So PyTorch's rule, written for every shape and every GPU at once, happens to keep at most 8 rows per
SM: 24 KB of `input` in the forward and 48 KB in the `input_grad` kernel, inside L1, where my kernels
are fastest. It gets there without knowing the GPU, because it gives each row a whole block. The
price is the step across 4 warps for every row, and, in the backward, two kernels.

**The extra addition.** The 20-step test lets PyTorch add 20 backwards onto `.grad`, and at the end
`.grad` equals my `input_grad` (Section 11.3). Eager PyTorch adds each new gradient onto `.grad` as a
separate step after the backward's kernels; for `input` that is an addition over 3.1 million numbers
that reads two tensors and writes one. My kernel adds onto `input_grad` in the same pass that
computes it. Counting bytes for one backward from step 2 on:

| | mine | PyTorch |
|---|---|---|
| `input_grad` | 50.3 MB, one kernel, the addition included | 37.7 MB (`input_grad` kernel) + 37.7 MB (the addition onto `.grad`) |
| `weight_grad`, `bias_grad` | in the same kernel | 25.2 MB (second kernel) |
| **total** | **50.3 MB** | **≈ 100.7 MB** |

*Table 22. Bytes to and from GPU memory for one backward, counting each pass after the first as a
cache hit. The addition onto `.grad` is inferred from how eager PyTorch accumulates gradients, not
profiled.*

### 11.7 Against PyTorch

| run | my forward | PyTorch forward | my backward | PyTorch backward |
|---|---:|---:|---:|---:|
| D1 | 0.181 | 0.184 | 0.499 | 0.542 |
| D2 | 0.179 | 0.173 | 0.463 | 0.540 |
| D3 | 0.176 | 0.155 | 0.373 | 0.504 |
| PyTorch takes | | 0.88–1.02x | | 1.09–1.35x |

*Table 23. The 20-step test at the usual launches, mean of steps 3–20 in ms, no warm-up. In every
run PyTorch's backward took 25.0–26.6 ms at step 2 and about 0.5 ms at every other step: step 2 is
its first addition onto the `.grad` that step 1 made, most likely the first use of the kernel that
adds. Left in, that one step made PyTorch's average look 3.8 times mine in D1.*

**The forward: the launch decides.** Both move the same 25.2 MB. At my usual launch PyTorch's
forward was as fast as mine in D1 and D2 and 12% faster in D3. Two things favour it: its rows fit in
L1 and mine do not (Section 11.5), and it reads each row twice, not three times. At my fastest
launch, 40 × 256, my forward took 0.117 ms in D3's part 4, against PyTorch's 0.155 ms in the same
run's 20-step test. Those two were not measured side by side: part 4 runs after a warm-up and the
20-step test without one, and in D3 my usual launch took 0.167 ms in part 4 and 0.176 ms in the
20-step test, 5% apart. Allowing for that, my forward at 40 × 256 is about 1.25x as fast as
PyTorch's: a launch that keeps its rows in L1, and rows that need no step across warps.

**The backward: fusion decides.** My backward was 1.09–1.35x faster at its usual launch. By Table
22, PyTorch moves about twice the bytes, and it moves them faster than mine: ≈ 100.7 MB in 0.504 ms
is about 200 GB/s in D3, against my 50.3 MB in 0.373 ms, 135 GB/s. So the lead comes from fusing the
backward into one kernel, not from its launch. The launch would add to it: 0.275 against 0.333 ms in
part 4, 17%.

**So why can a kernel with a launch of its own be faster than PyTorch, which works out its own?**

1. PyTorch's rules are written for every shape and every GPU at once: one block of 4 warps per row,
   whether the row has 768 numbers or 100,000. For 768 on a T4 the rule is a good one, mostly because
   it keeps few rows per SM, but it pays for its generality with a step across warps for every row.
2. A kernel written for one shape can choose for that shape: one warp per row, which needs no shared
   memory and no `__syncthreads()`, and as many rows per SM as the L1 holds. That choice comes from a
   measurement, not a rule: 40 × 256 suits a T4's 40 SMs and 64 KB of L1, not an A100's 108 SMs.
3. Owning the whole backward lets a kernel fuse what a library keeps apart: the weight and bias
   gradients, and the addition onto earlier gradients. That halves the bytes, and in the backward it
   is worth more than the launch.
4. Choosing for oneself also means choosing badly. My usual launch, one warp per row with the GPU
   filled, lost the forward to PyTorch, and only the measurement showed it.

### 11.8 Threats to validity

- **One run of part 4.** Part 1 found that small differences repeat over runs while whole runs move
  by up to 6%. The differences here are 17–30%, but one run cannot show how much they move, and the
  2,048-warp oddity of Section 11.4 may be noise.
- **The cache explanation** is inferred from where the forward's time steps up; no L1 hit rate was
  measured. The backward fits it only roughly.
- **Mixed conditions.** The 1.25x for the forward compares part 4, after a warm-up, with the 20-step
  test, without one; the 5% correction comes from one run.
- **PyTorch's launches** are read from its source, the same for this shape from 2.12.0 to 2.14.1.
  PyTorch's kernels were not profiled on Colab, and Colab's PyTorch version was not recorded; their
  registers per thread, and so whether 8 of their blocks fit an SM, are unknown.
- **The addition onto `.grad`** is inferred, not profiled. Without it PyTorch's backward would move
  ≈ 63 MB, at ≈ 125 GB/s, and its lead would be less clearly a matter of bytes.
- **Rows per SM** is the warps launched spread evenly over 40 SMs; how the T4 actually places blocks,
  and how many of the backward's small blocks fit beside their shared memory, was not checked.

### 11.9 Future work

- **Test the cache explanation**: a forward whose threads keep their 6 `float4`s of the row in
  registers and read it once. If the cache is the cause, it should take about the same time at
  512 × 256 as at 40 × 256. And Nsight Compute's L1 hit rate at both launches.
- **Time PyTorch side by side** with my kernels, both after the warm-up, at the usual and the fastest
  launch; and profile PyTorch's kernels on Colab (grid, block, registers, time of each), with its
  version recorded.
- **Repeat part 4** over several runs.
- **Then choose the launch from the GPU**: rows per SM from the L1 size and the bytes a row needs,
  blocks from the SM count, instead of a fixed 512 × 256 and 128 × 512.

### 11.10 Conclusion

LayerNorm's kernels give the same result with all 40 launches tried, to the last bit for the forward
and `input_grad`. Unlike the encoder's, they are fastest with far fewer warps than a T4 holds: the
forward takes 0.117 ms with 8 rows per SM and 0.167 ms with the 32 of its usual launch, the backward
0.275 against 0.333 ms. Most likely the re-reads of each row stop finding it in L1 once an SM's rows
pass 64 KB. PyTorch, which works out its launch from the shape, gives each row a block of 4 warps and
so keeps at most 8 rows per SM without knowing the GPU; at my usual launch its forward was as fast as
mine or faster. Mine is faster where it does what PyTorch's general code does not: a launch chosen
for this shape on this GPU, about 1.25x for the forward, and a backward fused into one kernel that
moves half the bytes, 1.09–1.35x. The launches in the code stay until the cache explanation is
tested.

## Reproducibility

Code at commit [`e84ce51`](https://github.com/ducbachsong/gpt2-small/tree/e84ce51) (batch C; batch A
is [`a1f1b7b`](https://github.com/ducbachsong/gpt2-small/tree/a1f1b7b)). On Google Colab with a T4
(PyTorch, and with it libtorch, are preinstalled):

```bash
git clone https://github.com/ducbachsong/gpt2-small.git
cd gpt2-small && git checkout e84ce51

TORCH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
nvcc -O3 -std=c++17 tests/test_encoder.cu -o test_encoder \
    -D_GLIBCXX_USE_CXX11_ABI=$ABI \
    -I$TORCH/include -I$TORCH/include/torch/csrc/api/include \
    -L$TORCH/lib -Xlinker --no-as-needed -ltorch -ltorch_cuda -ltorch_cpu -lc10_cuda -lc10 \
    -Xlinker -rpath=$TORCH/lib
./test_encoder
```

With no `-arch` flag, Colab's `nvcc` built machine code for the T4 (sm_75) directly; `cuobjdump -lelf
test_encoder` shows it. To log the clock while it runs:

```bash
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw \
           --format=csv,noheader,nounits -lms 200 > clocks.csv &
./test_encoder > run.log
kill %1
```

**Part 2** is at commit [`c730f66`](https://github.com/ducbachsong/gpt2-small/tree/c730f66) (batch D,
run D3), built the same way, with `TORCH` and `ABI` set as above and `-arch=native`, which builds for
the GPU in the machine, sm_75 on a T4:

```bash
git checkout c730f66
nvcc -O3 -std=c++17 -arch=native tests/test_layer_norm.cu -o test_layer_norm \
    -D_GLIBCXX_USE_CXX11_ABI=$ABI \
    -I$TORCH/include -I$TORCH/include/torch/csrc/api/include \
    -L$TORCH/lib -Xlinker --no-as-needed -ltorch -ltorch_cuda -ltorch_cpu -lc10_cuda -lc10 \
    -Xlinker -rpath=$TORCH/lib
./test_layer_norm
```

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
12. PyTorch. `aten/src/ATen/native/cuda/layer_norm_kernel.cu`, releases v2.12.0 to v2.14.1 (the
    launches for LayerNorm's forward, its `input_grad` and its weight and bias gradients).
    [github.com/pytorch/pytorch](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cuda/layer_norm_kernel.cu)
13. D. B. Song. *LayerNorm backward, broken down and rebuilt: nine chain-rule steps and one formula.*
    2026. [/posts/layer-norm-backward/](/posts/layer-norm-backward/)
14. NVIDIA. *Turing Tuning Guide* (the unified 96 KB of L1 and shared memory per SM, and its two
    splits). [docs.nvidia.com](https://docs.nvidia.com/cuda/turing-tuning-guide/)
15. B. P. Welford. *Note on a Method for Calculating Corrected Sums of Squares and Products.*
    Technometrics 4(3), 1962.

## Appendix A. Raw logs

Index: [README.txt](/files/encoder-launch-sizes/README.txt).

| file | what |
|---|---|
| [a-run1.log](/files/encoder-launch-sizes/a-run1.log) … [a-run6.log](/files/encoder-launch-sizes/a-run6.log) | batch A, commit `a1f1b7b`: every check, the 20-step timings and part 4, 6 runs |
| [b-summary.txt](/files/encoder-launch-sizes/b-summary.txt) | batch B: the summary of 5 runs (clock, part 4 at 40 × 128, part 5 slow-downs) and run 1's part 5 |
| [c-run1.log](/files/encoder-launch-sizes/c-run1.log) … [c-run5.log](/files/encoder-launch-sizes/c-run5.log) | batch C, commit `e84ce51`: every check, the timings, parts 4 and 5, 5 runs |
| [c-clocks-run1.csv](/files/encoder-launch-sizes/c-clocks-run1.csv) … [c-clocks-run5.csv](/files/encoder-launch-sizes/c-clocks-run5.csv) | batch C: timestamp, SM clock, memory clock (MHz), temperature (°C) and power (W), every 200 ms |
| [sass-loads-and-atomics.txt](/files/encoder-launch-sizes/sass-loads-and-atomics.txt) | the loads and atomics of each kernel, from `cuobjdump`, with how the program was compiled |
| [d-run1.log](/files/encoder-launch-sizes/d-run1.log) … [d-run3.log](/files/encoder-launch-sizes/d-run3.log) | part 2, batch D, `tests/test_layer_norm.cu`: every check and the 20-step timing; D3 also part 4, the 2 LayerNorm kernels at 40 launch sizes |

## Versions

The date at the top is when this entry was first published. Each version since is listed here,
with what it changed.

| version | date | what changed |
|---|---|---|
| 1 | October 7, 2026 | First version, from one run: the three encoder kernels at 32 launch sizes, three regions, and a bytes-in-flight model fitted to them. |
| 2 | October 7, 2026 | Rewritten after 16 runs (batches A–C): the model tested with a float4 load, which changed the kernel; the warm-up; the 768 × 128 forward; corrections to the build note, the forward's loads and the bandwidth figures. |
| 3 | October 8, 2026 | Section 6.3: what the float4 load changes and what it does not. |
| 4 | October 14, 2026 | Part 2, LayerNorm (Section 11), against the launch PyTorch works out itself; the title names both kernels; this list. |
