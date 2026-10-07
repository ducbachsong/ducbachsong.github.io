---
title: "How many threads? Launch sizes for GPT-2's encoder kernels on a T4"
collection: publications
date: 2026-10-07
permalink: /research/encoder-launch-sizes/
excerpt: "GPT-2's encoder as three CUDA kernels, timed on a Tesla T4 with 32 launch sizes each, in 16 runs. Every launch gives the same result. With fewer threads than the 40,960 a T4 holds at once, the backward into the token table, which loaded one float at a time, slowed down 2.2-2.7x; the others hardly at all. A bytes-in-flight model predicted the order, and a test that changed only the load width confirmed the cause: with one float4 per load the slow-down fell to 1.02-1.06x, so the kernel now loads float4s. Two surprises: without a warm-up the GPU clock made the first measurements unreliable, and one launch, 768 blocks of 128 threads, runs the forward 12% faster for a reason not yet known."
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
- **Versions.** The PyTorch, CUDA and driver versions of the Colab sessions were not recorded.

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

## Appendix A. Raw logs

Index: [README.txt](/files/encoder-launch-sizes/README.txt).

| file | what |
|---|---|
| [a-run1.log](/files/encoder-launch-sizes/a-run1.log) … [a-run6.log](/files/encoder-launch-sizes/a-run6.log) | batch A, commit `a1f1b7b`: every check, the 20-step timings and part 4, 6 runs |
| [b-summary.txt](/files/encoder-launch-sizes/b-summary.txt) | batch B: the summary of 5 runs (clock, part 4 at 40 × 128, part 5 slow-downs) and run 1's part 5 |
| [c-run1.log](/files/encoder-launch-sizes/c-run1.log) … [c-run5.log](/files/encoder-launch-sizes/c-run5.log) | batch C, commit `e84ce51`: every check, the timings, parts 4 and 5, 5 runs |
| [c-clocks-run1.csv](/files/encoder-launch-sizes/c-clocks-run1.csv) … [c-clocks-run5.csv](/files/encoder-launch-sizes/c-clocks-run5.csv) | batch C: timestamp, SM clock, memory clock (MHz), temperature (°C) and power (W), every 200 ms |
| [sass-loads-and-atomics.txt](/files/encoder-launch-sizes/sass-loads-and-atomics.txt) | the loads and atomics of each kernel, from `cuobjdump`, with how the program was compiled |
