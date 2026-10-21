---
title: "Tuning a matrix multiply on a T4: eighteen runs, what helped, what did not, and the last 10% to cuBLAS"
date: 2026-10-21
modified: 2026-10-21
version: 1
permalink: /posts/linear-kernel-tuning/
excerpt: "The linear layer kernel of my GPT-2 trainer started the day at 1.47 times PyTorch's time for one training step's matrix products, and ended it at 1.06. This note goes through every run in order: the idea behind it, the change, what the timing test and the profiler said, and what I kept. Five changes made it faster, five made no difference, two made it slower. The last run compares the kernel with cuBLAS's own at the level of machine code, and edits that machine code by hand to see where the remaining 10% comes from."
tags:
  - cuda
  - matmul
  - gpt-2
  - performance
  - notes
---

**What this is.** A log of one day of making a CUDA matrix multiply faster. The kernel is the linear
layer of my GPT-2 trainer [1]: `y = x Wᵀ + b` forward, and its two gradients `dW = dyᵀ x` and
`dx = dy W` backward. The [note on the two ways to multiply matrices](/posts/matmul-outer-product/) [2]
explains its design: each block of 256 threads makes a 128 × 128 square of the output, each thread an
8 × 8 patch, built up as a sum of outer products from pieces copied into shared memory. This note is
about what happened next: eighteen runs on a Tesla T4, each with a reason, a change, a measurement and a
decision.

**Sources.** Every number in this note was measured on a Colab T4 by the kernel's test file,
`tests/test_linear.cu` in [1], and by NVIDIA's profiler, Nsight Compute (`ncu`) [3]. The T4's register
banks are from Jia et al. [4]; the hand-written assembly results from Lai and Seznec [5], Gray [6],
Zhang et al. [7] and Yan et al. [8]; the assembler is CuAssembler [9].

* TOC
{:toc}

## 1. The starting point

### 1.1 The kernel

| | |
|---|---|
| block | 256 threads, makes 128 × 128 values of the product |
| thread | makes 8 × 8 of them, the 64 sums in registers |
| step | the block copies 8 values of the summed index for its 128 rows and 128 columns into shared memory, then each thread adds 8 outer products |
| forward | one kernel |
| backward | one kernel: its first blocks make squares of `dW`, the rest squares of `dx`; the blocks of the first column of `dW` also sum `db` |

*Table 1. The kernel at the start of the day: Boehm's kernel 5 [10], as in [2].*

### 1.2 How it was measured

The test times 20 steps of each layer of GPT-2 small, with 4 × 1024 = 4096 rows, against PyTorch's
`torch::linear` and its backward, which call NVIDIA's cuBLAS library. It averages steps 3 to 20 and
prints both times.

| layer | in_features | out_features | forward work |
|---|---|---|---|
| qkv | 768 | 2304 | 14.5 GFLOP |
| attention output | 768 | 768 | 4.8 GFLOP |
| MLP up | 768 | 3072 | 19.3 GFLOP |
| MLP down | 3072 | 768 | 19.3 GFLOP |
| head | 768 | 50304 | 316.5 GFLOP |

*Table 2. The five matrix products of GPT-2 small. Each layer's backward is twice its forward's work.*

Two things about measuring on this GPU turned out to matter more than I expected:

1. **The T4 changes its own clock.** It starts slow and speeds up under load, then slows down again when
   it gets warm. In one run, the same machine code was timed 33% apart, only because it was timed first
   while the clock was still rising. In another, a loop of 36 nearly identical kernels got steadily
   slower from first to last, by 13%. So a raw time from one run cannot be compared with a raw time from
   another.
2. **So I compare ratios, or use the profiler.** The test runs ours and PyTorch's in the same steps,
   under the same clock, so *our time ÷ PyTorch's* can be compared across runs. For changes smaller
   than about 5%, even that moves too much, and I use `ncu`: it locks the GPU at its base clock, 585 MHz,
   and its times repeat to 0.01 ms. Its times are longer than real ones, because of that low clock, but
   they compare fairly.

### 1.3 Where it started

| layer | forward, ours | forward, PyTorch | ratio | backward, ours | backward, PyTorch | ratio |
|---|---|---|---|---|---|---|
| qkv | 5.19 ms | 3.31 ms | 1.565 | 10.38 ms | 6.47 ms | 1.603 |
| attention output | 1.54 ms | 1.03 ms | 1.493 | 3.10 ms | 2.07 ms | 1.496 |
| MLP up | 6.63 ms | 4.30 ms | 1.543 | 13.61 ms | 8.49 ms | 1.604 |
| MLP down | 6.67 ms | 4.21 ms | 1.586 | 13.42 ms | 8.23 ms | 1.630 |
| head | 122.4 ms | 70.6 ms | 1.733 | 193.7 ms | 181.4 ms | 1.068 |

*Table 3. The first run. Ratio is our time ÷ PyTorch's: 1.565 means ours took 56.5% longer. About 2.8
TFLOPS against PyTorch's 4.4.*

Added up over one training step, 12 blocks of four layers and the head, forward and backward, ours
took **1042 ms** of matrix products against PyTorch's **709 ms**: **1.47 times**.

## 2. The runs

Figure 1 is the short version: the ratio after each change I kept.

![Our time ÷ PyTorch's after each change kept](/images/linear-kernel-tuning/fig1-progress.svg)

*Figure 1. The qkv layer's forward and backward, and the head's forward, after each kept change. The
sections below go through these and the runs that did not make it into the figure.*

### 2.1 Run 1: padding the pieces (kept, small)

**Idea.** The forward copies `x` and `W` into shared memory *turned round*: each thread reads 4 values
along a row of memory and writes them down a column of the piece, one at a time. In one warp, the
threads writing row `k` and row `k + 4` of a piece are 4 × 128 = 512 floats apart. Shared memory has 32
banks, one per 4 bytes, and 512 is a multiple of 32, so both halves of the warp hit the same 16 banks
and take turns.

**Change.** Each row of a piece is 132 floats long instead of 128: 4 unused at its end. Rows `k` and
`k + 4` are then 528 floats apart, 16 banks on, and both halves write together.

**Result.**

| layer | forward ratio, before → after | backward ratio, before → after |
|---|---|---|
| qkv | 1.565 → 1.493 | 1.603 → 1.592 |
| MLP up | 1.543 → 1.465 | 1.604 → 1.546 |
| MLP down | 1.586 → 1.502 | 1.630 → 1.627 |

*Table 4. 2 to 5% on the forward, 0 to 4% on the backward, which has fewer turned copies.*

**Conclusion.** A real conflict, but a small part of the time. Kept.

### 2.2 Run 2: double buffering, first try (no gain, undone)

**Idea.** Each step copies, waits at `__syncthreads()`, multiplies, waits again. While the copy waits for
GPU memory, about 300 cycles, the math units of that block sit idle. With two sets of pieces, each
thread can ask for the next step's values *before* this step's math, and write them into the other set
after it: the wait for GPU memory hides behind the math, and one `__syncthreads()` a step is enough.

**Result.** Within ±4% on every layer, both ways: no clear change. It also took more registers: the
forward from 109 to 128, the backward from 158 to 168.

**Conclusion.** Undone. The next run explains why it did not help: the copy wait was not what limited
the kernel.

### 2.3 Run 3: asking the profiler (the turning point)

Instead of guessing, `ncu` on the qkv forward and backward:

| | forward | backward |
|---|---|---|
| L1 / shared memory busy | **93%** | **81%** |
| GPU memory (DRAM) busy | 12% | 6% |
| shared memory reads that were bank-conflict repeats | **40%** | **40%** |
| shared memory writes that were repeats | 0 | 0 |

*Table 5. Shared memory was nearly full, and 2 of every 5 of its reads were repeats. GPU memory was
mostly idle. The padding of run 1 shows here too: the writes have no conflicts left.*

| a warp, for each instruction, waits … | cycles | share |
|---|---|---|
| for shared memory (`mio_throttle` + `short_scoreboard`) | 3.27 | 30% |
| for GPU memory (`long_scoreboard`) | 1.84 | 17% |
| at `__syncthreads()` (`barrier`) | 1.15 | 10% |

*Table 6. The forward's stalls, of 11.08 cycles per instruction. The biggest wait was shared memory,
which double buffering does not touch.*

**Where the repeats came from.** Each thread owned 8 neighbouring columns. At each step of the sum, the
16 threads of a half-warp read their columns of the right piece at 0, 8, 16, …, 120. Columns 0, 32, 64
and 96 all fall on banks 0 to 3, so a warp's read took 4 rounds where 2 would do.

### 2.4 Run 4: each thread's columns in two halves (kept, the biggest step)

**Change.** A thread's 8 columns become two groups of 4, 64 apart: thread 0 makes columns 0–3 and
64–67, thread 1 4–7 and 68–71, …, thread 15 60–63 and 124–127. A warp's read of its first halves is then
64 neighbouring floats, every bank used exactly twice, the fewest reads possible. The stores of the sums
get the same 64 neighbours.

**Result.**

| | before | after |
|---|---|---|
| shared memory read repeats, forward | 28.3 million | **0** |
| shared memory read repeats, backward | 56.6 million | **0** |
| L1 / shared memory busy, forward | 93% | 83% |
| math units busy, forward | 59% | 77% |
| qkv forward under ncu | 8.31 ms | **6.40 ms** |
| qkv forward ratio | 1.493 | **1.153** |
| qkv backward ratio | 1.592 | **1.462** |

*Table 7. 23% on every forward, 8 to 10% on every backward. The forward's biggest stalls are now its
math units being busy, which is what a matrix multiply should wait on.*

**Conclusion.** Kept. One indexing change, the largest gain of the day, and found only by the profiler.

### 2.5 Runs 5 to 7: the backward's registers (kept, after two tries)

The profiler now pointed at the backward: **158 registers a thread**. A T4 SM has 65,536 registers, so
at 158 only 1 block of 256 threads fits, 8 warps; when they wait, nothing else can run.

| run | change | result |
|---|---|---|
| 5 | `__launch_bounds__(256, 2)`: tell the compiler to fit 2 blocks, at most 128 registers | 128 registers, but **128 bytes spilled** to slow local memory; faster on 3 layers, slower on 2; undone |
| 6 | split the one backward kernel into a `dW` kernel and a `dx` kernel | `dx` 111 registers, 2 blocks an SM; `dW` still 158: the 158 was `dW`'s all along |
| 7 | add `db` inside the outer products, from the values each thread already reads, instead of a second loop that read them again | `dW` **119** registers, 2 blocks an SM |

*Table 8. One kernel gets the register count of its largest path; split, each gets its own. The bias
loop was reading 64 values a step that were already in registers.*

| | after run 6 | after run 7 |
|---|---|---|
| `dW` under ncu, qkv | 8.17 ms | **6.97 ms** |
| qkv backward ratio | 1.336 | **1.205** |
| MLP down backward ratio | 1.392 | **1.226** |

*Table 9. 10 to 12% on most backward layers. Committed: `8ecbae8`.*

### 2.6 Runs 8 and 9: double buffering again, now with room (kept)

With shared memory no longer the limit, the forward's waits were now GPU memory (1.09 cycles an
instruction) and the barrier (0.97): what double buffering hides. Tried again on the new code:

| run | change | registers: forward, dW, dx | blocks an SM | result |
|---|---|---|---|---|
| 8 | double buffering | 128, **133**, **130** | 2, **1**, **1** | forward 1–4% faster; backward slower: it fell back to 1 block |
| 9 | + `__launch_bounds__(256, 2)` on `dW` and `dx` | 128, 128, 127 | 2, 2, 2 | no spill this time: only 2 to 5 registers to save |

*Table 10. Run 5 asked the compiler to save 30 registers and it spilled; run 9 asked for 5 and it did
not.*

| | before | after run 9 |
|---|---|---|
| `dW` wait for GPU memory (cycles an instruction) | 0.71 | **0.02** |
| `dW` under ncu | 6.97 ms | **6.19 ms** |
| `dx` under ncu | 6.21 ms | **5.57 ms** |
| backward ratio: qkv, attention output, MLP up, MLP down | 1.205, 1.247, 1.200, 1.226 | **1.106, 1.077, 1.078, 1.119** |
| head backward ratio | 1.030 | **0.953**: faster than PyTorch |

*Table 11. Committed: `22c98fb`.*

**Conclusion.** The same idea failed in run 2 and worked in run 9: it hides GPU memory waits, which
only matter once shared memory is not the limit, and only pays if it does not cost a block per SM.

### 2.7 Run 10: numbering the forward's blocks in groups (kept)

**Idea.** The head's weight is 50304 × 768: 154 MB, against the T4's 4 MB L2 cache. Blocks were numbered
across first, so the 80 blocks running at once made 1 row × 80 columns of squares: one band of `x`, but
80 different bands of `W`. Each row of squares read all of `W` from GPU memory again: 32 × 154 MB = 4.9
GB per forward.

**Change.** Number the blocks in groups of 8 rows of squares, down a group first. The 80 blocks running
at once then make about 8 rows × 10 columns: 8 bands of `x` and 10 of `W`, shared through L2.

| | before | after |
|---|---|---|
| head forward, ours | 99.2 ms | **84.6 ms** |
| head forward ratio | 1.377 | **1.182** |
| other forward ratios | 1.084 to 1.180 | **1.060 to 1.135** |
| head forward: L2 hit rate, DRAM busy | – | 90%, 11% |

*Table 12. 15% on the head, 2 to 5% on the other forwards. Committed: `2529933`.*

### 2.8 Runs 11 to 13: three changes that did nothing (undone)

| run | idea | measured | why nothing |
|---|---|---|---|
| 11 | `__restrict__` on every pointer, and read the bias once instead of 16 times | ±1%; head forward under ncu 125.89 → 125.13 ms | the compiler already ordered the loads well; the 16 bias reads hit L1 |
| – | load the next shared memory values during this step's multiply-adds | not run: the stall it targets was 0.16 of 8.16 cycles | the 8-step loop is unrolled, so the compiler already moves those reads up |
| 12 | 32-bit index math instead of 64-bit `size_t` | under ncu, head forward 125.89 → 125.96 ms | a few percent of the instructions, and the math units, not the issue slots, were the limit |

*Table 13. All undone, so the code stayed simpler. Run 12 also taught the measuring rule of section
1.2: its timing test said "3% faster forward, 3% slower backward", and `ncu` said "the same".*

### 2.9 Run 14: no edge checks when the sizes fill the squares (measured, undone)

**Idea.** Every read and write asks "is this past the edge of the matrix?", for sizes that leave
part-full squares. All of GPT-2's sizes are multiples of 128 (50304 = 393 × 128), so a second copy of
each kernel, a template, could leave those tests out.

| qkv, under ncu | before | without edge checks |
|---|---|---|
| forward | 6.03 ms, 268.1 M instructions | **5.89 ms, 260.9 M** |
| `dW` | 6.19 ms | 6.34 ms |
| `dx` | 5.57 ms | 5.58 ms |

*Table 14. 2.3% on the forward, nothing on the backward. Undone to keep one copy of each kernel.*

## 3. The last 10%: comparing with cuBLAS itself

### 3.1 Same work, same resources

After run 10, ours was 1.06 to 1.14 times PyTorch on every forward but the head. `ncu` can profile
cuBLAS's own kernel inside the same test, which shows what it runs:

| qkv forward, under ncu | cuBLAS `volta_sgemm_128x64_tn` | ours |
|---|---|---|
| time | **5.36 ms** | **6.03 ms** |
| block | 128 threads, 128 × 64 values | 256 threads, 128 × 128 |
| per thread | 8 × 8 values | 8 × 8 values |
| registers per thread | 122 | 128 |
| warps per SM | 16 | 16 |
| multiply-adds (FFMA) | 7.257 billion | 7.248 billion |
| all instructions | 272.7 million | 268.1 million |
| cycles a warp waits per instruction | **7.06** | **7.84** |
| math units busy | about 91% | about 85% |

*Table 15. The same algorithm, the same amount of work, the same number of warps, and we even run fewer
instructions. cuBLAS's warps wait 10% less between instructions, which is the whole 12%.*

Two more things showed up in cuBLAS's backward: its `dW` splits the sum over the 4096 rows into **5
parts** (its grid is 6 × 36 × **5**), so its few blocks fill the GPU better, an idea still open for us.

### 3.2 Run 15: our kernel in cuBLAS's shape (slower, undone)

Is it the block shape? Our kernel with cuBLAS's shape: 64 rows × 128 columns, 128 threads, 4 blocks an
SM.

| | forward ratio, 128 × 128 → 64 × 128 | backward ratio |
|---|---|---|
| qkv | 1.091 → **1.158** | 1.104 → **1.263** |
| MLP down | 1.135 → **1.252** | 1.114 → **1.251** |

*Table 16. Slower on every layer, and `dW` spilled 64 bytes: the copy now held 3 float4s for the next
step instead of 2. cuBLAS fits the same shape in 122 registers.*

So the shape is not cuBLAS's secret. With the same shape, the compiler makes code that needs more
registers and runs slower. The difference is in the machine code.

### 3.3 Runs 16 to 18: editing the machine code by hand

The published work agrees: hand-written assembly matrix multiplies beat the compiler, and cuBLAS, on
Fermi (about 5% [5]), Maxwell (98% of peak, up to 4.8% over cuBLAS [6]) and Kepler (15% over cuBLAS
[7]); on the T4 itself, a half-precision one in assembly was 1.46 times cuBLAS 10.1 [8]. Gray writes that
NVIDIA hand-assembles its library routines rather than use its own compiler [6]. To see it on our own
kernel, I took the compiled kernel apart, edited it, and put it back together.

**The toolchain.** NVIDIA ships no assembler for its machine code (SASS). CuAssembler [9] turns a
compiled `.cubin` into editable text and back. Two problems on the way: Colab's CUDA 13 writes cubins in
a new format CuAssembler cannot read, so the C++ goes to PTX with CUDA 13 and to machine code with CUDA
12.6's `ptxas`; and CuAssembler did not know two of our instructions' encodings, which it learned from
the cubin itself. A test harness loads any `.cubin` and checks its output against the C++ kernel's to
the last bit.

**What the machine code showed.** 1,336 instructions, 512 of them FFMAs: the 8 × 64 multiply-adds of a
step, unrolled. The T4 keeps registers in 2 banks, even and odd, each serving 2 reads a cycle [4], so an
FFMA whose three sources are all read from one bank takes an extra cycle. A source marked `.reuse` comes
from a small cache instead. Counted that way, **78 of the 512 FFMAs (15%)** read all three from one
bank.

**The edit.** A program reorders the FFMAs within each run of consecutive FFMAs, swaps the two multiplied
operands where it helps (a × b = b × a, bit for bit), and sets the `.reuse` flags for the new order, so
the operand that would clash comes from the cache. It keeps every other instruction, register and
control code where it was, and checks that every FFMA still reads each value from the same instruction
as before, the waits for loads, and the distances between dependent instructions.

**A bug on the way.** The first edit gave wrong numbers. `LDS.U.128 R4` loads R4 to R7, but the
instruction text names only R4, so the program moved FFMAs that read R5 ahead of the wait for that load.
Counting each load as 4 registers fixed it, and 36 test variants, the whole edit and each of its 33 runs
alone, all came out the same to the last bit.

| qkv forward, under ncu | instructions | bank clashes (our count) | time | cycles per instruction |
|---|---|---|---|---|
| ptxas's machine code | 268,139,520 | 78 | **6.02 ms** | 7.82 |
| same order, our `.reuse` flags | 268,139,520 | 60 | 6.20 ms (+3.0%) | 8.08 |
| our order, no `.reuse` at all | 268,139,520 | 206 | 6.25 ms (+3.8%) | 8.10 |
| our order and our `.reuse` flags | 268,139,520 | **10** | 6.26 ms (+4.0%) | 8.18 |

*Table 17. The original, timed first and last: 6.02 ms both times. Every row runs the same instructions
and gives the same results to the last bit.*

**What it shows.**

1. **Machine code alone changes the speed.** The same instructions in a different order, with different
   `.reuse` flags, moved the time by up to 4%. Nothing else differs.
2. **But our order was worse than the compiler's, not better.** The version with 206 counted clashes and
   the one with 10 took the same time, so the clashes we counted are not what costs time: our model of
   the T4's banks, read from one paper, is too simple. `ptxas` schedules for the real hardware, and beating
   it takes knowing that hardware's exact rules, which is what the authors of [4] to [8] measured first.

## 4. Where it ended

| layer | forward ratio, start → end | backward ratio, start → end |
|---|---|---|
| qkv | 1.565 → **1.091** | 1.603 → **1.104** |
| attention output | 1.493 → **1.060** | 1.496 → **1.091** |
| MLP up | 1.543 → **1.100** | 1.604 → **1.086** |
| MLP down | 1.586 → **1.135** | 1.630 → **1.114** |
| head | 1.733 → **1.182** | 1.068 → **0.897** |

*Table 18. The committed kernel, `2529933`. The qkv forward went from 2.8 to 4.05 TFLOPS.*

| one training step's matrix products | ours | PyTorch | ratio |
|---|---|---|---|
| start of the day | 1042 ms | 709 ms | 1.47 |
| end of the day | **753 ms** | 709 ms | **1.06** |

*Table 19. 12 blocks of 4 layers and the head, forward and backward.*

| run | change | effect | kept |
|---|---|---|---|
| 1 | pad the pieces' rows to 132 floats | forward 2–5% faster | yes |
| 2 | double buffering, first try | none, ±4% | no |
| 3 | profile with `ncu` | found the 40% bank-conflict repeats | – |
| 4 | each thread's columns in two halves 64 apart | forward 23%, backward 8–10% | yes |
| 5 | `__launch_bounds__` on the one backward kernel | spilled 128 bytes | no |
| 6 | split the backward into `dW` and `dx` | backward 4–9% | yes |
| 7 | `db` inside the outer products | `dW` 158 → 119 registers, backward 3–12% | yes |
| 8 | double buffering, second try | forward faster, backward slower | with run 9 |
| 9 | + `__launch_bounds__` on `dW` and `dx` | backward 8–14% | yes |
| 10 | forward blocks in groups of 8 rows | head forward 15% | yes |
| 11 | `__restrict__`, bias read once | none | no |
| 12 | 32-bit index math | none | no |
| 14 | no edge checks for full squares | forward 2.3% | no, for simplicity |
| 15 | cuBLAS's block shape | 6–14% slower | no |
| 16–18 | hand-edited machine code | 3–4% slower | no |

*Table 20. Every run of the day. Run 13 is the next-values-from-shared-memory idea of Table 13, measured
by reading the profile instead of running it.*

## 5. What to remember

1. **Measure the same way each time.** A T4 changes its clock under load and heat. Compare our time ÷
   PyTorch's from the same run, and for small changes use `ncu`, which locks the clock.
2. **Ask the profiler before guessing.** The biggest gain of the day, 23% on the forward, came from one
   number in `ncu`, 40% of shared memory reads repeated, and a fix of one line of index math. Double
   buffering, the obvious guess, did nothing until that was fixed.
3. **Registers decide how many warps an SM runs.** At 158 registers, 1 block an SM; at 128, 2. Splitting a
   kernel, removing a duplicated loop and `__launch_bounds__` with a small target all gave back blocks;
   `__launch_bounds__` with a big target only spilled.
4. **The same idea can fail and then work.** Double buffering hides GPU memory waits; it only paid once
   shared memory was not the limit, and only once it did not cost a block per SM.
5. **The last 10% is in the machine code.** With the same algorithm, work, registers and warps as cuBLAS,
   ours waits 7.84 cycles per instruction and cuBLAS's 7.06. Rearranging the machine code by hand moved the
   time by 4% with identical instructions, but the wrong way: beating the compiler needs the hardware's
   exact rules, which NVIDIA has and a simple model from one paper does not.

## 6. What is left

1. **Split the sum of `dW`**, as cuBLAS does in 5 parts: `dW` is only 108 blocks for the qkv, 1.35 rounds
   of the GPU, so its second round runs mostly empty.
2. **Measure the T4's register banks and reuse cache** with small test kernels, as Jia et al. did [4],
   then reorder the machine code with the real rules.
3. **Half precision on the tensor cores.** The T4 does about 65 TFLOPS in FP16 on its tensor cores,
   against 8 in FP32 [8]: that, not a better FP32 kernel, is where large factors are left.

## References

1. D. B. Song. *gpt2-small*, branch `c-cuda-trainer`: GPT-2 small pre-training in C and CUDA.
   `llmc/kernels/linear.cuh` and `tests/test_linear.cu`, commits `8ecbae8`, `22c98fb`, `2529933`.
   [github.com/ducbachsong/gpt2-small](https://github.com/ducbachsong/gpt2-small/tree/c-cuda-trainer)
2. D. B. Song. *Two ways to multiply matrices: inner products, outer products, and why a GPU kernel uses
   the second.* 2026. [/posts/matmul-outer-product/](/posts/matmul-outer-product/)
3. NVIDIA. *Nsight Compute: Kernel Profiling Guide.*
   [docs.nvidia.com/nsight-compute/ProfilingGuide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
4. Z. Jia, M. Maggioni, J. Smith, D. P. Scarpazza. *Dissecting the NVidia Turing T4 GPU via
   Microbenchmarking.* 2019. [arxiv.org/abs/1903.07486](https://arxiv.org/abs/1903.07486)
5. J. Lai, A. Seznec. *Performance Upper Bound Analysis and Optimization of SGEMM on Fermi and Kepler
   GPUs.* CGO 2013. [hal-00789958](https://hal-univ-tlse3.archives-ouvertes.fr/INRIA/hal-00789958)
6. S. Gray. *maxas: Assembler for NVIDIA Maxwell architecture*, SGEMM wiki page.
   [github.com/NervanaSystems/maxas/wiki/SGEMM](https://github.com/NervanaSystems/maxas/wiki/SGEMM)
7. X. Zhang, G. Tan, S. Xue, J. Li, K. Zhou, M. Chen. *Understanding the GPU Microarchitecture to
   Achieve Bare-Metal Performance Tuning.* PPoPP 2017. DOI 10.1145/3018743.3018755
8. D. Yan, W. Wang, X. Chu. *Demystifying Tensor Cores to Optimize Half-Precision Matrix Multiply.*
   IPDPS 2020. [home.cse.ust.hk/~weiwa/papers/yan-ipdps20.pdf](https://home.cse.ust.hk/~weiwa/papers/yan-ipdps20.pdf)
9. cloudcores. *CuAssembler: an unofficial CUDA assembler.*
   [github.com/cloudcores/CuAssembler](https://github.com/cloudcores/CuAssembler)
10. S. Boehm. *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog.* 2022.
    [siboehm.com/articles/22/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM)
