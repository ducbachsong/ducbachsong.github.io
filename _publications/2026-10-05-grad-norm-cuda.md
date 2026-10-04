---
title: "Adding 124 million squares: a one-kernel gradient norm for GPT-2 in CUDA"
collection: publications
date: 2026-10-05
permalink: /research/grad-norm-cuda/
excerpt: "The norm of all of GPT-2 small's gradients in one CUDA launch over one flat buffer: on a Tesla T4 it reads the model's 124M values in 1.79 ms (278 GB/s, 87% of peak) whether they are 2 tensors or the real 148, while PyTorch takes 1.81x as long tensor by tensor and 1.24x with foreach. The same to the last bit on every run, within 1e-5 of PyTorch."
read_time: true
tags:
  - optimization
  - gradient-clipping
  - gpu
  - gpt-2
  - cuda
---

**Abstract.** The [previous post](/research/fused-adamw-cuda/) left one pass over memory that the
fused AdamW update cannot absorb: the norm of every gradient in the model, which clipping needs
before any value can change. I compute it with one CUDA kernel, launched once, over the flat buffer
that already holds every gradient. Each thread adds the squares of its share, the 32 threads of a
warp add their sums by exchanging registers, the 32 warps of a block add theirs through shared
memory, and the block that finishes last, found with a counter, adds the 512 block sums and takes
the square root. Every result agrees with PyTorch's `clip_grad_norm_` within $$10^{-5}$$, and the
same gradients give the same norm to the last bit on every run. On a Tesla T4, for GPT-2 small's
124,439,808 values, **the norm takes 1.79 ms (278 GB/s, 87% of peak) whether the values are 2
tensors or the model's real 148**. On the 148 tensors PyTorch takes **1.81x as long** computing
tensor by tensor (C++ `clip_grad_norm_`) and **1.24x** with its grouped foreach path (Python
`clip_grad_norm_`); on 2 tensors, where only the kernels differ, 1.05x and 1.08x. Getting there
needed a fair start: a first version read one float per request against PyTorch's vectorised
loads, and was 1.24x slower even on 2 tensors.

* TOC
{:toc}

## 1. Introduction

I am pre-training a GPT-2 small [1] from scratch, rebuilding the trainer in C and CUDA in the style
of Karpathy's llm.c [4]: one GPU allocation per kind of state, hand-written kernels, and a test
against PyTorch for every piece. The code is open source:
[github.com/ducbachsong/gpt2-small](https://github.com/ducbachsong/gpt2-small/tree/c-cuda-trainer)
(branch `c-cuda-trainer`; every link in this post is pinned to commit
[`ea4b2ab`](https://github.com/ducbachsong/gpt2-small/tree/ea4b2ab)).

The previous post [3] built the optimiser's update as one kernel and ended on what it could not do:
clipping divides every gradient by the same factor, and that factor depends on a sum over all 124M
of them. Its tests computed that norm on the CPU. This post builds the GPU kernel, tests it, times
it, and, because a sum over many threads is the most basic thing a GPU does that a CPU does not
need to think about, explains in detail how the GPU adds.

**Contributions.**

1. A gradient-norm kernel that reads every value once, in one launch, and gives the same result on
   every run: no atomic additions into the result, no second kernel (Section 4).
2. A step-by-step account of how CUDA threads, warps and blocks add numbers together, with worked
   numbers taken from the tests (Section 3).
3. Tests against PyTorch's `clip_grad_norm_` and against numbers worked out by hand, including how
   the work is split between the 512 blocks (Section 6).
4. Timings on a Tesla T4 in three runs: a first version with one float per load, the committed
   `float4` version, and the same kernel on 2 and on 148 tensors against PyTorch's two ways of
   computing the norm, with a simple model for each result (Section 7).

## 2. Background

### 2.1 The norm, and what clipping does with it

For gradients $$g_0, g_1, \ldots, g_{N-1}$$, all the model's values as one long list, the norm is

$$
\lVert g \rVert = \sqrt{g_0^2 + g_1^2 + \cdots + g_{N-1}^2}
$$

and clipping with a threshold $$c$$ multiplies every gradient by
$$\min\!\left(1, c / (\lVert g \rVert + 10^{-6})\right)$$ [2]. With gradients $$\{1, 2, 3, 4\}$$ and
$$c = 1$$, the norm is $$\sqrt{30} = 5.477$$ and every gradient is multiplied by $$0.183$$, so the new
norm is exactly 1; with a norm already below $$c$$ the factor is 1 and nothing changes.

The order of the additions does not change the result mathematically. PyTorch's
`clip_grad_norm_` [5] computes it tensor by tensor, the norm of each, then the norm of those norms,
which is the same number:

$$
\sqrt{\lVert g^{(1)} \rVert^2 + \lVert g^{(2)} \rVert^2 + \cdots} = \lVert g \rVert
$$

The C++ version launches one norm kernel per tensor, then a `stack` and one more norm. The Python
version calls `torch._foreach_norm`, which computes the norms of a list of tensors in fewer,
grouped launches, then the same last norm. Section 7 times both.

### 2.2 Why this is a hard problem for a GPU

The arithmetic is trivial: one multiply-add per value. The cost is reading the values: 124,439,808
floats are 498 MB, and the T4 reads at most 320 GB/s [7], so no implementation can take less than

$$
\frac{497.8 \text{ MB}}{320 \text{ GB/s}} = 1.556 \text{ ms}.
$$

The difficulty is combining. A GPU runs hundreds of thousands of threads, each adding part of the
list; their partial sums then have to become one number, and threads in different parts of the
chip cannot see each other's work, or wait for each other, without paying for it. The simplest
answer, every thread doing `atomicAdd` into one float, is also unreproducible: atomic additions
happen in whatever order the threads arrive, floating-point addition is not associative, so the last
digits of the norm change from run to run. The rest of this post is about combining in a fixed
order, cheaply.

## 3. How a GPU adds: threads, warps and blocks

This section explains the mechanisms the kernel uses, from the bottom up, with the numbers of the
real launch.

### 3.1 The launch

A *kernel* is a function marked `__global__`. It runs on the GPU, started from CPU code with a
launch that says how many copies to run:

```cpp
compute_grad_norm_kernel<<<512, 1024>>>(grad_norm, block_sums, blocks_done, grads, grads_numel);
```

This starts **512 blocks of 1024 threads: 524,288 threads**, all running the same function body
from top to bottom. Each has its own private copies of the local variables (`sum`, `i`), kept in
its own registers. What makes them different is four numbers the GPU fills in for each thread
before it starts; the program never sets them:

| built-in | meaning | here |
|---|---|---|
| `threadIdx.x` | this thread's number inside its block | 0 … 1023 |
| `blockIdx.x` | this block's number | 0 … 511 |
| `blockDim.x` | threads per block | 1024 |
| `gridDim.x` | blocks in the launch | 512 |

*Table 1. The built-in variables. nvcc compiles each to a read of a hardware register (`threadIdx.x`
becomes `%tid.x`).*

The hardware groups every block's threads, in order, into **warps of 32**: threads 0–31 are warp 0,
32–63 are warp 1, …, 992–1023 are warp 31. A warp is the GPU's real unit of execution: its 32
threads (its *lanes*) run each instruction together. A block runs on one *streaming
multiprocessor* (SM) from start to end; the T4 has 40 SMs, each holding at most 1024 threads at a
time, so the 512 blocks run in about 13 waves of 40, in an order the program does not control.

![Who the threads are, and what each one reads](/images/grad-norm-cuda/fig1-threads.svg)
*Figure 1. The launch, from the grid down to one warp's 32 lanes, and the gradients each thread
reads. Thread 5 of block 0 reads float4 group 5, then group 524,293, and so on.*

### 3.2 Sharing out the work: the grid-stride loop

Nothing hands values out to blocks or warps. The gradients lie in GPU memory as one flat array, and
each thread computes, from its own built-ins, which positions to read:

```cpp
size_t thread_number = (size_t)blockIdx.x * blockDim.x + threadIdx.x;      // 0 … 524,287
for (size_t g = thread_number; g < groups; g += (size_t)gridDim.x * blockDim.x)
    ... read group g ...
```

Thread 3077 (block 3, thread 5) starts at position 3077 and then jumps by 512 × 1024 = 524,288,
the number of threads. Each *round* of the loop, the 524,288 threads together read 524,288
consecutive positions, and the next round starts exactly where this one ended. The number of
values only decides how many rounds there are; the jump depends only on the launch.

The jump must equal the number of threads. With 4 threads and the 10 values $$1, 2, \ldots, 10$$
(true sum of squares 385):

| jump | thread 0 | thread 1 | thread 2 | thread 3 | sum of squares | norm |
|---|---|---|---|---|---:|---:|
| 2 (too small) | 0, 2, 4, 6, 8 | 1, 3, 5, 7, 9 | 2, 4, 6, 8 | 3, 5, 7, 9 | 765 | 27.66 ✗ |
| **4 (the thread count)** | **0, 4, 8** | **1, 5, 9** | **2, 6** | **3, 7** | **385** | **19.62** |
| 8 (too big) | 0, 8 | 1, 9 | 2 | 3 | 211 | 14.53 ✗ |

*Table 2. Positions read by each thread, for three jumps. A smaller jump reads positions twice; a
bigger one skips them. All threads always use the same jump; what matters is that it equals the
thread count, and that every thread starts at a different position.*

The kernel never types the number: it reads `gridDim.x * blockDim.x`, so the jump follows the launch.
The layout also keeps memory reads efficient: in every round, the 32 lanes of a warp read 32
neighbouring positions, which the GPU serves as a few wide memory transactions instead of 32
scattered ones.

### 3.3 Adding inside a warp: shuffles

After the loop each thread holds one `sum`. The first combining step uses a warp instruction that
needs no memory at all:

```cpp
for (int offset = 16; offset > 0; offset /= 2)
    sum += __shfl_xor_sync(0xffffffff, sum, offset);
```

`__shfl_xor_sync` [6] is one instruction executed by all 32 lanes at once: each lane hands over its
`sum` and receives the `sum` of lane (its lane number XOR `offset`), straight from that lane's
register. The mask `0xffffffff`, 32 ones, says all 32 lanes take part. Five steps, with offsets 16,
8, 4, 2 and 1, leave the warp's total in every lane.

![The butterfly fold inside a warp](/images/grad-norm-cuda/fig2-warp-fold.svg)
*Figure 2. The fold on 8 lanes holding 1 … 8. Every lane adds in every step; the number of
different sums halves, 8 → 4 → 2 → 1, until every lane holds 36.*

Every lane adds in every step, and from the first step on, lanes compute the same sums as their
partners: in Figure 2, lanes 0 and 4 both hold 6 after the first step. This duplication costs
nothing. The 32 lanes execute each instruction together, so lanes that sat a step out would not
make it faster, and the variant that keeps the result in lane 0 only (`__shfl_down_sync`) runs the
same 5 instructions. The kernel keeps lane 0's copy and ignores the others.

### 3.4 Adding inside a block: shared memory

A shuffle never crosses a warp, and a block has 32 warps. They pass their totals through *shared
memory*, a small memory on the SM with one copy per block, visible to all its threads:

```cpp
__shared__ float warp_totals[32];                       // one per warp
if (thread_in_warp == 0) warp_totals[warp] = sum;       // lane 0 of each warp writes its total
__syncthreads();                                        // wait until all 32 are written
float block_total = warp_totals[thread_in_warp];        // lane k reads warp k's total
// ... the same five-step shuffle loop on block_total ...
```

`__syncthreads()` [6] is a barrier: a thread that reaches it waits until all 1024 threads of its
block have. Without it, a warp that runs ahead could read a slot that a slower warp has not written
yet, and add a stale value. It only waits within one block, and every thread of the block must reach
it, so it never sits inside a branch that only some threads take.

Two rounds of the same fold are always enough: the first leaves one total per warp, and since a
block has at most 1024 threads, there are at most 32 totals, which one warp can fold. With 1024
threads there are exactly 32, one per lane, so the second round needs no "lanes past the end take 0"
check.

With the test `each_block_adds_its_share_of_the_values`, 4,194,304 ones, every thread reads two
float4s, 8 values:

| stage | every thread of a block holds | where |
|---|---:|---|
| after the loop | 8 | its registers |
| after round 1 | 32 × 8 = 256 | registers; lane 0 writes it to `warp_totals[warp]` |
| after round 2 | 32 × 256 = 8,192 | registers; the block's total |

*Table 3. One block's numbers in that test. All 512 block totals are 8,192, which is what the test
checks.*

### 3.5 Adding across blocks: the last block

Blocks cannot wait for each other: there is no barrier across the grid in an ordinary launch, and
with 512 blocks of 1024 threads, not all of them are even on the GPU at the same time. The usual
answer is a second kernel launch, which starts only after every block of the first has finished.
This kernel instead uses a counter in GPU memory, a technique from NVIDIA's CUDA samples [8]:

```cpp
if (threadIdx.x == 0) {
    block_sums[blockIdx.x] = block_total;                   // A: this block's sum, its own slot
    __threadfence();                                        // A is visible to the whole GPU before B
    float done_before = atomicAdd(blocks_done, 1.0f);       // B: count this block; get the count before
    is_last_block = done_before == 511;
}
__syncthreads();
if (!is_last_block) return;
```

`atomicAdd` makes the additions to the counter happen one block at a time, so exactly one block
sees 511 blocks done before it. That block knows every other block's sum is written, and adds the
512 sums itself. The other 511 return, all their threads together, since `is_last_block` is in
shared memory and is the same for the whole block. What they wrote to GPU memory stays there after
they return.

`__threadfence()` [6] is what makes the count trustworthy. Writes travel through caches and queues,
and two writes to different addresses can become visible in a different order than they were made:

| | block 7 (thread 0) | block 300 (thread 0), the last |
|---|---|---|
| 1 | writes `block_sums[7]`; the write is still on its way | |
| 2 | adds 1 to the counter; this write arrives first | |
| 3 | | adds 1 to the counter, sees 511 before it: "I am last" |
| 4 | | reads `block_sums[7]`: the old value ✗ |

*Table 4. What could happen without the fence. With it, a block's sum is visible everywhere before
its count goes up.*

The last block reads the sums with a `volatile` pointer, which forces a read from GPU memory instead
of a copy that might sit in its SM's cache. Its 1024 threads use `threadIdx.x`, now just a number
from 0 to 1023, as an index: threads 0–511 take `block_sums[threadIdx.x]`, threads 512–1023 take 0,
and the same two rounds of Section 3.4 add the 1024 values. Thread 0 writes the square root and sets
the counter back to 0 for the next call.

This is also what makes the result reproducible. Whichever block happens to finish last, it adds
`block_sums[0]` to `block_sums[511]` in the same positions, and every fold before that runs in a
fixed order too.

### 3.6 Three kinds of memory

| memory | written as | copies | who sees it | lives until | used for |
|---|---|---|---|---|---|
| registers | `float sum;` | one per thread | that thread | the thread ends | each thread's sum |
| shared | `__shared__ float warp_totals[32];` | one per block | the block's threads | the block ends | the 32 warp totals |
| GPU (global) | the flat buffer | one | every thread | freed by the program | gradients, block sums, counter, norm |

*Table 5. Where each number of the kernel lives.*

## 4. Method

### 4.1 The kernel

![The kernel as one data path](/images/grad-norm-cuda/fig3-kernel.svg)
*Figure 3. compute_grad_norm_kernel, stage by stage, with the number of values left after each.*

The three steps of Section 3 run in one kernel: the grid-stride loop (step 1), the two folds and
the counter (step 2), and the last block's two folds and square root (step 3). Steps 2 and 3 reuse
the same `warp_totals` array; the `__syncthreads()` after the counter guarantees that every thread
has finished reading it before step 3 writes it again.

The loop reads the gradients as `float4`: group $$g$$ is the four values $$4g$$ to $$4g+3$$, loaded
with one 16-byte request. GPT-2 small's 124,439,808 values are 31,109,952 groups, 59.3 rounds of
524,288, so each thread reads 59 or 60 float4s. If the count is not a multiple of 4, the 1–3 values
after the last whole group are read one float each by threads 0–2.

### 4.2 The class, and the flat buffer

`GradNorm` keeps no memory of its own. Its constructor takes one view of the flat buffer, of shape
$$\{2, 512\}$$: row 0 holds the 512 block sums and the first value of row 1 is the counter. The
other 511 values of row 1 are unused, and are there so that 1,024 values, a multiple of 4, keep
whatever follows in the buffer on 16 bytes. The counter is a float: `atomicAdd` on floats is exact
for whole numbers far beyond 512.

`compute(grads, grad_norm)` takes two views: the gradients (one group of the buffer: its address
and count) and the one-value result. It checks that the result is one value and that the gradients
start on 16 bytes, which `float4` loads require [6], and launches the kernel. Off 16 bytes, the GPU
would stop with "misaligned address" at some later call, far from the cause, the bug the previous
post found the hard way.

## 5. Implementation

The code is CUDA C++17. Comments are abridged in the listings; the linked files have them in full.

**Listing 1.** The kernel —
[`llmc/kernels/grad_norm.cuh`](https://github.com/ducbachsong/gpt2-small/blob/ea4b2ab/llmc/kernels/grad_norm.cuh)

```cpp
__global__ void compute_grad_norm_kernel(float* grad_norm, float* block_sums, float* blocks_done,
                                         const float* grads, size_t grads_numel) {
    __shared__ float warp_totals[32];
    __shared__ bool is_last_block;
    int thread_in_warp = threadIdx.x % 32, warp = threadIdx.x / 32;

    // 1. this thread's squares, 4 floats per load
    size_t thread_number = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t groups = grads_numel / 4;
    const float4* grads4 = reinterpret_cast<const float4*>(grads);
    float sum = 0.0f;
    for (size_t g = thread_number; g < groups; g += (size_t)gridDim.x * blockDim.x) {
        float4 v = grads4[g];
        sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    size_t leftover = groups * 4 + thread_number;            // 0 .. 3 values after the last float4
    if (leftover < grads_numel) sum += grads[leftover] * grads[leftover];

    // 2. this block's 1024 sums: two warp folds through shared memory
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);
    if (thread_in_warp == 0) warp_totals[warp] = sum;
    __syncthreads();
    float block_total = warp_totals[thread_in_warp];
    for (int offset = 16; offset > 0; offset /= 2) block_total += __shfl_xor_sync(0xffffffff, block_total, offset);
    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = block_total;
        __threadfence();
        float done_before = atomicAdd(blocks_done, 1.0f);
        is_last_block = done_before == GRAD_NORM_BLOCKS - 1;
    }
    __syncthreads();
    if (!is_last_block) return;

    // 3. only the last block: add the 512 block sums, take the square root
    float total = threadIdx.x < GRAD_NORM_BLOCKS ? ((volatile float*)block_sums)[threadIdx.x] : 0.0f;
    for (int offset = 16; offset > 0; offset /= 2) total += __shfl_xor_sync(0xffffffff, total, offset);
    if (thread_in_warp == 0) warp_totals[warp] = total;
    __syncthreads();
    float all_blocks_total = warp_totals[thread_in_warp];
    for (int offset = 16; offset > 0; offset /= 2)
        all_blocks_total += __shfl_xor_sync(0xffffffff, all_blocks_total, offset);
    if (threadIdx.x == 0) {
        *grad_norm = sqrtf(all_blocks_total);
        *blocks_done = 0.0f;
    }
}
```

**Listing 2.** The class, on the CPU side

```cpp
inline GradNorm::GradNorm(const Tensor& block_sums) {      // a {2, 512} view of the flat buffer
    // ... stops if the shape is not {2, 512} ...
    block_sums_ = block_sums[0];                           // the 512 block sums
    blocks_done_ = block_sums[1][0];                       // the counter: one value
}

inline void GradNorm::compute(const Tensor& grads, const Tensor& grad_norm) const {
    // ... stops if grad_norm is not one value, or grads does not start on 16 bytes ...
    compute_grad_norm_kernel<<<GRAD_NORM_BLOCKS, GRAD_NORM_THREADS>>>(
        grad_norm.data(), block_sums_.data(), blocks_done_.data(), grads.data(), grads.numel());
    cudaCheck(cudaGetLastError());
}
```

### 5.1 From two kernels to one

The first draft used two kernels: one wrote the 512 block sums, a second launch of one block added
them. That is the textbook design [9], and it is simpler, since the end of a launch is a barrier
across all blocks for free. I merged them so that the whole computation reads in one function, top
to bottom, with the counter of Section 3.5 replacing the second launch. A launch costs a few
microseconds against a 1.8 ms kernel, so the merge was not expected to change the speed; the
two-kernel version was not timed.

### 5.2 Why the kernel and `compute` stay separate

`compute` runs on the CPU and the kernel on the GPU, so they cannot be one function. The GPU cannot
use a `Tensor`, an object in CPU memory with a shape and a shared pointer that frees the GPU memory;
it receives plain addresses and counts, which `compute` takes out of the views. The checks
(`fprintf` and `exit` on a wrong shape, `cudaCheck` after the launch) also belong on the CPU, run
once rather than in 524,288 threads.

### 5.3 From `float` to `float4`

The first version that ran on the GPU read one float per loop round. It was correct and slower
than PyTorch (Section 7). Changing the loop to `float4` loads added the alignment check, the
leftover line for counts that are not a multiple of 4, and one test for those leftovers, and moved
the expected numbers of the two block tests: one round now covers 2,097,152 values instead of
524,288, so the tests use four times as many values to show the same thing.

## 6. Correctness evaluation

### 6.1 Reference and criterion

The answer key is PyTorch's `torch::nn::utils::clip_grad_norm_` on the same GPU, used the plain
way: separate parameter tensors, each with its gradient in `.grad`. It returns the norm of all of
them together; with a threshold of $$10^9$$, far above any norm here, it only measures. The
criterion is a difference of at most $$10^{-5}$$ of the norm: the two sides add the same squares
in different orders, so their last digits differ.

### 6.2 Tests

| test | what it establishes | result |
|---|---|---|
| a 1-layer GPT-2's 16 gradient tensors, 4,464 values, two rounds | the whole model against PyTorch, on both sides of a threshold of 1 (norms 67.6 and 0.067) | equal within $$10^{-5}$$ |
| GPT-2 small's size, 124,439,808 values, after each timing run (2 and 148 tensors) | the full size against both of PyTorch's ways | 11,155.1, all three |
| weight $$\{1, 2, 3, 4\}$$ and bias $$\{5, 6\}$$ in one buffer | the norm of all tensors as one list, $$\sqrt{91}$$ | 9.53939 |
| $$\{3, -4\}$$ | signs do not matter: exactly 5 | exact |
| 1,000 zeros, result tensor starting at 7 | 0, not NaN; the result is really written | exact |
| 4,194,304 ones | each of the 512 blocks reads its share: every block sum 8,192, norm 2,048 | exact |
| 4,000,000 values of 0.5 | a part-full last round: block sums 2,048, 1,600 and 1,024 for blocks 0, 464 and 511, norm 1,000 | exact |
| $$\{1, 1, 1, 1, 2, 2, 2\}$$ | the 3 values after the last whole float4 are counted: block sum 16, norm 4 (2 without them) | exact |
| 1,000,000 random values, 5 computes | the same norm every run, to the last bit; the counter back to 0 | bit-identical |
| a view of 2 values out of $$\{3, 4, 100, 100\}$$ | only the view is read: 5, not more | exact |
| one buffer of 7s around the values | nothing changes but the norm, the block sums and the counter | exact |

*Table 6. The 12 tests of
[`tests/test_grad_norm.cu`](https://github.com/ducbachsong/gpt2-small/blob/ea4b2ab/tests/test_grad_norm.cu),
all passing on a Colab Tesla T4. "Exact" tests compare with `==`: their sums are of numbers that
floats represent exactly in any order.*

The hand-worked tests check *how* the work is shared out, not only the final number. A wrong jump
in the grid-stride loop (Section 3.2) would change the block sums of the 4,194,304-ones test (16,384
and 12,288 instead of 8,192 with a jump half as long, for example), and the test prints the expected
and actual lists side by side. Of the C# trainer's tests, two concerned the norm: one is ported
directly (the $$\sqrt{91}$$ case), and the other, which compared the norm of the flat buffer with the
norm of each `.grad` after `backward()`, becomes the 16-tensor test with random gradients, since the
C trainer has no backward pass yet.

## 7. Performance evaluation

### 7.1 Setup

**Workloads.** GPT-2 small's 124,439,808 values, filled with random normal values, in two shapes:

- **2 tensors**, one matrix and one bias, $$162{,}030 \times 768 + 768$$. PyTorch launches only a
  few kernels here, so this compares the kernels themselves.
- **148 tensors**, the model's real shapes: wte ($$50{,}257 \times 768$$), wpe, the 12 tensors of
  each of the 12 layers, and the final LayerNorm. 50 of them are matrices; the other 98 are biases
  and LayerNorm weights of 768 to 3,072 values, 3 to 12 KB each.

On my side both are the same: one group of the flat buffer, one launch. On PyTorch's side each
tensor is its own tensor, as in any PyTorch model.

**Three implementations.** All three read the same values on the GPU:

| | timed region | launches, 2 tensors | launches, 148 tensors |
|---|---|---:|---:|
| **my GradNorm** | one `compute` | 1 | 1 |
| PyTorch per tensor | what C++ `clip_grad_norm_` does before it clips: `grad.norm()` for each tensor, then `stack` and the norm of the norms | 4 | 150 |
| PyTorch foreach | what Python's `clip_grad_norm_` does: `_foreach_norm` over the list, then `stack` and the norm of the norms | a few | a few, grouped |

*Table 7. What is timed. The clipping multiplication is left out of PyTorch's side: it is not part of
measuring.*

**Protocol.** 20 computes of each, interleaved step by step in the same process, each timed on the
GPU's own clock with a CUDA event before and after it. Step 1, which loads each side's kernels, is
shown but left out of the mean of steps 2–20. At the end, all three norms are compared (Section 6).

| | |
|---|---|
| hardware | Google Colab, NVIDIA Tesla T4 (16 GB, 320 GB/s peak [7]) |
| software | CUDA C++17, `nvcc -O3 -arch=native`; libtorch from Colab's preinstalled PyTorch |

*Table 8. Environment.*

**Three runs.** The numbers come from three Colab runs, in the order they happened:

| run | my kernel | workloads | PyTorch |
|---|---|---|---|
| 1 | first version, one `float` per load | 2 tensors | per tensor |
| 2 | committed version, `float4` loads | 2 tensors | per tensor |
| 3 | the same `float4` version | 2 and 148 tensors | per tensor and foreach |

*Table 9. The runs. The kernel of runs 2 and 3 is identical; run 3 only extends the timing test.*

### 7.2 Results

**Runs 1 and 2: a fair start.**

![Gradient norm time, float and float4 loads](/images/grad-norm-cuda/fig4-time-t4.svg)
*Figure 4. Runs 1 and 2, 2 tensors. Only the loads differ between my two versions; PyTorch was timed
in the same run as each.*

| run | my version | my time | PyTorch per tensor | my bandwidth | vs PyTorch |
|---|---|---:|---:|---:|---|
| 1 | `float` loads | 2.316 ms | 1.863 ms | 215 GB/s | 1.24x slower |
| 2 | **`float4` loads** | **1.790 ms** | 1.894 ms | **278 GB/s** | **1.06x faster** |

*Table 10. Mean of steps 2–20. Bandwidth is 497.8 MB divided by the time.*

**Run 3: 2 tensors against 148.**

![The same values as 2 tensors and as 148](/images/grad-norm-cuda/fig5-tensors-t4.svg)
*Figure 5. Run 3. The same 124,439,808 values as 2 tensors and as the model's 148.*

| | 2 tensors | 148 tensors | change | per extra tensor |
|---|---:|---:|---:|---:|
| **my GradNorm** | **1.786 ms** (279 GB/s) | **1.789 ms** (278 GB/s) | +0.003 ms | ≈ 0 |
| PyTorch per tensor | 1.874 ms (266 GB/s), 1.05x | 3.244 ms (153 GB/s), **1.81x** | +1.370 ms | 9.4 µs |
| PyTorch foreach | 1.921 ms (259 GB/s), 1.08x | 2.226 ms (224 GB/s), **1.24x** | +0.305 ms | 2.1 µs |

*Table 11. Mean of steps 2–20, run 3. "1.81x" means PyTorch takes 1.81 times as long as mine. Per
extra tensor: the change divided by the 146 extra tensors.*

All the steps are steady: in run 3 mine stays within 1.786–1.797 ms on 148 tensors, PyTorch per
tensor within 3.223–3.299 ms and foreach within 2.204–2.276 ms. Run 3's 2-tensor numbers repeat run
2's within 1%, which gives a sense of the run-to-run noise.

### 7.3 Analysis

**Run 1 was not a fair comparison.** PyTorch's reduction kernel reads its input vectorised, several
values per memory request [11]; my first version read one float per request. What run 1 measured was
mostly that difference, not the design. The reason it matters is how GPU memory is read. A read
takes roughly half a microsecond to come back, and to keep memory streaming at full speed the GPU
must have enough bytes requested and not yet returned, *in flight*, to cover that wait. By Little's
law [10]:

$$
\text{bytes in flight needed} \approx 320 \text{ GB/s} \times 0.5\ \mu\text{s} \approx 160 \text{ KB}.
$$

A thread can send a request and keep going; it only stops when it uses the value. In the `float`
loop each thread uses each value right away, so it has about one 4-byte request in flight:

| version | bytes per request | upper bound in flight on a T4 (40 SMs × 1024 threads) |
|---|---:|---:|
| `float` loads | 4 | 160 KB, just at the need, and less in practice |
| `float4` loads | 16 | 640 KB |

*Table 12. A rough model, not a measurement: the half-microsecond is an estimate, and the compiler
may overlap some requests by itself. It explains the direction of the result, not its exact size.*

With `float4` the loop runs 59–60 rounds per thread instead of 237–238, and the same threads keep
four times as many bytes in flight: from 215 to 278 GB/s. From run 2 on, both sides read
vectorised, and the comparison is like for like.

**With 2 tensors the kernels are close.** On 2 tensors, where PyTorch launches only a few kernels,
mine is 1.05x faster than PyTorch per tensor and 1.08x faster than foreach. Fewer launches explain
little of that (PyTorch's 3 extra launches are worth perhaps 0.01–0.015 ms of the 0.09 ms gap); the
rest is in the kernels' internals, which I have not profiled. The honest summary is that, once both
read vectorised, my kernel is at least as good as PyTorch's on one large array.

**With 148 tensors the flat buffer decides.** The values do not change between the two workloads;
only the number of tensors does, and only PyTorch notices. Tensor by tensor, each of the 146 extra
tensors costs about 9.4 µs: one more kernel launch and one more reduction that has to start, finish
and write a result, whatever its size. For the 98 small tensors of 768 to 3,072 values, that fixed
cost is nearly all of the work. PyTorch's effective bandwidth falls from 266 to 153 GB/s. Foreach
groups the tensors into fewer launches and pays about 2.1 µs per extra tensor, falling to 224 GB/s.
My kernel pays nothing: the 148 tensors are one array in the flat buffer, the grid-stride loop runs
over it as over any other, and the kernel does not know where one tensor ends and the next begins.
It stays at 278 GB/s.

**What is left.** 278 GB/s is 87% of the datasheet figure. Part of the remaining gap is the last of
the ~13 waves of blocks, in which 8 of the 40 SMs have nothing to do; part is that each thread still
waits once per `float4`. Unrolling the loop, so each thread issues two to four loads before using
any, is the obvious next step.

## 8. Discussion

The result has two parts, and the runs separate them cleanly. The **kernel** had to be right first:
with one float per request it lost to PyTorch even on 2 tensors, and with `float4` it matched and
slightly beat PyTorch's kernel on one large array. The **flat buffer and the single launch** are
what win on the real model: 1.81x against PyTorch tensor by tensor and 1.24x against its foreach
path, a margin that comes entirely from the number of tensors, since the values, the GPU and the run
are the same. The flat buffer also made the `float4` change simple: the whole model's gradients are
one array starting on 16 bytes, so one loop covers every tensor, with at most three values left over,
instead of an alignment case per tensor.

The previous post estimated the norm at "roughly 2 ms per step at full size". It takes 1.79 ms, at a
higher bandwidth than the update itself (278 against 238 GB/s), plausibly because a pure read is
easier on memory than a mix of reads and writes. The optimiser's work per step at full size is now
1.79 ms for the norm plus 14.65 ms for the update, about 16.4 ms, of which the norm is 11%.

Reproducibility came with the last-block design, and it has a use beyond tidiness: if a training run
diverges, the same gradients give the same clip factor on a rerun, so the cause can be searched for
step by step.

## 9. Threats to validity

- **Launch costs in isolation.** The timed region covers PyTorch's launches back to back. If the CPU
  enqueues 150 small kernels more slowly than the GPU runs them, the GPU waits, and the CUDA events
  count that wait. Inside a full training step the CPU may queue this work while the GPU is still
  busy with the backward pass, hiding part of it. The 148-tensor gap may therefore be larger here
  than it would be in a real step.
- **One run each.** Each figure is the mean of 19 steps from one run, from three Colab runs that may
  not have been on the same machine. Run 3's 2-tensor numbers repeat run 2's within 1%.
- **Different block tests.** Between runs 1 and 2 the two block tests changed (4x more values) and
  one test was added; the timed workload is identical in all three runs.
- **One dtype, one GPU.** All gradients are fp32 on one GPU, the case PyTorch's foreach path groups
  best. Mixed precision or several devices were not tested.
- **Versions.** The PyTorch, CUDA and driver versions of the Colab sessions were not recorded.
- **The in-flight model.** Table 12 is a back-of-the-envelope model with an estimated latency; it was
  not checked with a profiler, nor were the kernels' internals behind the 2-tensor gap.
- **The two-kernel version** was replaced before being timed (Section 5.1), so the claim that
  merging did not change the speed is an estimate from launch cost, not a measurement.

## 10. Future work

- **Unroll the loop**, two to four `float4`s per thread before using any, and see how much of the
  remaining 13% it recovers.
- **Put `GradNorm` into the optimiser:** its $$\{2, 512\}$$ block goes in the optimiser's buffer
  before `grad_norm`, and each step becomes one norm launch plus one update launch, timed together.
- **Time it inside a training step**, once the rest of the C/CUDA trainer exists, to see how much of
  PyTorch's per-tensor cost survives when the CPU can queue work ahead.

## 11. Conclusion

The norm of all of GPT-2 small's gradients is now one CUDA launch over the flat gradient buffer: each
thread adds its squares, warps fold their sums with shuffles, blocks fold through shared memory, and
the last block, found with a counter, adds the block sums. It agrees with PyTorch within
$$10^{-5}$$, gives the same result on every run, and on a Tesla T4 reads 124M values in 1.79 ms,
278 GB/s, whether they are 2 tensors or 148. With vectorised loads on both sides the kernels are
close on one large array; on the model's real 148 tensors PyTorch takes 1.81x as long tensor by
tensor and 1.24x with foreach, because it pays for every tensor and the flat buffer pays for none.

## Reproducibility

Code at commit [`ea4b2ab`](https://github.com/ducbachsong/gpt2-small/tree/ea4b2ab). On Google Colab
with a T4 (PyTorch, and with it libtorch, are preinstalled):

```bash
git clone https://github.com/ducbachsong/gpt2-small.git
cd gpt2-small && git checkout ea4b2ab

TORCH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
nvcc -O3 -std=c++17 -arch=native tests/test_grad_norm.cu -o test_grad_norm \
    -D_GLIBCXX_USE_CXX11_ABI=$ABI \
    -I$TORCH/include -I$TORCH/include/torch/csrc/api/include \
    -L$TORCH/lib -Xlinker --no-as-needed -ltorch -ltorch_cuda -ltorch_cpu -lc10_cuda -lc10 \
    -Xlinker -rpath=$TORCH/lib
./test_grad_norm
```

`--no-as-needed` keeps `libtorch_cuda` linked although no call names it: PyTorch's GPU operations
are registered from inside it. `-arch=native` builds for the GPU in the machine.

## References

1. A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever. *Language Models are Unsupervised
   Multitask Learners.* OpenAI, 2019.
2. R. Pascanu, T. Mikolov, Y. Bengio. *On the difficulty of training recurrent neural networks.*
   ICML 2013. [arXiv:1211.5063](https://arxiv.org/abs/1211.5063)
3. D. B. Song. *One launch: a fused AdamW kernel for GPT-2 in CUDA.* 2026.
   [/research/fused-adamw-cuda/](/research/fused-adamw-cuda/)
4. A. Karpathy. *llm.c: LLM training in simple, raw C/CUDA.*
   [github.com/karpathy/llm.c](https://github.com/karpathy/llm.c)
5. PyTorch documentation, `torch.nn.utils.clip_grad_norm_`.
   [pytorch.org](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
6. NVIDIA. *CUDA C++ Programming Guide* (warp shuffle functions, `__syncthreads`, memory fence
   functions, atomic functions, vector types and their alignment).
   [docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
7. NVIDIA. *Tesla T4 Tensor Core GPU* (datasheet: 320 GB/s memory bandwidth).
   [nvidia.com](https://www.nvidia.com/en-us/data-center/tesla-t4/)
8. NVIDIA. *CUDA Samples: threadFenceReduction* (a single-pass reduction where the last block to
   finish adds the partial sums). [github.com/NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)
9. M. Harris. *Optimizing Parallel Reduction in CUDA.* NVIDIA, 2007.
10. J. D. C. Little. *A Proof for the Queuing Formula: L = λW.* Operations Research 9(3), 1961.
11. PyTorch source, `aten/src/ATen/native/cuda/Reduce.cuh` (vectorised input loads in the
    reduction kernel). [github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)

## Appendix A. Raw logs

Index: [README.txt](/files/grad-norm-cuda/README.txt).

| log | what |
|---|---|
| [test-grad-norm-t4-float.log](/files/grad-norm-cuda/test-grad-norm-t4-float.log) | run 1, the first version (`float` loads): every check and the timing run (2 tensors) |
| [test-grad-norm-t4-float4.log](/files/grad-norm-cuda/test-grad-norm-t4-float4.log) | run 2, the `float4` version: every check and the timing run (2 tensors) |
| [test-grad-norm-t4-148-tensors.log](/files/grad-norm-cuda/test-grad-norm-t4-148-tensors.log) | run 3: 2 and 148 tensors against PyTorch per tensor and foreach, and every check |
