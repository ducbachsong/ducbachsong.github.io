---
title: "How a CUDA kernel uses threads, warps, blocks and registers"
date: 2026-10-05
permalink: /posts/gpu-threads-blocks-registers/
excerpt: "How a CUDA launch becomes threads, warps and blocks on the GPU's SMs, how several blocks share one SM, how to choose between many small blocks and a few big ones, what registers are and who decides how many a thread gets, how to spend them to cut the time spent waiting for memory, and what happens when they run out. Explained on one small kernel, with the numbers of a Tesla T4, and based on NVIDIA's documentation and published measurements."
tags:
  - cuda
  - gpu
  - notes
---

**What this is.** A CUDA kernel runs as many threads, grouped into warps and blocks, spread over the
GPU's SMs, each thread computing in its own registers. This note explains how those pieces relate,
how a kernel uses them, where their limits are and how to use them well, on one small kernel and a
Tesla T4, the GPU of Google Colab. It is a note to keep what I learned, not a research entry: nothing
new is measured here.

**Sources.** Each statement is backed by NVIDIA's documentation (the CUDA C++ Programming Guide [3],
the Best Practices Guide [4], the Runtime API [5]) or by published work: measurements of the T4 by
Jia et al. [6], and Volkov's studies of latency hiding and occupancy [7, 8, 9]. Quotes are marked as
quotes. Numbers that come from arithmetic on documented rules say so, and estimates are labelled as
estimates.

* TOC
{:toc}

## 1. The example kernel

All the explanations use one kernel: the forward pass of GPT-2's encoder, from my C/CUDA trainer
[1]. It is small and its threads never work together, which keeps the hardware in view. For every
token it adds two rows of 768 numbers, 4 numbers (one `float4`) per step. The post launches it as
2,048 blocks of 256 threads, so that several blocks share each SM:

```cpp
#define ENCODER_BLOCKS 2048         // blocks in every launch
#define ENCODER_THREADS 256         // threads per block: 2048 x 256 = 524,288 threads in all

__global__ void encoder_forward_kernel(float4* encoded, const int* token_ids, const float4* wte,
                                       const float4* wpe, size_t B, size_t T, size_t embed_dim) {
    size_t thread_number = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t groups_per_row = embed_dim / 4;
    size_t groups = B * T * groups_per_row;                     // float4s to write
    for (size_t group = thread_number; group < groups; group += (size_t)gridDim.x * blockDim.x) {
        size_t row = group / groups_per_row;                    // which token: row = b x T + t
        size_t column = group % groups_per_row;                 // which float4 of its 768 numbers
        size_t t = row % T;
        size_t token = token_ids[row];
        float4 token_part = wte[token * groups_per_row + column];
        float4 position_part = wpe[t * groups_per_row + column];
        encoded[group] = make_float4(token_part.x + position_part.x, token_part.y + position_part.y,
                                     token_part.z + position_part.z, token_part.w + position_part.w);
    }
}

// launched from the CPU:
encoder_forward_kernel<<<ENCODER_BLOCKS, ENCODER_THREADS>>>(encoded, token_ids, wte, wpe, B, T, embed_dim);
```

With a batch of B = 4 rows of T = 1,024 tokens, there are 4 × 1,024 × 192 = **786,432 float4s** to
write.

| Tesla T4 (TU104, compute capability 7.5) | value | source |
|---|---|---|
| SMs | 40 | [6, Table 3.1] |
| clock | up to 1,590 MHz | [6, Table 3.1] |
| per SM: FP32 units, warp schedulers | 64, 4 | [3, *Compute Capability 7.x*] |
| per SM: resident threads, warps, blocks at most | 1,024, 32, 16 | [3, Table 21] |
| per SM: 32-bit registers | 65,536 (64 K) | [3, Table 21] |
| per SM: L1 data cache + shared memory | 96 KB, of which 32 or 64 KB shared | [6, Figure 3.1, Table 3.1] |
| L2 cache | 4 MB | [6, Table 3.1] |
| GPU memory | 16 GB GDDR6, 320 GB/s | [6, Section 3] |

*Table 1. The hardware the numbers come from.*

## 2. The pieces, and how they fit

### 2.1 Thread, warp, block, grid, SM

A launch `<<<2048, 256>>>` asks for **2,048 blocks of 256 threads**. Some of the words name what the
*program* asks for (grid, block, thread), one names what the *hardware* has (the SM), and the warp
sits in between:

| piece | what it is | decided by | in the example |
|---|---|---|---|
| **thread** | one copy of the kernel function, with its own registers | the launch | 524,288 |
| **warp** | 32 threads of one block that the SM schedules together | the hardware | 8 per block |
| **block** | a group of threads that can share memory and wait for each other | the launch: 1 to 1,024 threads | 2,048 blocks of 256 |
| **grid** | all the blocks of one launch | the launch | 2,048 blocks |
| **SM** (streaming multiprocessor) | a processor on the chip, with its own registers, caches, shared memory and schedulers | the hardware | 40 on a T4 |

*Table 2. The five pieces [3, Thread Hierarchy; Hardware Implementation].*

![From the launch to the hardware](/images/gpu-threads-blocks-registers/fig1-hierarchy.svg)
*Figure 1. The 2,048 blocks of the launch on a T4's 40 SMs. Each SM holds 4 blocks of 256 threads, 8
warps each, which fill its 32 warp slots; a warp is 32 threads.*

The Programming Guide describes the mapping in one paragraph: "When a CUDA program on the host CPU
invokes a kernel grid, the blocks of the grid are enumerated and distributed to multiprocessors with
available execution capacity. The threads of a thread block execute concurrently on one
multiprocessor, and multiple thread blocks can execute concurrently on one multiprocessor. As thread
blocks terminate, new blocks are launched on the vacated multiprocessors" [3, Hardware
Implementation].

A thread learns who it is from four numbers the GPU fills in before it starts: `threadIdx.x`
(0–255, its number in the block), `blockIdx.x` (0–2,047), `blockDim.x` (256) and `gridDim.x`
(2,048). The example turns them into one number, 0 to 524,287:

```cpp
size_t thread_number = (size_t)blockIdx.x * blockDim.x + threadIdx.x;   // block 3, thread 5 → 3 × 256 + 5 = 773
```

Which values a thread reads and writes is arithmetic on this number, written by the programmer.

### 2.2 The warp

A warp is **32 threads**; an SM holds **up to 32 warps** on a T4 [3, Table 21]. A block of 256
threads is therefore 8 warps, and four such blocks fill an SM. The guide: "The multiprocessor
creates, manages, schedules, and executes threads in groups of 32 parallel threads called warps",
and "each warp contains threads of consecutive, increasing thread IDs with the first warp containing
thread 0" [3, SIMT Architecture]. Threads 0–31 of a block are its warp 0, 32–63 its warp 1, …,
224–255 its warp 7.

In the example, the 32 threads of a warp run each load of `wte` together, each for its own `group`.
This is why the Best Practices Guide asks for whole warps: "Threads per block should be a multiple
of warp size to avoid wasting computation on under-populated warps" [4, 11.3]. A block of 100 threads
is 4 warps, the last with 4 busy threads out of 32.

### 2.3 A block lives on one SM

"All threads of a block are expected to reside on the same streaming multiprocessor core and must
share the limited memory resources of that core. On current GPUs, a thread block may contain up to
1024 threads" [3, Thread Hierarchy]. Two of those shared resources explain why a block cannot be
spread over two SMs:

- **shared memory** (`__shared__` variables), a fast memory inside the SM that all threads of the
  block can read and write;
- **`__syncthreads()`**, a barrier where every thread of the block waits for all the others.

Another SM has its own shared memory, and could not see this one's.

Blocks, on the other hand, must not depend on each other: "Thread blocks are required to execute
independently: It must be possible to execute them in any order, in parallel or in series" [3, Thread
Hierarchy]. Which SM runs which block, and when, is the GPU's choice.

### 2.4 Several blocks on one SM

An SM is a pool of resources, and each block takes a part of it when it arrives: "The number of
blocks and warps that can reside and be processed together on the multiprocessor for a given kernel
depends on the amount of registers and shared memory used by the kernel and the amount of registers
and shared memory available on the multiprocessor. There are also a maximum number of resident
blocks and a maximum number of resident warps per multiprocessor" [3, Hardware Multithreading].

For the example's blocks of 256 threads, at 32 registers each (Section 5.5), every resource gives a
limit, and the smallest one decides:

| T4 SM resource | total [3, Table 21] | one block of 256 threads takes | blocks it allows |
|---|---:|---:|---:|
| warp slots | 32 | 8 | **4** |
| registers | 65,536 | 256 × 32 = 8,192 | 8 |
| shared memory | 64 KB | none | no limit |
| block slots | 16 | 1 | 16 |

*Table 3. Four blocks of the example fit on one SM; warp slots run out first (arithmetic).*

The four blocks stay separate: each thread keeps its own registers, each block its own shared
memory, and `__syncthreads()` waits only for the threads of its own block. The SM switches between
their warps for free: "The execution context (program counters, registers, and so on) for each warp
processed by a multiprocessor is maintained on-chip during the entire lifetime of the warp.
Therefore, switching from one execution context to another has no cost" [3, Hardware
Multithreading]. On compute capability 7.x, "an SM statically distributes its warps among its
schedulers. Then, at every instruction issue time, each scheduler issues one instruction for one of
its assigned warps that is ready to execute, if any" [3, Compute Capability 7.x]. Jia et al. found
the rule on the T4: warp *w* goes to scheduler *w* mod 4 [6, Section 2.2]. Warps of all four blocks
are mixed on the same schedulers; a block cannot tell that others share its SM.

## 3. From a launch to running threads

### 3.1 Launched is not running

A launch is a list of work. The GPU places as many blocks as fit, and as they end, "new blocks are
launched on the vacated multiprocessors" [3, Hardware Implementation]:

| GPU | SMs | threads per SM | blocks of 256 per SM | blocks resident at once | threads resident at once |
|---|---:|---:|---:|---:|---:|
| Tesla T4 (7.5) | 40 [6] | 1,024 [3] | 4 | 160 | 40,960 |
| A100 (8.0) | 108 [10] | 2,048 [3] | 8 | 864 | 221,184 |

*Table 4. Arithmetic on the limits. Of the example's 524,288 threads, about 41 thousand are on a T4
at any moment.*

On a T4, 160 blocks run at once, and the 2,048 blocks go through in 2,048 / 160 = 12.8 **waves**
(arithmetic). In the last wave, blocks 1,920–2,047 fill 32 SMs and 8 SMs have nothing to do. A wave is
a simplification, though: blocks do not start and end together, so an SM takes a new block whenever
one of its four ends.

### 3.2 Sizing a launch: one thread per value, or a fixed grid and a loop

![Two ways to size a launch](/images/gpu-threads-blocks-registers/fig2-grid-stride.svg)
*Figure 2. The same 10 values, done by one thread each (A) or by 4 threads that loop (B).*

**A. One thread per value.** The number of blocks follows the work, rounded up so that the last,
part-full block is not left out; the threads past the end return without doing anything:

```cpp
size_t blocks = (groups + 256 - 1) / 256;        // 786,432 float4s / 256 = 3,072 blocks
kernel<<<blocks, 256>>>(...);
// in the kernel:
size_t group = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
if (group >= groups) return;
```

**B. A fixed grid and a loop**, a *grid-stride loop* [11]. The launch is always the same, and each
thread jumps by the number of threads until the work runs out, as in the example:

```cpp
for (size_t group = thread_number; group < groups; group += (size_t)gridDim.x * blockDim.x)
```

The jump must be exactly the number of threads: smaller and some values are done twice, bigger and
some are skipped. Reading it from `gridDim.x * blockDim.x` keeps it right if the launch changes.

| | A: one thread per value | B: fixed grid + loop |
|---|---|---|
| blocks | grows with the work | fixed: 2,048 |
| each thread does | 1 value | as many rounds as needed |
| the example (786,432 float4s) | 3,072 blocks of 256 | 1.5 rounds: threads 0–262,143 do 2, the rest 1 |

*Table 5. Both give the same result.*

### 3.3 Work and threads

With a grid-stride loop, the amount of work and the number of threads are separate: more work means
more rounds for the same threads.

| rows B | float4s to write | rounds per thread |
|---:|---:|---:|
| 4 | 786,432 | 1.5 |
| 64 | 12,582,912 | 24 |
| 1,000 | 196,608,000 | 375 |

*Table 6. The same launch for any amount of work (arithmetic).*

The amount of work is bounded by memory instead: the inputs and results must fit in the GPU's 16 GB.

### 3.4 How many blocks

The Best Practices Guide: "The number of blocks in a grid should be larger than the number of
multiprocessors so that all multiprocessors have at least one block to execute. Furthermore, there
should be multiple active blocks per multiprocessor so that blocks that aren't waiting for a
`__syncthreads()` can keep the hardware busy. [...] To scale to future devices, the number of blocks
per kernel launch should be in the thousands" [4, 11.3]. The example's 2,048 blocks of 256 follow
this; 512 blocks of 1,024 would not.

When threads work alone, as in the example, any count gives the right answer, and the last wave
matters:

| blocks of 256 on a T4 (160 at once) | waves | last wave |
|---:|---:|---|
| 80 | 0.5 | 20 of 40 SMs busy: half the GPU idle the whole time |
| 160 | 1 | full |
| 2,048 | 12.8 | 32 of 40 SMs busy |
| 2,080 | 13 | full |

*Table 7. Arithmetic on the T4's 40 SMs. A fixed count fits one GPU better than another: the same
2,048 on an A100 (864 at once) is 2.4 waves.*

When the threads of a block add their results together, the count can also be part of the design:
in my gradient norm kernel [2], each block leaves one partial sum, and the last block reads all of
them in one go. The grid size can also be taken from the GPU while the program runs:
`cudaGetDeviceProperties` gives `multiProcessorCount`, and
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` gives how many blocks of a kernel fit on one SM
[4, 11.1.1].

## 4. Threads per block

### 4.1 How block sizes fill an SM

An SM's 32 warp slots are filled by whole blocks. The block size decides how many blocks fit, and
how many slots are left over:

![How block sizes fill an SM](/images/gpu-threads-blocks-registers/fig3-block-sizes.svg)
*Figure 3. One T4 SM's 32 warp slots with different block sizes.*

| threads per block | warps | blocks per SM | slots used | what happens |
|---:|---:|---:|---:|---|
| 1,024 | 32 | 1 | 32 / 32 | full |
| 992 | 31 | 1 | 31 / 32 | each SM leaves 1 slot empty (3%) |
| **1,056** | 33 | — | — | **the launch fails** with `cudaErrorInvalidConfiguration` |
| 512 | 16 | 2 | 32 / 32 | full |
| 384 | 12 | 2 | 24 / 32 | a third block would need 1,152 threads: 25% empty |
| 256 | 8 | 4 | 32 / 32 | full (the example) |
| 32 | 1 | 16 | 16 / 32 | the 16-block limit stops it: half empty |

*Table 8. Arithmetic on the limits of Table 1. The error is documented as: "a kernel launch is
requesting resources that can never be satisfied by the current device. Requesting more shared
memory per block than the device supports will trigger this error, as will requesting too many
threads or blocks" [5].*

The Best Practices Guide's rules: "Threads per block should be a multiple of warp size", "A minimum
of 64 threads per block should be used, and only if there are multiple concurrent blocks per
multiprocessor", and "Between 128 and 256 threads per block is a good initial range for
experimentation with different block sizes" [4, 11.3]. Sizes that divide 1,024 (128, 256, 512,
1,024) also fill a T4 SM exactly.

### 4.2 Many small blocks or a few big ones, at the same number of threads

The same 524,288 threads can be launched as a few big blocks or many small ones. With the example's
32 registers per thread, the T4 sees:

| threads per block | blocks in the launch | blocks per SM | warps per SM | blocks at once | waves | registers per thread that still let one block launch | threads in one `__syncthreads()` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 512 | 1 | 32 | 40 | 12.8 | 64 | 1,024 |
| 512 | 1,024 | 2 | 32 | 80 | 12.8 | 128 | 512 |
| **256** | **2,048** | **4** | **32** | **160** | **12.8** | **255** | **256** |
| 128 | 4,096 | 8 | 32 | 320 | 12.8 | 255 | 128 |
| 64 | 8,192 | 16 | 32 | 640 | 12.8 | 255 | 64 |
| 32 | 16,384 | 16 (block limit) | 16 | 640 | 25.6 | 255 | 32 |

*Table 9. The same 524,288 threads on a T4 (arithmetic on Table 1). Down to 64 threads, every choice
fills the SMs and needs the same number of waves; at 32, the 16-block limit halves the warps.*

When the SM is equally full, what differs is how the work is cut, and each side has something going
for it.

**Smaller blocks**

- *The SM refills in smaller steps.* A block's slots are freed only when its last warp ends. With 1
  block of 1,024, one slow warp holds the whole SM; with 4 blocks of 256, the other three blocks keep
  running, and a new block can move into the freed quarter. The Best Practices Guide: "Use several
  smaller thread blocks rather than one large thread block per multiprocessor if latency affects
  performance. This is particularly beneficial to kernels that frequently call `__syncthreads()`"
  [4, 11.3], and "there should be multiple active blocks per multiprocessor so that blocks that aren't
  waiting for a `__syncthreads()` can keep the hardware busy" [4, 11.3].
- *A barrier stops fewer warps.* `__syncthreads()` makes the block's warps wait for its slowest; with
  256 threads that is 8 warps out of the SM's 32, while the other 24 can go on.
- *Register needs degrade gracefully.* A kernel that needs 72 registers per thread cannot launch with
  blocks of 1,024 at all (72 × 1,024 = 73,728 > 65,536), but runs with blocks of 256, three per SM
  (Section 5.7). The guide's own example shows the block size alone changing occupancy at the same
  register count: on compute capability 7.0, "a kernel with 128-thread blocks using 37 registers per
  thread results in an occupancy of 75% with 12 active 128-thread blocks per multi-processor, whereas
  a kernel with 320-thread blocks using the same 37 registers per thread results in an occupancy of
  63% because only four 320-thread blocks can reside on a multiprocessor" [4, 11.1.1].

**Bigger blocks**

- *More threads can work together.* Only threads of one block share shared memory and a barrier. When
  threads combine results, as in a norm, LayerNorm or softmax, a bigger block does more of the
  combining in fast memory and leaves fewer partial results: in my gradient norm, 1,024 threads per
  block leave 512 block sums where 256 would leave 2,048 [2].
- *Work done once per block is shared by more threads*, for example loading a tile of data into
  shared memory that every thread of the block then reads.

**Too small**

- Below 64 threads per block the per-SM block limit starts to bite: 16 blocks of 32 threads fill only
  half a T4 SM (Table 9). Hence the guide's minimum of 64 [4, 11.3].

For a kernel whose threads work alone, like the example, the guide's starting range of 128–256 [4,
11.3] is the reasonable default; for one whose threads combine results, the block size follows the
combining. In both cases the answer is final only after timing, since the result of the kernel does
not depend on the block size.

## 5. Registers

### 5.1 Why registers exist: load, compute, store

A register is a 32-bit storage slot inside the SM. Instructions take their operands from registers:
"accessing a register consumes zero extra clock cycles per instruction" [4, 10.2.7], while "if some
input operand resides in off-chip memory, the latency is much higher: typically hundreds of clock
cycles" [3, Multiprocessor Level]. So a kernel loads data from GPU memory into registers, computes on
registers, and stores the results back. One round of the example, written out roughly as the GPU
runs it:

```
load   r1        ← token_ids[row]                 the id
load   r2..r5    ← wte[token × 192 + column]      4 floats: token_part
load   r6..r9    ← wpe[t × 192 + column]          4 floats: position_part
add    r2 = r2 + r6                               4 adds, registers only
add    r3 = r3 + r7
add    r4 = r4 + r8
add    r5 = r5 + r9
store  encoded[group] ← r2..r5                    4 floats out
```

Every `wte[...]` in the code becomes a load into registers; every `+` becomes an add on registers.
An arithmetic result can be used about 4 cycles after it is computed ("the latency of most arithmetic
instructions is typically 4 cycles on devices of compute capability 7.0" [4, 11.2]; 4 cycles for FMA
on the T4 [6, Chapter 4]); a value from GPU memory takes 296 cycles on the T4 [6, Figure 3.5].

### 5.2 Each thread owns its registers

A T4 SM has **65,536** 32-bit registers [3, Table 21]: four processing blocks, each with "a physical
register file of 16,384, 32-bit elements" [6, Section 3.5.1]. They are handed out, not shared: the
threads of a warp "have their own instruction address counter and register state" [3, SIMT
Architecture], and "separate registers are allocated to all active threads", so "no swapping of
registers or other state need occur when switching among GPU threads. Resources stay allocated to
each thread until it completes its execution" [4, 3.1]. "Registers are allocated to an entire block
all at once" [4, 11.1.1].

![Registers on an SM](/images/gpu-threads-blocks-registers/fig4-registers.svg)
*Figure 4. Top: an SM's registers with four blocks of 256 threads at 32 registers each; half of them
are left free. Middle: in warp 0, every thread has its own registers. Bottom: what decides how many a
kernel needs.*

A thread cannot read another thread's registers, with one exception: the warp shuffle functions
(`__shfl_sync`, `__shfl_xor_sync`, …) exchange a variable between the threads of one warp [3, Warp
Shuffle Functions]; my gradient norm adds 32 values with five of them [2]. Each level of the hierarchy
has its own way to move values between threads:

| between | how | source |
|---|---|---|
| threads of one warp | shuffles: register to register | [3, Warp Shuffle Functions] |
| warps of one block | shared memory, then `__syncthreads()` | [3, Shared Memory] |
| different blocks | GPU memory | [3, Thread Hierarchy] |

*Table 10. Three ways threads exchange values.*

### 5.3 The compiler decides the count, per kernel

The number of registers per thread is fixed for each kernel when it is compiled: the compiler "uses
heuristics to minimize register usage while keeping register spilling [...] and instruction count to
a minimum" [3, Launch Bounds], and `nvcc --ptxas-options=-v` reports the result [4, 11.1.1]. On the
GPU, "register allocations are rounded up to the nearest 256 registers per warp" [4, 11.1.1], that is,
in increments of 8 per thread [6, Section 3.5.1].

The count matters because it decides how many blocks fit, and the steps are sharp. The Programming
Guide's example, for compute capability 6.x: "if a kernel uses 64 registers and each block has 512
threads and requires very little shared memory, then two blocks (i.e., 32 warps) can reside on the
multiprocessor since they require 2x512x64 registers, which exactly matches the number of registers
available on the multiprocessor. But as soon as the kernel uses one more register, only one block
(i.e., 16 warps) can be resident" [3, Multiprocessor Level].

### 5.4 Live values, not lines of code

A register is needed only while its value is *live*: from where it is computed to its last use.
Values whose lives do not overlap can share a register. This is the classic way compilers allocate
registers, by interference between live ranges [12]:

```cpp
float a = x * 2.0f;     // a live
float b = a + 1.0f;     // a's last use: b can take its register
float c = b * b;        // b's last use: c takes the same register
out[i] = c - 3.0f;
```

Three named values, and at most one is live at any point, so one register can hold all three. Named
temporaries cost nothing by themselves; what costs registers is many values live **at the same
time**, such as many running sums kept for a whole loop. One case never fits in registers: arrays
"for which it cannot determine that they are indexed with constant quantities" go to local memory
[3, Device Memory Accesses].

### 5.5 Counting a kernel's registers

A register holds 32 bits [3, Table 21], so wider types take more than one:

| type | bytes | registers |
|---|---:|---:|
| `float`, `int` | 4 | 1 |
| `size_t`, a pointer (64-bit) | 8 | 2 |
| `float4` | 16 | 4 |

*Table 11. Registers per variable (arithmetic).*

In the example, the two `float4`s are 8 registers; the eight `size_t` indices are up to 16, fewer
when some are no longer live; the parameters and computed addresses need a few more. **My estimate is
20 to 30 registers per thread**, which the GPU rounds up to 24 or 32. The real number is what
`--ptxas-options=-v` or `cudaFuncGetAttributes` reports (Section 8.4).

### 5.6 The limit per thread depends on the block size

All the threads on an SM share its 65,536 registers, and one thread can have at most 255
[3, Table 21]:

| threads on the SM | most registers each thread can have |
|---:|---:|
| 1,024 (for example, 4 blocks of 256) | 65,536 / 1,024 = 64 |
| 512 | 128 |
| 256 (a single block of 256) | 255 (the per-thread maximum) |

*Table 12. Arithmetic on the limits.*

"If there are not enough registers or shared memory available per multiprocessor to process at least
one block, the kernel will fail to launch" [3, Hardware Multithreading]. With blocks of 256, that
failure only comes past 255 registers, where the compiler spills instead (Section 8.3); what changes
first is how many of the blocks fit at once.

### 5.7 Occupancy

"Occupancy is the ratio of the number of active warps per multiprocessor to the maximum number of
possible active warps" [4, 11.1]. Registers decide it through the number of blocks that fit:

| registers per thread | block size | registers per block | blocks per SM | warps | occupancy |
|---:|---:|---:|---:|---:|---:|
| 32 | 256 | 8,192 | 4 | 32 | 100% |
| 64 | 256 | 16,384 | 4 | 32 | 100% |
| 72 | 256 | 18,432 | 3 | 24 | 75% |
| 88 | 256 | 22,528 | 2 | 16 | 50% |
| 128 | 256 | 32,768 | 2 | 16 | 50% |
| 72 | 1,024 | 73,728 | 0: the launch fails | — | — |

*Table 13. Arithmetic with the rules of Section 5.3 on a T4 SM (32 warps, 65,536 registers): blocks
per SM is the smallest of 4 (warps), 65,536 / registers per block, and 16.*

Occupancy matters because "executing other warps when one warp is paused or stalled is the only way
to hide latencies and keep the hardware busy", but it is not a goal in itself: "Higher occupancy does
not always equate to higher performance—there is a point above which additional occupancy does not
improve performance. However, low occupancy always interferes with the ability to hide memory
latency" [4, 11.1]. Section 7 uses this freedom.

## 6. Getting data in and out

### 6.1 The memory levels

Every value a kernel needs has to come in from GPU memory at least once. On the way it passes caches
that are smaller and faster the closer they are to the SM. Jia et al. measured each level on the T4:

![Where the numbers live](/images/gpu-threads-blocks-registers/fig5-memory.svg)
*Figure 5. A T4's memories and their measured read latencies [6]. The red path is a spill (Section 8).*

| place | where | size on a T4 | read latency on a T4 | who sees it |
|---|---|---|---|---|
| registers | each SM | 256 KB per SM | no extra cycles [4] | one thread |
| shared memory | each SM | 32 or 64 KB per SM | 19 cycles | one block |
| L1 data cache | each SM | the rest of 96 KB | 32 cycles | the SM |
| L2 cache | the chip | 4 MB | ≈ 188 cycles | all SMs |
| GPU memory | separate chips | 16 GB | 296 cycles; 616 on a TLB miss | everyone |

*Table 14. Sizes and latencies from Jia et al. [6, Table 3.1, Figure 3.5], in cycles at 1,590 MHz,
measured one access at a time. The 616 includes a miss in the address-translation cache (TLB).*

### 6.2 Hiding the wait

A thread needs new data for every round of its loop, and each read from GPU memory takes about 300
cycles. The SM keeps busy by switching warps, which costs nothing (Section 2.4): while one warp waits
for its data, the schedulers issue instructions for other warps, which send their own requests.

![Hiding the wait for memory](/images/gpu-threads-blocks-registers/fig6-latency-hiding.svg)
*Figure 6. An illustration: every warp waits about 300 cycles for its data, but the waits overlap,
and after the first answers, data arrives continuously.*

How many warps are needed depends on the code: "more warps are required if the ratio of the number
of instructions with no off-chip memory operands [...] to the number of instructions with off-chip
memory operands is low (this ratio is commonly called the arithmetic intensity of the program)"
[3, Multiprocessor Level]. The example does 4 adds per 3 loads, a very low ratio, so it needs many
requests in flight; its speed is then set by the memory bandwidth, 320 GB/s on paper and 220 GiB/s
measured by Jia et al.'s benchmark [6, Table 3.1].

Two details:

- **A store has no result to wait for.** A warp stalls only when an instruction needs an operand that
  is not ready yet [3, Multiprocessor Level]; a store produces no value that a later instruction of
  the thread depends on, so the warp goes on to its next instruction.
- **Neighbouring threads, neighbouring addresses.** "The concurrent accesses of the threads of a warp
  will coalesce into a number of transactions equal to the number of 32-byte transactions necessary
  to service all of the threads of the warp" [4, 10.2.1]. In the example, the 32 threads of a warp
  read 32 `float4`s that lie next to each other: 512 bytes, 16 transactions of 32 bytes, the fewest
  possible.

## 7. Spending registers to cut the time spent on memory

### 7.1 Two ways registers save memory time

The example uses an estimated 26 registers per thread out of the 64 it may have at full occupancy
(Figure 4): more than half of each SM's registers are idle. Registers can cut the time a kernel spends
on memory in two different ways:

| way | what the registers hold | what it saves | needs |
|---|---|---|---|
| **reuse** | a value that will be used again | reads: the value is loaded once instead of several times | the same value used more than once |
| **more loads in flight** | several loaded values that are not used yet | waiting: several waits overlap in one thread | independent loads |

*Table 15. Neither changes the result of the kernel; both cost registers.*

### 7.2 Reuse: load once, use many times

A value in a register costs nothing to read again [4, 10.2.7], so a value used by several outputs
should be loaded once and kept.

**The example has little to reuse.** Each `wte` value is used once. The only shared input is `wpe`:
position *t*'s row is added to the B = 4 rows of the batch, and the example loads it 4 times, by 4
different threads. Giving each thread one position and one column, and all B rows, loads it once:

```cpp
// one thread per (t, column): wpe[t][column] is loaded once and used for all B rows
float4 position_part = wpe[t * groups_per_row + column];
for (size_t b = 0; b < B; b++) {
    size_t row = b * T + t;
    float4 token_part = wte[(size_t)token_ids[row] * groups_per_row + column];
    encoded[row * groups_per_row + column] = make_float4(token_part.x + position_part.x, ...);
}
```

| | float4 loads of `wpe` | all float4 loads |
|---|---:|---:|
| the example | 786,432 | 1,572,864 |
| one thread per (t, column) | 196,608 | 983,040 (−37.5%) |

*Table 16. Arithmetic, for B = 4, T = 1,024.*

The bytes from GPU memory change much less than the table suggests: `wpe` is 3 MB and fits in the
T4's 4 MB L2 cache [6], so the repeated loads mostly hit L2 already (≈ 188 cycles instead of 296).
What the change saves is requests to L2, not trips to GPU memory. It also has a cost: 196,608 threads
with work instead of 524,288, each doing 4 rows. Whether it is faster has to be measured.

**Matrix multiplication is where reuse pays.** In `encoded @ W` with `W` of 768 × 768, each input
value is multiplied by 768 weights. A thread that computes a tile of outputs, kept in registers, uses
each loaded value many times (arithmetic):

```
1 output per thread:       per step load 1 x and 1 w  →  1 multiply-add           (0.5 per load)
8 × 8 outputs per thread:  per step load 8 x and 8 w  →  8 × 8 = 64 multiply-adds (4 per load)
```

This *register blocking* is how Volkov and Demmel's matrix multiplication reached near-peak speed
[9], and the Best Practices Guide describes the same idea: "some operations common to each element
can be performed by the thread once, amortizing the cost over the number of shared memory elements
processed by the thread" [4, 11.4]. Its price is 64 running sums, so 64 registers or more, which
lowers occupancy (Table 13).

### 7.3 More loads in flight: ask for several before using any

When each value is used once, registers can still save time, by **overlapping waits inside one
thread**. In the example, a thread asks for one `float4` of `wte` and one of `wpe`, waits about 300
cycles, adds, stores, and only then asks for the next. If it asks for four before it uses any, the
four waits overlap:

![Loads in flight](/images/gpu-threads-blocks-registers/fig7-ilp.svg)
*Figure 7. One thread over 1,000 cycles, with one float4 per round or four. Illustrative timings.*

```cpp
#define GROUPS_PER_THREAD 4

__global__ void encoder_forward_kernel_x4(float4* encoded, const int* token_ids, const float4* wte,
                                          const float4* wpe, size_t B, size_t T, size_t embed_dim) {
    size_t threads = (size_t)gridDim.x * blockDim.x;
    size_t first = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t groups_per_row = embed_dim / 4, groups = B * T * groups_per_row;
    for (size_t base = first; base < groups; base += threads * GROUPS_PER_THREAD) {
        float4 token_part[GROUPS_PER_THREAD], position_part[GROUPS_PER_THREAD];
        #pragma unroll
        for (int k = 0; k < GROUPS_PER_THREAD; k++) {           // 1. ask for all of them
            size_t group = base + k * threads;
            if (group < groups) {
                size_t row = group / groups_per_row, column = group % groups_per_row;
                token_part[k] = wte[(size_t)token_ids[row] * groups_per_row + column];
                position_part[k] = wpe[(row % T) * groups_per_row + column];
            }
        }
        #pragma unroll
        for (int k = 0; k < GROUPS_PER_THREAD; k++) {           // 2. then use them
            size_t group = base + k * threads;
            if (group < groups)
                encoded[group] = make_float4(token_part[k].x + position_part[k].x, token_part[k].y + position_part[k].y,
                                             token_part[k].z + position_part[k].z, token_part[k].w + position_part[k].w);
        }
    }
}
```

Three details make it work:

- **The values live in registers, not local memory.** `token_part[k]` is an array, and an array
  indexed by a value known only at run time goes to local memory [3, Device Memory Accesses]. The
  `#pragma unroll` on loops with a constant count turns every `k` into a constant, so the compiler can
  keep each element in its own registers. This must be checked: `localSizeBytes` must stay 0
  (Section 8.4).
- **Each round still coalesces.** The k-th load of every thread in a warp is `base + k × threads`, and
  neighbouring threads have neighbouring `base`, so each of the four loads of a warp reads 512
  neighbouring bytes, as before.
- **The loads are independent.** None of the four needs another's result, so the warp can issue all
  of them before it stalls on the first add [3, Multiprocessor Level].

This is what Volkov calls instruction-level parallelism (ILP): fewer threads, each with more
independent work, can hide the same latency [8]. The Best Practices Guide agrees: "with a high
degree of exposed instruction-level parallelism (ILP) it is, in some cases, possible to fully cover
latency with a low occupancy" [4, 11.3]. Why more requests in flight help at all is Little's law:
the data in flight must equal the latency times the throughput [7]. The latencies of Table 14 are
measured one access at a time; under full load the latency is higher [7], so more data has to be in
flight than that number suggests. My own gradient norm kernel is an example: reading 16 bytes per
request instead of 4, with the same threads, took it from 215 to 278 GB/s on a T4 [2].

### 7.4 How far to go

Each extra `float4` in flight costs 8 registers (4 for the `wte` part, 4 for the `wpe` part), and
past 64 per thread, fewer blocks of 256 fit (Table 13). For the example on a T4:

| float4s per round | registers per thread (estimate) | rounded | blocks of 256 per SM | warps | float4 loads in flight per SM, at most |
|---:|---:|---:|---:|---:|---:|
| 1 | ≈ 26 | 32 | 4 | 32 | 32 × 32 × 2 = 2,048 (32 KB) |
| 2 | ≈ 34 | 40 | 4 | 32 | 4,096 (64 KB) |
| 4 | ≈ 50 | 56 | 4 | 32 | 8,192 (128 KB) |
| 8 | ≈ 82 | 88 | 2 | 16 | 16 × 32 × 16 = 8,192 (128 KB) |

*Table 17. Register counts are my estimates; the rest is arithmetic with the rules of Section 5.
"At most" assumes every warp has issued all its loads.*

In this model, 4 per round keeps every warp slot full and has 4 times as many requests in flight; 8
per round loses half the warps and gains nothing more. The model leaves out the extra index
arithmetic, the larger last round and the fact that wpe mostly comes from L2, so it only says which
versions are worth timing. **I have not timed these versions**: the next step is to build each one,
check its registers and local memory with the test of Section 8.4, and time it against the example.

## 8. When registers run out

### 8.1 Spilling

If a kernel needs more registers than it may use, the compiler keeps some variables in local memory
instead. The guide lists what goes there: arrays it cannot prove are indexed with constants, "large
structures or arrays that would consume too much register space", and "any variable if the kernel
uses more registers than available (this is also known as register spilling)"
[3, Device Memory Accesses]. A spilled variable is stored to local memory and loaded back when it is
next used. **The results stay correct**; only the speed changes.

### 8.2 Local memory

"Local memory is so named because its scope is local to the thread, not because of its physical
location. In fact, local memory is off-chip. Hence, access to local memory is as expensive as access
to global memory" [4, 10.2.4]. It lives in GPU memory, on the same chips as all the kernel's data
(Figure 5), and "local memory accesses have the same high latency and low bandwidth as global memory
accesses" [3, Device Memory Accesses].

Two things soften the cost. The layout: "local memory is however organized such that consecutive
32-bit words are accessed by consecutive thread IDs. Accesses are therefore fully coalesced as long as
all threads in a warp access the same relative address" [3, Device Memory Accesses], which is the case
for a spilled variable. And the cache: "an L2 cache shared by all SMs [...] is used to cache accesses
to local or global memory, including temporary register spills" [3, Compute Capability 7.x]; an L2
hit takes ≈ 188 cycles on the T4, against 296 from GPU memory [6].

So spending registers (Section 7) has a ceiling: a version that needs more registers than it may have
spills, and the spill traffic goes back to memory, the very thing the extra registers were meant to
save.

### 8.3 When a kernel spills, and when its launch fails instead

| situation | result | source |
|---|---|---|
| the kernel needs ≤ 255 registers, no limit given | no spill | [3, Table 21] |
| it needs more than a limit set with `__launch_bounds__` | "the compiler reduces it further until it becomes less or equal to L, usually at the expense of more local memory usage and/or higher number of instructions" | [3, Launch Bounds] |
| a limit set for a whole file with `-maxrregcount` | the same, except for kernels with launch bounds, where it is ignored | [3, Launch Bounds], [4, 10.2.7.1] |
| no limit, but registers × threads per block > 65,536 | no spill, but the launch fails with `cudaErrorLaunchOutOfResources`, "too many resources requested for launch" | [3, Hardware Multithreading], [4, 11.1], [5] |

*Table 18. Without launch bounds, the compiler does not know the block size the kernel will be
launched with.*

The Runtime API describes the last error as one that "usually indicates that the user has attempted
to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for
the kernel's register count" [5]. The Best Practices Guide's advice is to tell the compiler the block
size: "developers should include the single argument `__launch_bounds__(maxThreadsPerBlock)` which
specifies the largest block size that the kernel will be launched with. Failure to do so could lead to
'too many resources requested for launch' errors" [4, 11.1]. The two-argument form,
`__launch_bounds__(256, 4)`, also asks for 4 resident blocks of 256, and the compiler then keeps the
kernel within the registers that allows, 64 per thread on a T4 [3, Launch Bounds]: a direct way to
hold the versions of Table 17 to full occupancy, at the risk of spills.

### 8.4 Checking a kernel

**The compiler.** `nvcc --ptxas-options=-v` (short: `-Xptxas -v`) prints the registers per thread of
every kernel [4, 11.1.1] and "total local memory usage per kernel" [4, 10.2.4]. The output looks like:

```
ptxas info    : Function properties for _Z22encoder_forward_kernelP6float4PKiPKS_S4_mmm
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 26 registers, 392 bytes cmem[0]
```

(The numbers show the format only. The long name is the kernel's name with its argument types
encoded.) Registers × threads per block must be ≤ 65,536, and the spill bytes should be 0.

**The program.** `cudaFuncGetAttributes` returns, for a kernel [5]:

| field | documented meaning |
|---|---|
| `numRegs` | "The number of registers used by each thread of this function." |
| `maxThreadsPerBlock` | "The maximum number of threads per block, beyond which a launch of the function would fail. This number depends on both the function and the device on which the function is currently loaded." |
| `localSizeBytes` | "The size in bytes of local memory used by each thread of this function." |

*Table 19. From the Runtime API's `cudaFuncAttributes` [5].*

For a kernel using 72 registers, `maxThreadsPerBlock` would be 896: a warp takes 72 × 32 = 2,304
registers, and 65,536 / 2,304 = 28 whole warps (arithmetic). A test can check every kernel of a file
this way, as the encoder's tests do:

```cpp
static void every_kernel_fits_a_block_of_encoder_threads(void)
{
    const void *kernels[] = {(const void *)encoder_forward_kernel, (const void *)encoder_backward_wte_kernel,
                             (const void *)encoder_backward_wpe_kernel};
    const char *names[] = {"encoder_forward_kernel", "encoder_backward_wte_kernel", "encoder_backward_wpe_kernel"};
    for (int k = 0; k < 3; k++)
    {
        cudaFuncAttributes attributes;
        cudaCheck(cudaFuncGetAttributes(&attributes, kernels[k]));
        printf("    %s: %d registers per thread, largest block %d threads, %zu bytes local memory\n", names[k],
               attributes.numRegs, attributes.maxThreadsPerBlock, attributes.localSizeBytes);
        ASSERT_EQ(attributes.maxThreadsPerBlock >= ENCODER_THREADS, true);   // the registers fit the block
        ASSERT_EQ(attributes.localSizeBytes, (size_t)0);                      // nothing spilled
    }
}
```

For the versions of Section 7, the same test answers both questions at once: `numRegs` shows how
many registers the extra loads really cost, and `localSizeBytes` shows whether the arrays stayed in
registers.

**The launch.** Following every launch with `cudaCheck(cudaGetLastError())` reports a failed launch
right where it happens, instead of leaving wrong numbers behind.

**A profiler.** Nsight Compute shows registers per thread, occupancy and memory traffic on a real
run, and includes an occupancy calculator [4, 11.1.1].

### 8.5 What the check costs

`cudaFuncGetAttributes` is a CPU call that reads what the compiler recorded about the kernel; it
launches nothing. Two one-time costs can land on it:

- **Starting CUDA.** The runtime creates its context "at the first runtime function which requires an
  active context on this device", and the guide warns to "keep this in mind when timing runtime
  function calls" [3, Initialization]. Whichever CUDA call comes first pays for it.
- **Loading the kernel.** With lazy loading, the default, CUDA "delays loading of CUDA modules and
  kernels from program initialization closer to kernels execution" [3, Lazy Loading], so the first
  question about a kernel may load it.

Since the answer only changes when the code is recompiled, it belongs in a one-time check such as a
test, not before every launch.

### 8.6 Fuse first, split last

Splitting a kernel into two gives each part its own registers, but it moves the values between the
parts through GPU memory: the first kernel's results must be stored before it ends, and the second
must load them again. That is a spill of a whole tensor (arithmetic):

```
fused:   load x → [step 1 → step 2, in registers] → store y
split:   kernel 1: load x → step 1 → store tmp         ← tmp goes out to GPU memory
         kernel 2: load tmp → step 2 → store y         ← and comes back
```

| | extra traffic | can L2 (4 MB) hold it? |
|---|---|---|
| spilling a few variables | a few bytes per thread | often |
| splitting, with a `tmp` of 3.1 million floats | 12.6 MB out and 12.6 MB back | no |

*Table 20. Two ways to run out of registers.*

My fused AdamW is an example: one kernel instead of 10 moved about 3.3× fewer bytes for the same
arithmetic [13]. The order to try things:

1. Write it as **one kernel**.
2. Check registers and spills (Section 8.4). Zero spill: done.
3. If it spills: keep **fewer values live at once**, give each thread less work (fewer float4s per
   round in Section 7.4), or use **smaller blocks**, which allow more registers per thread (Table 12).
4. Only then, and measured, **split**.

Good reasons to split are rarely registers: two parts that need different thread layouts, or a part
that needs all of another part finished first, from every block, which only the end of a kernel
guarantees (Section 2.3).

## 9. The limits in one table

| limit | Tesla T4 (7.5) | past it |
|---|---|---|
| threads per block | 1,024 | the launch fails: `cudaErrorInvalidConfiguration` [5] |
| threads per warp | 32 | fixed |
| resident warps (threads) per SM | 32 (1,024) | fewer blocks fit at once |
| resident blocks per SM | 16 | fewer blocks fit at once |
| 32-bit registers per SM | 65,536 | fewer blocks fit at once |
| registers per block | 65,536 | the launch fails: `cudaErrorLaunchOutOfResources` [5] |
| registers per thread | 255 | the compiler spills |
| shared memory per block | 48 KB static; up to 64 KB dynamic, with an opt-in | the launch fails: `cudaErrorInvalidConfiguration` [5] |
| local memory per thread | 512 KB | — |
| SMs | 40 | blocks wait for a free SM: more waves |

*Table 21. From the Programming Guide [3, Table 21; Compute Capability 7.x], except the SM count [6].*

## 10. Using them well

| | do | source |
|---|---|---|
| 1 | threads per block: a multiple of 32, at least 64; start with 128–256 and measure | [4, 11.3] |
| 2 | prefer several blocks per SM to one big block, unless threads must combine results across the whole block | [4, 11.3], Section 4.2 |
| 3 | more blocks than SMs, several per SM; thousands per launch to scale to future GPUs, or a grid-stride loop over a fixed grid | [4, 11.3], [11] |
| 4 | neighbouring threads read neighbouring addresses, so a warp's loads coalesce | [4, 10.2.1] |
| 5 | spend idle registers: keep reused values in registers, and ask for several independent loads before using any | [9], [8], [4, 11.3], Section 7 |
| 6 | keep registers per thread at or under 65,536 ÷ the threads wanted on an SM (64 for 4 blocks of 256 on a T4), but do not chase occupancy for its own sake | [4, 11.1], [8] |
| 7 | give kernels `__launch_bounds__`, so the compiler knows the block size | [4, 11.1], [3, Launch Bounds] |
| 8 | read each value once and write it once; fuse before splitting | Section 8.6 |
| 9 | check registers and local memory with `--ptxas-options=-v`, or in a test with `cudaFuncGetAttributes`, and time every version | [4, 10.2.4], [5] |

*Table 22. A checklist.*

## References

1. D. B. Song. *gpt2-small*, branch `c-cuda-trainer`: GPT-2 small pre-training in C and CUDA.
   [github.com/ducbachsong/gpt2-small](https://github.com/ducbachsong/gpt2-small/tree/c-cuda-trainer)
2. D. B. Song. *Adding 124 million squares: a one-kernel gradient norm for GPT-2 in CUDA.* 2026.
   [/research/grad-norm-cuda/](/research/grad-norm-cuda/)
3. NVIDIA. *CUDA C++ Programming Guide*, release 12.6. Sections Thread Hierarchy, Initialization,
   Hardware Implementation (SIMT Architecture, Hardware Multithreading), Multiprocessor Level, Device
   Memory Accesses, Launch Bounds, Warp Shuffle Functions, Lazy Loading, Compute Capability 7.x, and
   Table 21 "Technical Specifications per Compute Capability".
   [docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html)
4. NVIDIA. *CUDA C++ Best Practices Guide.* Sections 3.1 (Differences between Host and Device), 10.2.1
   (Coalesced Access to Global Memory), 10.2.4 (Local Memory), 10.2.7 (Registers), 11.1 (Occupancy),
   11.2 (Hiding Register Dependencies), 11.3 (Thread and Block Heuristics), 11.4 (Effects of Shared
   Memory).
   [docs.nvidia.com/cuda/cuda-c-best-practices-guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
5. NVIDIA. *CUDA Runtime API*: `cudaFuncAttributes`, and the error codes `cudaErrorInvalidConfiguration`
   and `cudaErrorLaunchOutOfResources`.
   [docs.nvidia.com/cuda/cuda-runtime-api](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaFuncAttributes.html)
6. Z. Jia, M. Maggioni, J. Smith, D. P. Scarpazza. *Dissecting the NVidia Turing T4 GPU via
   Microbenchmarking.* arXiv:1903.07486, 2019. [arxiv.org/abs/1903.07486](https://arxiv.org/abs/1903.07486)
7. V. Volkov. *Understanding Latency Hiding on GPUs.* PhD thesis, UC Berkeley, technical report
   UCB/EECS-2016-143, 2016.
   [eecs.berkeley.edu](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-143.html)
8. V. Volkov. *Better Performance at Lower Occupancy.* GPU Technology Conference, 2010.
   [nvidia.com](https://www.nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf)
9. V. Volkov, J. W. Demmel. *Benchmarking GPUs to Tune Dense Linear Algebra.* SC 2008.
10. NVIDIA. *NVIDIA A100 Tensor Core GPU Architecture* (whitepaper), 2020: 108 SMs.
    [nvidia.com](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
11. M. Harris. *CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops.* NVIDIA Technical Blog,
    2013. [developer.nvidia.com](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/)
12. G. J. Chaitin. *Register Allocation & Spilling via Graph Coloring.* SIGPLAN Symposium on Compiler
    Construction, 1982. [doi.org/10.1145/872726.806984](https://doi.org/10.1145/872726.806984)
13. D. B. Song. *One launch: a fused AdamW kernel for GPT-2 in CUDA.* 2026.
    [/research/fused-adamw-cuda/](/research/fused-adamw-cuda/)
