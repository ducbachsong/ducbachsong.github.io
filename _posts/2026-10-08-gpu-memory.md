---
title: "Where a CUDA kernel's numbers live: registers, local, shared, L1, L2 and GPU memory"
date: 2026-10-08
permalink: /posts/gpu-memory/
excerpt: "A GPU has many memories, and their names mix up where a memory is with what it is for: local memory is not near the thread, and a warp has no memory of its own. This note goes through them one level at a time, the thread, the warp, the block, the SM, the chip and the card, and says for each what memory it has, what that memory is for and who decides what goes in it. Then it follows a value through them: a load, a store, an atomic add, a spill, shared memory and a shuffle. With the numbers of a Tesla T4 and the encoder kernels of my GPT-2 trainer."
tags:
  - cuda
  - gpu
  - notes
---

**What this is.** A CUDA kernel reads its inputs from one memory, computes in another, and may pass
through four or five more on the way. This note goes through them one level at a time: the
**thread**, the **warp**, the **block**, the **SM**, the **chip** and the **card**. For each level it
says what memory it has, what that memory is for, who decides what goes in it, and how it connects to
the others. It follows the [note on threads, warps, blocks and registers](/posts/gpu-threads-blocks-registers/)
[2], which explains registers and spills in more depth, and uses the same GPU: a Tesla T4, the GPU of
Google Colab. Nothing new is measured here.

**Sources.** NVIDIA's documentation (the CUDA C++ Programming Guide [3], the Best Practices Guide [4],
the Runtime API [5]), Jia et al.'s measurements of the T4 [6], and the compiled code of my encoder
kernels [7]. Quotes are marked as quotes; numbers worked out by hand say so.

* TOC
{:toc}

## 1. The map

### 1.1 Every level and the memory it has

![The memories of a Tesla T4](/images/gpu-memory/fig1-map.svg)
*Figure 1. Where each memory is on a T4. Each of the 40 SMs has its own register file, shared memory,
L1 and constant cache; the chip has one L2; the card has the DRAM.*

| level | the memory it has | size on a T4 | who can see it |
|---|---|---|---|
| **thread** | its registers | up to 255 of 32 bits (1,020 bytes) | that thread only |
| | its local memory, in DRAM | up to 512 KB | that thread only |
| **warp** | none of its own: it is 32 threads, each with its own registers | — | shuffles move values between the 32 |
| **block** | its shared memory | up to 48 KB, 64 KB if the kernel asks for it | the threads of that block |
| **SM** | the register file | 65,536 × 4 bytes = 256 KB | split between its threads |
| | shared memory + L1 cache | 96 KB together, split 32 + 64 or 64 + 32 | shared: per block; L1: the SM |
| | the constant cache | 8 KB | the SM |
| **chip** | the L2 cache | 4 MB | all 40 SMs |
| **card** | GPU memory (DRAM): global, local and constant memory | 16 GB | every thread; the CPU copies in and out |

*Table 1. From the Programming Guide [3, Table 21; Compute Capability 7.x] and Jia et al. [6, Table
3.1]. The register file, 256 KB, is bigger than the SM's L1 and shared memory together.*

### 1.2 Two kinds of names: places and uses

The names are confusing because some say **where** a memory is and others say **what it is used
for**:

- **Places** are pieces of hardware: the register file, the 96 KB of on-chip memory, the constant
  cache, L2, DRAM.
- **Uses** are what the program sees: registers, *local*, *shared*, *global* and *constant* memory.

Each use lives in some place:

| use in the program | place in the hardware | cached on the way in | who decides what goes in |
|---|---|---|---|
| registers | the register file, in the SM | — | the compiler |
| **local** memory | **DRAM** | L1, L2 | the compiler |
| **shared** memory | the SM's 96 KB of on-chip memory | — (it is on the chip already) | you, with `__shared__` |
| **global** memory | DRAM | L1, L2 | you, with `cudaMalloc` |
| **constant** memory | DRAM | the SM's constant cache | you, with `__constant__`; also the kernel's parameters |

*Table 2. Uses and places.*

The row that surprises is local memory: "Local memory is so named because its scope is local to the
thread, not because of its physical location. In fact, local memory is off-chip. Hence, access to
local memory is as expensive as access to global memory" [4, 10.2.4]. *Local* says who can see it,
not where it is.

### 1.3 How far each one is

| place | read latency on a T4 |
|---|---:|
| registers | no extra cycles [4, 10.2.7] |
| shared memory | 19 cycles |
| L1 | 32 cycles |
| L2 | ≈ 188 cycles |
| DRAM | 296 cycles (616 on a TLB miss) |

*Table 3. Jia et al. [6, Figure 3.5], one access at a time, at 1,590 MHz. Under full load the waits
are longer.*

Every step outwards is bigger, slower and shared by more threads. The rest of this note goes through
the levels from the inside out.

## 2. The thread: registers and local memory

### 2.1 Registers

**What they are for:** every calculation. An add, a multiply or a compare takes its inputs from
registers and writes its result to a register. Data in GPU memory has to be loaded into registers
before a thread can do anything with it, and stored back afterwards.

**Who decides:** the compiler. It turns the kernel's variables into registers and fixes, per kernel,
how many each thread gets. The SM hands them out from its 65,536 when a block arrives, and a thread
keeps them until it ends [4, 3.1].

**How many:** at most 255 per thread [3, Table 21], and all the threads on an SM share the 65,536. With
1,024 threads on an SM, that is 64 each.

**An example.** The backward kernel for `wte` in my trainer [1] loads 4 floats of `encoded_grad` and
adds them onto a row of `wte_grad`. Its compiled code [7], shortened here, shows the 4 floats landing
in 4 registers, R12 to R15, and each one going out from its own register:

```
LDG.E.128  R12, [R8]              load 16 bytes: R12, R13, R14, R15 = encoded_grad[index]
RED.E.ADD  [R4],      R12         wte_grad row[0] += R12
RED.E.ADD  [R4+0x4],  R13         wte_grad row[1] += R13
RED.E.ADD  [R4+0x8],  R14         wte_grad row[2] += R14
RED.E.ADD  [R4+0xc],  R15         wte_grad row[3] += R15
```

A `float4` is 4 registers; there is no register that holds all 16 bytes at once.

### 2.2 Local memory

**What it is for:** a thread's values that cannot stay in registers. The compiler puts them in local
memory in three cases [3, Device Memory Accesses]:

| case | example | why not a register |
|---|---|---|
| too many values at once (a **spill**) | 80 running sums in a loop, when the thread may have 64 registers | there are not enough |
| an array indexed by a number known only while running | `float v[8]; v[k] = x;` with `k` from a loop the compiler cannot unroll | a register has no address, so `v[k]` cannot pick one |
| a stack | a function call that is not inlined; `printf` in a kernel | calls need memory that has addresses |

*Table 4. When the compiler uses local memory.*

**Where it is:** in DRAM, next to your tensors. Each thread has its own piece, up to 512 KB [3, Table
21], and no other thread can see it. Two things make it cheaper than it sounds. It is cached in L1
and L2: "an L2 cache shared by all SMs [...] is used to cache accesses to local or global memory,
including temporary register spills" [3, Compute Capability 7.x]. And it is laid out so a warp's 32
threads touch neighbouring addresses: "consecutive 32-bit words are accessed by consecutive thread
IDs. Accesses are therefore fully coalesced as long as all threads in a warp access the same relative
address" [3, Device Memory Accesses].

**Why to avoid it:** a value in a register is free to use again; a value in local memory is a store
now and a load later, and the load waits like any other load. The results stay right; only the speed
suffers.

**How to check it:** `cudaFuncGetAttributes` returns `localSizeBytes`, "the size in bytes of local
memory used by each thread of this function" [5], and `nvcc -Xptxas -v` prints the spills of every
kernel. The encoder's tests require 0 for all three kernels; the counting is per thread, so at 1,024
threads per SM, 16 bytes each would be 16 KB of DRAM per SM (arithmetic).

## 3. The warp: no memory of its own

A warp is 32 threads of one block that the SM runs together, one instruction for all 32 [3, SIMT
Architecture]. It has **no memory of its own**. What looks like warp memory is three things:

**Its threads' registers.** The register file hands registers out per warp, "rounded up to the nearest
256 registers per warp" [4, 11.1.1], which is 8 per thread. Each thread still sees only its own.

**Shuffles.** `__shfl_sync`, `__shfl_xor_sync` and their kin let a thread read a register of another
thread in the same warp [3, Warp Shuffle Functions], with no memory in between. My gradient norm adds
32 values into one with 5 of them (16 + 8 + 4 + 2 + 1) [8].

**One memory request for 32 threads.** When a warp runs a load, the 32 addresses go out together, and
"the concurrent accesses of the threads of a warp will coalesce into a number of transactions equal
to the number of 32-byte transactions necessary to service all of the threads of the warp" [4,
10.2.1]. This is why the warp matters for memory even without memory of its own:

| a warp's 32 `float4` loads | bytes wanted | 32-byte pieces fetched | bytes fetched |
|---|---:|---:|---:|
| neighbouring, as in the encoder | 512 | 16 | 512 |
| scattered, each in a different piece | 512 | 32 | 1,024 |

*Table 5. Arithmetic with the 32-byte rule.*

## 4. The block: shared memory

**What it is for:** threads of one block passing values to each other, or reading one piece of data
many times. It is the only memory that is both fast and seen by more than one thread.

**Who decides:** you. A `__shared__` variable is one copy per block; every thread of the block reads
and writes the same copy. It exists from the block's start to its end, and the next block on the SM
gets its own fresh copy.

```cpp
__shared__ float partial[32];          // one copy for the whole block
if (lane == 0) partial[warp] = sum;    // the first thread of each warp writes its warp's sum
__syncthreads();                       // wait until every warp has written
if (warp == 0) sum = partial[lane];    // warp 0 reads all 32
```

`__syncthreads()` is what makes this safe: every thread of the block waits there until all have
arrived, so the reads come after the writes.

**Where it is:** in the SM's 96 KB of on-chip memory, the same memory as L1. The split is 32 KB of
shared memory and 64 KB of L1, or 64 KB and 32 KB [6, Table 3.1]: more shared memory means less L1.
A block can have 48 KB, or up to 64 KB if the kernel asks for it [3, Table 21].

**How fast:** 19 cycles [6], but only when the 32 threads of a warp spread over its 32 *banks*.
Shared memory is cut into 32 banks of 4 bytes; neighbouring 4-byte words are in neighbouring banks,
and two threads of a warp that need different words of the same bank are served one after the other
[3, Compute Capability 5.x, Shared Memory].

**What it costs:** it limits how many blocks fit on an SM. With 64 KB of shared memory on the SM and
blocks that take 40 KB each, only one block fits, whatever the threads and registers allow
(arithmetic).

The encoder does not use shared memory: its threads never pass values to each other.

## 5. The SM: register file, L1 and the constant cache

Each of the T4's 40 SMs has three memories of its own (Figure 1).

### 5.1 The register file

The 256 KB the registers of Section 2.1 come from. It is the SM's biggest memory.

### 5.2 L1

**What it is for:** keeping recently read data close to the SM, so a second read of the same bytes by
any warp of that SM takes 32 cycles instead of ≈ 188 from L2 or 296 from DRAM [6]. It holds copies of
global and local memory, in the part of the 96 KB that shared memory does not take.

**Who decides:** the hardware. You never name L1 in the code; what stays in it depends on what was
read and how much has been read since.

**What it does not do:**

- **It does not keep the 40 SMs in step.** Each SM has its own L1, and nothing updates SM 5's L1 when
  SM 0 writes. This is one reason blocks must not wait on each other's results inside one kernel
  [3, Thread Hierarchy].
- **It is not where writes and atomic adds end up.** A store goes on to L2, and an atomic add on global
  memory is done in L2, where every SM sees the same copy (Section 6).

**What it does for the encoder:** little reuse. In the forward, each `wpe` value is needed by 4
threads, one for each row of the batch, but those threads are 196,608 `float4`s apart, in different
blocks and usually on different SMs, so they do not meet in one L1. L1's main help there is collecting
each warp's 32 neighbouring loads into whole 32-byte pieces.

### 5.3 The constant cache

**What it is for:** values every thread reads and nobody changes while the kernel runs: `__constant__`
variables and the kernel's parameters, which are passed through constant memory [3, Function
Parameters]. The memory itself is 64 KB in DRAM; each SM keeps an 8 KB working set close by [3, Table
21].

**Best case:** all 32 threads of a warp read the same address, and one read serves them all. When they
read different addresses, "a request is then split into as many separate requests as there are
different memory addresses in the initial request" [3, Device Memory Accesses]. So it suits the
parameters, such as `embed_dim` or the pointer to `wte`, which are the same for every thread.

The compiler reports the parameters' space as `cmem[0]` in `nvcc -Xptxas -v`'s output [2, 8.4].

## 6. The chip: L2

There is **one** L2 on the chip, 4 MB on a T4 [6], and all 40 SMs use it. Everything that goes to or
from DRAM passes through it.

**It keeps data for every SM.** A value read by SM 0 and then by SM 5 can come from L2 the second time,
≈ 188 cycles instead of 296. It only works if the second read comes before the value is pushed out by
newer data. In the encoder's forward, `wpe` (3.1 MB) would fit in 4 MB, but between two reads of the
same `wpe` row about 25 MB of `wte` and `encoded` stream through. My measurement of the forward's
bandwidth suggests `wpe` is read from DRAM again for each row of the batch [7, 6.5].

**It is the copy every SM agrees on.** L2 is single, so a value written there is the same value for
every SM. That is why atomic adds on global memory are carried out in L2, not in the threads and not in
L1. When thread A on SM 0 and thread B on SM 5 both add onto `wte_grad[3][0]`:

```
thread A sends "add 0.3 to wte_grad[3][0]"   and goes on
thread B sends "add 0.5 to wte_grad[3][0]"   and goes on
L2:  0.0  →  + 0.3 = 0.3  →  + 0.5 = 0.8      one after the other, in arrival order
```

Each add is read, add and write as one step, so none is lost. If the thread does not use the old value
that `atomicAdd` returns, the compiler emits `RED`, which the thread sends without waiting for an
answer [7, 6.1]. The order is whichever arrives first, so a sum of many floats can differ in its last
bits from run to run.

**It caches local memory too**, as quoted in Section 2.2, so a spill that is loaded back soon may not
reach DRAM at all.

## 7. The card: GPU memory (DRAM)

The T4 has 16 GB of GDDR6 on separate chips next to the GPU, at 320 GB/s [9]. It is the only memory
that lasts longer than a kernel: everything a kernel produces has to end up here. It holds three uses:

| use | what is in it | who writes it | lives until |
|---|---|---|---|
| global memory | your tensors: `wte`, `wpe`, `encoded`, the gradients | you: `cudaMalloc`, `cudaMemcpy`, kernels | `cudaFree` |
| local memory | each thread's spills, arrays and stack | the compiler's code, per thread | the thread ends |
| constant memory | `__constant__` variables, kernel parameters | the CPU, before the kernel | the program ends |

*Table 6. The three uses of DRAM.*

The encoder's kernels are limited by this memory: they do almost no arithmetic, so their time is the
time to move their bytes through DRAM, about three quarters of the 320 GB/s at the usual launch
[7, 6.5].

## 8. How a value moves

![How a value moves between the memories](/images/gpu-memory/fig2-paths.svg)
*Figure 2. The six kinds of move. A load passes L2 and L1 on its way in; a store and an atomic add go
out to L2; a spill goes all the way to DRAM unless a cache keeps it; shared memory and shuffles never
leave the SM.*

| move | path | started by | does the thread wait? |
|---|---|---|---|
| load `x = a[i]` | DRAM → L2 → L1 → registers | you | only when it uses `x` |
| store `a[i] = x` | registers → L2 → DRAM later | you | no |
| `atomicAdd`, result unused (`RED`) | registers → L2, which adds | you | no |
| `atomicAdd`, result used (`ATOM`) | registers → L2, which adds and sends the old value back | you | when it uses the old value |
| spill and reload | registers → local memory (L1, L2, DRAM) → registers | the compiler | on the reload |
| shared memory | registers ↔ shared memory | you | about 19 cycles on a read |
| shuffle | register → register, inside one warp | you | a few cycles |

*Table 7. The moves. "Does the thread wait": a GPU thread only stops when an instruction needs a value
that is not there yet [3, Multiprocessor Level].*

**One job of the encoder's `wte` backward**, from its compiled code [7], ties the levels together:

| step | instruction | from | to |
|---|---|---|---|
| 1 | `LDG.E` the token id | global memory, through L2 and L1 | 1 register |
| 2 | `LDG.E.128` 16 bytes of `encoded_grad` | global memory, through L2 and L1 | 4 registers, R12–R15 |
| 3 | 4 × `RED.E.ADD`, one per float | 4 registers | L2, which does the 4 adds |

*Table 8. Steps 1 and 2 are sent together; step 3 waits for them, then sends four adds and goes on to
the next job without waiting.*

Two things follow. The load is one request for 16 bytes, so each thread keeps more bytes on their way
and fewer threads are needed to keep DRAM busy; that is what made this kernel stop slowing down at
small launches [7, 6.3]. The adds are still four, one per float, carried out by L2, one after the
other when two land on the same number: a T4 has no instruction that adds a `float4` at once, neither
as an atomic nor in registers.

## 9. All of it in one table

| memory | place | size on a T4 | read latency | seen by | decided by | lives as long as |
|---|---|---|---:|---|---|---|
| registers | register file, in each SM | 256 KB per SM; ≤ 255 per thread | none | one thread | the compiler | the thread |
| local memory | DRAM (cached in L1, L2) | ≤ 512 KB per thread | as global | one thread | the compiler | the thread |
| shared memory | each SM's 96 KB | 32 or 64 KB per SM; ≤ 48 (64) KB per block | 19 | one block | you | the block |
| L1 | each SM's 96 KB | 64 or 32 KB per SM | 32 | one SM | the hardware | until pushed out |
| constant cache | each SM | 8 KB | — | one SM | the hardware | until pushed out |
| L2 | the chip | 4 MB | ≈ 188 | all SMs | the hardware | until pushed out |
| global memory | DRAM | 16 GB, all uses together | 296 | every thread; the CPU by copies | you | `cudaFree` |
| constant memory | DRAM (cached per SM) | 64 KB | — | every thread, read-only | you | the program |

*Table 9. Latencies in cycles from Jia et al. [6]; sizes from [3, Table 21] and [6, Table 3.1].*

**What to remember:**

1. Calculations happen only in **registers**; every other memory is a place to load from or store to.
2. **Local** memory is the thread's own, but it is in DRAM: keep `localSizeBytes` at 0.
3. A **warp** has no memory; it is the unit of a memory request, so neighbouring threads should read
   neighbouring addresses.
4. **Shared** memory is the one fast memory several threads see, the threads of one block, and it
   takes space from L1.
5. **L1** is per SM and not kept in step between SMs; **L2** is one for the chip, and the atomic adds
   happen there.
6. **DRAM** holds global, local and constant memory, and is the only one that outlives a kernel.

## References

1. D. B. Song. *gpt2-small*, branch `c-cuda-trainer`: GPT-2 small pre-training in C and CUDA.
   [github.com/ducbachsong/gpt2-small](https://github.com/ducbachsong/gpt2-small/tree/c-cuda-trainer)
2. D. B. Song. *How a CUDA kernel uses threads, warps, blocks and registers.* 2026.
   [/posts/gpu-threads-blocks-registers/](/posts/gpu-threads-blocks-registers/)
3. NVIDIA. *CUDA C++ Programming Guide*, release 12.6. Sections Thread Hierarchy, SIMT Architecture,
   Multiprocessor Level, Device Memory Accesses, Function Parameters, Warp Shuffle Functions, Compute
   Capability 5.x (Shared Memory), Compute Capability 7.x, and Table 21 "Technical Specifications per
   Compute Capability".
   [docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html)
4. NVIDIA. *CUDA C++ Best Practices Guide.* Sections 3.1, 10.2.1 (Coalesced Access to Global Memory),
   10.2.4 (Local Memory), 10.2.7 (Registers), 11.1.1 (Calculating Occupancy).
   [docs.nvidia.com/cuda/cuda-c-best-practices-guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
5. NVIDIA. *CUDA Runtime API*: `cudaFuncAttributes`.
   [docs.nvidia.com/cuda/cuda-runtime-api](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaFuncAttributes.html)
6. Z. Jia, M. Maggioni, J. Smith, D. P. Scarpazza. *Dissecting the NVidia Turing T4 GPU via
   Microbenchmarking.* arXiv:1903.07486, 2019. [arxiv.org/abs/1903.07486](https://arxiv.org/abs/1903.07486)
7. D. B. Song. *How many threads? Launch sizes for GPT-2's encoder kernels on a T4.* 2026.
   [/research/encoder-launch-sizes/](/research/encoder-launch-sizes/); the compiled code is in its
   [Appendix A](/files/encoder-launch-sizes/sass-loads-and-atomics.txt).
8. D. B. Song. *Adding 124 million squares: a one-kernel gradient norm for GPT-2 in CUDA.* 2026.
   [/research/grad-norm-cuda/](/research/grad-norm-cuda/)
9. NVIDIA. *Tesla T4 Tensor Core GPU* (datasheet: 320 GB/s memory bandwidth).
   [nvidia.com](https://www.nvidia.com/en-us/data-center/tesla-t4/)
