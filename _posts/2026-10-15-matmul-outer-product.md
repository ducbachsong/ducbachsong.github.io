---
title: "Two ways to multiply matrices: inner products, outer products, and why a GPU kernel uses the second"
date: 2026-10-15
modified: 2026-10-15
version: 1
permalink: /posts/matmul-outer-product/
excerpt: "A matrix product is usually taught as row times column: each output is one dot product, an inner product. The same numbers come out of a second order: take one column of the left matrix and one row of the right, multiply every value by every value into a whole grid, an outer product, and add one grid for each value of the summed index. This note works both ways through one small example, shows that they do the same products in the same order, and counts why the linear layer kernel of my GPT-2 trainer uses the second: for each value it reads, it does 4 multiply-adds in a thread and 64 in a block, where the first way does half of one."
tags:
  - cuda
  - matmul
  - gpt-2
  - notes
---

**What this is.** The forward pass of a linear layer is a matrix product, $$y = x\, W^\top$$ (plus a
bias, which this note leaves out). At school it is worked out **row times column**: each output is the
dot product of a row of the left matrix and a column of the right. The linear kernel of my GPT-2
trainer [1] does not work it out that way. It goes **one value of the summed index at a time**, and at
each one multiplies a column by a row into a whole grid. This note shows the two ways side by side on
one small example, shows why they give the same numbers, and counts why the kernel uses the second.

**Sources.** The example is worked out by hand and checked with NumPy. The kernel is `linear.cuh` in
[1]; its sizes are those of Boehm's kernel 5 [3]. The register and shared memory limits are a Tesla
T4's, from the notes on threads and registers [4] and on GPU memory [5].

* TOC
{:toc}

## 1. The product

### 1.1 The formula

$$
y_{t,j} = \sum_{i=1}^{n_\text{in}} x_{t,i}\, W_{j,i}
$$

Output $$(t, j)$$ is row $$t$$ of $$x$$ dotted with row $$j$$ of $$W$$. $$W$$ has the output index
first, because that is how PyTorch's `nn.Linear` keeps it, `{out_features, in_features}`; as a matrix
product, that is $$x\, W^\top$$ [2].

| symbol | meaning | in GPT-2 small's qkv layer |
|---|---|---|
| $$t$$ | row: which token | $$T$$ = 4 × 1024 = 4096 rows |
| $$i$$ | input column: the index that is summed | $$n_\text{in}$$ = 768 |
| $$j$$ | output column | $$n_\text{out}$$ = 2304 |
| $$x$$ | input, `{T, n_in}` | `{4096, 768}` |
| $$W$$ | weight, `{n_out, n_in}`: row $$j$$ holds the weights of output $$j$$ | `{2304, 768}` |
| $$y$$ | output, `{T, n_out}` | `{4096, 2304}` |

*Table 1. The names, as in the [note on the linear backward](/posts/linear-backward/) [2].*

### 1.2 The inner dimension

Write the shapes of the product next to each other:

$$
\underbrace{\{T,\ n_\text{in}\}}_{x} \cdot \underbrace{\{n_\text{in},\ n_\text{out}\}}_{W^\top}
= \underbrace{\{T,\ n_\text{out}\}}_{y}
$$

$$n_\text{in}$$ sits **between** the two: it is the **inner dimension**, and it is summed away, so it
is not in $$y$$. $$T$$ and $$n_\text{out}$$ sit **outside**: they are the outer dimensions, and they
become $$y$$'s shape.

One warning, because the words are close: "inner dimension" names a **place in the shapes**, not a way
of multiplying. Both ways below sum over the same inner dimension. The second way, the outer product
way, is the one that **walks** the inner dimension, one value at a time.

### 1.3 The example

Two rows, three input columns, two output columns: $$T = 2$$, $$n_\text{in} = 3$$,
$$n_\text{out} = 2$$.

$$
x = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}
\qquad
W = \begin{bmatrix} 1 & 0 & 2 \\ 3 & 1 & 0 \end{bmatrix}
\qquad
W^\top = \begin{bmatrix} 1 & 3 \\ 0 & 1 \\ 2 & 0 \end{bmatrix}
$$

$$y$$ has 4 values, and each needs 3 products: 12 multiply-adds in all.

![The two ways, on the example](/images/matmul-outer-product/fig1-two-ways.svg)
*Figure 1. (a) Inner products: each output is one row of $$x$$ times one column of $$W^\top$$. (b)
Outer products: each $$i$$ gives a whole grid, column $$i$$ of $$x$$ times row $$i$$ of $$W^\top$$, and
the three grids add up to $$y$$. The thick cells are $$y_{1,1}$$ and its three products.*

## 2. Way 1: inner products, one output at a time

Take one output, and add up its products over the whole inner dimension:

```
for each row t:
    for each output column j:
        sum = 0
        for each i:                      # the inner dimension, innermost
            sum += x[t][i] * W[j][i]
        y[t][j] = sum
```

Each round of the two outer loops finishes **one** output. In the example:

| output | row of $$x$$ · column of $$W^\top$$ | value |
|---|---|---|
| $$y_{1,1}$$ | $$1 \cdot 1 + 2 \cdot 0 + 3 \cdot 2$$ | 7 |
| $$y_{1,2}$$ | $$1 \cdot 3 + 2 \cdot 1 + 3 \cdot 0$$ | 5 |
| $$y_{2,1}$$ | $$4 \cdot 1 + 5 \cdot 0 + 6 \cdot 2$$ | 16 |
| $$y_{2,2}$$ | $$4 \cdot 3 + 5 \cdot 1 + 6 \cdot 0$$ | 17 |

*Table 2. Way 1 on the example: four dot products.*

A dot product takes two lists of numbers and gives **one** number. Its other name is the **inner
product**, and that is why this is the inner product way.

## 3. Way 2: outer products, one i at a time

The **outer product** goes the other way: it takes a column of $$m$$ numbers and a row of $$n$$
numbers, and multiplies every one by every one into a grid of $$m \times n$$:

$$
\begin{bmatrix} a_1 \\ a_2 \end{bmatrix} \otimes \begin{bmatrix} b_1 & b_2 \end{bmatrix}
= \begin{bmatrix} a_1 b_1 & a_1 b_2 \\ a_2 b_1 & a_2 b_2 \end{bmatrix}
$$

A matrix product is a sum of these, one for each value of the inner dimension:

$$
x\, W^\top = \sum_{i=1}^{n_\text{in}} (\text{column } i \text{ of } x) \otimes (\text{row } i \text{ of } W^\top)
$$

Row $$i$$ of $$W^\top$$ is $$W_{j,i}$$ for every $$j$$. As loops, the inner dimension moves from the
inside to the **outside**:

```
y = all zeros
for each i:                              # the inner dimension, outermost
    for each row t:
        for each output column j:
            y[t][j] += x[t][i] * W[j][i]
```

Each round of the outer loop adds **one product to every output**. In the example:

| $$i$$ | column $$i$$ of $$x$$ | row $$i$$ of $$W^\top$$ | grid |
|---|---|---|---|
| 1 | $$(1, 4)$$ | $$(1, 3)$$ | $$\begin{bmatrix} 1 & 3 \\ 4 & 12 \end{bmatrix}$$ |
| 2 | $$(2, 5)$$ | $$(0, 1)$$ | $$\begin{bmatrix} 0 & 2 \\ 0 & 5 \end{bmatrix}$$ |
| 3 | $$(3, 6)$$ | $$(2, 0)$$ | $$\begin{bmatrix} 6 & 0 \\ 12 & 0 \end{bmatrix}$$ |
| sum | | | $$\begin{bmatrix} 7 & 5 \\ 16 & 17 \end{bmatrix}$$ |

*Table 3. Way 2 on the example: three grids, and their sum is Table 2's $$y$$.*

## 4. Why the two ways are equal

The two programs have the same line inside, `y[t][j] += x[t][i] * W[j][i]`, with the loops in a
different order. So each output gets the **same products**: $$y_{1,1}$$ gets $$1 \cdot 1$$ at
$$i = 1$$, $$2 \cdot 0$$ at $$i = 2$$ and $$3 \cdot 2$$ at $$i = 3$$. Way 1 adds them in one go; way 2
adds one in each grid. They are added in the same order too, $$i = 1, 2, 3$$, starting from 0.

| | way 1: inner products | way 2: outer products |
|---|---|---|
| outermost loop | each output $$(t, j)$$ | each $$i$$ |
| innermost loop | each $$i$$ | each output $$(t, j)$$ |
| one round of the outermost loop | finishes one output | adds one product to every output |
| what it keeps while it works | one sum | every sum |
| what it reads in one round | a whole row of $$x$$ and of $$W$$: $$n_\text{in}$$ values each | one column of $$x$$ and one row of $$W^\top$$ |
| the result | $$y$$ | the same $$y$$ |

*Table 4. The two ways. Way 1 keeps one sum still and moves two long rows past it. Way 2 keeps all the
sums still and moves short columns past them.*

The difference is not in the result. It is in **what is read, and how often**, and that is what decides
the speed on a GPU.

## 5. Why the kernel uses way 2: count the reads

A multiply-add is one instruction. Reading a value is the slow part: on a T4, 296 cycles from GPU
memory, 19 from shared memory, none from a register [5]. So the number to look at is
**multiply-adds per value read**: the higher, the less time is spent waiting.

### 5.1 One thread

Say a thread makes a square of $$a \times a$$ outputs.

- **Way 1.** For each output it reads a row of $$x$$ and a row of $$W$$, $$2\, n_\text{in}$$ values,
  for $$n_\text{in}$$ multiply-adds: **0.5 per value read**, for any $$a$$. To read a row of $$x$$ once
  and use it for all its $$a$$ outputs, it would have to keep the row, 768 values, and a thread has at
  most 255 registers [4].
- **Way 2.** At each $$i$$ it reads $$a$$ values of $$x$$ (its rows) and $$a$$ of $$W$$ (its
  columns), $$2a$$ values, and does every pair, $$a^2$$ multiply-adds: **$$a / 2$$ per value read**.
  Each value read is used $$a$$ times.

| square per thread | sums it keeps (registers) | values read per $$i$$ | multiply-adds per $$i$$ | per value read |
|---|---|---|---|---|
| 1 × 1 | 1 | 2 | 1 | 0.5 |
| 2 × 2 | 4 | 4 | 4 | 1 |
| 4 × 4 | 16 | 8 | 16 | 2 |
| **8 × 8** | **64** | **16** | **64** | **4** |
| 16 × 16 | 256 | 32 | 256 | 8 |

*Table 5. Way 2 in one thread. The bigger the square, the more each value is used, but the sums live
in registers: at 16 × 16, the 256 sums alone pass the 255 registers a thread can have, and the rest
spills to slow memory [4]. 8 × 8 is the biggest square that fits, with room for the 16 values and the
indexes.*

### 5.2 One block

The same trick works one level up. The block's 256 threads stand 16 × 16, each with an 8 × 8 square,
so together they make a **128 × 128** square of $$y$$. The block brings $$x$$ and $$W$$ in from GPU
memory into shared memory, and at each $$i$$ it needs 128 values of $$x$$ (its rows) and 128 of
$$W$$ (its columns): 256 values, for 128 × 128 = 16,384 multiply-adds, **64 per value read**.

Way 2 also fits in shared memory, and way 1 does not. Way 2 needs only a few $$i$$ at a time: the
kernel takes 8, so the block holds 8 × 128 values of $$x$$ and 8 × 128 of $$W$$, **8 KB**. Way 1
needs whole rows: 128 rows of $$x$$ and 128 of $$W$$, 768 values each, **768 KB**, sixteen times
the 48 KB a block can have [5].

### 5.3 Both levels, and the whole layer

| level | reads from → into | square of $$y$$ | values read per $$i$$ | multiply-adds per $$i$$ | per value read |
|---|---|---|---|---|---|
| block | GPU memory → shared memory | 128 × 128 | 256 | 16,384 | **64** |
| thread | shared memory → registers | 8 × 8 | 16 | 64 | **4** |
| way 1, at either level | — | 1 × 1 | 2 | 1 | 0.5 |

*Table 6. The kernel uses way 2 twice: the block's square cuts the reads from GPU memory, the thread's
square cuts the reads from shared memory.*

For the qkv layer, $$4096 \times 2304 \times 768 = 7.25$$ billion multiply-adds. The values read from
GPU memory, before any cache helps:

| | values read | bytes |
|---|---|---|
| way 1: 2 per multiply-add | 14.5 billion | 58 GB |
| way 2 in 128 × 128 blocks: 576 blocks × 768 $$i$$ × 256 values | 113 million | 453 MB |

*Table 7. 128 times fewer reads, for the same multiply-adds and the same $$y$$.*

## 6. Where it is in the kernel

`linear_forward_kernel` in `linear.cuh` [1] is way 2. Its loop over $$i$$, 8 values per step, 96 steps
for 768; inside each step, for each of the 8 values of $$i$$:

```c
// x[t][i] of its 8 rows t, and W[j][i] of its 8 columns j: column i of x, row i of Wᵀ
float input_values[8]  = {...};
float weight_values[8] = {...};
for (int t = 0; t < 8; t++)
    for (int j = 0; j < 8; j++) sums[t][j] += input_values[t] * weight_values[j];   // one 8 × 8 grid
```

| in way 2 | in the kernel |
|---|---|
| $$y$$, all the sums | `sums[8][8]`: the thread's 64 outputs, in registers |
| column $$i$$ of $$x$$ | `input_values[8]`: $$x_{t,i}$$ for its 8 rows $$t$$ |
| row $$i$$ of $$W^\top$$ | `weight_values[8]`: $$W_{j,i}$$ for its 8 columns $$j$$ |
| one grid | the two loops above: 64 multiply-adds |
| the loop over $$i$$ | 96 steps × 8 values of $$i$$ |
| a few $$i$$ at a time in shared memory | `input_piece[8][128]` and `weight_piece[8][128]` |

*Table 8. Way 2's names in the kernel. `LINEAR_INNER_PER_STEP` = 8 is 8 values of the inner
dimension per step, Section 1.2's "inner", not an inner product.*

## 7. What way 2 asks of the layout

At one $$i$$, a thread needs $$x_{t,i}$$ for its 8 rows $$t$$: a **column** of $$x$$. In $$x$$, which
keeps $$i$$ last, those 8 values are 768 floats apart. So as the block copies its 8 values of $$i$$
into shared memory, it **turns them round**: `input_piece[i][t]`, with $$t$$ last. Now a thread's 8
values at one $$i$$ sit side by side, and it reads them with two `float4` (4 floats at once) instead of
eight single reads. `weight_piece[i][j]` is turned round the same way. Below, `t16i0` is $$x_{t,i}$$
at row 16 and $$i = 0$$, for the thread whose rows are 16 to 23:

```
input_piece[128][8], as x is:     t16i0 t16i1 ... t16i7 t17i0 ...     8 floats apart: 8 reads
input_piece[8][128], turned round: t16i0 t17i0 t18i0 ... t23i0       side by side:  2 float4 reads
```

| layout of each piece | a thread's 8 values at one $$i$$ | reads per step, both pieces |
|---|---|---|
| `[128][8]`, as in memory | 8 floats apart | 8 $$i$$ × (8 + 8) = 128 |
| `[8][128]`, turned round | side by side | 8 $$i$$ × (2 + 2) = 32 |

*Table 9. The price is in the copy: each thread writes its 4 values one at a time instead of one
`float4`. The copy is once per step; the reads are 8 times per step.*

## 8. What to remember

1. **Two orders, the same products.** Way 1 makes one output at a time, with $$i$$ innermost: inner
   products. Way 2 goes one $$i$$ at a time, with $$i$$ outermost: a sum of outer products. Each output
   gets the same products, in the same order.
2. **"Inner dimension" is a place, not a method.** It is the index that sits between the two shapes and
   is summed away. The outer product way is the one that walks it.
3. **Way 2 keeps the outputs still and moves the inputs past them.** In an $$a \times a$$ square, each
   value read is used $$a$$ times. Way 1 uses each value once.
4. **The kernel uses it twice.** 128 × 128 per block: 64 multiply-adds per value read from GPU memory.
   8 × 8 per thread: 4 per value read from shared memory. The sizes are set by what fits: 48 KB of
   shared memory and 255 registers.
5. **It asks for the values of one $$i$$ to sit together.** That is why the pieces in shared memory are
   `[8][128]`, turned round from how $$x$$ and $$W$$ sit in memory.

## References

1. D. B. Song. *gpt2-small*, branch `c-cuda-trainer`: GPT-2 small pre-training in C and CUDA.
   `llmc/kernels/linear.cuh`. [github.com/ducbachsong/gpt2-small](https://github.com/ducbachsong/gpt2-small/tree/c-cuda-trainer)
2. D. B. Song. *Linear backward, broken down and rebuilt: three chain-rule steps and three matrix
   products.* 2026. [/posts/linear-backward/](/posts/linear-backward/)
3. S. Boehm. *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog.* 2022.
   Kernel 5: 2D blocktiling, 128 × 128 per block, 8 inner values at a time, 8 × 8 per thread.
   [siboehm.com/articles/22/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM)
4. D. B. Song. *How a CUDA kernel uses threads, warps, blocks and registers.* 2026.
   [/posts/gpu-threads-blocks-registers/](/posts/gpu-threads-blocks-registers/)
5. D. B. Song. *Where a CUDA kernel's numbers live: registers, local, shared, L1, L2 and GPU memory.*
   2026. [/posts/gpu-memory/](/posts/gpu-memory/)
