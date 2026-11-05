---
title: "Master weights: why a bfloat16 model keeps a float copy of itself"
date: 2026-11-05
permalink: /posts/master-weights/
excerpt: "My GPT-2 trainer runs its model in bfloat16 but keeps a second copy of every weight in float, the master weights, that only the optimizer touches. This note works out why from two facts: bfloat16 throws away any change smaller than half the gap to the next value, and an AdamW step moves a weight by at most about the learning rate. Put together, GPT-2's 19,200 LayerNorm weights would never move at all, and the other weights lose their updates in the first 71 steps and the last 3,800. With a float master the error the model sees stays below half a gap forever, instead of growing every step; the kernel does it in the same one pass, with the same 28 bytes per weight and the same memory as training in float."
tags:
  - mixed-precision
  - bfloat16
  - adamw
  - notes
---

**What this is.** My GPT-2 trainer [1] runs the model in **bfloat16**: the weights, activations and
gradients are 2 bytes each, which halves the memory and lets the tensor cores run the matmuls. But the
optimizer does not update those bfloat16 weights directly. It keeps a second copy of every weight in
**float**, called the **master weights**, does the AdamW step on that copy, and then writes it, rounded,
back into the bfloat16 weights the model uses.

That looks like a waste: if the model only ever sees the rounded value, why keep the exact one? This note
works out the answer from two facts. bfloat16 throws away any change smaller than half the gap to the
next value (Section 2), and an AdamW step moves a weight by at most about the learning rate (Section 3).
Together they say which of GPT-2's weights would stop learning without the master (Section 4). Then it
shows why the master fixes it: the error the model sees stays small instead of growing (Section 5), and
what it costs in the kernel (Section 6).

The idea is from *Mixed Precision Training* [3]; my earlier note on half and bfloat16 [2] has the bits
and the rounding this one builds on.

* TOC
{:toc}

## 1. Two copies of every weight

| | code name | type | who reads it | who writes it |
|---|---|---|---|---|
| model weights $$\hat{w}$$ | `params_ptr` | bfloat16 | forward and backward (the matmuls) | the optimizer, once a step |
| master weights $$w$$ | `master_params_ptr` | float | only the optimizer | only the optimizer |

*Table 1. The two copies. The hat on $$\hat{w}$$ marks the rounded one.*

Each step goes:

```
1. forward + backward with ŵ (bfloat16)              → the gradients g
2. AdamW on the master:   w ← w − Δ                  (in float)
3. round the master into the model:   ŵ ← round(w)   (to bfloat16)
```

The model never sees $$w$$ itself, only $$\hat{w} = \text{round}(w)$$. The question is why step 2
cannot just be $$\hat{w} \leftarrow \text{round}(\hat{w} - \Delta)$$, with no float copy at all.

## 2. When bfloat16 loses an update

### 2.1 The gap between two neighbours

A bfloat16 has 8 significant bits: a hidden 1 and 7 fraction bits [2]. Between two powers of two,
$$2^e$$ and $$2^{e+1}$$, it can hold $$2^7 = 128$$ values, evenly spaced. The band is $$2^e$$ wide, so the
gap between two neighbours is

$$
\text{gap} = \frac{2^e}{2^7} = 2^{e-7}
$$

| band | $$e$$ | gap $$2^{e-7}$$ | half the gap |
|---|---|---|---|
| 0.5 .. 1 | −1 | $$2^{-8} = 0.0039$$ | 0.0020 |
| 1 .. 2 | 0 | $$2^{-7} = 0.0078$$ | 0.0039 |
| 0.015625 .. 0.03125 | −6 | $$2^{-13} = 0.000122$$ | 0.000061 |

*Table 2. Three bands GPT-2's weights live in. The gap grows with the value: it is always between
$$2^{-8}$$ and $$2^{-7}$$ of it.*

### 2.2 The rule

Rounding to the nearest value moves $$x$$ to whichever neighbour is closer. So $$\hat{w} - \Delta$$ rounds
back to $$\hat{w}$$ itself whenever $$\hat{w}$$ is still the closest one, that is whenever the move is less
than half the gap:

$$
\text{round}(\hat{w} - \Delta) = \hat{w} \quad \text{when} \quad |\Delta| < \tfrac{1}{2}\, \text{gap}
$$

The update is computed, and then thrown away by the rounding. Example: a weight of 1.0 moving down by
0.0006. Below 1.0 the band is 0.5 .. 1, the gap is 0.0039 and half of it 0.0020. 0.9994 is 0.0006 from
1.0 and 0.0033 from the next value down, 0.99609375, so it rounds back to 1.0.

## 3. How big an AdamW update is

AdamW's step, without weight decay, is

$$
\Delta = \text{lr} \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon},
\qquad \hat{m} = \frac{m}{1 - \beta_1^t}, \qquad \hat{v} = \frac{v}{1 - \beta_2^t}
$$

with $$m \leftarrow \beta_1 m + (1 - \beta_1) g$$ the running average of the gradient and
$$v \leftarrow \beta_2 v + (1 - \beta_2) g^2$$ the running average of its square, $$\beta_1 = 0.9$$,
$$\beta_2 = 0.95$$.

### 3.1 The first step: exactly the learning rate

At $$t = 1$$, $$m$$ and $$v$$ start at 0, so one update gives

$$
m_1 = (1 - \beta_1)\, g, \qquad v_1 = (1 - \beta_2)\, g^2
$$

The bias corrections divide by the same factors, $$1 - \beta_1^1 = 1 - \beta_1$$ and
$$1 - \beta_2^1 = 1 - \beta_2$$, so they cancel:

$$
\hat{m}_1 = \frac{(1 - \beta_1)\, g}{1 - \beta_1} = g, \qquad
\hat{v}_1 = \frac{(1 - \beta_2)\, g^2}{1 - \beta_2} = g^2
$$

and, leaving out $$\epsilon$$ (1e-8, tiny next to $$|g|$$),

$$
\Delta_1 = \text{lr} \cdot \frac{g}{\sqrt{g^2}} = \text{lr} \cdot \frac{g}{|g|} = \pm\, \text{lr}
$$

The size of the gradient has cancelled out: whether $$g$$ is 0.5 or 0.0001, the first step moves the
weight by exactly the learning rate, only the sign comes from $$g$$. With $$g = 0.5$$ and lr = 0.0006:
$$m_1 = 0.05$$, $$v_1 = 0.0125$$, step size $$0.0006 / 0.1 = 0.006$$, $$\sqrt{v_1} \cdot (1/\sqrt{0.05}) =
0.1118 \cdot 4.472 = 0.5$$, and $$\Delta_1 = 0.006 \cdot 0.05 / 0.5 = 0.0006$$.

### 3.2 Later steps: at most about the learning rate

If the gradients keep coming with an average $$\mu$$ and a spread $$\sigma$$ around it, then after many
steps $$\hat{m}$$ settles near $$\mu$$ and $$\hat{v}$$ near the average of $$g^2$$, which is
$$\mu^2 + \sigma^2$$. So

$$
|\Delta| \approx \text{lr} \cdot \frac{|\mu|}{\sqrt{\mu^2 + \sigma^2}} \le \text{lr}
$$

the learning rate when the gradient always points the same way ($$\sigma = 0$$), and less when it is
noisy. Adam's paper [4] gives the same bound for each single step, not only on average, whenever
$$1 - \beta_1 \le \sqrt{1 - \beta_2}$$; here $$0.1 \le 0.224$$. So **an AdamW step moves a weight by at most
the learning rate**: 0.0006 at GPT-2's peak, and less in the warmup and as the cosine brings it down to 0.

## 4. Which of GPT-2's weights lose their updates

Put Sections 2 and 3 together: a bfloat16 weight with no master loses its update whenever

$$
\text{lr} \;<\; \tfrac{1}{2}\, \text{gap at } \hat{w}
$$

and since $$|\Delta| \le \text{lr}$$, the update is then lost no matter what the gradient is.

**The LayerNorm weights: never move.** They start at 1.0. Half the gap is 0.0020 going down and 0.0039
going up, both more than the peak learning rate of 0.0006. Every update of every step rounds back to 1.0.
GPT-2 small has 25 LayerNorms of 768 weights each: **19,200 weights that would never learn**. (They get
no weight decay, so nothing else moves them either.)

**A weight of 0.02**, the size GPT-2 starts its matrices at: half the gap is 0.000061. The peak learning
rate is 10 times that, so the middle of training is fine. But the learning rate is below 0.000061

- in the **first 71 steps** of the warmup, which climbs from $$6 \times 10^{-4} / 700 = 8.6 \times 10^{-7}$$,
  and reaches 0.000061 at step 71;
- in the **last 3,800 steps** of the cosine (of 19,073), where
  $$0.0006 \cdot \tfrac{1}{2}(1 + \cos \pi p) < 0.000061$$ for $$p > 0.793$$ of the way down.

And that is with $$|\Delta|$$ at its largest; with noisy gradients (Section 3.2) it is lost earlier.

**Kept, but the wrong size.** An update between half a gap and one gap is not lost, but rounded to a whole
gap. At 0.02 with $$\Delta = 0.0001$$, each step moves the weight by the gap, 0.000122, 22% too far:

| 100 steps of $$\Delta$$ from 0.02 | the true value | without a master | with a master |
|---|---|---|---|
| $$\Delta = 0.0001$$ | 0.010000 | 0.007812 (moved 22% too far) | 0.010010 |
| $$\Delta = 0.00003$$ | 0.017000 | 0.020020 (never moved) | 0.016968 |

*Table 3. The model's weight after 100 equal steps, worked out with bfloat16's round to nearest, ties to
even. "With a master" is round(w) of the float master.*

## 5. The master: why the error stops growing

Take one weight that the gradient keeps pushing the same way, $$\Delta$$ every step, and compare the two
ways of storing it after $$k$$ steps. The true value is $$w_0 - k\Delta$$.

**Without a master**, each step starts from the rounded value: $$\hat{w}_k = \text{round}(\hat{w}_{k-1} -
\Delta)$$. If $$|\Delta|$$ is under half the gap, every step gives back $$\hat{w}_{k-1}$$, so
$$\hat{w}_k = \hat{w}_0$$, and the error is

$$
|\hat{w}_k - (w_0 - k\Delta)| = k\, |\Delta|
$$

It grows by $$|\Delta|$$ every step, without end: what is lost in one step is never given back, because
the next step starts from the same rounded value.

**With a master**, the float copy keeps every step: $$w_k = w_0 - k\Delta$$ (float's own gap at 1.0 is
$$2^{-24} = 6 \times 10^{-8}$$, so 0.0006 is kept). The model gets that value rounded once:
$$\hat{w}_k = \text{round}(w_k)$$. Rounding moves a value by at most half a gap, so

$$
|\hat{w}_k - w_k| \le \tfrac{1}{2}\, \text{gap at } w_k \le 2^{-8}\, |w_k|
$$

the same bound at step 1 and at step 1,000. The rounding still happens, but each step rounds the true
value, not the last step's rounded one, so the errors cannot add up.

| step $$k$$ | master $$w_k$$ (float) | model, with master: round($$w_k$$) | model, without master |
|---|---|---|---|
| 0 | 1.000000 | 1.0 | 1.0 |
| 1 | 0.999400 | 1.0 | 1.0 |
| 3 | 0.998200 | 1.0 | 1.0 |
| 4 | 0.997600 | **0.99609375** | 1.0 |
| 10 | 0.994000 | 0.99218750 | 1.0 |
| 100 | 0.940002 | 0.94140625 | 1.0 |
| 1000 | 0.400015 | 0.40039062 | 1.0 |

*Table 4. One weight from 1.0, moved down 0.0006 each step. With a master the model's value is a little
behind or ahead, never by more than half a gap: it moves at step 4, when the master passes 0.998047, the
middle between 1.0 and 0.99609375. Without one it is still 1.0 after 1,000 steps, 0.6 away from the truth.*

So the model does see a rounded weight, as the question in Section 1 said. But the rounded weight of the
**right** value is a small, fixed blur; the rounded weight of a value that stopped updating is wrong by
more every step.

## 6. In the kernel: one pass, same bytes

The AdamW kernel [1] (`llmc/kernels/adamw.cuh`) does steps 2 and 3 of Section 1 in one pass. Each thread
takes 4 weights:

```cpp
float4 p = master_params_ptr != nullptr ? reinterpret_cast<float4*>(master_params_ptr)[index]   // the master
                                        : load4(params_ptr + VALUES_PER_FLOAT4 * index);         // float: the params
...                                                       // clip, decay, m, v, the step: all in float
if (master_params_ptr != nullptr) reinterpret_cast<float4*>(master_params_ptr)[index] = p;   // master, exact
store4(params_ptr + VALUES_PER_FLOAT4 * index, p);       // the model's copy, rounded to bfloat16
```

In float training there is no master: the caller passes `nullptr` and the kernel reads and writes the
params themselves. In bfloat16 the master is read in place of the params, and the params are only written.
Per weight, the memory each step moves:

| | read | write | total |
|---|---|---|---|
| float | p 4 + g 4 + m 4 + v 4 | p 4 + m 4 + v 4 | 28 bytes |
| bfloat16 + master | master 4 + g 2 + m 4 + v 4 | master 4 + m 4 + v 4 + p 2 | 28 bytes |

*Table 5. Bytes per weight in one AdamW step. The kernel is limited by memory, so the same bytes is the
same time.*

And the memory kept per weight is the same too:

| | weight | gradient | master | $$m$$ | $$v$$ | total |
|---|---|---|---|---|---|---|
| float | 4 | 4 | — | 4 | 4 | 16 bytes |
| bfloat16 + master | 2 | 2 | 4 | 4 | 4 | 16 bytes |

*Table 6. Bytes kept per weight. For GPT-2 small's 124,439,808 weights the master is 498 MB on one GPU;
on 8 GPUs each keeps the master only for its own slice, 62 MB.*

So bfloat16 training saves nothing on the weights and the optimizer: it saves on the activations, which
are far larger, and it runs the matmuls on the tensor cores. The master is filled once before the first
step by copying the bfloat16 weights into it, and that copy is exact: every bfloat16 is a float with its
last 16 bits zero.

There is a way with no master at all: round the new value up or down at random, with chances set so the
average is exact (stochastic rounding, as llm.c does [5]; the math is in [2], Section 4.2). That saves the
4 bytes, but each weight's value is then a random walk around the truth instead of the nearest value to it.

## 7. What to remember

1. **Two copies.** The model computes with bfloat16 weights; the optimizer keeps a float master copy,
   updates it, and rounds it into the model's copy once a step.
2. **A lost update.** bfloat16 rounds back any change under half the gap, about $$2^{-9}$$ to $$2^{-8}$$ of
   the weight.
3. **AdamW's step is at most the learning rate**, whatever the gradient's size: exactly lr at step 1.
4. **So without a master**, the 19,200 LayerNorm weights at 1.0 never move, and weights of 0.02 lose every
   update in the first 71 and the last 3,800 steps. The error grows by $$|\Delta|$$ each step.
5. **With a master**, the model's weight is the true value rounded once: off by at most half a gap, at
   every step. Same 28 bytes per weight in the kernel, same 16 bytes of memory per weight as float.

## References

1. D. B. Song. *gpt2-small*, branch `c-cuda-trainer`: GPT-2 small pre-training in C and CUDA;
   `llmc/kernels/adamw.cuh` (the kernel) and `llmc/optimizer.cuh` (the master, and its slices on 8 GPUs).
   [github.com/ducbachsong/gpt2-small](https://github.com/ducbachsong/gpt2-small/tree/c-cuda-trainer)
2. D. B. Song. *Half and bfloat16: two ways to fit a float into 16 bits.* 2026. The bits, the gap, and
   rounding to the nearest. [/posts/half-and-bfloat16/](/posts/half-and-bfloat16/)
3. P. Micikevicius et al. *Mixed Precision Training.* 2017. Float master weights.
   [arxiv.org/abs/1710.03740](https://arxiv.org/abs/1710.03740)
4. D. P. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization.* 2014. Section 2.1: the bound on the
   step size. [arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
5. A. Karpathy. *llm.c*: the AdamW kernel's stochastic rounding of bfloat16 parameters.
   [github.com/karpathy/llm.c](https://github.com/karpathy/llm.c)
