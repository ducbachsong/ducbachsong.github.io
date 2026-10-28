---
title: "Residual add backward, broken down and rebuilt: one plus sign and a gradient that passes through"
date: 2026-10-28
modified: 2026-10-28
version: 2
permalink: /posts/residual-backward/
excerpt: "Every block of GPT-2 adds what its attention or MLP gives back onto the row it was given: y = r + b. The backward pass of that add is the shortest of all, the gradient going on unchanged to both, but the residual is also the branch's input, so its gradient has a second part, and that part is where the add earns its place. This note takes one add apart into four small steps, each derived with the chain rule written out in full, and works one row of 4 numbers through all of them. Then it puts them back together, for one add and for 24 of them stacked, and shows why the gradient reaches the first block of a deep network at all: the product of 24 slopes becomes a product of 24 times (1 + slope), and its first term, 1, is a path that skips every branch."
tags:
  - residual-connection
  - backpropagation
  - gpt-2
  - notes
---

**What this is.** Every block of GPT-2 small does two things, attention and an MLP. After each, it adds
what it computed back onto the row it was given. That add, the **residual add**, is one line forward
and one line backward:

$$
y = r + b \qquad\qquad dr = dy \qquad db = dy
$$

with $$r$$ the row coming in, $$b$$ what the layers computed from it, and $$y$$ the row going on to the
next layer (Section 1.1 shows each with a number). The backward line is correct, and it is what the
kernel in my GPT-2 trainer [1] runs, but it is only half the story: $$b$$ was computed from $$r$$, so
$$dr = dy$$ is only the first part of $$r$$'s gradient. This note **breaks it down** into four small steps, B1 to B4, each derived
with the chain rule written out in full. Then it **rebuilds** them, in two moves, C1 and C2: one add,
then 24 adds stacked, as in GPT-2 small. One row of 4 numbers goes through every step, so each formula
has a number next to it.

**Sources.** The derivation is worked out by hand, with the same three rules as the notes on
[LayerNorm's backward](/posts/layer-norm-backward/) [2], [the linear layer's](/posts/linear-backward/) [3],
[GELU's](/posts/gelu-backward/) [4] and [the classifier's](/posts/classifier-backward/) [5]. The
residual add is from He et al. [6]; the expansion of C2 is the argument of their second paper [7]; the
small branches at the start are GPT-2's [8]; the kernel is llm.c's `residual_forward` [9].

* TOC
{:toc}

## 1. The forward pass

### 1.1 What the residual add does: two paths, then an add

A block of GPT-2 takes a row of 768 numbers, one token, and gives a row of 768 back. Look at one half of
a block, the attention half. The row coming in goes along **two paths at once**:

```
                 ┌──► LayerNorm ──► attention ──► linear ──┐    top path: works out something from the row
row coming in ───┤                                         + ──► row going out, to the next layer
                 └─────────────────────────────────────────┘    bottom path: a plain wire, carries the row as it is
```

At the end, the two paths are added, value by value. For one value of the row:

| | value |
|---|---:|
| the row coming in | 1 |
| the top path works out, from it | 0.5 |
| the bottom path carries, unchanged | 1 |
| the row going out: the two added | 1 + 0.5 = 1.5 |

*Table 1. One value through one residual add.*

Each of these has a name in the rest of the note:

| name | what it is | in Table 1 |
|---|---|---:|
| $$r$$, the **residual** | the row coming in | 1 |
| $$f$$, the **branch** | the top path: LayerNorm, then attention (or the MLP), then a linear layer | — |
| $$b = f(r)$$, the **branch output** | what the top path works out from $$r$$ | 0.5 |
| the **skip** | the bottom path: it does nothing, it only carries $$r$$ around the branch | — |
| $$y = r + b$$, the **output** | the row going out, to the next layer | 1.5 |

*Table 2. The names. Only the branch has weights and learns; the skip and the add have nothing to
learn.*

**Why two paths, and not only the top one.** With only the top path, the branch would have to work out
the whole new row, 1.5, by itself: $$y = f(r)$$. With the skip, the row 1 is already there, and the
branch only has to work out the **change**, +0.5. A branch that gives 0 changes nothing, $$y = r$$, and
the row goes on as it came. A new block starts close to that, and learns only what to add.

GPT-2 small has 12 blocks and two adds in each, 24 in all, one after the attention and one after the
MLP:

```
r ──┬── LayerNorm ── attention ── linear ──► b ──┐
    └──────────────────────────────────────────── + ──► y ──┬── LayerNorm ── MLP ── linear ──► b' ──┐
                                                            └────────────────────────────────────── + ──► next block
```

The row that goes from add to add, through all 12 blocks, is the **residual stream**.

### 1.2 One step for one value

The add works on one value at a time: values never mix. For one value, there is one step:

$$
\text{F1:} \qquad y = r + b
$$

with $$b = f(r)$$ given by the branch. F1 is all the add does; the branch is another layer, with its
own note (LayerNorm [2], linear [3], GELU [4]). Here it matters only that the branch **uses $$r$$**.

### 1.3 The names

| symbol | meaning | shape in GPT-2 small |
|---|---|---|
| $$r$$ | residual: the block's input row, goes around the branch | `{4, 1024, 768}` |
| $$f$$ | the branch: LayerNorm, then attention or the MLP, then a linear layer | — |
| $$b$$ | branch output: $$b = f(r)$$ | `{4, 1024, 768}` |
| $$y$$ | output: $$r + b$$, the next residual | `{4, 1024, 768}` |
| $$L$$ | the loss: one number, how wrong the model is | 1 |
| $$\partial$$ | partial derivative: change one variable, hold all others fixed | — |
| $$da$$ | short for $$\partial L / \partial a$$, for any variable $$a$$; same shape as $$a$$ | — |

*Table 3. $$dy$$ is given: it is what the layer after the add sends back. The job is to find $$db$$,
which goes on into the branch's backward, and $$dr$$, which goes on to the block before. The add has
no weight and no bias.*

### 1.4 The running example

One row of 4 values. The real branch mixes the values of the row (LayerNorm, attention), which makes
its numbers long; the example uses a **stand-in** branch that does not mix them, so every number can be
worked out by eye. Each value is multiplied by its own number:

$$
b_i = w_i \cdot r_i \qquad\qquad w = [0.5,\ -1,\ 0,\ 0.5]
$$

And a stand-in loss whose gradient is exactly a chosen $$dy$$: $$L = \sum_i dy_i\, y_i$$.

| | value 0 | value 1 | value 2 | value 3 |
|---|---:|---:|---:|---:|
| residual $$r$$ | 1 | 2 | 3 | 4 |
| stand-in branch: $$b = w \cdot r$$ | $$0.5 \times 1 = 0.5$$ | $$-1 \times 2 = -2$$ | $$0 \times 3 = 0$$ | $$0.5 \times 4 = 2$$ |
| F1: $$y = r + b$$ | **1.5** | **0** | **3** | **6** |
| $$dy$$, from the layer after | 1 | −2 | 0.5 | 3 |

*Table 4. The example. Value 1's branch takes away exactly what the skip brings: $$y_1 = 2 - 2 = 0$$.
The stand-in loss is $$L = 1 \times 1.5 + (-2) \times 0 + 0.5 \times 3 + 3 \times 6 = 21$$.*

Why the stand-in loss has gradient $$dy$$: $$L = dy_0\, y_0 + dy_1\, y_1 + dy_2\, y_2 + dy_3\, y_3$$, and
with respect to $$y_i$$ every term is a constant but $$dy_i\, y_i$$, which is a constant times $$y_i$$:

$$
\frac{\partial L}{\partial y_i} = 0 + \cdots + dy_i + \cdots + 0 = dy_i
$$

### 1.5 Where each value is used

![One residual add, every use of every value](/images/residual-backward/fig1-graph.svg)
*Figure 1. Each arrow is one use of a value. The backward pass walks the same arrows the other way.
$$r$$ has two arrows going out, the skip and the branch, so its gradient has two parts.*

Figure 1 is the whole plan:

| step | arrow walked back | gives |
|---|---|---|
| B1 | F1: $$b \to y$$ | $$db$$, sent into the branch's backward |
| B2 | F1: $$r \to y$$, the skip | $$dr^{(1)}$$, the first part of $$dr$$ |
| B3 | the branch: $$r \to b$$ | $$dr^{(2)}$$, the second part of $$dr$$ |
| B4 | — | $$dr = dr^{(1)} + dr^{(2)}$$, sent to the block before |

*Table 5. B1 and B2 are the add's own. B3 is the branch's backward, run by the branch's kernels; it is
here because without it $$dr$$ is not finished.*

## 2. Three rules and one derivative

Every step below uses the same three rules as the earlier notes [2, 3, 4, 5], and nothing else.

| rule | when | what it says |
|---|---|---|
| **chain rule** | $$a$$ affects $$L$$ only through $$b$$ | $$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial b} \cdot \frac{\partial b}{\partial a}$$, written short as $$da = db \cdot \partial b / \partial a$$ |
| **shared value: sum** | $$a$$ is used in many places | add one chain-rule term per place: $$\frac{\partial L}{\partial a} = \sum_k \frac{\partial L}{\partial b_k} \cdot \frac{\partial b_k}{\partial a}$$ |
| **used twice: add the parts** | $$a$$ is used by two different forward steps | each step gives one part; $$da$$ = part 1 + part 2 |

*Table 6. The three rules. The add needs the first and the third: $$r$$ is used twice, by F1 and by
the branch.*

**The chain rule, in full.** Every gradient in this note is a $$\partial L / \partial(\cdot)$$: how much
the loss $$L$$ moves per unit that one value moves. $$da$$ is only a short name for
$$\partial L / \partial a$$. When $$a$$ affects $$L$$ only through $$b$$, there is one road,
$$a \to b \to \cdots \to L$$: a nudge of $$a$$ moves $$b$$ by $$\partial b / \partial a$$ per unit, and
each unit of $$b$$ moves $$L$$ by $$\partial L / \partial b$$. So the two rates multiply:

$$
\underbrace{\frac{\partial L}{\partial a}}_{da} = \underbrace{\frac{\partial L}{\partial b}}_{db} \cdot \frac{\partial b}{\partial a}
$$

$$\partial L / \partial b$$ is already known, from the step before: the steps go backwards from the
loss. $$\partial b / \partial a$$ is the local derivative, one forward formula differentiated. If $$a$$
had two roads to $$L$$, there would be one such product per road, added: that is the used-twice rule.
Each step below writes the full form first, then the short one.

And one habit: **each step looks at one input and holds the others fixed.** F1, $$y = r + b$$, has two
inputs: B1 takes the derivative with respect to $$b$$ and holds $$r$$ fixed; B2 takes it with respect
to $$r$$ and holds $$b$$ fixed, even though $$b$$ was made from $$r$$. The road through $$b$$ is not
lost: B3 walks it.

One derivative is needed, of a sum, with two rules:

| rule | what it says | why |
|---|---|---|
| the derivative of $$x$$ | $$\frac{d}{dx}\, x = 1$$ | nudge $$x$$ by $$h$$: it moves by $$h$$, so 1 per unit |
| a sum | the derivative of a sum is the sum of the derivatives; a term without $$x$$ gives 0 | a term without $$x$$ does not move when $$x$$ moves |

*Table 7. The two rules behind $$\partial y / \partial r$$ and $$\partial y / \partial b$$.*

## 3. Breaking it down: B1 to B4

Each step first works on one value, then adds the row and column index for the whole batch. Each step
ends with the example, and with what its number means: nudge the variable by 0.01 and watch how much
the stand-in loss moves.

### 3.1 B1: the branch output, $$db$$

**One value.** From F1, $$y = r + b$$, with respect to $$b$$, holding the other input $$r$$ fixed:

$$
\begin{aligned}
\frac{\partial y}{\partial b}
  &= \frac{\partial r}{\partial b} + \frac{\partial b}{\partial b}
  && \text{derivative of a sum = sum of derivatives} \\
  &= 0 + 1 = 1
  && r \text{ is held fixed, so it is a constant; } \tfrac{d}{db}\, b = 1 \\
\frac{\partial L}{\partial b} &= \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial b}
  && \text{chain rule: } b \text{ affects } L \text{ only through } y \\
db &= dy \cdot 1 = dy
  && \text{the same, written short}
\end{aligned}
$$

$$b$$ is used once, by F1, so there is one road and nothing to add.

**The whole batch.** $$b_{t,i}$$ belongs to one token $$t$$ and one column $$i$$. It is used in one
place only, $$y_{t,i}$$: one road, so no sum.

$$
db_{t,i} = dy_{t,i}
$$

This is what goes into the branch's backward: the gradient arriving at its last linear layer.

**Example.** $$db = dy = [1,\ -2,\ 0.5,\ 3]$$.

**What it means.** Raise $$b_1$$ by 0.01, from −2 to −1.99, holding $$r$$ fixed:

| | $$b_1$$ | $$y_1 = r_1 + b_1$$ | $$L$$ |
|---|---:|---:|---:|
| before | −2 | 0 | 21 |
| after | −1.99 | 0.01 | 20.98 |
| change | 0.01 | 0.01 | −0.02 |

*Table 8. $$L$$ moved −0.02 when $$b_1$$ moved 0.01: −2 per unit, $$db_1$$. $$y_1$$ moved exactly as
much as $$b_1$$, so $$L$$ moved as it would for $$y_1$$ itself: $$dy_1 = -2$$.*

**Why 1.** An add passes any change of one input to the output whole: $$b$$ up by 0.01 is $$y$$ up by
0.01, whatever $$r$$ is. So the gradient comes back whole too.

### 3.2 B2: the residual, part 1, $$dr^{(1)}$$

**One value.** From F1 again, now with respect to $$r$$, holding the other input $$b$$ fixed:

$$
\begin{aligned}
\frac{\partial y}{\partial r}
  &= \frac{\partial r}{\partial r} + \frac{\partial b}{\partial r}\Big\rvert_{b \text{ fixed}}
  && \text{derivative of a sum = sum of derivatives} \\
  &= 1 + 0 = 1
  && b \text{ is held fixed: the road through it is B3's} \\
\frac{\partial L}{\partial r}\Big\rvert_{\text{through the skip}} &= \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial r}
  && \text{chain rule, along the arrow F1} \\
dr^{(1)} &= dy \cdot 1 = dy
  && \text{the same, written short}
\end{aligned}
$$

**The whole batch.** $$r_{t,i}$$ is used once in F1, in $$y_{t,i}$$: no sum.

$$
dr^{(1)}_{t,i} = dy_{t,i}
$$

**Why "(1)".** In Figure 1, $$r$$ has two arrows going out, so its gradient has two parts:

| part | forward step that uses $$r$$ | gradient part | worked out in |
|---|---|---|---|
| 1 | F1: $$y = r + b$$, the skip | $$dr^{(1)} = dy$$ | B2, here |
| 2 | the branch: $$b = f(r)$$ | $$dr^{(2)}$$ | B3, next |

*Table 9. $$dr = dr^{(1)} + dr^{(2)}$$. After B2 it is not finished, and nothing may go further back
with it until part 2 is added (B4).*

**Example.** $$dr^{(1)} = dy = [1,\ -2,\ 0.5,\ 3]$$: the same as $$db$$.

**What it means.** Raise $$r_2$$ by 0.01, from 3 to 3.01, and **hold $$b_2$$ fixed**, as if the branch
did not see the change: $$y_2$$ goes from 3 to 3.01, and $$L$$ from 21 to 21.005. That is
$$0.005 / 0.01 = 0.5$$ per unit, $$dr^{(1)}_2 = dy_2$$.

**The add copies the gradient, it does not split it.** $$db$$ and $$dr^{(1)}$$ are both all of $$dy$$,
not half of it each. A mean splits ($$\tfrac{1}{R}$$ each, as the classifier note found for its rows
[5]), because each input counts for a part of a mean. Each input of an add counts in full.

### 3.3 B3: the residual, part 2, through the branch, $$dr^{(2)}$$

**One value.** $$r$$ is the branch's input: $$b = f(r)$$. With respect to $$r$$, along that road:

$$
\begin{aligned}
\frac{\partial L}{\partial r}\Big\rvert_{\text{through the branch}} &= \frac{\partial L}{\partial b} \cdot \frac{\partial b}{\partial r}
  && \text{chain rule, along the branch} \\
dr^{(2)} &= db \cdot \frac{\partial b}{\partial r}
  && \text{the same, written short; } db = dy \text{ from B1}
\end{aligned}
$$

$$\partial b / \partial r$$ is the slope of the branch: how much its output moves per unit its input
moves. It is all the branch's backward does, worked out by its own kernels.

**The whole batch.** The real branch **mixes** the row: LayerNorm takes the mean of all 768 values,
attention mixes tokens. So $$r_{t,i}$$ is used by many outputs of the branch, $$b_{t',j}$$, and by the
shared-value rule:

$$
\frac{\partial L}{\partial r_{t,i}}\Big\rvert_{\text{through the branch}} = \sum_{t', j} \frac{\partial L}{\partial b_{t',j}} \cdot \frac{\partial b_{t',j}}{\partial r_{t,i}}
\qquad\text{written short:}\qquad dr^{(2)}_{t,i} = \sum_{t', j} db_{t',j}\, \frac{\partial b_{t',j}}{\partial r_{t,i}}
$$

In GPT-2 that sum is the branch's backward, ending in LayerNorm's: its `input_grad` is
$$dr^{(2)}$$ [2].

**With the stand-in branch.** $$b_j = w_j\, r_j$$ does not mix: $$b_j$$ holds $$r_i$$ only when
$$j = i$$, so the sum has one term:

$$
\begin{aligned}
\frac{\partial b_j}{\partial r_i} &= \begin{cases} w_i & j = i \\ 0 & j \ne i \end{cases}
  && w_i \text{ is a constant factor; } b_j \text{ for } j \ne i \text{ holds no } r_i \\
dr^{(2)}_i &= db_i \cdot w_i
  && \text{one term of the sum left}
\end{aligned}
$$

**Example.**

| | value 0 | value 1 | value 2 | value 3 |
|---|---:|---:|---:|---:|
| $$db$$ (B1) | 1 | −2 | 0.5 | 3 |
| $$w$$, the stand-in's slope | 0.5 | −1 | 0 | 0.5 |
| $$dr^{(2)} = db \cdot w$$ | 0.5 | 2 | 0 | 1.5 |

*Table 10. B3 in the example. Value 2's branch is flat ($$w = 0$$), so nothing comes back through it.*

**What it means.** Raise $$r_3$$ by 0.01 and let **only the branch** see it, holding the skip fixed:
$$b_3 = 0.5\, r_3$$ goes up by 0.005, $$y_3$$ by 0.005, and $$L$$ by $$3 \times 0.005 = 0.015$$. That is
1.5 per unit, $$dr^{(2)}_3$$.

### 3.4 B4: the residual, complete, $$dr$$

**One value.** Both parts are now known. $$r$$ reaches the loss along two roads (Figure 1), so by the
used-twice rule they are added. Written out as the two roads, each a chain of the chain rule:

$$
\begin{aligned}
dr &= \underbrace{\frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial r}}_{\text{road 1: } r \to y\text{, the skip}}
    + \underbrace{\frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial b} \cdot \frac{\partial b}{\partial r}}_{\text{road 2: } r \to b \to y\text{, the branch}} \\
   &= dy \cdot 1 \;+\; dy \cdot 1 \cdot \frac{\partial b}{\partial r}
   && \text{B2; and B1, B3} \\
   &= dr^{(1)} + dr^{(2)}
\end{aligned}
$$

**The whole batch.** Both parts are one per value, so the sum is too:
$$dr_{t,i} = dr^{(1)}_{t,i} + dr^{(2)}_{t,i}$$. This is the gradient sent to the block before.

**Example.**

| | value 0 | value 1 | value 2 | value 3 |
|---|---:|---:|---:|---:|
| $$dr^{(1)}$$ (B2): road 1, the skip | 1 | −2 | 0.5 | 3 |
| $$dr^{(2)}$$ (B3): road 2, the branch | 0.5 | 2 | 0 | 1.5 |
| $$dr = dr^{(1)} + dr^{(2)}$$ | **1.5** | **0** | **0.5** | **4.5** |

*Table 11. The two parts of $$dr$$.*

**What it means.** Raise each $$r_i$$ by 0.01, and let both roads see it: $$y_i = r_i + w_i\, r_i$$
moves by $$0.01\,(1 + w_i)$$, and $$L$$ by $$dy_i$$ times that:

| | value 0 | value 1 | value 2 | value 3 |
|---|---:|---:|---:|---:|
| $$y_i$$ moves by $$0.01\,(1 + w_i)$$ | 0.015 | 0 | 0.01 | 0.015 |
| $$L$$ moves by $$dy_i$$ times that | 0.015 | 0 | 0.005 | 0.045 |
| per unit: $$dr_i$$ | 1.5 | 0 | 0.5 | 4.5 |

*Table 12. Value 1: the branch's slope is −1, so whatever the skip brings the branch takes away,
$$y_1 = r_1 - r_1 = 0$$ always, and $$r_1$$ has no say in the loss. Value 2: the branch is flat, and
only the skip carries the gradient.*

| part | what happens when $$r$$ changes | gradient part |
|---|---|---|
| 1 | $$y$$ changes along the skip, one for one | $$dy$$: always all of it |
| 2 | the branch's output changes, so $$y$$ changes | $$db \cdot \partial b / \partial r$$: as much as the branch's slope lets through |

*Table 13. Part 1 needs nothing from the branch: whatever the branch's slope, $$dy$$ reaches $$r$$.*

### 3.5 All four steps

| step | from | gradient of | formula | example, value 3 |
|---|---|---|---|---:|
| B1 | F1 | branch output | $$db = dy$$ | 3 |
| B2 | F1 | residual, part 1 | $$dr^{(1)} = dy$$ | 3 |
| B3 | the branch | residual, part 2 | $$dr^{(2)} = db \cdot \partial b / \partial r$$ | 1.5 |
| B4 | — | residual, complete | $$dr = dr^{(1)} + dr^{(2)}$$ | 4.5 |

*Table 14. The backward pass, broken down. B1 and B2 need only $$dy$$; B3 needs B1 and the branch; B4
needs both parts.*

Side by side with the four earlier notes:

| | LayerNorm [2] | linear [3] | GELU [4] | classifier [5] | residual add |
|---|---|---|---|---|---|
| steps | 9 | 3 | 7 | 7 | 4, one of them the branch's |
| a gradient summed over rows | $$d\gamma$$, $$d\beta$$ | $$dW$$, $$db$$ | none: no weights | none: no weights | none: no weights |
| a gradient summed over a row | $$dr$$, $$dv$$, $$d\mu$$ | $$dx$$, over the outputs | none | $$dm$$, and it is 0 | none |
| a value used twice | $$x$$ and $$d$$ | none | $$x$$ | $$a_t$$ and $$z$$ | $$r$$ |

*Table 15. The add is the simplest of the five on its own, two copies. Its weight is in the value it
uses twice.*

## 4. Rebuilding: C1 and C2

### 4.1 C1: one add, one line

Put B1 to B4 together, with $$db = dy$$ written into B3:

$$
\begin{aligned}
dr &= dr^{(1)} + dr^{(2)}
   && \text{B4} \\
   &= dy + dy \cdot \frac{\partial b}{\partial r}
   && \text{B2; B3 with B1's } db = dy \\
   &= dy \cdot \Big(1 + \frac{\partial b}{\partial r}\Big)
   && \text{take } dy \text{ out}
\end{aligned}
$$

$$
\boxed{\;dr = dy \cdot \big(1 + f'(r)\big), \qquad db = dy\;}
$$

with $$f'(r) = \partial b / \partial r$$, the branch's slope (for the real branch, a matrix of slopes,
and the 1 a matrix of 1s down its diagonal). **The block's slope is the branch's slope plus 1.** The 1
is the skip.

In the example, value 0: $$1 \times (1 + 0.5) = 1.5$$; value 1: $$-2 \times (1 - 1) = 0$$; B4's
numbers.

### 4.2 C2: 24 adds stacked

The residual stream goes through 24 adds. Going back, each add multiplies the gradient by its
$$1 + f_k'$$, by C1, one after the other: that is the chain rule along the stream. Take the simplest
case, one value and every branch with a slope $$s_k$$ of its own. From the loss back to the first
block:

$$
dr_{\text{first}} = dy_{\text{last}} \cdot (1 + s_{24})(1 + s_{23}) \cdots (1 + s_1)
$$

**A small case first, 2 adds.** Multiply the brackets out:

$$
\begin{aligned}
(1 + s_2)(1 + s_1) &= 1 \cdot 1 + 1 \cdot s_1 + s_2 \cdot 1 + s_2\, s_1
  && \text{every term of one bracket times every term of the other} \\
                   &= \underbrace{1}_{\text{skip, skip}} + \underbrace{s_1}_{\text{branch 1 only}} + \underbrace{s_2}_{\text{branch 2 only}} + \underbrace{s_2\, s_1}_{\text{both branches}}
\end{aligned}
$$

Each term is one road from the loss back to $$r$$: at each add the road takes the skip (a factor 1) or
the branch (a factor $$s$$). With $$s_1 = 0.1$$ and $$s_2 = -0.2$$:
$$1.1 \times 0.8 = 0.88 = 1 + 0.1 - 0.2 - 0.02$$.

**24 adds.** The same, with $$2^{24} = 16{,}777{,}216$$ roads, and one of them is the **1**: the road
that takes the skip at every add and reaches the first block with the gradient untouched. Without the
skips there is only one road, through every branch, and the gradient is the product of 24 slopes:

| every branch's slope $$s$$ | with the skip: $$(1 + s)^{24}$$ | without: $$s^{24}$$ |
|---:|---:|---:|
| 0 | 1 | 0 |
| 0.05 | 3.2250999 | $$6.0 \times 10^{-32}$$ |
| −0.05 | 0.2919890 | $$6.0 \times 10^{-32}$$ |
| 0.5 | 16,834 | $$5.96 \times 10^{-8}$$ |

*Table 16. Without the skip, a branch with a small slope shrinks the gradient at every block: after 24,
nothing is left for the first layers to learn from. With it, small slopes keep the gradient near 1.*

![The gradient through 24 adds](/images/residual-backward/fig2-through-24-adds.svg)
*Figure 2. The gradient after going back through $$k$$ adds, starting from 1, for $$s = \pm 0.05$$ with
the skip and $$s = 0.05$$ without. The y axis is logarithmic.*

Table 16's last line shows the other side: large slopes make the product with the skip grow too. So
the skip keeps the gradient near 1 **when the branches start small**, and GPT-2 makes sure they do.

### 4.3 Why GPT-2 starts its branches small

GPT-2 sets the weights of the last linear layer of every branch, the one that makes $$b$$, $$\sqrt{N}$$
times smaller than the others at the start, with $$N = 24$$ the number of adds [8]. llm.c does the same:
standard deviation $$0.02 / \sqrt{2 \times 12} = 0.0040825$$ instead of 0.02 [9].

The reason is in the forward. The residual stream after the last block is the first row plus 24 branch
outputs:

$$
y_{24} = r_1 + b_1 + b_2 + \cdots + b_{24}
$$

If the $$b_k$$ are random and independent, each with variance $$\sigma^2$$, their variances add up: the
sum has variance $$24\, \sigma^2$$, a spread $$\sqrt{24} = 4.90$$ times each one's. Making each $$b_k$$
$$\sqrt{24}$$ times smaller makes their sum as spread as one of them. Small $$b_k$$ also means small
branch slopes, the $$s$$ of Table 16: the gradient starts near the "with the skip, $$s \approx 0$$" line.

## 5. Three checks

**Moving a number from one input to the other changes nothing.** Add $$c$$ to $$r$$ and take $$c$$ from
$$b$$: $$y = (r + c) + (b - c) = r + b$$, the same output, so the same loss. The loss moves by
$$dr^{(1)} \cdot c - db \cdot c$$ for a small $$c$$, which must be 0, so $$dr^{(1)} = db$$: B1 and B2
must agree, and they do, both $$dy$$.

**A branch that gives 0 makes the block the identity.** If the branch gives $$b = 0$$ for every $$r$$
(its slope 0), then $$y = r$$ and C1 gives $$dr = dy \cdot (1 + 0) = dy$$: the row passes through
forward, and the gradient passes through backward, both unchanged. A block that has learned nothing
yet does not stop the gradient on its way to the blocks before it.

**A branch with slope −1 cuts the block off.** If $$f'(r) = -1$$, then $$dr = dy \cdot (1 - 1) = 0$$: the
branch takes away whatever the skip brings, and $$r$$ has no say in the loss. Value 1 of the example is
this case: $$y_1 = 2 - 2 = 0$$ and $$dr_1 = 0$$ (Table 12). It is the one slope the skip cannot pass.

## 6. What the rebuilt formula gives the kernel

The add's own steps are B1 and B2: two copies of $$dy$$. B3 is the branch's backward and B4 is an
add-onto, both run by other kernels. So the kernel in my trainer [1] (`residual.cuh`) is small:

| in the note | in the kernel | where |
|---|---|---|
| $$y = r + b$$, F1 | `output` | `residual_forward_kernel`: `residual4.x + branch_output4.x`, ... |
| $$dy$$ | `output_grad` | given by the layer after |
| $$db = dy$$, B1 | `branch_output_grad` | `residual_backward_kernel`: written over (`=`) |
| $$dr^{(1)} = dy$$, B2 | `residual_grad` | `residual_backward_kernel`: written over (`=`) |
| $$dr^{(2)}$$, B3, and $$dr = dr^{(1)} + dr^{(2)}$$, B4 | `residual_grad` again | `layer_norm_backward`: its `input_grad` added onto (`+=`) |

*Table 17. B4's add is done by LayerNorm's backward: it adds its input gradient onto what is there
instead of writing over it, for exactly this reason [2].*

**`=` for the add's gradients, `+=` for LayerNorm's**, from the rules of Section 2:

| gradient | the variable is used | so |
|---|---|---|
| $$db$$ | by the add alone: the branch's output goes nowhere else | one part: written over |
| $$dr$$ | by the add and by the branch's LayerNorm: used twice | two parts: the add writes part 1 over, then LayerNorm's backward adds part 2 onto it |

*Table 18. So the order matters: the add's backward runs first, then the branch's, from its last linear
layer back to its LayerNorm.*

**The backward can be no kernel at all.** B1 and B2 only copy $$dy$$. If $$db$$ and $$dr^{(1)}$$ are
given $$dy$$'s own memory, there is nothing to copy: the branch's last linear layer reads $$dy$$ as its
output gradient, and LayerNorm's backward adds onto the same memory. llm.c's CUDA trainer works this
way and runs no residual backward [9]. Mine keeps `residual_backward` so the backward reads like every
other layer's, and it may be called with all three tensors the same: each value is read before it is
written, by the same thread.

Each value is on its own, so, as in GELU's kernel [4], there is nothing to share between threads: one
`float4` per thread, 786,432 of them for `{4, 1024, 768}`, in 3072 blocks of 256. The time is GPU memory:
the forward reads $$r$$ and $$b$$ and writes $$y$$, the backward reads $$dy$$ and writes two copies,
$$3 \times 12.6$$ MB each. At a T4's 320 GB/s, at least 0.118 ms each. How close the kernel gets to that
is for a later entry: nothing here is timed.

## 7. What to remember

1. **An add passes the gradient whole, to both inputs.** $$\partial(r + b) / \partial r = 1$$ and
   $$\partial(r + b) / \partial b = 1$$: $$dr^{(1)} = db = dy$$. It copies, it does not split.
2. **The residual is used twice.** By the add and by the branch, so its gradient has two parts:
   $$dr = dy + dy \cdot f'(r)$$, and the add's backward is not finished until the branch's has added
   part 2.
3. **The block's slope is the branch's plus 1.** $$dr = dy\,(1 + f'(r))$$. The 1 is the skip, and it
   is there whatever the branch does.
4. **Through 24 adds, one road skips every branch.** The product $$\prod (1 + s_k)$$ has a term 1, so
   the gradient reaches the first block even when every branch's slope is small, where a product
   $$\prod s_k$$ would vanish. GPT-2 starts its branches small to stay near that road.
5. **The kernel is two copies, or none.** `=` for both gradients, LayerNorm adds its part onto
   `residual_grad`, and a trainer can skip the copies by sharing $$dy$$'s memory.

## References

1. D. B. Song. *gpt2-small*, branch `c-cuda-trainer`: GPT-2 small pre-training in C and CUDA.
   [github.com/ducbachsong/gpt2-small](https://github.com/ducbachsong/gpt2-small/tree/c-cuda-trainer)
2. D. B. Song. *LayerNorm backward, broken down and rebuilt: nine chain-rule steps and one formula.* 2026.
   [/posts/layer-norm-backward/](/posts/layer-norm-backward/)
3. D. B. Song. *Linear backward, broken down and rebuilt: three chain-rule steps and three matrix
   products.* 2026. [/posts/linear-backward/](/posts/linear-backward/)
4. D. B. Song. *GELU backward, broken down and rebuilt: seven chain-rule steps and one slope.* 2026.
   [/posts/gelu-backward/](/posts/gelu-backward/)
5. D. B. Song. *Classifier backward, broken down and rebuilt: softmax, cross-entropy and prob minus
   one.* 2026. [/posts/classifier-backward/](/posts/classifier-backward/)
6. K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* 2015.
   [arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
7. K. He, X. Zhang, S. Ren, J. Sun. *Identity Mappings in Deep Residual Networks.* 2016. Section 3: the
   gradient of a stack of residual units has a term that reaches every unit unchanged.
   [arxiv.org/abs/1603.05027](https://arxiv.org/abs/1603.05027)
8. A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever. *Language Models are Unsupervised
   Multitask Learners.* 2019. Section 2.3: the weights of residual layers scaled by $$1/\sqrt{N}$$ at
   the start, $$N$$ the number of residual layers.
   [openai.com](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
9. A. Karpathy. *llm.c*, `train_gpt2.cu`: `residual_forward`, and the residual stream's gradient shared
   by the layers that use it. [github.com/karpathy/llm.c](https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu)
