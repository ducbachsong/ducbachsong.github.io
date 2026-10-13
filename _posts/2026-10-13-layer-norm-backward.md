---
title: "LayerNorm backward, broken down and rebuilt: nine chain-rule steps and one formula"
date: 2026-10-13
permalink: /posts/layer-norm-backward/
excerpt: "The backward pass of LayerNorm is usually given as one line: dx = rstd · (dx̂ − mean(dx̂) − x̂ · mean(dx̂ · x̂)). This note takes it apart into nine small steps, one per forward step, each derived with nothing more than the chain rule, and works one row of 4 numbers through all of them. Then it puts the steps back together, in seven moves, into the one line a GPU kernel runs: two sums per row instead of eight values per number. Every number is checked against PyTorch."
tags:
  - layer-norm
  - backpropagation
  - gpt-2
  - notes
---

**What this is.** GPT-2 small runs LayerNorm 25 times in every forward pass, and the backward
pass of each one is usually written as a single line:

$$
dx_i = r \left( d\hat{x}_i - \frac{1}{n}\sum_j d\hat{x}_j - \hat{x}_i \cdot \frac{1}{n}\sum_j d\hat{x}_j\, \hat{x}_j \right)
$$

That line is correct, and it is what the kernel in my GPT-2 trainer [1] runs, but it is hard to see
where it comes from. This note **breaks it down** into nine small steps, B1 to B9, one for each
forward step, each derived with nothing more than the chain rule. Then it **rebuilds** the line from
those steps, in seven moves, C1 to C7. One row of 4 numbers goes through every step, so each formula
has a number next to it.

**Sources.** The derivation is worked out by hand. Every number in it was checked against PyTorch's
`layer_norm` and its autograd in double precision [3]; Section 6 gives the results. LayerNorm itself
is from Ba et al. [2]; the one-line formula is the one llm.c uses [4].

* TOC
{:toc}

## 1. The forward pass

### 1.1 Six steps for one row

LayerNorm works on one row at a time: the $$n = 768$$ numbers of one token. Every row is normalized
on its own, and rows never mix. For one row:

$$
\begin{aligned}
&\text{F1:} && \mu = \frac{1}{n}\sum_{j=1}^{n} x_j && \text{the mean} \\
&\text{F2:} && d_i = x_i - \mu && \text{each value, centered} \\
&\text{F3:} && v = \frac{1}{n}\sum_{j=1}^{n} d_j^2 && \text{the variance} \\
&\text{F4:} && r = (v + \epsilon)^{-1/2} && \text{1 / the standard deviation (rstd)} \\
&\text{F5:} && \hat{x}_i = d_i \cdot r && \text{normalized: mean 0, standard deviation 1} \\
&\text{F6:} && y_i = \gamma_i\, \hat{x}_i + \beta_i && \text{each column's own scale and shift}
\end{aligned}
$$

$$\epsilon = 10^{-5}$$ only keeps a flat row (all values equal, $$v = 0$$) from dividing by 0.

### 1.2 The names

| symbol | meaning | shape in GPT-2 small |
|---|---|---|
| $$t$$ | row index: which token, $$t = 1 \ldots T$$ | — |
| $$i, j$$ | column index: which feature, $$1 \ldots n$$ | — |
| $$T$$ | number of rows = batch size × sequence length | 4 × 1024 = 4096 |
| $$n$$ | number of columns = embedding size | 768 |
| $$x, d, \hat{x}, y$$ | input, centered, normalized, output: one per value | `{4, 1024, 768}` |
| $$\mu, v, r$$ | mean, variance, rstd: one per row | `{4, 1024}` |
| $$\gamma, \beta$$ | weight (starts at 1) and bias (starts at 0): one per column, shared by all rows | `{768}` |
| $$L$$ | the loss: one number, how wrong the model is | 1 |
| $$\partial$$ | partial derivative: change one variable, hold all others fixed | — |
| $$da$$ | short for $$\partial L / \partial a$$, for any variable $$a$$; same shape as $$a$$ | — |

*Table 1. $$dy$$ is given: it is what the layer after LayerNorm sends back. The job is to find
$$d\beta$$, $$d\gamma$$ and $$dx$$.*

Careful with one name: $$d_i$$ (the centered value) and $$dd_i$$ (its gradient) are different
things. The kernel calls $$\hat{x}$$ `norm` and $$d\hat{x}$$ `norm_grad`.

### 1.3 The running example

One row of $$n = 4$$ numbers, with $$\gamma = 1$$ and $$\beta = 0$$ (their starting values), and
$$\epsilon$$ left out so the numbers stay round (with $$\epsilon = 10^{-5}$$ every number below is the
same to 4 digits):

| | values |
|---|---|
| input $$x$$ | $$[2,\ 4,\ 6,\ 8]$$ |
| F1: $$\mu$$ | $$(2 + 4 + 6 + 8) / 4 = 5$$ |
| F2: $$d$$ | $$[-3,\ -1,\ 1,\ 3]$$ |
| F3: $$v$$ | $$(9 + 1 + 1 + 9) / 4 = 5$$ |
| F4: $$r$$ | $$1 / \sqrt{5} = 0.4472$$ |
| F5: $$\hat{x}$$ | $$[-1.3416,\ -0.4472,\ 0.4472,\ 1.3416]$$ |
| F6: $$y$$ | the same as $$\hat{x}$$, since $$\gamma = 1, \beta = 0$$ |
| $$dy$$, from the next layer | $$[1,\ 0,\ 0,\ 0]$$: the loss cares only about $$y_1$$ |

*Table 2. The example row, forward.*

### 1.4 Where each value is used

![LayerNorm's forward for one row](/images/layer-norm-backward/fig1-graph.svg)
*Figure 1. Each arrow is one use of a value. The backward pass walks the same arrows the other way,
one step per arrow. x and d each have two arrows going out, so their gradients have two parts.*

Figure 1 is the whole plan. The backward pass starts at $$y$$, where $$dy$$ is known, and walks every
arrow backwards, from right to left: B1 to B3 at F6, B4 and B5 at F5, B6a at F4, B6b at F3, B7 and
B8 at F2, and B9 at F1.

## 2. Three rules

Every step below uses the same three rules and nothing else.

| rule | when | what it says |
|---|---|---|
| **chain rule** | $$a$$ affects $$L$$ through $$b$$ | $$da = db \cdot \partial b / \partial a$$ |
| **shared value: sum** | $$a$$ is used in many places (all rows, or all columns of a row) | add one chain-rule term per place: $$da = \sum_k db_k \cdot \partial b_k / \partial a$$ |
| **used twice: add the parts** | $$a$$ is used by two different forward steps | each step gives one part; $$da$$ = part 1 + part 2, and $$da$$ is not finished until both are in |

*Table 3. The three rules. The second and third are the same rule, the multivariable chain rule
(add the effect of every path), seen from two sides.*

And one habit: **each step looks only at its direct inputs.** F5 is $$\hat{x}_i = d_i \cdot r$$. Its
direct inputs are $$d_i$$ and $$r$$. When B4 takes the derivative with respect to $$d_i$$, it holds
$$r$$ fixed, even though $$r$$ is itself computed from the $$d$$s. That other road,
$$d_i \to v \to r \to \hat{x}$$, is not lost: it is its own arrow in Figure 1, and B6b walks it.

**Why the sums do not depend on the loss.** Whether a step has a sum comes only from where the
variable is used in the forward pass. The loss and the layers after LayerNorm only decide the values
of $$dy$$, some of which may even be 0. The only assumptions are that the variable affects $$L$$ only
through the uses shown in Figure 1, and that $$L$$ is differentiable.

## 3. Breaking it down: B1 to B9

Each step first works on one row (no index $$t$$), then adds the row index for the whole batch.

### 3.1 B1: the bias, $$d\beta$$

**One row.** From F6, $$y_i = \gamma_i \hat{x}_i + \beta_i$$:

$$
\begin{aligned}
\frac{\partial y_i}{\partial \beta_i}
  &= \frac{\partial}{\partial \beta_i}\left(\gamma_i \hat{x}_i\right) + \frac{\partial}{\partial \beta_i}\left(\beta_i\right)
  && \text{derivative of a sum = sum of derivatives} \\
  &= 0 + 1 = 1
  && \gamma_i \hat{x}_i \text{ does not contain } \beta_i \text{, so it is a constant} \\
d\beta_i &= dy_i \cdot \frac{\partial y_i}{\partial \beta_i} = dy_i \cdot 1 = dy_i
  && \text{chain rule}
\end{aligned}
$$

**The whole batch.** The same $$\beta_i$$ is used in every row: $$y_{t,i} = \gamma_i \hat{x}_{t,i} + \beta_i$$
for every $$t$$. Changing $$\beta_i$$ changes $$y_{1,i}, y_{2,i}, \ldots, y_{T,i}$$ all at once, so there are
$$T$$ paths to the loss, and by the shared-value rule:

$$
d\beta_i = \frac{\partial L}{\partial y_{1,i}}\frac{\partial y_{1,i}}{\partial \beta_i} + \cdots + \frac{\partial L}{\partial y_{T,i}}\frac{\partial y_{T,i}}{\partial \beta_i}
         = \sum_{t=1}^{T} dy_{t,i} \cdot 1
         = \sum_{t=1}^{T} dy_{t,i}
$$

**Example.** One row, so $$d\beta = dy = [1, 0, 0, 0]$$.

### 3.2 B2: the weight, $$d\gamma$$

**One row.** From F6 again, this time with respect to $$\gamma_i$$:

$$
\begin{aligned}
\frac{\partial y_i}{\partial \gamma_i}
  &= \frac{\partial}{\partial \gamma_i}\left(\gamma_i \hat{x}_i\right) + \frac{\partial}{\partial \gamma_i}\left(\beta_i\right)
  = \hat{x}_i + 0 = \hat{x}_i
  && \hat{x}_i \text{ is a constant factor; } \beta_i \text{ a constant} \\
d\gamma_i &= dy_i \cdot \hat{x}_i
  && \text{chain rule}
\end{aligned}
$$

**The whole batch.** $$\gamma_i$$ is shared by every row too, but each row has its **own**
$$\hat{x}_{t,i}$$:

$$
d\gamma_i = \sum_{t=1}^{T} dy_{t,i} \cdot \frac{\partial y_{t,i}}{\partial \gamma_i} = \sum_{t=1}^{T} dy_{t,i}\, \hat{x}_{t,i}
$$

**The difference from B1.** In B1 the local derivative was 1 in every row, so it disappeared. Here it
is $$\hat{x}_{t,i}$$, different in every row, so it stays inside the sum. $$\gamma$$ multiplies
$$\hat{x}$$: where $$\hat{x}$$ is large, a small change of $$\gamma$$ moves $$y$$ a lot; where
$$\hat{x} = 0$$, changing $$\gamma$$ does nothing to that row.

**Example.** $$d\gamma = dy \cdot \hat{x} = [1 \times (-1.3416),\ 0,\ 0,\ 0] = [-1.3416,\ 0,\ 0,\ 0]$$.

### 3.3 B3: the normalized value, $$d\hat{x}$$

**One row.** From F6, with respect to $$\hat{x}_i$$:

$$
\begin{aligned}
\frac{\partial y_i}{\partial \hat{x}_i} &= \gamma_i + 0 = \gamma_i
  && \gamma_i \text{ is a constant factor} \\
d\hat{x}_i &= dy_i\, \gamma_i
  && \text{chain rule}
\end{aligned}
$$

**The whole batch.** Unlike $$\gamma$$ and $$\beta$$, $$\hat{x}_{t,i}$$ belongs to one row and one
column. It is used in **one place only**, $$y_{t,i}$$: one path, so no sum.

$$
d\hat{x}_{t,i} = dy_{t,i}\, \gamma_i
$$

| step | variable | used in how many places | sum? | result |
|---|---|---|---|---|
| B1 | $$\beta_i$$ | every row | yes, over rows | one per column: `{768}` |
| B2 | $$\gamma_i$$ | every row | yes, over rows | one per column: `{768}` |
| B3 | $$\hat{x}_{t,i}$$ | only $$y_{t,i}$$ | **no** | one per value: `{4, 1024, 768}` |

*Table 4. B1 and B2 give the gradients the optimizer uses to update $$\gamma$$ and $$\beta$$. B3 starts
the road back to $$dx$$, the gradient sent to the layer before.*

If $$\gamma_i = 3$$, a small change of $$\hat{x}_i$$ is 3 times bigger in $$y_i$$, so the gradient
coming back is 3 times bigger too. If $$\gamma_i = 0$$, the column is switched off and nothing comes
back through it.

**Example.** $$d\hat{x} = dy \cdot \gamma = [1, 0, 0, 0]$$.

### 3.4 B4: the centered value $$d$$, part 1

**One row.** From F5, $$\hat{x}_i = d_i \cdot r$$, with respect to $$d_i$$, holding the other direct
input $$r$$ fixed (Section 2):

$$
\begin{aligned}
\frac{\partial \hat{x}_i}{\partial d_i} &= r \cdot \frac{\partial d_i}{\partial d_i} = r \cdot 1 = r \\
dd_i^{(1)} &= d\hat{x}_i \cdot r
  && \text{chain rule, along the arrow F5}
\end{aligned}
$$

**The whole batch.** Each row has its own $$r_t$$, the `rstd` the forward saved, `{4, 1024}`. Inside
F5, $$d_{t,i}$$ is used once, so there is no sum:

$$
dd_{t,i}^{(1)} = d\hat{x}_{t,i}\, r_t
$$

**Why "(1)".** In Figure 1, $$d_i$$ has two arrows going out, so its gradient has two parts:

| part | forward step that uses $$d_i$$ | gradient part | worked out in |
|---|---|---|---|
| 1 | F5: $$\hat{x}_i = d_i \cdot r$$ | $$dd_i^{(1)} = d\hat{x}_i \cdot r$$ | B4, here |
| 2 | F3: $$v = \frac{1}{n}\sum_j d_j^2$$ | $$dd_i^{(2)}$$ | B6b, later |

*Table 5. $$dd_i = dd_i^{(1)} + dd_i^{(2)}$$. After B4 it is not finished, and nothing may go further
back with it until part 2 is added.*

**Example.** $$dd^{(1)} = [1, 0, 0, 0] \times 0.4472 = [0.4472,\ 0,\ 0,\ 0]$$.

In the forward, $$d_i$$ was multiplied by $$r$$ to become $$\hat{x}_i$$, so any change of $$d_i$$
reaches $$\hat{x}_i$$ multiplied by $$r$$. The gradient goes back along the same road and is
multiplied by the same $$r$$.

### 3.5 B5: the rstd, $$dr$$

**One row.** From F5 again, now with respect to $$r$$, holding $$d_i$$ fixed:

$$
\frac{\partial \hat{x}_i}{\partial r} = d_i \cdot \frac{\partial r}{\partial r} = d_i
$$

The difference from B4 is where $$r$$ is used. There is **one** $$r$$ for the whole row, and every
column uses it:
$$\hat{x}_1 = d_1 r,\ \hat{x}_2 = d_2 r,\ \ldots,\ \hat{x}_n = d_n r$$.
That is $$n$$ paths, so by the shared-value rule:

$$
dr = \frac{\partial L}{\partial \hat{x}_1}\frac{\partial \hat{x}_1}{\partial r} + \cdots + \frac{\partial L}{\partial \hat{x}_n}\frac{\partial \hat{x}_n}{\partial r}
   = \sum_{i=1}^{n} d\hat{x}_i\, d_i
$$

**The whole batch.** $$r_t$$ is used in every column of row $$t$$, and in no other row, so the sum is
over the **columns** $$i$$, not over the rows. One number per row, `{4, 1024}`, the shape of `rstd`:

$$
dr_t = \sum_{i=1}^{n} d\hat{x}_{t,i}\, d_{t,i}
$$

| step | variable | shared by | sum over |
|---|---|---|---|
| B1, B2 | $$\beta_i$$, $$\gamma_i$$ | all rows, one column | rows $$t$$ |
| B3, B4 | $$\hat{x}_{t,i}$$, $$d_{t,i}$$ (part 1) | nothing: one value | no sum |
| B5 | $$r_t$$ | all columns, one row | columns $$i$$ |

*Table 6. Sum over whatever the variable is shared across.*

**Example.** $$dr = (1)(-3) + (0)(-1) + (0)(1) + (0)(3) = -3$$.

Changing $$r$$ stretches or shrinks the whole row at once: if $$r$$ goes up by a small
$$\Delta r$$, each $$\hat{x}_i$$ moves by $$d_i \cdot \Delta r$$, and the loss changes by
$$\sum_i d\hat{x}_i\, d_i\, \Delta r = dr \cdot \Delta r$$. So $$dr$$ answers one question: would the
loss like the whole row stretched (negative $$dr$$) or shrunk (positive $$dr$$)? Here $$dr = -3$$,
stretched: $$d\hat{x}_1 = 1$$ says the loss goes down when $$\hat{x}_1$$ goes down, and
$$\hat{x}_1 = -3r$$ goes down when $$r$$ goes up.

### 3.6 B6a: the variance, $$dv$$

**One row.** From F4, $$r = (v + \epsilon)^{-1/2}$$, a function inside a function. Call the inside
$$u = v + \epsilon$$, so $$r = u^{-1/2}$$:

$$
\begin{aligned}
\frac{\partial u}{\partial v} &= 1 + 0 = 1
  && \epsilon \text{ is a constant} \\
\frac{\partial r}{\partial u} &= -\tfrac{1}{2}\, u^{-1/2 - 1} = -\tfrac{1}{2}\, u^{-3/2}
  && \text{power rule } \tfrac{\partial}{\partial u} u^k = k\, u^{k-1} \text{, with } k = -\tfrac{1}{2} \\
\frac{\partial r}{\partial v} &= \frac{\partial r}{\partial u} \cdot \frac{\partial u}{\partial v} = -\tfrac{1}{2}\,(v + \epsilon)^{-3/2}
  && \text{chain rule} \\
  &= -\tfrac{1}{2}\, r^3
  && r^3 = \left((v + \epsilon)^{-1/2}\right)^3 = (v + \epsilon)^{-3/2} \\
dv &= dr \cdot \frac{\partial r}{\partial v} = -\tfrac{1}{2}\, r^3\, dr
  && v \text{ is used once, in } r
\end{aligned}
$$

**The whole batch.** $$v_t$$ is used only in $$r_t$$: one path, no sum, one number per row.

$$
dv_t = -\tfrac{1}{2}\, r_t^3\, dr_t
$$

**Why write $$r^3$$ instead of $$(v + \epsilon)^{-3/2}$$.** They are the same number:

| form | with $$v = 5$$ | result |
|---|---|---|
| $$(v + \epsilon)^{-3/2}$$ | $$1 / (\sqrt{5} \cdot \sqrt{5} \cdot \sqrt{5}) = 1 / 11.18$$ | $$0.0894$$ |
| $$r^3$$ | $$0.4472 \times 0.4472 \times 0.4472$$ | $$0.0894$$ |

*Table 7. Same number, two ways.*

But $$r$$ is already saved from the forward, so $$r^3$$ is two multiplies where the other form needs
a square root again. And writing everything in $$r$$ is what lets terms cancel when the steps are
rebuilt in Section 4.

**Example.** $$dv = -\tfrac{1}{2} \times 0.08944 \times (-3) = 0.13416$$.

**The minus sign.** A bigger variance means a wider row, a bigger standard deviation, and a smaller
$$r = 1/\sigma$$: $$r$$ moves against $$v$$. So in the example the sign flips: $$dr = -3$$ says the
loss wants $$r$$ up, which means $$v$$ down, so $$dv$$ is positive.

**Why $$r^3$$.** The curve $$r = (v + \epsilon)^{-1/2}$$ is steep where $$v$$ is small and flat where
$$v$$ is large, and $$r^3$$ is big exactly where $$v$$ is small. A row with a narrow spread is very
sensitive to its variance; a wide row is not.

### 3.7 B6b: the centered value $$d$$, part 2, and complete

**One row.** From F3, $$v = \frac{1}{n}\sum_j d_j^2$$. The sum uses $$j$$ so it does not mix with the
$$i$$ we differentiate for:

$$
\begin{aligned}
\frac{\partial v}{\partial d_i}
  &= \frac{1}{n} \cdot \frac{\partial}{\partial d_i}\left(d_1^2 + \cdots + d_i^2 + \cdots + d_n^2\right)
  && \tfrac{1}{n} \text{ is a constant factor} \\
  &= \frac{1}{n}\left(0 + \cdots + \frac{\partial}{\partial d_i} d_i^2 + \cdots + 0\right)
  && \text{every other } d_j^2 \text{ is a constant} \\
  &= \frac{1}{n} \cdot 2\, d_i = \frac{2}{n}\, d_i
  && \text{power rule, } k = 2 \\
dd_i^{(2)} &= dv \cdot \frac{2}{n}\, d_i
  && \text{chain rule, along the arrow F3}
\end{aligned}
$$

Now both parts are known, and by the used-twice rule:

$$
dd_i = dd_i^{(1)} + dd_i^{(2)} = d\hat{x}_i \cdot r + dv \cdot \frac{2}{n}\, d_i
$$

**The whole batch.** $$d_{t,i}$$ is used in F3 once, in its own row's $$v_t$$: no sum.

$$
dd_{t,i} = d\hat{x}_{t,i}\, r_t + dv_t \cdot \frac{2}{n}\, d_{t,i}
$$

**Example.** $$dv \cdot \frac{2}{n} = 0.13416 \times \frac{2}{4} = 0.06708$$, then

| | values |
|---|---|
| $$dd^{(1)}$$ (B4) | $$[0.4472,\ 0,\ 0,\ 0]$$ |
| $$dd^{(2)} = 0.06708 \times [-3, -1, 1, 3]$$ | $$[-0.2012,\ -0.0671,\ 0.0671,\ 0.2012]$$ |
| $$dd = dd^{(1)} + dd^{(2)}$$ | $$[0.2460,\ -0.0671,\ 0.0671,\ 0.2012]$$ |

*Table 8. Part 2 gives every column a gradient, although only $$dy_1$$ is not 0.*

**Why $$d_i$$ is in part 2.** The variance squares each $$d_i$$, so a value far from the center
moves the variance a lot when nudged, and a value at the center ($$d_i = 0$$) not at all. **Why
$$\frac{1}{n}$$.** The variance is an average, and each value is only one of $$n$$ in it. With 768
columns, one value alone barely moves it.

| part | what happens when $$d_i$$ changes | gradient part |
|---|---|---|
| 1 | $$\hat{x}_i$$ changes directly | $$d\hat{x}_i \cdot r$$ |
| 2 | the variance changes, so $$r$$ changes, so **every** $$\hat{x}$$ of the row is rescaled | $$dv \cdot \frac{2}{n} d_i$$ |

*Table 9. Part 2 is why the gradient of one value depends on the whole row.*

### 3.8 B7 and B8: the input $$x$$, part 1, and the mean, $$d\mu$$

Both come from F2, $$d_i = x_i - \mu$$, which has two direct inputs, just as B4 and B5 both came from
F5.

**B7, one row.** With respect to $$x_i$$, holding $$\mu$$ fixed (the road
$$x_i \to \mu \to d$$ is B8 and B9's):

$$
\begin{aligned}
\frac{\partial d_i}{\partial x_i} &= \frac{\partial x_i}{\partial x_i} - \frac{\partial \mu}{\partial x_i}\Big\rvert_{\mu \text{ fixed}} = 1 - 0 = 1 \\
dx_i^{(1)} &= dd_i \cdot 1 = dd_i
  && x_i \text{ is used in one } d \text{ only}
\end{aligned}
$$

Like $$d$$, $$x_i$$ has two arrows in Figure 1, F2 and F1, so this is only part 1. Part 2 is B9.

**B8, one row.** With respect to $$\mu$$, holding $$x_i$$ fixed:

$$
\frac{\partial d_i}{\partial \mu} = 0 - \frac{\partial \mu}{\partial \mu} = -1
$$

There is one $$\mu$$ for the whole row, used in every column:
$$d_1 = x_1 - \mu,\ \ldots,\ d_n = x_n - \mu$$. That is $$n$$ paths:

$$
d\mu = \sum_{i=1}^{n} dd_i \cdot \frac{\partial d_i}{\partial \mu} = \sum_{i=1}^{n} dd_i \cdot (-1) = -\sum_{i=1}^{n} dd_i
$$

This needs the **complete** $$dd_i$$, both parts. That is why B6b had to come first.

**The whole batch.** $$x_{t,i}$$ is used in one $$d$$; $$\mu_t$$ in every column of row $$t$$:

$$
dx_{t,i}^{(1)} = dd_{t,i} \qquad\qquad d\mu_t = -\sum_{i=1}^{n} dd_{t,i}
$$

**Example.** $$dx^{(1)} = dd = [0.2460,\ -0.0671,\ 0.0671,\ 0.2012]$$, and
$$d\mu = -(0.2460 - 0.0671 + 0.0671 + 0.2012) = -0.4472$$.

| variable | local derivative | used in | gradient |
|---|---|---|---|
| $$x_i$$ | $$+1$$ | only $$d_i$$ | $$dx_i^{(1)} = dd_i$$, no sum |
| $$\mu$$ | $$-1$$ | every $$d_i$$ of the row | $$d\mu = -\sum_i dd_i$$, sum over columns |

*Table 10. Subtraction passes the gradient through unchanged to $$x_i$$: if $$x_i$$ goes up by 0.1
with $$\mu$$ fixed, $$d_i$$ goes up by exactly 0.1. And if $$\mu$$ goes up by 0.1, every $$d_i$$ goes
down by 0.1 at once, so the loss changes by $$-0.1 \sum_i dd_i$$.*

### 3.9 B9: the input $$x$$, part 2, and complete

**One row.** From F1, $$\mu = \frac{1}{n}\sum_j x_j$$:

$$
\begin{aligned}
\frac{\partial \mu}{\partial x_i}
  &= \frac{1}{n} \cdot \frac{\partial}{\partial x_i}\left(x_1 + \cdots + x_i + \cdots + x_n\right)
   = \frac{1}{n}\left(0 + \cdots + 1 + \cdots + 0\right) = \frac{1}{n} \\
dx_i^{(2)} &= d\mu \cdot \frac{1}{n} = \frac{d\mu}{n}
  && \text{chain rule, along the arrow F1}
\end{aligned}
$$

Both parts of $$dx_i$$ are now known:

$$
dx_i = dx_i^{(1)} + dx_i^{(2)} = dd_i + \frac{d\mu}{n}
$$

**The whole batch.** $$x_{t,i}$$ is used once in $$\mu_t$$: no sum.

$$
dx_{t,i} = dd_{t,i} + \frac{d\mu_t}{n}
$$

This is the gradient sent to the layer before, `{4, 1024, 768}`, the shape of the input.

**Example.** $$d\mu / n = -0.4472 / 4 = -0.1118$$, taken off every column:

$$
dx = [0.2460 - 0.1118,\ -0.0671 - 0.1118,\ 0.0671 - 0.1118,\ 0.2012 - 0.1118] = [0.1342,\ -0.1789,\ -0.0447,\ 0.0894]
$$

Part 2 spreads $$d\mu$$ **equally** over the columns, because every $$x_i$$ counts equally,
$$\frac{1}{n}$$, in the mean.

### 3.10 All nine steps

| step | gradient of | formula | one per | example |
|---|---|---|---|---|
| B1 | bias | $$d\beta_i = \sum_t dy_{t,i}$$ | column | $$[1,\ 0,\ 0,\ 0]$$ |
| B2 | weight | $$d\gamma_i = \sum_t dy_{t,i}\, \hat{x}_{t,i}$$ | column | $$[-1.3416,\ 0,\ 0,\ 0]$$ |
| B3 | normalized | $$d\hat{x}_i = dy_i\, \gamma_i$$ | value | $$[1,\ 0,\ 0,\ 0]$$ |
| B4 | centered, part 1 | $$dd_i^{(1)} = d\hat{x}_i\, r$$ | value | $$[0.4472,\ 0,\ 0,\ 0]$$ |
| B5 | rstd | $$dr = \sum_i d\hat{x}_i\, d_i$$ | row | $$-3$$ |
| B6a | variance | $$dv = -\tfrac{1}{2} r^3\, dr$$ | row | $$0.13416$$ |
| B6b | centered, complete | $$dd_i = dd_i^{(1)} + dv \cdot \tfrac{2}{n} d_i$$ | value | $$[0.2460,\ -0.0671,\ 0.0671,\ 0.2012]$$ |
| B7 | input, part 1 | $$dx_i^{(1)} = dd_i$$ | value | $$[0.2460,\ -0.0671,\ 0.0671,\ 0.2012]$$ |
| B8 | mean | $$d\mu = -\sum_i dd_i$$ | row | $$-0.4472$$ |
| B9 | input, complete | $$dx_i = dx_i^{(1)} + \tfrac{d\mu}{n}$$ | value | $$[0.1342,\ -0.1789,\ -0.0447,\ 0.0894]$$ |

*Table 11. The backward pass, broken down. The order matters only where a step needs a finished
gradient: B8 needs the complete $$dd$$ of B6b.*

## 4. Rebuilding: C1 to C7

Table 11 is correct but long. Run as it stands, it would keep five gradients per value
($$d\hat{x}$$, $$dd^{(1)}$$, $$dd$$, $$dx^{(1)}$$, $$dx$$), and $$d$$ itself, and three per row. This
section folds the steps into one formula, using two facts about the forward:

$$
d_i = \frac{\hat{x}_i}{r} \quad \text{(F5, turned around)}
\qquad\qquad
\sum_{i=1}^{n} \hat{x}_i = 0 \quad \text{(each row of } \hat{x} \text{ has mean 0)}
$$

The second holds because $$\sum_i d_i = \sum_i x_i - n\mu = 0$$, and $$\hat{x}_i = d_i r$$ with the same
$$r$$ for the whole row. To keep the formulas short, write the two row means that will appear as

$$
m_1 = \frac{1}{n}\sum_{j=1}^{n} d\hat{x}_j
\qquad\qquad
m_2 = \frac{1}{n}\sum_{j=1}^{n} d\hat{x}_j\, \hat{x}_j
$$

In the example, $$m_1 = 1/4 = 0.25$$ and $$m_2 = (1 \times -1.3416)/4 = -0.3354$$.

**C1: $$dr$$ in $$\hat{x}$$.** Put $$d_j = \hat{x}_j / r$$ into B5:

$$
dr = \sum_j d\hat{x}_j\, \frac{\hat{x}_j}{r} = \frac{n\, m_2}{r}
$$

Example: $$4 \times (-0.3354) / 0.4472 = -3$$, as in B5.

**C2: $$dv$$ in $$\hat{x}$$.** Put C1 into B6a. $$r^3 / r = r^2$$:

$$
dv = -\tfrac{1}{2}\, r^3 \cdot \frac{n\, m_2}{r} = -\tfrac{1}{2}\, r^2\, n\, m_2
$$

Example: $$-\tfrac{1}{2} \times 0.2 \times 4 \times (-0.3354) = 0.13416$$, as in B6a.

**C3: part 2 of $$dd$$.** Put C2 and $$d_i = \hat{x}_i / r$$ into B6b's part 2. The $$n$$s cancel, the
$$2$$ cancels the $$\tfrac{1}{2}$$, and $$r^2 / r = r$$:

$$
dd_i^{(2)} = dv \cdot \frac{2}{n}\, d_i
           = \left(-\tfrac{1}{2}\, r^2\, n\, m_2\right) \cdot \frac{2}{n} \cdot \frac{\hat{x}_i}{r}
           = -r\, \hat{x}_i\, m_2
$$

This is the cancelling that $$r^3$$ in B6a was kept for. Example:
$$-0.4472 \times \hat{x} \times (-0.3354) = 0.15 \times \hat{x} = [-0.2012,\ -0.0671,\ 0.0671,\ 0.2012]$$,
as in Table 8.

**C4: the complete $$dd$$.** Add part 1 from B4, and take $$r$$ out:

$$
dd_i = r\, d\hat{x}_i - r\, \hat{x}_i\, m_2 = r\left(d\hat{x}_i - \hat{x}_i\, m_2\right)
$$

**C5: $$d\mu$$.** Put C4 into B8, and use $$\sum_i \hat{x}_i = 0$$:

$$
d\mu = -\sum_i r\left(d\hat{x}_i - \hat{x}_i\, m_2\right)
     = -r \sum_i d\hat{x}_i + r\, m_2 \underbrace{\sum_i \hat{x}_i}_{=\,0}
     = -r\, n\, m_1
$$

The whole of part 2 of $$dd$$ drops out of $$d\mu$$. Example: $$-0.4472 \times 4 \times 0.25 = -0.4472$$,
as in B8.

**C6: the complete $$dx$$.** Put C4 and C5 into B9:

$$
dx_i = dd_i + \frac{d\mu}{n} = r\left(d\hat{x}_i - \hat{x}_i\, m_2\right) - r\, m_1
$$

**C7: one formula.** Take $$r$$ out once more, write $$m_1$$ and $$m_2$$ out, and put in B3's
$$d\hat{x}_i = dy_i\, \gamma_i$$:

$$
\boxed{\;dx_i = r\left(d\hat{x}_i - \frac{1}{n}\sum_{j} d\hat{x}_j - \hat{x}_i \cdot \frac{1}{n}\sum_{j} d\hat{x}_j\, \hat{x}_j\right),
\qquad d\hat{x}_i = dy_i\, \gamma_i\;}
$$

For the whole batch every symbol gets the row index $$t$$, and both sums stay inside the row.

**Example**, straight from the formula, with $$m_1 = 0.25$$ and $$m_2 = -0.3354$$:

| $$i$$ | $$d\hat{x}_i$$ | $$\hat{x}_i$$ | $$d\hat{x}_i - m_1 - \hat{x}_i\, m_2$$ | $$\times\, r = dx_i$$ | B9 gave |
|---|---:|---:|---:|---:|---:|
| 1 | 1 | −1.3416 | $$1 - 0.25 - 0.45 = 0.30$$ | 0.1342 | 0.1342 |
| 2 | 0 | −0.4472 | $$0 - 0.25 - 0.15 = -0.40$$ | −0.1789 | −0.1789 |
| 3 | 0 | 0.4472 | $$0 - 0.25 + 0.15 = -0.10$$ | −0.0447 | −0.0447 |
| 4 | 0 | 1.3416 | $$0 - 0.25 + 0.45 = 0.20$$ | 0.0894 | 0.0894 |

*Table 12. The rebuilt formula gives the same $$dx$$ as the nine steps.*

**What each part means.** Inside the brackets, three things are taken from the gradient:

| part | what it is | why it is there |
|---|---|---|
| $$d\hat{x}_i$$ | the gradient as if $$\mu$$ and $$r$$ were fixed | the direct road, B4 and B7 |
| $$-m_1$$ | the row's average gradient, taken off every column | shifting the whole row does not change $$\hat{x}$$ (B8, B9) |
| $$-\hat{x}_i\, m_2$$ | the part that points along $$\hat{x}$$ itself | stretching the whole row does not change $$\hat{x}$$ (B5 to B6b) |

*Table 13. LayerNorm forgets where a row was and how wide it was, so the gradient can have no part
that would only move or stretch the row.*

## 5. Two checks

**The gradient of each row adds up to 0.** Add C6 over the columns:

$$
\sum_i dx_i = r\left(\sum_i d\hat{x}_i - m_2 \sum_i \hat{x}_i\right) - n\, r\, m_1
            = r\, n\, m_1 - 0 - n\, r\, m_1 = 0
$$

In the example: $$0.1342 - 0.1789 - 0.0447 + 0.0894 = 0$$. Adding the same amount to every input
changes nothing after the mean is taken off, so the gradient can have no "shift everything" part.

**The gradient of each row is at right angles to $$\hat{x}$$.** Multiply C6 by $$\hat{x}_i$$ and add.
$$\sum_i \hat{x}_i^2 = r^2 \sum_i d_i^2 = n\, v\, r^2 = n\,\frac{v}{v + \epsilon}$$, so

$$
\sum_i \hat{x}_i\, dx_i = r\left(n\, m_2 - m_2\, n\,\frac{v}{v + \epsilon}\right) - r\, m_1 \cdot 0
                       = r\, n\, m_2\, \frac{\epsilon}{v + \epsilon}
$$

This is exactly 0 without $$\epsilon$$, and very close to 0 with it. In the example:
$$0.1342 \times (-1.3416) + (-0.1789)(-0.4472) + (-0.0447)(0.4472) + 0.0894 \times 1.3416 = 0$$.
Stretching a row changes nothing after it is divided by its standard deviation, so the gradient has
no "stretch everything" part either, except for the little that $$\epsilon$$ lets through.

## 6. Checked against PyTorch

| test | compared | largest difference |
|---|---|---|
| the example row, $$\epsilon = 0$$ and $$\epsilon = 10^{-5}$$ | every number of Tables 2, 8, 11 and 12 with PyTorch's autograd | none in the 4 digits shown |
| 6 random rows of 768, $$x \sim 3\,\mathcal{N}(0,1) + 2$$, random $$\gamma$$, $$\beta$$, $$dy$$, $$\epsilon = 10^{-5}$$ | C7 against `layer_norm`'s $$dx$$ | $$4.4 \times 10^{-16}$$ |
| the same | B1, B2 against $$d\beta$$, $$d\gamma$$ | $$0$$ and $$4.4 \times 10^{-15}$$ |
| the same | $$\sum_i dx_i$$ per row | $$1.1 \times 10^{-14}$$ |
| the same | $$\sum_i \hat{x}_i\, dx_i$$ per row | $$1.4 \times 10^{-5}$$, the $$\epsilon$$ term of Section 5 |

*Table 14. PyTorch [3], `torch.nn.functional.layer_norm` and `backward`, in double precision.
Differences near $$10^{-16}$$ are rounding.*

## 7. What the rebuilt formula gives the kernel

The nine steps and the one formula give the same numbers, but they are very different programs.

| | broken down (Table 11) | rebuilt (C7) |
|---|---|---|
| per value, besides $$dx$$ | $$d$$, $$d\hat{x}$$, $$dd^{(1)}$$, $$dd$$, $$dx^{(1)}$$ | none: $$\hat{x}$$ and $$d\hat{x}$$ are worked out again when needed |
| per row | $$dr$$, $$dv$$, $$d\mu$$ | two sums: $$\sum_j d\hat{x}_j$$ and $$\sum_j d\hat{x}_j\, \hat{x}_j$$ |
| passes over the row | one per step | two: the two sums, then $$dx$$ |
| saved by the forward and needed | $$\mu$$, $$r$$ | the same: $$\mu$$, $$r$$ |

*Table 15. In GPT-2 small, one tensor of one value per number is 4096 × 768 × 4 bytes = 12.6 MB,
for each of the 25 LayerNorms.*

The kernel in my trainer [1] (`layer_norm.cuh`) is C7 written out, one warp of 32 threads per row:

| in C7 | in the kernel | where |
|---|---|---|
| $$\hat{x}_i = (x_i - \mu)\, r$$ | `norm` | worked out again from `input`, `mean`, `rstd` in both passes |
| $$d\hat{x}_i = dy_i\, \gamma_i$$ | `norm_grad` | from `output_grad` and `weight` |
| $$m_1$$ | `norm_grad_mean` | pass 1: each thread adds its part, then 5 shuffles add across the warp |
| $$m_2$$ | `norm_grad_norm_mean` | pass 1, in the same loop and the same shuffles |
| $$dx_i$$ | `input_grad` | pass 2: `rstd * (norm_grad - norm_grad_mean - norm * norm_grad_norm_mean)` |
| $$d\beta_i$$, $$d\gamma_i$$ (B1, B2) | `bias_grad`, `weight_grad` | pass 2: summed over the block's rows in shared memory, then one add per column per block into GPU memory |

*Table 16. B1 and B2 are sums over rows, while $$dx$$ is a sum over columns: the kernel splits the
rows between warps, so $$dx$$ needs no other warp, but $$d\gamma$$ and $$d\beta$$ need every warp.*

The kernel never works out $$d$$, $$dr$$, $$dv$$ or $$d\mu$$: C1 to C6 folded them away. All that is
left from the nine steps is in the shape of the formula: the $$-m_1$$ is B8 and B9, the
$$-\hat{x}_i\, m_2$$ is B5 to B6b, and the $$r$$ in front is B4.

## 8. What to remember

1. **One step per arrow.** The backward pass walks Figure 1's arrows backwards. Each step uses only
   the chain rule, and holds its other direct inputs fixed.
2. **A sum comes from sharing.** $$\gamma$$ and $$\beta$$ are shared by every row, so their gradients
   sum over rows. $$\mu$$, $$v$$ and $$r$$ are shared by every column of a row, so theirs sum over
   columns. A value used in one place has no sum.
3. **Two uses, two parts.** $$x$$ and $$d$$ are each used by two forward steps, and their gradients are
   the two parts added. Nothing may use them before both parts are in.
4. **Rebuilding needs two facts:** $$d = \hat{x} / r$$ and $$\sum_i \hat{x}_i = 0$$. With them, $$r^3$$
   cancels, part 2 drops out of $$d\mu$$, and nine steps become one line with two sums per row.
5. **The line says what LayerNorm ignores.** It takes off the row's average gradient and the part along
   $$\hat{x}$$, so each row's $$dx$$ sums to 0 and, without $$\epsilon$$, is at right angles to
   $$\hat{x}$$.

## References

1. D. B. Song. *gpt2-small*, branch `c-cuda-trainer`: GPT-2 small pre-training in C and CUDA.
   [github.com/ducbachsong/gpt2-small](https://github.com/ducbachsong/gpt2-small/tree/c-cuda-trainer)
2. J. L. Ba, J. R. Kiros, G. E. Hinton. *Layer Normalization.* arXiv:1607.06450, 2016.
   [arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450)
3. PyTorch. `torch.nn.functional.layer_norm`.
   [docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.layer_norm.html)
4. A. Karpathy. *llm.c*, `doc/layernorm/layernorm.md`: LayerNorm forward and backward in PyTorch and C.
   [github.com/karpathy/llm.c](https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md)
