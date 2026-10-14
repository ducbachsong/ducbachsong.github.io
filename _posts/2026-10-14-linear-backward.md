---
title: "Linear backward, broken down and rebuilt: five chain-rule steps and three matrix products"
date: 2026-10-14
permalink: /posts/linear-backward/
excerpt: "The backward pass of a linear layer is usually given as three lines: dx = dy · W, dW = dyᵀ · x, db = the column sums of dy. This note takes it apart into five small steps, one per forward step, each derived with nothing more than the chain rule, and works two rows of 4 numbers through all of them. Then it puts the steps back together, in five moves, into the three matrix products a GPU kernel runs, and shows where each ᵀ comes from: from which index a sum runs over. Two checks follow from the formulas, and every number is checked against PyTorch."
tags:
  - linear-layer
  - backpropagation
  - gpt-2
  - notes
---

**What this is.** GPT-2 small runs 49 linear layers in every forward pass, and the backward pass of
each one is usually written as three lines:

$$
dx = dy\, W \qquad\qquad dW = dy^\top x \qquad\qquad db = dy^\top \mathbf{1}
$$

Those lines are correct, and they are what the kernel in my GPT-2 trainer [1] runs, but they hide two
things: where the sums come from, and why the $$\top$$ sits where it does. This note **breaks them
down** into five small steps, B1 to B5, one for each forward step, each derived with nothing more than
the chain rule. Then it **rebuilds** the three lines from those steps, in five moves, C1 to C5. One
example of 2 rows and 4 numbers goes through every step, so each formula has a number next to it.

**Sources.** The derivation is worked out by hand, with the same three rules as the
[note on LayerNorm's backward](/posts/layer-norm-backward/) [2]. Every number in it was checked against
PyTorch's `linear` and its autograd in double precision [4]; Section 6 gives the results. The three
lines are the ones llm.c's C code runs [5].

* TOC
{:toc}

## 1. The forward pass

### 1.1 Three steps for one row

A linear layer works on one row at a time: the $$n_\text{in}$$ numbers of one token. Every row is
multiplied by the same weight, and rows never mix. For one row:

$$
\begin{aligned}
&\text{F1:} && p_{j,i} = x_i\, W_{j,i} && \text{each input times its weight to each output: } n_\text{out} \times n_\text{in} \text{ products} \\
&\text{F2:} && z_j = \sum_{i=1}^{n_\text{in}} p_{j,i} && \text{the dot product of } x \text{ with row } j \text{ of } W \\
&\text{F3:} && y_j = z_j + b_j && \text{each output column's own bias}
\end{aligned}
$$

Written as one line, $$y_j = \sum_i x_i\, W_{j,i} + b_j$$. The kernel never stores $$p$$ or $$z$$:
they are names for the steps, so that each backward step has one small forward step to undo.

### 1.2 The names

| symbol | meaning | shape in GPT-2 small (the qkv layer) |
|---|---|---|
| $$t$$ | row index: which token, $$t = 1 \ldots T$$ | — |
| $$i$$ | input column: which input feature, $$1 \ldots n_\text{in}$$ | — |
| $$j$$ | output column: which output feature, $$1 \ldots n_\text{out}$$ | — |
| $$T$$ | number of rows = batch size × sequence length | 4 × 1024 = 4096 |
| $$n_\text{in}, n_\text{out}$$ | number of input and output columns | 768 and 2304 |
| $$x$$ | input: one per value | `{4, 1024, 768}` |
| $$W$$ | weight: row $$j$$ holds the $$n_\text{in}$$ weights of output $$j$$; shared by all rows | `{2304, 768}` |
| $$b$$ | bias: one per output column, shared by all rows | `{2304}` |
| $$p$$ | the products: $$n_\text{out} \times n_\text{in}$$ per row | `{4, 1024, 2304, 768}`, never stored |
| $$z, y$$ | dot product and output: one per value | `{4, 1024, 2304}` |
| $$L$$ | the loss: one number, how wrong the model is | 1 |
| $$\partial$$ | partial derivative: change one variable, hold all others fixed | — |
| $$da$$ | short for $$\partial L / \partial a$$, for any variable $$a$$; same shape as $$a$$ | — |

*Table 1. $$dy$$ is given: it is what the layer after the linear layer sends back. The job is to find
$$db$$, $$dW$$ and $$dx$$.*

Careful with one thing: $$W$$ has the **output** index first, $$W_{j,i}$$. That is how PyTorch's
`nn.Linear` keeps its weight, as `{out_features, in_features}` [3], and it is where the $$W^\top$$ of
the usual forward formula comes from (C4).

GPT-2 small has five kinds of linear layer:

| linear layer | $$n_\text{in}$$ | $$n_\text{out}$$ | how many | its input is the output of |
|---|---:|---:|---:|---|
| qkv | 768 | 2304 | 12 | the first LayerNorm of the block |
| attention output | 768 | 768 | 12 | attention |
| MLP up | 768 | 3072 | 12 | the second LayerNorm of the block |
| MLP down | 3072 | 768 | 12 | GELU |
| head (no bias; its weight is the token table `wte`) | 768 | 50304 | 1 | the last LayerNorm |

*Table 2. 49 linear layers. The last column matters in Section 7.*

### 1.3 The running example

Two rows of $$n_\text{in} = 4$$ numbers, and $$n_\text{out} = 4$$ outputs. The weight is chosen so
that every output is easy to work out by eye: output 1 copies $$x_1$$, output 2 copies $$x_2$$, output 3
adds the row up, and output 4 adds it with signs $$+ - + -$$:

$$
W_1 = [1,\ 0,\ 0,\ 0] \qquad W_2 = [0,\ 1,\ 0,\ 0] \qquad W_3 = [1,\ 1,\ 1,\ 1] \qquad W_4 = [1,\ -1,\ 1,\ -1]
\qquad\qquad b = [10,\ 20,\ 30,\ 40]
$$

where $$W_j$$ is row $$j$$ of $$W$$. Then:

| | row 1 | row 2 |
|---|---|---|
| input $$x$$ | $$[1,\ 2,\ 3,\ 4]$$ | $$[5,\ 6,\ 7,\ 8]$$ |
| F1: $$p_{1,i} = x_i W_{1,i}$$ | $$[1,\ 0,\ 0,\ 0]$$ | $$[5,\ 0,\ 0,\ 0]$$ |
| F1: $$p_{2,i}$$ | $$[0,\ 2,\ 0,\ 0]$$ | $$[0,\ 6,\ 0,\ 0]$$ |
| F1: $$p_{3,i}$$ | $$[1,\ 2,\ 3,\ 4]$$ | $$[5,\ 6,\ 7,\ 8]$$ |
| F1: $$p_{4,i}$$ | $$[1,\ -2,\ 3,\ -4]$$ | $$[5,\ -6,\ 7,\ -8]$$ |
| F2: $$z$$ | $$[1,\ 2,\ 10,\ -2]$$ | $$[5,\ 6,\ 26,\ -2]$$ |
| F3: $$y$$ | $$[11,\ 22,\ 40,\ 38]$$ | $$[15,\ 26,\ 56,\ 38]$$ |
| $$dy$$, from the next layer | $$[1,\ 0,\ 1,\ 0]$$ | $$[0,\ 0,\ 1,\ 0]$$ |

*Table 3. The example, forward. Each $$z_j$$ is the sum of its line of $$p$$: for row 2,
$$z_3 = 5 + 6 + 7 + 8 = 26$$. The loss cares about three outputs: $$y_{1,1}$$, $$y_{1,3}$$ and
$$y_{2,3}$$.*

### 1.4 Where each value is used

![A linear layer's forward for one row](/images/linear-backward/fig1-graph.svg)
*Figure 1. Each arrow is one use of a value. The backward pass walks the same arrows the other way, one
step per arrow. $$x_i$$ has $$n_\text{out}$$ arrows going out, one per output, so its gradient is a sum.
$$W$$ and $$b$$ are used again in every row, so theirs are sums over the rows.*

Figure 1 is the whole plan. The backward pass starts at $$y$$, where $$dy$$ is known, and walks every
arrow backwards, from right to left: B1 and B2 at F3, B3 at F2, B4 and B5 at F1.

## 2. Three rules

Every step below uses the same three rules as the LayerNorm note [2], and nothing else.

| rule | when | what it says |
|---|---|---|
| **chain rule** | $$a$$ affects $$L$$ through $$b$$ | $$da = db \cdot \partial b / \partial a$$ |
| **shared value: sum** | $$a$$ is used in many places (all rows, or all outputs of a row) | add one chain-rule term per place: $$da = \sum_k db_k \cdot \partial b_k / \partial a$$ |
| **used twice: add the parts** | $$a$$ is used by two different forward steps | each step gives one part; $$da$$ = part 1 + part 2 |

*Table 4. The three rules. A linear layer needs only the first two for its gradients. The third comes
back in Section 7, for one choice the kernel makes.*

And one habit: **each step looks only at its direct inputs.** F1 is $$p_{j,i} = x_i\, W_{j,i}$$, with
two direct inputs. B4 takes the derivative with respect to $$W_{j,i}$$ and holds $$x_i$$ fixed; B5 takes
it with respect to $$x_i$$ and holds $$W_{j,i}$$ fixed.

Whether a step has a sum comes only from where the variable is used in the forward pass, Figure 1. The
loss and the layers after decide only the values of $$dy$$.

## 3. Breaking it down: B1 to B5

Each step first works on one row (no index $$t$$), then adds the row index for the whole batch.

### 3.1 B1: the bias, $$db$$

**One row.** From F3, $$y_j = z_j + b_j$$:

$$
\begin{aligned}
\frac{\partial y_j}{\partial b_j}
  &= \frac{\partial z_j}{\partial b_j} + \frac{\partial b_j}{\partial b_j}
  && \text{derivative of a sum = sum of derivatives} \\
  &= 0 + 1 = 1
  && z_j \text{ does not contain } b_j \text{, so it is a constant} \\
db_j &= dy_j \cdot \frac{\partial y_j}{\partial b_j} = dy_j \cdot 1 = dy_j
  && \text{chain rule}
\end{aligned}
$$

**The whole batch.** The same $$b_j$$ is used in every row: $$y_{t,j} = z_{t,j} + b_j$$ for every
$$t$$. Changing $$b_j$$ changes $$y_{1,j}, y_{2,j}, \ldots, y_{T,j}$$ all at once, so there are $$T$$
paths to the loss, and by the shared-value rule:

$$
db_j = \frac{\partial L}{\partial y_{1,j}}\frac{\partial y_{1,j}}{\partial b_j} + \cdots + \frac{\partial L}{\partial y_{T,j}}\frac{\partial y_{T,j}}{\partial b_j}
     = \sum_{t=1}^{T} dy_{t,j} \cdot 1
     = \sum_{t=1}^{T} dy_{t,j}
$$

The sum of column $$j$$ of $$dy$$. This is the LayerNorm note's B1 again: a bias added to every row.

**Example.** Column 1 of $$dy$$ is $$[1, 0]$$, column 3 is $$[1, 1]$$, columns 2 and 4 are 0:

$$
db = [1 + 0,\ 0 + 0,\ 1 + 1,\ 0 + 0] = [1,\ 0,\ 2,\ 0]
$$

$$db_3 = 2$$: $$b_3$$ moves output 3 of both rows at once, and both rows want their output 3 changed.

### 3.2 B2: the dot product, $$dz$$

**One row.** From F3 again, this time with respect to $$z_j$$:

$$
\begin{aligned}
\frac{\partial y_j}{\partial z_j} &= 1 + 0 = 1
  && b_j \text{ is a constant} \\
dz_j &= dy_j \cdot 1 = dy_j
  && \text{chain rule}
\end{aligned}
$$

**The whole batch.** Unlike $$b_j$$, $$z_{t,j}$$ belongs to one row and one column. It is used in
**one place only**, $$y_{t,j}$$: one path, so no sum.

$$
dz_{t,j} = dy_{t,j}
$$

**Example.** $$dz = dy$$: row 1 $$[1, 0, 1, 0]$$, row 2 $$[0, 0, 1, 0]$$.

Adding a constant passes the gradient through unchanged: if $$z_j$$ goes up by 0.1, $$y_j$$ goes up by
exactly 0.1.

| step | variable | used in how many places | sum? | result |
|---|---|---|---|---|
| B1 | $$b_j$$ | every row | yes, over rows | one per column: `{2304}` |
| B2 | $$z_{t,j}$$ | only $$y_{t,j}$$ | **no** | one per value: `{4, 1024, 2304}` |

*Table 5. B1 gives the gradient the optimizer uses to update $$b$$. B2 starts the road back to $$dW$$
and $$dx$$.*

### 3.3 B3: the products, $$dp$$

**One row.** From F2, $$z_j = \sum_k p_{j,k}$$. The sum uses $$k$$ so it does not mix with the $$i$$ we
differentiate for:

$$
\begin{aligned}
\frac{\partial z_j}{\partial p_{j,i}}
  &= \frac{\partial}{\partial p_{j,i}}\left(p_{j,1} + \cdots + p_{j,i} + \cdots + p_{j,n_\text{in}}\right) \\
  &= 0 + \cdots + 1 + \cdots + 0 = 1
  && \text{every other } p_{j,k} \text{ is a constant} \\
dp_{j,i} &= dz_j \cdot 1 = dz_j
  && \text{chain rule, along the arrow F2}
\end{aligned}
$$

**The whole batch.** $$p_{t,j,i}$$ is used once, in its own row's $$z_{t,j}$$: no sum.

$$
dp_{t,j,i} = dz_{t,j}
$$

**Example.** Every product in $$z_j$$ gets the whole $$dz_j$$:

| | row 1 | row 2 |
|---|---|---|
| $$dp_{1,i}$$ | $$[1,\ 1,\ 1,\ 1]$$, from $$dz_{1,1} = 1$$ | $$[0,\ 0,\ 0,\ 0]$$ |
| $$dp_{2,i}$$ | $$[0,\ 0,\ 0,\ 0]$$ | $$[0,\ 0,\ 0,\ 0]$$ |
| $$dp_{3,i}$$ | $$[1,\ 1,\ 1,\ 1]$$, from $$dz_{1,3} = 1$$ | $$[1,\ 1,\ 1,\ 1]$$, from $$dz_{2,3} = 1$$ |
| $$dp_{4,i}$$ | $$[0,\ 0,\ 0,\ 0]$$ | $$[0,\ 0,\ 0,\ 0]$$ |

*Table 6. $$dp$$ in the example: each line is its $$dz_j$$, copied 4 times.*

B3 is B1 turned round. In B1, **one** value was used in many places, and its gradient **summed** many
terms. In B3, **many** values are added into one, and each gets the **same** gradient, copied:

| in the forward | in the backward | where |
|---|---|---|
| one value used in many places | its gradient is a sum, one term per place | $$b_j$$ (B1), and soon $$W_{j,i}$$ (B4) and $$x_i$$ (B5) |
| many values added into one | each gets the same gradient | the $$n_\text{in}$$ products of $$z_j$$ (B3) |

*Table 7. A copy in the forward is a sum in the backward, and a sum in the forward is a copy in the
backward.*

### 3.4 B4: the weight, $$dW$$

**One row.** From F1, $$p_{j,i} = x_i\, W_{j,i}$$, with respect to $$W_{j,i}$$, holding the other direct
input $$x_i$$ fixed (Section 2):

$$
\begin{aligned}
\frac{\partial p_{j,i}}{\partial W_{j,i}} &= x_i \cdot \frac{\partial W_{j,i}}{\partial W_{j,i}} = x_i \cdot 1 = x_i
  && x_i \text{ is a constant factor} \\
dW_{j,i} &= dp_{j,i} \cdot x_i
  && \text{chain rule, along the arrow F1}
\end{aligned}
$$

Within one row, $$W_{j,i}$$ is used once: it joins input $$i$$ to output $$j$$, and nothing else.

**The whole batch.** $$W_{j,i}$$ is shared by every row: $$p_{t,j,i} = x_{t,i}\, W_{j,i}$$ for every
$$t$$. That is $$T$$ paths, each with its **own** $$x_{t,i}$$:

$$
dW_{j,i} = \sum_{t=1}^{T} dp_{t,j,i} \cdot \frac{\partial p_{t,j,i}}{\partial W_{j,i}} = \sum_{t=1}^{T} dp_{t,j,i}\, x_{t,i}
$$

**The difference from B1.** In B1 the local derivative was 1 in every row, so it disappeared. Here it
is $$x_{t,i}$$, different in every row, so it stays inside the sum. This is the LayerNorm note's B2,
$$d\gamma$$, again: a weight shared by every row, multiplying a value that is different in every row.

**Example.** $$dW_{j,i} = dp_{1,j,i}\, x_{1,i} + dp_{2,j,i}\, x_{2,i}$$, with $$dp$$ from Table 6:

| row of $$dW$$ | row 1's part + row 2's part | result |
|---|---|---|
| $$dW_1$$ | $$[1, 1, 1, 1] \odot [1, 2, 3, 4] + 0$$ | $$[1,\ 2,\ 3,\ 4]$$ |
| $$dW_2$$ | $$0 + 0$$ | $$[0,\ 0,\ 0,\ 0]$$ |
| $$dW_3$$ | $$[1, 1, 1, 1] \odot [1, 2, 3, 4] + [1, 1, 1, 1] \odot [5, 6, 7, 8]$$ | $$[6,\ 8,\ 10,\ 12]$$ |
| $$dW_4$$ | $$0 + 0$$ | $$[0,\ 0,\ 0,\ 0]$$ |

*Table 8. $$dW$$ in the example. $$\odot$$ multiplies value by value:
$$[a_1, a_2] \odot [c_1, c_2] = [a_1 c_1,\ a_2 c_2]$$. Row 3 has a part from each row of the input, as
$$db_3$$ did.*

$$W$$ multiplies $$x$$: where $$x_{t,i}$$ is large, a small change of $$W_{j,i}$$ moves $$y_{t,j}$$ a
lot; where $$x_{t,i} = 0$$, changing $$W_{j,i}$$ does nothing to that row. So $$dW_3 = [6, 8, 10, 12]$$
grows with the inputs: $$W_{3,4}$$ multiplies the largest ones, 4 and 8.

### 3.5 B5: the input, $$dx$$

**One row.** From F1 again, now with respect to $$x_i$$, holding $$W_{j,i}$$ fixed:

$$
\frac{\partial p_{j,i}}{\partial x_i} = \frac{\partial x_i}{\partial x_i} \cdot W_{j,i} = W_{j,i}
$$

The difference from B4 is where $$x_i$$ is used. One $$x_i$$ goes into **every** output of the row:
$$p_{1,i} = x_i W_{1,i},\ p_{2,i} = x_i W_{2,i},\ \ldots,\ p_{n_\text{out},i} = x_i W_{n_\text{out},i}$$.
That is $$n_\text{out}$$ paths, so by the shared-value rule:

$$
dx_i = \frac{\partial L}{\partial p_{1,i}}\frac{\partial p_{1,i}}{\partial x_i} + \cdots + \frac{\partial L}{\partial p_{n_\text{out},i}}\frac{\partial p_{n_\text{out},i}}{\partial x_i}
     = \sum_{j=1}^{n_\text{out}} dp_{j,i}\, W_{j,i}
$$

**The whole batch.** $$x_{t,i}$$ is used in every output of row $$t$$, and in no other row, so the sum
is over the **outputs** $$j$$, not over the rows:

$$
dx_{t,i} = \sum_{j=1}^{n_\text{out}} dp_{t,j,i}\, W_{j,i}
$$

This is the gradient sent to the layer before, `{4, 1024, 768}`, the shape of the input.

| step | variable | shared by | sum over |
|---|---|---|---|
| B1, B4 | $$b_j$$, $$W_{j,i}$$ | all rows | rows $$t$$ |
| B2, B3 | $$z_{t,j}$$, $$p_{t,j,i}$$ | nothing: one value | no sum |
| B5 | $$x_{t,i}$$ | all outputs of its row | outputs $$j$$ |

*Table 9. Sum over whatever the variable is shared across.*

**Example.** In row 1, only $$dp_{1,i}$$ and $$dp_{3,i}$$ are not 0 (Table 6):

$$
dx_1 = [1, 1, 1, 1] \odot W_1 + [1, 1, 1, 1] \odot W_3 = [1,\ 0,\ 0,\ 0] + [1,\ 1,\ 1,\ 1] = [2,\ 1,\ 1,\ 1]
$$

and in row 2, only $$dp_{3,i}$$: $$dx_2 = [1, 1, 1, 1] \odot W_3 = [1,\ 1,\ 1,\ 1]$$.

Each output sends its gradient back along its own row of weights, and each input collects from all the
outputs of its row. Output 3 adds the row up, so it sends 1 back to every input; output 1 copies
$$x_1$$, so it sends 1 back to $$x_1$$ alone. That is why $$dx_{1,1} = 2$$: $$x_{1,1}$$ reaches the loss
through two outputs.

### 3.6 All five steps

| step | gradient of | formula | one per | example |
|---|---|---|---|---|
| B1 | bias | $$db_j = \sum_t dy_{t,j}$$ | output column | $$[1,\ 0,\ 2,\ 0]$$ |
| B2 | dot product | $$dz_{t,j} = dy_{t,j}$$ | output value | $$[1, 0, 1, 0]$$, $$[0, 0, 1, 0]$$ |
| B3 | products | $$dp_{t,j,i} = dz_{t,j}$$ | product | Table 6 |
| B4 | weight | $$dW_{j,i} = \sum_t dp_{t,j,i}\, x_{t,i}$$ | weight | Table 8 |
| B5 | input | $$dx_{t,i} = \sum_j dp_{t,j,i}\, W_{j,i}$$ | input value | $$[2,\ 1,\ 1,\ 1]$$, $$[1,\ 1,\ 1,\ 1]$$ |

*Table 10. The backward pass, broken down. B4 and B5 both need B3's $$dp$$, and nothing from each
other.*

## 4. Rebuilding: C1 to C5

Table 10 is correct but costly. Run as it stands, it would keep $$dp$$: one value per product,
$$768 \times 2304 = 1{,}769{,}472$$ per row of the qkv layer, 7.2 billion for its 4096 rows, 29 GB in
float32. This section folds $$dp$$ away, then reads the sums that are left as matrix products.

**C1: $$dp$$ in $$dy$$.** Put B2 into B3:

$$
dp_{t,j,i} = dz_{t,j} = dy_{t,j}
$$

The index $$i$$ is gone from the right side: every product of output $$j$$ gets the same
$$dy_{t,j}$$. Nothing per product has to be stored. Example: row 1's $$dp_{3,i} = [1, 1, 1, 1]$$ is
$$dy_{1,3} = 1$$ four times, as in Table 6.

**C2: $$dW$$ in $$dy$$.** Put C1 into B4:

$$
dW_{j,i} = \sum_{t=1}^{T} dy_{t,j}\, x_{t,i}
$$

Example: $$dW_{3,2} = dy_{1,3}\, x_{1,2} + dy_{2,3}\, x_{2,2} = 1 \times 2 + 1 \times 6 = 8$$, as in
Table 8.

**C3: $$dx$$ in $$dy$$.** Put C1 into B5:

$$
dx_{t,i} = \sum_{j=1}^{n_\text{out}} dy_{t,j}\, W_{j,i}
$$

Example: $$dx_{1,1} = 1 \times 1 + 0 \times 0 + 1 \times 1 + 0 \times 1 = 2$$, as in B5.

![Where the sums run](/images/linear-backward/fig2-sums.svg)
*Figure 2. The sums of C3, C2 and B1, in the example. Each coloured cell is one term.*

**C4: reading a sum as a product.** A matrix product $$C = A\,B$$ is, one value at a time,

$$
C_{r,c} = \sum_k A_{r,k}\, B_{k,c}
$$

So a sum is a matrix product **when the summed index is the second index of the left factor and the
first of the right factor**. A factor that has it the other way round is read turned round, with
$$(A^\top)_{r,k} = A_{k,r}$$. Apply it to the forward, $$y_{t,j} = \sum_i x_{t,i}\, W_{j,i} + b_j$$: the
summed index $$i$$ is second in $$x$$, as it should be, but second in $$W$$ too, where it should be
first. So $$W$$ is read turned round:

$$
y = x\, W^\top + b
$$

This is where the forward's $$\top$$ comes from: from $$W$$ keeping the output index first.

**C5: the three gradients as products.** The same reading for C3, C2 and B1:

| gradient | the sum | summed index | where it sits | product |
|---|---|---|---|---|
| input | $$\sum_j dy_{t,j}\, W_{j,i}$$ | $$j$$ | 2nd in $$dy$$, 1st in $$W$$: both right | $$dx = dy\, W$$ |
| weight | $$\sum_t dy_{t,j}\, x_{t,i}$$ | $$t$$ | 1st in $$dy$$: wrong, so $$dy^\top$$; 1st in $$x$$: right | $$dW = dy^\top x$$ |
| bias | $$\sum_t dy_{t,j}$$ | $$t$$ | 1st in $$dy$$: wrong, so $$dy^\top$$ | $$db = dy^\top \mathbf{1}$$ |

*Table 11. $$\mathbf{1}$$ is a column of $$T$$ ones: multiplying by it adds up each row of $$dy^\top$$,
which is each column of $$dy$$.*

$$
\boxed{\;dx = dy\, W, \qquad dW = dy^\top x, \qquad db = dy^\top \mathbf{1}\;}
$$

For the whole batch, the batch and sequence dims are laid flat into the $$T$$ rows: $$x$$ is
`{4096, 768}` and $$dy$$ is `{4096, 2304}`.

**Example**, straight from the three products:

| | from C5 | the five steps gave |
|---|---|---|
| $$dx = dy\, W$$ | row 1: $$1 \cdot W_1 + 1 \cdot W_3 = [2,\ 1,\ 1,\ 1]$$; row 2: $$1 \cdot W_3 = [1,\ 1,\ 1,\ 1]$$ | the same (B5) |
| $$dW = dy^\top x$$ | row $$j$$ is $$\sum_t dy_{t,j}\, x_t$$: $$x_1$$, $$0$$, $$x_1 + x_2$$, $$0$$ | Table 8 |
| $$db = dy^\top \mathbf{1}$$ | the column sums of $$dy$$: $$[1,\ 0,\ 2,\ 0]$$ | the same (B1) |

*Table 12. The rebuilt products give the same numbers as the five steps.*

**The shapes check it.** A gradient has the shape of its variable, and the products give exactly
that:

| | left | right | result | the shape of |
|---|---|---|---|---|
| $$y = x\, W^\top$$ | `{4096, 768}` | `{768, 2304}` | `{4096, 2304}` | $$y$$ |
| $$dx = dy\, W$$ | `{4096, 2304}` | `{2304, 768}` | `{4096, 768}` | $$x$$ |
| $$dW = dy^\top x$$ | `{2304, 4096}` | `{4096, 768}` | `{2304, 768}` | $$W$$ |
| $$db = dy^\top \mathbf{1}$$ | `{2304, 4096}` | `{4096}` | `{2304}` | $$b$$ |

*Table 13. The qkv layer. The shapes are a quick way to remember the products, but they are a check,
not a reason: the reason is Table 11.*

**What each product means.**

| product | what it is | why |
|---|---|---|
| $$dx = dy\, W$$ | each row of $$dy$$, sent back through $$W$$: a sum of rows of $$W$$ | the forward goes from $$x$$ to $$y$$ through $$W^\top$$; the gradient comes back through $$W$$ (B5) |
| $$dW = dy^\top x$$ | for each output $$j$$, the input rows added up, each scaled by $$dy_{t,j}$$ | $$W$$ is shared by every row, and multiplies a different $$x$$ in each (B4) |
| $$db = dy^\top \mathbf{1}$$ | the column sums of $$dy$$ | $$b$$ is shared by every row, and multiplies nothing (B1) |

*Table 14. The three products, read back into words.*

## 5. Two checks

**The bias is a weight on an input of 1.** Add a 1 to the end of every input row, and put $$b$$ into
$$W$$ as its last column:

$$
x'_t = [x_t,\ 1] \qquad W'_j = [W_j,\ b_j] \qquad\Rightarrow\qquad x'_t \cdot W'_j = x_t \cdot W_j + b_j = y_{t,j}
$$

The layer is now one without a bias. C2 gives the gradient of the new last column of $$W'$$, whose
input is always 1:

$$
dW'_{j,\,n_\text{in}+1} = \sum_t dy_{t,j} \cdot 1 = \sum_t dy_{t,j} = db_j
$$

which is B1. In the example, $$x'_1 = [1, 2, 3, 4, 1]$$ and $$x'_2 = [5, 6, 7, 8, 1]$$, and the last
column of $$dW' = dy^\top x'$$ is $$[1, 0, 2, 0] = db$$. So the bias needs no rule of its own: B1 is B4
with an input that is always 1.

**Stretching $$x$$ or stretching $$W$$ changes the loss by the same amount.** $$z = x\, W^\top$$ is
linear in $$x$$ and in $$W$$. Multiply $$x$$ by $$1 + \epsilon$$ and $$z$$ becomes
$$(1 + \epsilon)\, z$$; multiply $$W$$ by $$1 + \epsilon$$ instead, and $$z$$ becomes the same
$$(1 + \epsilon)\, z$$. The loss sees the same change, so the gradients must agree:

$$
\sum_{t,i} x_{t,i}\, dx_{t,i} = \sum_{j,i} W_{j,i}\, dW_{j,i} = \sum_{t,j} z_{t,j}\, dy_{t,j}
$$

From C3, with the sums swapped:

$$
\sum_{t,i} x_{t,i}\, dx_{t,i} = \sum_{t,i} x_{t,i} \sum_j dy_{t,j}\, W_{j,i}
                              = \sum_{t,j} dy_{t,j} \underbrace{\sum_i x_{t,i}\, W_{j,i}}_{=\, z_{t,j}}
$$

and from C2 the same way for $$W$$. In the example, all three are 37:

| sum | worked out | total |
|---|---|---|
| $$\sum x \cdot dx$$ | row 1: $$1 \times 2 + 2 \times 1 + 3 \times 1 + 4 \times 1 = 11$$; row 2: $$5 + 6 + 7 + 8 = 26$$ | 37 |
| $$\sum W \cdot dW$$ | $$W_1 \cdot dW_1 = 1$$; $$W_3 \cdot dW_3 = 6 + 8 + 10 + 12 = 36$$; rows 2 and 4 of $$dW$$ are 0 | 37 |
| $$\sum z \cdot dy$$ | row 1: $$z_1 + z_3 = 1 + 10 = 11$$; row 2: $$z_3 = 26$$ | 37 |

*Table 15. The second check in the example. Both checks hold for any $$x$$, $$W$$, $$b$$ and $$dy$$.*

## 6. Checked against PyTorch

| test | compared | largest difference |
|---|---|---|
| the example of Section 1.3 | every number of Tables 3, 6, 8, 10, 12 and 15, and the first check, with `F.linear` and autograd | none: the same whole numbers |
| 16 random rows, 768 in, 2304 out; $$x$$, $$b$$, $$dy$$ from $$\mathcal{N}(0, 1)$$, $$W$$ from $$0.02\,\mathcal{N}(0, 1)$$ | the five steps B1 to B5, with $$p$$ and $$dp$$ kept, against autograd | $$db$$: 0; $$dx$$: $$2.7 \times 10^{-15}$$; $$dW$$: $$7.1 \times 10^{-15}$$ |
| 256 random rows, the same sizes | C2 and C3, written index by index (`einsum`), against autograd | 0 and 0 |
| the same | C5's products $$dy\, W$$, $$dy^\top x$$ and $$dy^\top \mathbf{1}$$ | 0, 0 and $$3.6 \times 10^{-14}$$ |
| the same | the first check: the last column of $$dW'$$ against $$db$$ | $$3.6 \times 10^{-14}$$ |
| the same | the second check: the three sums, each about 502.589 | $$1.0 \times 10^{-15}$$ of the sum |
| input `{4, 64, 768}`, three dims as in GPT-2 | $$dx$$ against $$dy\, W$$; $$dW$$ against $$dy^\top x$$ and $$db$$ against the column sums, with the 256 rows laid flat | 0, 0 and 0 |
| one weight, $$W_{6,8}$$, 256 random rows | C2 against a finite difference, $$(L(W + \epsilon) - L(W - \epsilon)) / 2\epsilon$$ with $$\epsilon = 10^{-6}$$ | $$-9.855569485$$ against $$-9.855569488$$: $$2.9 \times 10^{-9}$$, the finite difference's own error |

*Table 16. PyTorch 2.12 [4], `torch.nn.functional.linear` and `backward`, in double precision.
Differences near $$10^{-15}$$ are rounding; a difference of 0 is equal to the last bit.*

## 7. What the rebuilt formulas give the kernel

The five steps and the three products give the same numbers, but they are very different programs.

| | broken down (Table 10) | rebuilt (C5) |
|---|---|---|
| per product $$p_{t,j,i}$$ | $$p$$ and $$dp$$: 7.2 billion values each for the qkv layer | none: C1 folded them away |
| per output value | $$z$$, $$dz$$ | none: $$dz = dy$$ |
| the sums | B4 over rows, B5 over outputs, B1 over rows | the same sums, as the inner dims of three products |
| saved by the forward and needed | $$x$$, $$W$$ | the same: $$x$$, $$W$$ |

*Table 17. In float32, the qkv layer's $$p$$ alone would be 29 GB, for one of GPT-2's 49 linear layers.*

The kernel in my trainer [1] (`linear.cuh`) is C4 and C5 written out:

| in C4, C5 | in the kernel | where |
|---|---|---|
| $$y = x\, W^\top + b$$ | `output` | `linear_matmul<false, true>`: $$b$$ added as each value is stored |
| $$dy$$ | `output_grad` | given by the layer after |
| $$dx = dy\, W$$ | `input_grad` | `linear_matmul<false, false>`: written over (`=`) |
| $$dW = dy^\top x$$ | `weight_grad` | `linear_matmul<true, false>`: added onto (`+=`) |
| $$db = dy^\top \mathbf{1}$$ | `bias_grad` | `linear_bias_grad_kernel`: each block owns 32 columns and sums all their rows: added onto (`+=`) |

*Table 18. The two flags of `linear_matmul<LEFT_TRANSPOSED, RIGHT_TRANSPOSED>` say whether the left and
right of `product = left · right` are read turned round. Nothing is ever turned round in memory:
$$W^\top$$ and $$dy^\top$$ are only read in a different order.*

**One kernel for all three products.** Each block makes a 128 × 128 square of the product and walks the
inner dim 8 values at a time through shared memory; each of its 256 threads keeps an 8 × 8 square of
sums in registers. These are the sizes of Boehm's kernel 5 [6]. The inner dim is the index C5 sums
over: $$n_\text{in}$$ for $$y$$, $$n_\text{out}$$ for $$dx$$, and the rows for $$dW$$. All three
products have the same $$T \cdot n_\text{in} \cdot n_\text{out}$$ multiply-adds, so the backward costs
twice the forward.

**`=` for $$dx$$, `+=` for $$dW$$ and $$db$$** comes from the rules of Section 2:

| gradient | the variable is used | so |
|---|---|---|
| $$dx$$ | by this linear layer alone (Table 2's last column): one use | one gradient, nothing to add: written over |
| $$dW$$, $$db$$ | by every micro-batch of a training step | the shared-value rule, across calls: each call adds its part |
| $$dW$$ of the head | by the head and by the encoder, both reading `wte` | used twice: add the parts |

*Table 19. So $$dW$$ and $$db$$ are set to 0 once per step, and every call adds onto them.*

The kernel never works out $$p$$, $$dp$$ or $$dz$$: C1 folded them away. All that is left from the five
steps is where the sums run: B5's sum over outputs is the inner dim of $$dx = dy\, W$$, B4's sum over
rows is the inner dim of $$dW = dy^\top x$$, and B1's is the column sums. Why squares of that size, and
how fast the kernel is next to cuBLAS, are for a later entry: nothing here is timed.

## 8. What to remember

1. **One step per arrow.** The backward pass walks Figure 1's arrows backwards. Each step uses only the
   chain rule, and holds its other direct inputs fixed.
2. **A sum comes from sharing.** $$W$$ and $$b$$ are shared by every row, so their gradients sum over
   rows. $$x_i$$ is shared by every output of its row, so its gradient sums over outputs. A value used in
   one place has no sum.
3. **A sum in the forward is a copy in the backward.** $$z_j$$ adds $$n_\text{in}$$ products, and each
   gets the whole $$dz_j$$. So $$dp = dy$$, and nothing per product has to be stored.
4. **A $$\top$$ comes from the summed index.** Write the sum, and see where its index sits in each
   factor: second in the left and first in the right, or read turned round. The shapes check it.
5. **The formulas say what a linear layer is.** The bias is a weight on an input of 1, and stretching
   $$x$$ or $$W$$ changes the loss the same: $$\sum x \cdot dx = \sum W \cdot dW = \sum z \cdot dy$$.

## References

1. D. B. Song. *gpt2-small*, branch `c-cuda-trainer`: GPT-2 small pre-training in C and CUDA.
   [github.com/ducbachsong/gpt2-small](https://github.com/ducbachsong/gpt2-small/tree/c-cuda-trainer)
2. D. B. Song. *LayerNorm backward, broken down and rebuilt: nine chain-rule steps and one formula.* 2026.
   [/posts/layer-norm-backward/](/posts/layer-norm-backward/)
3. PyTorch. `torch.nn.Linear`.
   [docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)
4. PyTorch. `torch.nn.functional.linear`.
   [docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.linear.html)
5. A. Karpathy. *llm.c*, `train_gpt2.c`: `matmul_forward` and `matmul_backward`, a linear layer forward
   and backward in C. [github.com/karpathy/llm.c](https://github.com/karpathy/llm.c/blob/master/train_gpt2.c)
6. S. Boehm. *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog.* 2022.
   Kernel 5: 2D blocktiling, 128 × 128 per block, 8 inner values at a time, 8 × 8 per thread.
   [siboehm.com/articles/22/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM)
