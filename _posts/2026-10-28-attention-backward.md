---
title: "Attention backward, broken down and rebuilt: nine chain-rule steps, a softmax over a whole row and four matrix products"
date: 2026-10-28
modified: 2026-10-29
version: 2
permalink: /posts/attention-backward/
excerpt: "Causal self-attention is how each token of GPT-2 takes in the tokens before it: a query asks, keys answer, and values are passed on, mixed by weights that add up to 1. Its backward pass is usually given as four matrix products and a softmax line. This note takes one head apart into six forward steps and nine backward steps, each derived with the chain rule written out in full, and works one head of 3 tokens through all of them. The softmax here gets a gradient on every weight, not only the target's as in the classifier, and its backward comes out as one line: ds = a (da − row_dot). Then it puts the steps back together into the four products a GPU runs with cuBLAS, and shows where each ᵀ and each sum comes from."
tags:
  - attention
  - softmax
  - backpropagation
  - gpt-2
  - notes
---

**What this is.** Every block of GPT-2 small has one attention layer. In it, each token looks at the
tokens before it and takes in a mix of what they say. Its backward pass is usually written as four
matrix products and one softmax line:

$$
dV = A^\top dY \qquad dA = dY\, V^\top \qquad dS = A \odot \big(dA - \text{rowsum}(A \odot dA)\big) \qquad dQ = \tfrac{1}{\sqrt{d}}\, dS\, K \qquad dK = \tfrac{1}{\sqrt{d}}\, dS^\top Q
$$

Those lines are correct, and they are what the kernel in my GPT-2 trainer [1] runs, but they hide
where each sum and each $$\top$$ comes from, and why the softmax's gradient has that shape. This note
**breaks it down** into nine small steps, B1 to B9, one for each arrow of the forward pass, each derived
with the chain rule written out in full. Then it **rebuilds** the lines from those steps, in two moves,
C1 and C2. One head of 3 tokens goes through every step, so each formula has a number next to it.

**Sources.** The derivation is worked out by hand, with the same three rules as the notes on
[LayerNorm](/posts/layer-norm-backward/) [2], [the linear layer](/posts/linear-backward/) [3],
[GELU](/posts/gelu-backward/) [4], [the classifier](/posts/classifier-backward/) [5] and
[the residual add](/posts/residual-backward/) [6]. Attention is from Vaswani et al. [7], with GPT-2's
causal mask [8]; the kernel follows llm.c's [9], with the online softmax of Milakov and
Gimelshein [10].

* TOC
{:toc}

## 1. The forward pass

### 1.1 What attention does: a query asks, keys answer, values are passed on

Take one token, token 2 of a sentence. To work out what it means here, it may look at itself and at
the tokens before it, 0 and 1, never after. Attention gives every token three rows of numbers, made by
the qkv linear layer from the token's row:

| row | what it is for | example, token 2 |
|---|---|---|
| query $$q$$ | what this token looks for | $$q_2 = [2,\ 2,\ 0,\ 0]$$ |
| key $$k$$ | what this token offers to the tokens that look at it | $$k_2 = [1,\ 1,\ 0,\ 0]$$ |
| value $$v$$ | what this token passes on to the tokens that look at it | $$v_2 = [0,\ 0,\ 1,\ 0]$$ |

*Table 1. Three rows per token, here with 4 numbers each.*

Token 2 then does three things:

1. **Scores.** It compares its query with every key it may see, a dot product each: a large score means
   "this token has what I look for". $$q_2 \cdot k_0 = 2$$, $$q_2 \cdot k_1 = 2$$, $$q_2 \cdot k_2 = 4$$, each
   then divided by $$\sqrt{4} = 2$$: scores 1, 1, 2.
2. **Weights.** Softmax turns the scores into weights between 0 and 1 that add up to 1:
   0.2119416, 0.2119416, 0.5761169. Token 2 looks at itself most.
3. **Mix.** It takes that mix of the values: $$0.2119416\, v_0 + 0.2119416\, v_1 + 0.5761169\, v_2$$.

The mix is token 2's output: what it takes in from the tokens it can see. GPT-2 small does this in
**12 heads** at once, each with its own 64 columns of $$q$$, $$k$$ and $$v$$, so each head can look for
something different. The 12 outputs of 64 are put side by side, 768 per token.

### 1.2 The names

| symbol | meaning | shape in GPT-2 small |
|---|---|---|
| `{4, 12, 1024, 64}` | how a shape reads: 4 sequences in a batch, 12 heads, $$T$$ tokens, $$d$$ columns | — |
| $$T$$ | tokens per sequence | 1024 |
| $$t$$ | the token that looks: a row | — |
| $$j$$ | the token looked at: a column, $$j \le t$$ | — |
| $$d$$ | head size: columns per head | 64 |
| $$c$$ | one of the head's $$d$$ columns | — |
| $$h$$ | one of the 12 heads | — |
| qkv | the qkv linear's output: per token, a $$q$$, a $$k$$ and a $$v$$ part of $$768 = 12 \times 64$$ columns each | `{4, 1024, 2304}` |
| $$q, k, v$$ | query, key, value: per head, one row of $$d$$ per token | `{4, 12, 1024, 64}` each |
| $$s, a$$ | scores and weights: one per pair $$j \le t$$ | `{4, 12, 1024, 1024}` |
| $$m_t, S_t$$ | row $$t$$'s largest score, and the sum of its $$e$$ | one per row |
| $$e$$ | $$\exp(s - m)$$: one per pair $$j \le t$$ | — |
| $$y$$ | the head's output: one row of $$d$$ per token | `{4, 12, 1024, 64}` |
| $$Q, K, V, Y$$ | one head's $$q, k, v, y$$ rows stacked as a matrix: row $$t$$ of $$Q$$ is $$q_t$$ | `{1024, 64}` per head |
| $$S, A$$ | one head's $$s$$ and $$a$$ as a matrix: row $$t$$, column $$j$$ is $$s_{tj}$$, $$a_{tj}$$; the cells after the token ($$j > t$$) are masked. A capital with no index: not $$S_t$$, the sum | `{1024, 1024}` per head |
| $$X^\top$$ | matrix $$X$$ read turned round: row $$j$$, column $$t$$ of $$X^\top$$ is row $$t$$, column $$j$$ of $$X$$ | — |
| $$\odot$$, rowsum | multiply two matrices of the same shape place by place; add up each row | — |
| $$L$$ | the loss: one number, how wrong the model is | 1 |
| $$\partial$$ | partial derivative: change one variable, hold all others fixed | — |
| $$da$$ | short for $$\partial L / \partial a$$, for any variable $$a$$; same shape as $$a$$. So $$dS_t$$ is one number per row, the sum's gradient, and $$dS$$ is the matrix of $$ds$$ | — |

*Table 2. $$dy$$ is given: it is what the layer after sends back, cut into heads (B1). The job is to
find $$dq$$, $$dk$$ and $$dv$$. Attention has no weight and no bias of its own: those are in the qkv
linear before it and the output linear after it.*

Heads never mix, and nor do the 4 sequences of the batch: everything below is for one head of one
sequence, and GPT-2 runs $$4 \times 12 = 48$$ of them side by side.

### 1.3 Six steps for one head

For one head, with $$d$$ columns per head ($$d = 64$$ in GPT-2), token $$t$$ and an earlier token
$$j \le t$$:

$$
\begin{aligned}
&\text{F1:} \qquad q_t,\ k_t,\ v_t = \text{the head's columns of qkv row } t && \text{split: only moves values} \\
&\text{F2:} \qquad s_{tj} = \frac{q_t \cdot k_j}{\sqrt{d}} = \frac{1}{\sqrt{d}} \sum_{c=1}^{d} q_t[c]\, k_j[c] && \text{the score of token } j \text{ for token } t \\
&\text{F3:} \qquad \text{only } j \le t && \text{the mask: a token cannot see later tokens} \\
&\text{F4:} \qquad m_t = \max_{j \le t} s_{tj}, \quad e_{tj} = \exp(s_{tj} - m_t), \quad S_t = \sum_{j \le t} e_{tj}, \quad a_{tj} = \frac{e_{tj}}{S_t} && \text{softmax over the row} \\
&\text{F5:} \qquad y_t = \sum_{j \le t} a_{tj}\, v_j && \text{the mix: } d \text{ numbers} \\
&\text{F6:} \qquad \text{output row } t = \text{the 12 heads' } y_t \text{ side by side} && \text{merge: only moves values}
\end{aligned}
$$

F4 is the softmax of the classifier note [5], step by step: the largest score $$m_t$$ is taken away
first so that exp cannot overflow, and it changes no weight. Why $$\sqrt{d}$$ in F2: a dot product of
$$d$$ random numbers grows like $$\sqrt{d}$$, so dividing by it keeps the scores near 1 for any $$d$$,
and the softmax away from weights that are all 0 and one 1 [7].

**The steps, to play with.** The pictures below are the steps above with numbers in them. Tokens 0 to 2 are the running example of Section 1.4; tokens 3 to 7 are added so the mask's staircase shows, and the buttons switch between the two. Every number is worked out in the page, with the formulas above.

<script src="/assets/js/attention-widgets.js" defer></script>

<div class="aw" data-widget="split">This picture needs JavaScript.</div>

*Interactive 1. F1 and F6 at GPT-2's real sizes. Each head takes the same 64 columns out of the q, k and v parts of every qkv row, and its output goes back to the same 64 columns of the output row. Nothing is worked out: values only move.*

<div class="aw" data-widget="F2">This picture needs JavaScript.</div>

*Interactive 2. F2. Score $$s_{tj}$$ is row $$t$$ of $$Q$$ dotted with row $$j$$ of $$K$$, over the head's 4 columns, divided by $$\sqrt{4} = 2$$. $$K$$ is used by rows, so in the product it is read turned round, $$K^\top$$. The striped cells are after the token: cuBLAS works them out too, and F3 masks them.*

<div class="aw" data-widget="softmax-forward">This picture needs JavaScript.</div>

*Interactive 3. F3 and F4 on one row. Only the columns up to $$t$$ take part. The largest score $$m_t$$ is taken away before exp, and adding one number to every score of the row moves the scores and $$m_t$$ together, so no weight changes.*

<div class="aw" data-widget="F5">This picture needs JavaScript.</div>

*Interactive 4. F5. **Row $$t$$ of $$A$$ is the only row used** for $$y_t$$: its lit cells, columns 0 to $$t$$, are the weights, and the striped cells after it are the masked future, 0. **The lit rows of $$V$$** are the ones those weights multiply; rows after $$t$$ get weight 0 and add nothing. **The shape line** is $$(1 \times (t+1)) \cdot ((t+1) \times 4) = 1 \times 4$$: the shared length grows with $$t$$, but **$$y_t$$ is always one row of 4 numbers**, 64 in GPT-2, however many tokens are mixed. $$A$$'s staircase is the lower triangle the mask makes: row 0 has 1 visible cell, row 7 has 8. The GPU works out the full $$A V$$ in one go, and the zeros above the diagonal make it equal to the cut version.*

### 1.4 The running example

One head of 3 tokens with $$d = 4$$, so $$\sqrt{d} = 2$$. The rows are chosen so every number can be
worked out by eye; each $$v$$ has one 1, so the output shows the weights themselves.

| token | $$q$$ | $$k$$ | $$v$$ |
|---|---|---|---|
| 0 | 2 0 0 0 | 1 0 0 0 | 1 0 0 0 |
| 1 | 0 2 0 0 | 0 1 0 0 | 0 1 0 0 |
| 2 | 2 2 0 0 | 1 1 0 0 | 0 0 1 0 |

*Table 3. The example's rows.*

| step | token 0 | token 1 | token 2 |
|---|---|---|---|
| $$q_t \cdot k_j$$, $$j \le t$$ | 2 | 0, 2 | 2, 2, 4 |
| F2: $$s_{tj}$$, divided by 2 | 1 | 0, 1 | 1, 1, 2 |
| F4: $$m_t$$ | 1 | 1 | 2 |
| F4: $$e_{tj} = \exp(s_{tj} - m_t)$$ | 1 | 0.3678794, 1 | 0.3678794, 0.3678794, 1 |
| F4: $$S_t$$ | 1 | 1.3678794 | 1.7357589 |
| F4: $$a_{tj} = e_{tj} / S_t$$ | **1** | **0.2689414, 0.7310586** | **0.2119416, 0.2119416, 0.5761169** |
| F5: $$y_t$$ | 1 0 0 0 | 0.2689414 0.7310586 0 0 | 0.2119416 0.2119416 0.5761169 0 |

*Table 4. The example, forward. Token 0 sees only itself, so its weight is 1 and it passes on its own
$$v$$.*

For the backward, take $$dy_t = [1,\ 0,\ 0,\ 0]$$ for every token: the loss wants column 0 of every
output up, which is each token's weight on token 0. And a stand-in loss whose gradient is exactly that
$$dy$$, $$L = \sum_t dy_t \cdot y_t$$: here the sum of column 0 of $$y$$,
$$L = 1 + 0.2689414 + 0.2119416 = 1.4808830$$. (Why its gradient is $$dy$$: each term of the sum is
$$dy_t[c]$$ times $$y_t[c]$$, a constant times the value, as in the residual note [6].)

### 1.5 Where each value is used

![One head: every use of every value](/images/attention-backward/fig1-graph.svg)
*Figure 1. Each arrow is one use of a value. The backward pass walks the same arrows the other way, one
step per arrow. $$e_{tj}$$ has two arrows going out, so its gradient has two parts.*

Figure 1 is the whole plan, from the end back:

| step | arrow walked back | gives |
|---|---|---|
| B1 | F6: the merge | $$dy$$, cut into heads |
| B2 | F5: $$a_{tj} \to y_t$$ | $$da$$ |
| B3 | F5: $$v_j \to y_t$$ | $$dv$$ |
| B4 | F4: $$S_t \to a_{tj}$$ | $$dS_t$$ |
| B5 | F4: $$e_{tj} \to a_{tj}$$ and $$e_{tj} \to S_t$$ | $$de$$, two parts added |
| B6 | F4: $$s_{tj} \to e_{tj}$$ | $$ds$$ (and $$dm = 0$$) |
| B7 | F2: $$q_t \to s_{tj}$$ | $$dq$$ |
| B8 | F2: $$k_j \to s_{tj}$$ | $$dk$$ |
| B9 | F1: the split | $$dq, dk, dv$$ back into qkv's places |

*Table 5. F3, the mask, has no step of its own: it decides which pairs exist. Every sum below runs only
over the pairs $$j \le t$$.*

## 2. Three rules and three derivatives

Every step below uses the same three rules as the earlier notes [2, 3, 4, 5, 6], and nothing else.

| rule | when | what it says |
|---|---|---|
| **chain rule** | $$a$$ affects $$L$$ only through $$b$$ | $$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial b} \cdot \frac{\partial b}{\partial a}$$, written short as $$da = db \cdot \partial b / \partial a$$ |
| **shared value: sum** | $$a$$ is used in many places | add one chain-rule term per place: $$\frac{\partial L}{\partial a} = \sum_k \frac{\partial L}{\partial b_k} \cdot \frac{\partial b_k}{\partial a}$$ |
| **used twice: add the parts** | $$a$$ is used by two different forward steps | each step gives one part; $$da$$ = part 1 + part 2 |

*Table 6. The three rules. Attention needs all three, the second most: $$q_t$$ is in every score of its
row, $$k_j$$ and $$v_j$$ in every later token's.*

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

And one habit: **each step looks at one input and holds the others fixed.** F4's
$$a_{tj} = e_{tj} / S_t$$ has two inputs: B4 takes the derivative with respect to $$S_t$$ holding
$$e_{tj}$$ fixed, and B5 with respect to $$e_{tj}$$ holding $$S_t$$ fixed, even though $$S_t$$ was made
from $$e_{tj}$$. The road through $$S_t$$ is not lost: B5 adds it.

Three derivatives are needed beyond the rules:

| function | derivative | why |
|---|---|---|
| $$x \cdot w$$, $$w$$ held fixed | $$w$$ | a constant times $$x$$ |
| $$1 / S$$ | $$-1 / S^2$$ | below |
| $$\exp a$$ | $$\exp a$$ | exp is its own slope (the classifier note [5]) |

*Table 7. The three derivatives. A dot product $$q \cdot k = q[1]\, k[1] + \cdots + q[d]\, k[d]$$ is a sum
of products, so with respect to $$q[c]$$ only the term $$q[c]\, k[c]$$ holds it, and its derivative is
$$k[c]$$: the first line.*

**The derivative of $$1/S$$.** Nudge $$S$$ by a small $$h$$:

$$
\begin{aligned}
\frac{1}{S + h} - \frac{1}{S} &= \frac{S - (S + h)}{S\,(S + h)}
  && \text{one fraction, over } S\,(S + h) \\
  &= \frac{-h}{S\,(S + h)} \approx \frac{-h}{S^2}
  && h \text{ small: } S + h \approx S \\
\frac{d}{dS}\, \frac{1}{S} &= -\frac{1}{S^2}
  && \text{divide by } h
\end{aligned}
$$

A small case: $$S = 2$$, $$h = 0.01$$: $$1/2.01 - 1/2 = -0.0024876$$, and $$-0.01 / 2^2 = -0.0025$$. The
minus: a bigger sum makes each share smaller.

## 3. Breaking it down: B1 to B9

Each step first works on one head of one sequence, then says what changes for the whole batch. Each
ends with the example, and with what its number means: nudge one value by 0.01, work the stand-in loss
out again, and compare.

### 3.1 B1: the head outputs, $$dy$$

**One head: head 0.** F6 only copies. Head 0's outputs come first in the output row: $$y_t[c]$$ is
copied to output row $$t$$, column $$c$$, for $$c = 0$$ to 63. So $$\text{output}_t[c] = y_t[c]$$, and
$$y_t[c]$$ is used in that one place only. By the chain rule:

$$
\begin{aligned}
\frac{\partial L}{\partial y_t[c]} &= \frac{\partial L}{\partial \text{output}_t[c]} \cdot \frac{\partial\, \text{output}_t[c]}{\partial y_t[c]}
  && \text{chain rule: } y_t[c] \text{ is used in one place only} \\
&= \frac{\partial L}{\partial \text{output}_t[c]} \cdot 1
  && \text{output}_t[c] = y_t[c] \text{: slope 1} \\
dy_t[c] &= d\text{output}_t[c]
  && \text{the same, written short}
\end{aligned}
$$

A copy has slope 1 from a value to its copy and 0 to every other place, so the gradient comes back
unchanged: head 0's $$dy_t$$ is columns 0 to 63 of $$d\text{output}_t$$.

**Every head.** Write $$y^{(h)}$$ for head $$h$$'s $$y$$. Head $$h$$'s columns come after the $$h$$ heads
before it, $$h \cdot d$$ columns, so F6 copies $$y^{(h)}_t[c]$$ to column $$h \cdot d + c$$:

| head $$h$$ | its columns of output row $$t$$ | $$y^{(h)}_t[c]$$ goes to column |
|---|---|---|
| 0 | 0 to 63 | $$c$$ |
| 1 | 64 to 127 | $$64 + c$$ |
| 2 | 128 to 191 | $$128 + c$$ |
| 11 | 704 to 767 | $$704 + c$$ |

*Table 8. Where F6 puts each head. Head 2's $$y_t[5]$$, for example, is output column $$128 + 5 = 133$$.*

The same chain rule as for head 0, with column $$h d + c$$ in place of $$c$$:

$$
\begin{aligned}
\frac{\partial L}{\partial y^{(h)}_t[c]} &= \frac{\partial L}{\partial \text{output}_t[h d + c]} \cdot \frac{\partial\, \text{output}_t[h d + c]}{\partial y^{(h)}_t[c]}
  && \text{chain rule: one place only} \\
dy^{(h)}_t[c] &= d\text{output}_t[h d + c] \cdot 1
  && h = 0 \text{ gives head 0's line}
\end{aligned}
$$

**The whole batch.** Each of the 4 sequences has its own output rows, and nothing crosses from one
sequence to another, so sequence $$b$$ only adds an index: $$dy^{(b, h)}_t[c] = d\text{output}_{b, t}[h d + c]$$.
Every value of every head of every sequence is copied to one place: no sums. The gradient of the layer
after, `{4, 1024, 768}`, is cut into heads, `{4, 12, 1024, 64}`. Interactive 1, read from the output row
up, shows where each value of the gradient goes.

**Example.** One head, so $$dy$$ is the given $$[1,\ 0,\ 0,\ 0]$$ for each token.

### 3.2 B2: the weights, $$da$$

**One head.** From F5, $$y_t = \sum_{j \le t} a_{tj}\, v_j$$. It is $$d$$ numbers, one per column $$c$$:
$$y_t[c] = a_{t0}\, v_0[c] + a_{t1}\, v_1[c] + \cdots$$. With respect to $$a_{tj}$$, holding $$v$$ fixed,
only one term of each holds it:

$$
\frac{\partial y_t[c]}{\partial a_{tj}} = v_j[c]
\qquad \text{a constant times } a_{tj} \text{ (Table 7)}
$$

$$a_{tj}$$ is used by all $$d$$ numbers of $$y_t$$, so by the shared-value rule there is one term per
column:

$$
\begin{aligned}
\frac{\partial L}{\partial a_{tj}} &= \sum_{c=1}^{d} \frac{\partial L}{\partial y_t[c]} \cdot \frac{\partial y_t[c]}{\partial a_{tj}}
  && \text{chain rule, one term per column of } y_t \\
da_{tj} &= \sum_{c=1}^{d} dy_t[c]\, v_j[c] = dy_t \cdot v_j
  && \text{the same, written short: a dot product}
\end{aligned}
$$

**The whole batch.** $$a_{tj}$$ belongs to one pair of one head: it is used only in $$y_t$$ of that
head. No sum over rows.

**Example.** $$dy_t = [1, 0, 0, 0]$$ and $$v_j$$ has its 1 in column $$j$$, so $$dy_t \cdot v_j$$ is 1
for $$j = 0$$ and 0 otherwise:

| | $$j = 0$$ | $$j = 1$$ | $$j = 2$$ |
|---|---:|---:|---:|
| $$da_{0j}$$ | 1 | | |
| $$da_{1j}$$ | 1 | 0 | |
| $$da_{2j}$$ | 1 | 0 | 0 |

*Table 9. B2 in the example: the loss wants every token's weight on token 0 up.*

**What it means.** Raise $$a_{20}$$ by 0.01, holding everything else: $$y_2$$ moves by
$$0.01\, v_0 = [0.01, 0, 0, 0]$$, and $$L$$, the sum of column 0, by 0.01: $$da_{20} = 1$$. Raise
$$a_{22}$$ instead: $$y_2$$ moves by $$0.01\, v_2 = [0, 0, 0.01, 0]$$, column 2, which $$L$$ does not count:
$$da_{22} = 0$$.

<div class="aw" data-widget="B2">This picture needs JavaScript.</div>

*Interactive 5. B2. $$da_{tj}$$ is $$dy_t$$ dotted with $$v_j$$. $$V$$ is used by rows, so it is read as $$V^\top$$. With 8 tokens, every token whose $$v$$ has a number in column 0 (tokens 0, 4, 6 and 7) gets a non-zero $$da$$: the loss counts column 0 only.*

### 3.3 B3: the values, $$dv$$

**One head.** From F5 again, now with respect to $$v_j[c]$$, holding the weights fixed:

$$
\frac{\partial y_t[c]}{\partial v_j[c]} = a_{tj}
\qquad \text{a constant times } v_j[c]
$$

Which $$y$$ hold $$v_j$$? Every token that can see token $$j$$: $$t = j, j + 1, \ldots$$ to the last.
That is many places, so by the shared-value rule:

$$
\begin{aligned}
\frac{\partial L}{\partial v_j[c]} &= \sum_{t \ge j} \frac{\partial L}{\partial y_t[c]} \cdot \frac{\partial y_t[c]}{\partial v_j[c]}
  && \text{chain rule, one term per token that sees } j \\
dv_j &= \sum_{t \ge j} a_{tj}\, dy_t
  && \text{the same, written short, all } d \text{ columns at once}
\end{aligned}
$$

**The whole batch.** $$v_j$$ belongs to one token of one head: its sum runs over the later tokens of
the same head and sequence, never across heads.

**Example.** Token 0 is seen by all three, token 2 only by itself:

| | sum | $$dv_j$$ |
|---|---|---|
| $$dv_0$$ | $$1 \cdot dy_0 + 0.2689414\, dy_1 + 0.2119416\, dy_2$$ | 1.4808830 0 0 0 |
| $$dv_1$$ | $$0.7310586\, dy_1 + 0.2119416\, dy_2$$ | 0.9430001 0 0 0 |
| $$dv_2$$ | $$0.5761169\, dy_2$$ | 0.5761169 0 0 0 |

*Table 10. B3 in the example. Each value's gradient is the weight every later token gives it, times that
token's $$dy$$.*

**What it means.** Raise $$v_0[0]$$ by 0.01: every token's output column 0 moves by its weight on token 0
times 0.01, and $$L$$ by $$0.01 \times (1 + 0.2689414 + 0.2119416) = 0.0148088$$: $$dv_0[0] = 1.4808830$$.

<div class="aw" data-widget="B3">This picture needs JavaScript.</div>

*Interactive 6. B3. $$dv_j$$ mixes the rows of $$dY$$ of every token that sees token $$j$$, weighted by **column** $$j$$ of $$A$$: a column, so $$A^\top$$. Token 0 is seen by every token, so its column is the longest; the last token is seen only by itself. With 8 tokens, $$dv_0$$ has 5 more terms than Table 10, because tokens 3 to 7 look back at token 0 too.*

### 3.4 B4: the softmax's sum, $$dS_t$$

**One head.** From F4, $$a_{tj} = e_{tj} \cdot \frac{1}{S_t}$$, with respect to $$S_t$$, holding
$$e_{tj}$$ fixed, with Table 7:

$$
\frac{\partial a_{tj}}{\partial S_t} = e_{tj} \cdot \Big(-\frac{1}{S_t^2}\Big) = -\frac{e_{tj}}{S_t^2}
\qquad e_{tj} \text{ is a constant factor}
$$

One $$S_t$$ for the whole row, used by every weight of it, $$j = 0 \ldots t$$: by the shared-value rule,

$$
\begin{aligned}
\frac{\partial L}{\partial S_t} &= \sum_{j \le t} \frac{\partial L}{\partial a_{tj}} \cdot \frac{\partial a_{tj}}{\partial S_t}
  && \text{chain rule, one term per weight of the row} \\
dS_t &= -\sum_{j \le t} da_{tj}\, \frac{e_{tj}}{S_t^2} = -\frac{1}{S_t} \sum_{j \le t} da_{tj}\, a_{tj}
  && \text{the same, written short; } e_{tj} / S_t = a_{tj}
\end{aligned}
$$

**The whole batch.** One $$S_t$$ per row of each head: the sum runs over that row only.

**Example.**

| row | $$\sum_j da_{tj}\, a_{tj}$$ | $$dS_t$$ |
|---|---|---:|
| 0 | $$1 \times 1 = 1$$ | $$-1 / 1 = -1$$ |
| 1 | $$1 \times 0.2689414 + 0 = 0.2689414$$ | $$-0.2689414 / 1.3678794 = -0.1966119$$ |
| 2 | $$1 \times 0.2119416 + 0 + 0 = 0.2119416$$ | $$-0.2119416 / 1.7357589 = -0.1221031$$ |

*Table 11. B4 in the example.*

**What it means.** Raise $$S_1$$ by 0.01, holding $$e_{10}$$ and $$e_{11}$$: each weight of row 1 shrinks,
$$a_{10} = 0.3678794 / 1.3778794 = 0.2669895$$, down 0.0019519, and $$L$$ with it, since it counts
$$a_{10}$$: about $$0.01 \times (-0.1966119)$$.

### 3.5 B5: the softmax's exps, $$de$$, two parts

**One head.** $$e_{tj}$$ has two arrows in Figure 1: into its own weight, $$a_{tj} = e_{tj} / S_t$$, and
into the row's sum, $$S_t = e_{t0} + \cdots + e_{tt}$$.

$$
\begin{aligned}
\frac{\partial a_{tj}}{\partial e_{tj}} &= \frac{1}{S_t}
  && S_t \text{ held fixed: a constant factor} \\
\frac{\partial S_t}{\partial e_{tj}} &= 0 + \cdots + 1 + \cdots + 0 = 1
  && \text{only one term of the sum is } e_{tj}
\end{aligned}
$$

By the used-twice rule, one chain per road, added:

$$
\begin{aligned}
\frac{\partial L}{\partial e_{tj}} &= \underbrace{\frac{\partial L}{\partial a_{tj}} \cdot \frac{\partial a_{tj}}{\partial e_{tj}}}_{\text{road 1: into its weight}} + \underbrace{\frac{\partial L}{\partial S_t} \cdot \frac{\partial S_t}{\partial e_{tj}}}_{\text{road 2: into the sum}}
  && \text{chain rule, two roads} \\
de_{tj} &= \frac{da_{tj}}{S_t} + dS_t
  && \text{the same, written short}
\end{aligned}
$$

**The whole batch.** $$e_{tj}$$ is in one weight and one sum, both of its own row: no other sums.

**Example.**

| | $$j = 0$$ | $$j = 1$$ | $$j = 2$$ |
|---|---|---|---|
| row 0 | $$1/1 - 1 = 0$$ | | |
| row 1 | $$1/1.3678794 - 0.1966119 = 0.5344466$$ | $$0 - 0.1966119 = -0.1966119$$ | |
| row 2 | $$1/1.7357589 - 0.1221031 = 0.4540138$$ | $$-0.1221031$$ | $$-0.1221031$$ |

*Table 12. B5 in the example. Road 1 is $$da / S$$; road 2, $$dS_t$$, is the same for every column of the
row.*

**What it means.** Raise $$e_{10}$$ by 0.01 and let only road 1 see it, $$S_1$$ held: $$a_{10}$$ goes up by
$$0.01 / 1.3678794 = 0.0073106$$, and $$L$$ with it. Road 2 alone, $$S_1$$ up 0.01: $$L$$ down 0.0019519
(B4). Together, about $$0.01 \times 0.5344466$$.

### 3.6 B6: the scores, $$ds$$

**One head.** From F4, $$e_{tj} = \exp(s_{tj} - m_t)$$, with respect to $$s_{tj}$$, holding $$m_t$$ fixed:

$$
\begin{aligned}
\frac{\partial e_{tj}}{\partial s_{tj}} &= \exp(s_{tj} - m_t) \cdot 1 = e_{tj}
  && \text{exp is its own slope; } \tfrac{\partial}{\partial s}(s - m) = 1 \\
\frac{\partial L}{\partial s_{tj}} &= \frac{\partial L}{\partial e_{tj}} \cdot \frac{\partial e_{tj}}{\partial s_{tj}}
  && \text{chain rule: } s_{tj} \text{ is used in } e_{tj} \text{ only (with } m \text{ held)} \\
ds_{tj} &= de_{tj} \cdot e_{tj}
  && \text{the same, written short}
\end{aligned}
$$

The road through the max, $$s \to m_t \to e$$, carries nothing: $$dm_t = -\sum_j de_{tj}\, e_{tj} =
-\sum_j ds_{tj}$$, and Section 5 shows each row of $$ds$$ adds up to 0, as the classifier note found
for its own $$m$$ [5].

**The whole batch.** One per pair of each head: no sums.

**Example.**

| | $$j = 0$$ | $$j = 1$$ | $$j = 2$$ |
|---|---|---|---|
| row 0 | $$0 \times 1 = 0$$ | | |
| row 1 | $$0.5344466 \times 0.3678794 = 0.1966119$$ | $$-0.1966119 \times 1 = -0.1966119$$ | |
| row 2 | $$0.4540138 \times 0.3678794 = 0.1670223$$ | $$-0.1221031 \times 0.3678794 = -0.0449192$$ | $$-0.1221031 \times 1 = -0.1221031$$ |

*Table 13. B6 in the example: each row adds up to 0. Token 0's one score gets 0: its only weight is
always 1, whatever its score.*

**What it means.** Raise each score by 0.01 in turn, and work the weights and $$L$$ out again:

| score | $$0.01 \times ds_{tj}$$ | $$L$$'s change |
|---|---:|---:|
| $$s_{10}$$ | 0.0019661 | 0.0019707 |
| $$s_{21}$$ | −0.0004492 | −0.0004505 |
| $$s_{22}$$ | −0.0012210 | −0.0012201 |

*Table 14. Raising token 2's score of token 0 would lift its weight on token 0, which the loss wants;
raising its score of itself takes weight away from token 0.*

![The example's weights and the gradient of the scores](/images/attention-backward/fig2-weights-and-grad.svg)
*Figure 2. The weights $$a$$ and the scores' gradient $$ds$$ of the example. The masked pairs have
weight 0 and gradient 0.*

### 3.7 B7: the queries, $$dq$$

**One head.** From F2, $$s_{tj} = \frac{1}{\sqrt{d}}\left(q_t[1]\, k_j[1] + \cdots + q_t[d]\, k_j[d]\right)$$,
with respect to $$q_t[c]$$, holding $$k$$ fixed. Only one term holds it:

$$
\frac{\partial s_{tj}}{\partial q_t[c]} = \frac{k_j[c]}{\sqrt{d}}
\qquad k_j[c] / \sqrt{d} \text{ is a constant factor (Table 7)}
$$

$$q_t$$ is in every score of its row, $$j = 0 \ldots t$$: by the shared-value rule,

$$
\begin{aligned}
\frac{\partial L}{\partial q_t[c]} &= \sum_{j \le t} \frac{\partial L}{\partial s_{tj}} \cdot \frac{\partial s_{tj}}{\partial q_t[c]}
  && \text{chain rule, one term per score of the row} \\
dq_t &= \frac{1}{\sqrt{d}} \sum_{j \le t} ds_{tj}\, k_j
  && \text{the same, written short, all } d \text{ columns at once}
\end{aligned}
$$

**The whole batch.** $$q_t$$ belongs to one token of one head: the sum runs over its row only.

**Example.**

| | sum | $$dq_t$$ |
|---|---|---|
| $$dq_0$$ | $$\tfrac{1}{2}\,(0 \cdot k_0)$$ | 0 0 0 0 |
| $$dq_1$$ | $$\tfrac{1}{2}\,(0.1966119\, k_0 - 0.1966119\, k_1)$$ | 0.0983060 −0.0983060 0 0 |
| $$dq_2$$ | $$\tfrac{1}{2}\,(0.1670223\, k_0 - 0.0449192\, k_1 - 0.1221031\, k_2)$$ | 0.0224596 −0.0835112 0 0 |

*Table 15. B7 in the example. Token 1's query should turn toward $$k_0 = [1, 0, 0, 0]$$ and away from
$$k_1 = [0, 1, 0, 0]$$, to score token 0 higher.*

**What it means.** Raise $$q_2[0]$$ by 0.01: token 2's scores of tokens 0 and 2, whose keys have a 1 in
column 0, go up by $$0.01 / 2$$, and $$L$$ by 0.0002243, about $$0.01 \times 0.0224596$$.

<div class="aw" data-widget="B7">This picture needs JavaScript.</div>

*Interactive 7. B7. $$dq_t$$ mixes the keys of the tokens $$t$$ can see, weighted by row $$t$$ of $$dS$$, and halved. A positive $$ds_{tj}$$ turns the query toward $$k_j$$, a negative one away. Token 0's row of $$dS$$ is 0, so $$dq_0 = 0$$.*

### 3.8 B8: the keys, $$dk$$

**One head.** From F2 again, now with respect to $$k_j[c]$$, holding $$q$$ fixed:

$$
\frac{\partial s_{tj}}{\partial k_j[c]} = \frac{q_t[c]}{\sqrt{d}}
$$

$$k_j$$ is in the score of every token that sees token $$j$$, $$t = j, j + 1, \ldots$$: by the
shared-value rule,

$$
\begin{aligned}
\frac{\partial L}{\partial k_j[c]} &= \sum_{t \ge j} \frac{\partial L}{\partial s_{tj}} \cdot \frac{\partial s_{tj}}{\partial k_j[c]}
  && \text{chain rule, one term per token that sees } j \\
dk_j &= \frac{1}{\sqrt{d}} \sum_{t \ge j} ds_{tj}\, q_t
  && \text{the same, written short}
\end{aligned}
$$

**The whole batch.** As $$v_j$$'s, the sum runs over the later tokens of the same head and sequence.

**Example.**

| | sum | $$dk_j$$ |
|---|---|---|
| $$dk_0$$ | $$\tfrac{1}{2}\,(0 \cdot q_0 + 0.1966119\, q_1 + 0.1670223\, q_2)$$ | 0.1670223 0.3636343 0 0 |
| $$dk_1$$ | $$\tfrac{1}{2}\,(-0.1966119\, q_1 - 0.0449192\, q_2)$$ | −0.0449192 −0.2415312 0 0 |
| $$dk_2$$ | $$\tfrac{1}{2}\,(-0.1221031\, q_2)$$ | −0.1221031 −0.1221031 0 0 |

*Table 16. B8 in the example. Token 0's key should turn toward the queries that look at it.*

**What it means.** Raise $$k_0[0]$$ by 0.01: $$L$$ goes up by 0.0016750, about $$0.01 \times 0.1670223$$.
Raise $$k_1[1]$$ by 0.01: $$L$$ goes down by 0.0024121, about $$0.01 \times 0.2415312$$.

<div class="aw" data-widget="B8">This picture needs JavaScript.</div>

*Interactive 8. B8. $$dk_j$$ mixes the queries of the tokens that see token $$j$$, weighted by **column** $$j$$ of $$dS$$, so $$dS^\top$$, and halved. It is B3's shape again, with $$dS$$ for $$A$$ and $$Q$$ for $$dY$$.*

### 3.9 B9: back into qkv

**One head.** F1 only copied, so as in B1, each gradient goes back to the place its value was copied
from: $$dq_t$$ into the query part of qkv row $$t$$, at the head's columns, and $$dk_t$$, $$dv_t$$ into the
key and value parts. No sums: every value of qkv was copied to one place.

**Example.** qkv row $$t$$ is $$q_t \,\vert\, k_t \,\vert\, v_t$$, so its gradient is
$$dq_t \,\vert\, dk_t \,\vert\, dv_t$$: row 0 is $$0\ 0\ 0\ 0 \,\vert\, 0.1670223\ 0.3636343\ 0\ 0
\,\vert\, 1.4808830\ 0\ 0\ 0$$.

### 3.10 All nine steps

| step | from | gradient of | formula | sum over |
|---|---|---|---|---|
| B1 | F6 | head outputs | $$dy$$ = output's gradient, cut into heads | — |
| B2 | F5 | weights | $$da_{tj} = dy_t \cdot v_j$$ | the head's columns $$c$$ |
| B3 | F5 | values | $$dv_j = \sum_{t \ge j} a_{tj}\, dy_t$$ | tokens that see $$j$$ |
| B4 | F4 | the row's sum | $$dS_t = -\frac{1}{S_t} \sum_{j \le t} da_{tj}\, a_{tj}$$ | the row |
| B5 | F4 | exps, two parts | $$de_{tj} = da_{tj} / S_t + dS_t$$ | — |
| B6 | F4 | scores | $$ds_{tj} = de_{tj}\, e_{tj}$$ | — |
| B7 | F2 | queries | $$dq_t = \frac{1}{\sqrt{d}} \sum_{j \le t} ds_{tj}\, k_j$$ | the row |
| B8 | F2 | keys | $$dk_j = \frac{1}{\sqrt{d}} \sum_{t \ge j} ds_{tj}\, q_t$$ | tokens that see $$j$$ |
| B9 | F1 | qkv | $$dq, dk, dv$$ back in their places | — |

*Table 17. The backward pass, broken down.*

Side by side with the earlier notes:

| | LayerNorm [2] | linear [3] | classifier [5] | attention |
|---|---|---|---|---|
| steps | 9 | 3 | 7 | 9 |
| a gradient summed over rows | $$d\gamma$$, $$d\beta$$ | $$dW$$, $$db$$ | none | none: no weights |
| a gradient summed over a row | $$dr$$, $$dv$$, $$d\mu$$ | $$dx$$ | $$dm$$, 0 | $$dS_t$$, $$dq$$ |
| summed over later tokens | — | — | — | $$dv$$, $$dk$$ |
| a value used twice | $$x$$, $$d$$ | none | $$a_t$$, $$z$$ | $$e$$ |

*Table 18. Attention is the only layer where tokens mix: $$k_j$$ and $$v_j$$ get gradients from every
token after $$j$$.*

## 4. Rebuilding: C1 and C2

### 4.1 C1: the softmax's backward in one line

B4, B5 and B6 are one softmax. Put them together for one row, with $$a_{tj} = e_{tj} / S_t$$:

$$
\begin{aligned}
ds_{tj} &= de_{tj}\, e_{tj}
  && \text{B6} \\
        &= \Big(\frac{da_{tj}}{S_t} + dS_t\Big)\, e_{tj}
  && \text{B5} \\
        &= \frac{e_{tj}}{S_t}\, da_{tj} - \frac{e_{tj}}{S_t} \sum_{k \le t} da_{tk}\, a_{tk}
  && \text{B4: } dS_t = -\tfrac{1}{S_t} \textstyle\sum_k da_{tk}\, a_{tk} \\
        &= a_{tj}\, \Big(da_{tj} - \sum_{k \le t} a_{tk}\, da_{tk}\Big)
  && e_{tj} / S_t = a_{tj}
\end{aligned}
$$

$$
\boxed{\;ds_{tj} = a_{tj}\, \big(da_{tj} - \text{row\_dot}_t\big), \qquad \text{row\_dot}_t = \sum_{k \le t} a_{tk}\, da_{tk}\;}
$$

$$\text{row\_dot}_t$$ is the weighted mean of the row's $$da$$: each weight's gradient is pushed by how much
its $$da$$ is above that mean, scaled by the weight itself. At row 1:
$$0.2689414 \times (1 - 0.2689414) = 0.1966119$$, B6's number. The kernel works out only this line:
$$S$$, $$e$$ and $$de$$ never exist.

<div class="aw" data-widget="softmax-backward">This picture needs JavaScript.</div>

*Interactive 9. C1 on one row. $$\text{row\_dot}_t$$ is the row's $$da$$, weighted by $$a$$. Each $$ds$$ is its weight times how far its $$da$$ is above that mean, so each row of $$ds$$ adds up to 0. With 8 tokens, tokens 4, 6 and 7 also have a number in column 0 of their $$v$$, so their $$da$$ is not 0 either.*

### 4.2 C2: four matrix products

Write each head's rows as matrices: $$Q$$, $$K$$, $$V$$, $$Y$$ are $$\{T, d\}$$, one row per token; $$S$$
and $$A$$ are $$\{T, T\}$$, row $$t$$ column $$j$$, with 0 above the diagonal (F3). Then each sum of
Table 17 is a matrix product, read as in the linear note [3]: **a sum is a product when the summed index
is the second index of the left factor and the first of the right**, and a factor that has it the other
way is read turned round, $$\top$$:

| step | the sum | summed index | where it sits | product |
|---|---|---|---|---|
| F2 | $$\sum_c q_t[c]\, k_j[c]$$ | $$c$$ | 2nd in $$Q$$: right; 2nd in $$K$$: wrong, so $$K^\top$$ | $$S = \tfrac{1}{\sqrt{d}}\, Q K^\top$$ |
| F5 | $$\sum_j a_{tj}\, v_j[c]$$ | $$j$$ | 2nd in $$A$$, 1st in $$V$$: both right | $$Y = A V$$ |
| B2 | $$\sum_c dy_t[c]\, v_j[c]$$ | $$c$$ | 2nd in $$dY$$; 2nd in $$V$$, so $$V^\top$$ | $$dA = dY\, V^\top$$ |
| B3 | $$\sum_t a_{tj}\, dy_t[c]$$ | $$t$$ | 1st in $$A$$, so $$A^\top$$; 1st in $$dY$$ | $$dV = A^\top dY$$ |
| B7 | $$\sum_j ds_{tj}\, k_j[c]$$ | $$j$$ | 2nd in $$dS$$, 1st in $$K$$ | $$dQ = \tfrac{1}{\sqrt{d}}\, dS\, K$$ |
| B8 | $$\sum_t ds_{tj}\, q_t[c]$$ | $$t$$ | 1st in $$dS$$, so $$dS^\top$$; 1st in $$Q$$ | $$dK = \tfrac{1}{\sqrt{d}}\, dS^\top Q$$ |

*Table 19. The two forward products and the four backward ones. The sums "over $$j \le t$$" and "over
$$t \ge j$$" need nothing special: $$A$$ and $$dS$$ are 0 above the diagonal, so the full products add 0
for every masked pair.*

Interactives 2, 4, 5, 6, 7 and 8 are these six products. In each, the values one cell uses are lit. Where they run down a column of a stored matrix, the summed index is that matrix's first, and it is read turned round: $$K^\top$$ in F2, $$V^\top$$ in B2, $$A^\top$$ in B3, $$dS^\top$$ in B8.

$$
\boxed{\;dA = dY\, V^\top, \quad dV = A^\top dY, \quad dS = A \odot (dA - \text{row\_dot}), \quad dQ = \tfrac{1}{\sqrt{d}}\, dS\, K, \quad dK = \tfrac{1}{\sqrt{d}}\, dS^\top Q\;}
$$

The lines at the top of this note. The shapes check them: $$dY\, V^\top$$ is
$$\{T, d\} \cdot \{d, T\} = \{T, T\}$$, the shape of $$A$$; $$A^\top dY$$ is
$$\{T, T\} \cdot \{T, d\} = \{T, d\}$$, the shape of $$V$$.

## 5. Three checks

**Each row of $$ds$$ adds up to 0.** Sum C1 over the row:

$$
\sum_{j \le t} ds_{tj} = \sum_j a_{tj}\, da_{tj} - \text{row\_dot}_t \sum_j a_{tj} = \text{row\_dot}_t - \text{row\_dot}_t \times 1 = 0
$$

Adding one number to every score of a row changes no weight (the shift of Section 1.3), so the loss
cannot care about that direction. In the example: $$0.1670223 - 0.0449192 - 0.1221031 = 0$$. This is also
why the max's road carries nothing (B6).

**The classifier's gradient falls out of C1.** The classifier's loss is $$\ell = -\log a_t$$ for the
target column $$t$$ [5]. Then $$da_j = -1 / a_t$$ at the target and 0 elsewhere, so
$$\text{row\_dot} = a_t \times (-1/a_t) = -1$$, and C1 gives:

$$
ds_j = a_j\, (da_j + 1) = \begin{cases} a_t\,(-1/a_t + 1) = a_t - 1 & j = t \\ a_j\, (0 + 1) = a_j & j \ne t \end{cases}
$$

$$p - 1$$ at the target, $$p$$ elsewhere: the classifier note's C2. Attention's softmax is the general
case, with a gradient arriving on every weight.

**The first token passes its gradient only to its value.** Token 0 sees only itself: its one weight is
$$\exp(0) / \exp(0) = 1$$ whatever its score, so its score's gradient is $$ds_{00} = 1 \times (da_{00} -
1 \times da_{00}) = 0$$, and nothing goes to $$q_0$$ from its own row. In the example $$dq_0 = 0$$; $$dk_0$$
is not 0, because tokens 1 and 2 look at token 0.

## 6. What the rebuilt formulas give the kernel

The kernel in my trainer [1] (`attention.cuh`) is C1 and C2 written out, as llm.c's is [9]:

| step | in the kernel | how |
|---|---|---|
| F1 | `attention_split_qkv_kernel` | one `float4` per thread: each head's rows next to each other |
| F2 | `cublasSgemmStridedBatched`, scale $$1/\sqrt{d}$$ | 48 products $$\{1024, 64\} \cdot \{64, 1024\}$$ at once |
| F3, F4 | `attention_softmax_forward_kernel` | one warp per row: the online softmax of the classifier [5, 10], 0 after the token |
| F5 | `cublasSgemmStridedBatched` | 48 products $$\{1024, 1024\} \cdot \{1024, 64\}$$ |
| F6 | `attention_merge_heads_kernel` | the heads side by side again |
| B1 | `attention_split_heads_kernel` | F6 undone |
| B2, B3 | two `cublasSgemmStridedBatched` | $$dA = dY V^\top$$, $$dV = A^\top dY$$ |
| B4–B6 | `attention_softmax_backward_kernel` | one warp per row: C1, `row_dot` added up with shuffles |
| B7, B8 | two `cublasSgemmStridedBatched`, scale $$1/\sqrt{d}$$ | $$dQ = dS\, K$$, $$dK = dS^\top Q$$ |
| B9 | `attention_merge_qkv_grad_kernel` | F1 undone |

*Table 20. "Strided batched": one call makes many products of the same size, each one stride further
along in memory: here, one per head of each sequence. cuBLAS reads matrices by columns, so each product
is asked for turned round, as the linear note's cuBLAS version does [3].*

**Keeping the weights.** The backward needs $$A$$ (B3, C1). The forward keeps it: 4 × 12 × 1024 × 1024
floats, 201 MB per layer, 2.4 GB for GPT-2 small's 12. The other way is to keep only $$m_t$$ and $$S_t$$ per
row, as the classifier does, and work $$s = QK^\top / \sqrt{d}$$ out again in the backward, one more
product; FlashAttention goes further and never writes $$A$$ at all [11]. Mine keeps $$A$$, as llm.c's does,
for the simplest kernel.

**`=` for qkv's gradient.** qkv is the qkv linear's output and attention reads it alone, so its gradient
has one part: written over. The scratch tensors ($$dY$$, $$dA$$, $$dQ$$, $$dK$$, $$dV$$) are written over
too; $$dA$$ becomes $$dS$$ in place, since C1 needs each $$da$$ only at its own place.

**The work.** Each of the six products is $$48 \times 1024 \times 1024 \times 64$$ multiply-adds,
6.4 billion operations; at a T4's 8.1 TFLOPS for floats, at least 0.80 ms each, 1.6 ms forward and 3.2 ms
backward. Half of each multiplies masked zeros, which cuBLAS does not know to skip. How close the kernel
gets is for the test to print: nothing here is timed.

## 7. What to remember

1. **A query asks, keys answer, values are passed on.** $$s = q \cdot k / \sqrt{d}$$, softmax over the
   tokens a token can see, then the weighted mix of their values.
2. **Every sum comes from sharing.** $$q_t$$ is in its whole row, so $$dq_t$$ sums over the row; $$k_j$$ and
   $$v_j$$ are in every later token's, so $$dk_j$$ and $$dv_j$$ sum over the tokens that see $$j$$.
3. **The softmax's backward is one line.** $$ds = a\,(da - \text{row\_dot})$$: each weight's gradient
   minus the row's weighted mean, times the weight. With one $$-1/a_t$$ it is the classifier's $$p - 1$$.
4. **Four products, each $$\top$$ from its summed index.** $$dA = dY V^\top$$, $$dV = A^\top dY$$,
   $$dQ = dS K / \sqrt{d}$$, $$dK = dS^\top Q / \sqrt{d}$$; the mask is in the zeros of $$A$$ and $$dS$$.
5. **The weights are kept.** 201 MB per layer, so the backward can read them; the price of the simplest
   kernel.

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
6. D. B. Song. *Residual add backward, broken down and rebuilt: one plus sign and a gradient that passes
   through.* 2026. [/posts/residual-backward/](/posts/residual-backward/)
7. A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin.
   *Attention Is All You Need.* 2017. Section 3.2.1: scaled dot-product attention, and why
   $$1/\sqrt{d}$$. [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
8. A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever. *Language Models are Unsupervised
   Multitask Learners.* 2019.
   [openai.com](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
9. A. Karpathy. *llm.c*, `train_gpt2.cu`: `attention_forward` and `attention_backward`, with
   `cublasSgemmStridedBatched` for the products. [github.com/karpathy/llm.c](https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu)
10. M. Milakov, N. Gimelshein. *Online normalizer calculation for softmax.* 2018.
    [arxiv.org/abs/1805.02867](https://arxiv.org/abs/1805.02867)
11. T. Dao, D. Y. Fu, S. Ermon, A. Rudra, C. Ré. *FlashAttention: Fast and Memory-Efficient Exact
    Attention with IO-Awareness.* 2022. [arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
