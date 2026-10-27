---
title: "Classifier backward, broken down and rebuilt: softmax, cross-entropy and prob minus one"
date: 2026-10-26
modified: 2026-10-27
version: 2
permalink: /posts/classifier-backward/
excerpt: "The last step of GPT-2 turns each row of 50257 scores into one loss, and its backward pass is usually given as one line: dz = p − 1 at the right word. This note takes the forward apart into five small steps, from the logits to a shift by the row's largest score, an exp, a sum and a log, and the backward into seven, each derived line by line with the chain rule, and works one row of four words through all of them. The sum turns every column's gradient into its probability, the target's column gets one more part, −1, and the shift drops out. Then it puts the steps back together into the one line a GPU kernel runs, and shows how the kernel finds the row's largest score and its sum in one read of the row: the online softmax."
tags:
  - softmax
  - cross-entropy
  - backpropagation
  - gpt-2
  - notes
---

**What this is.** GPT-2 small ends every forward pass with a classifier: each of its 4096 rows of
logits, one score for every word of the vocabulary, becomes probabilities with softmax, and the loss is
how unlikely the right next word was. The backward pass of it is usually written as one line:

$$
dz_j = d\ell \cdot \big(p_j - [j = t]\big) \qquad\qquad p_j = \text{softmax}(z)_j
$$

with $$[j = t]$$ equal to 1 in the right word's column and 0 in every other. That line is correct, and
it is what the kernel in my GPT-2 trainer [1] runs, but it hides two things: why a gradient should be a
probability, and where the $$-1$$ comes from. This note **breaks it down** into seven small steps, B1
to B7, one for each arrow of the forward pass, each derived with nothing more than the chain rule. Then
it **rebuilds** the line from those steps, in two moves, C1 and C2. One row of 4 words goes through
every step, so each formula has a number next to it.

**Sources.** The derivation is worked out by hand, with the same three rules as the notes on
[LayerNorm's backward](/posts/layer-norm-backward/) [2], [the linear layer's](/posts/linear-backward/) [3]
and [GELU's](/posts/gelu-backward/) [4]; every number is worked out from the formulas, to 7 digits. The
one-read forward of Section 6 is the online softmax of Milakov and Gimelshein [5]; llm.c's CUDA
classifier uses it too [6].

* TOC
{:toc}

## 1. The forward pass

### 1.1 Softmax and cross-entropy

The head, GPT-2's last linear layer, gives every position one score per word: the **logits** $$z$$.
Softmax turns a row of scores into probabilities, all between 0 and 1 and adding up to 1:

$$
p_j = \frac{\exp(z_j)}{\sum_k \exp(z_k)}
$$

exp makes every score positive, even a negative one, and dividing by the sum makes the row add up to 1.
The loss of the row is the **cross-entropy**, minus the log of the right word's probability:

$$
\ell = -\log p_t
$$

The log turns the product of the probabilities of a whole text into a sum, one term per position; the
minus makes it a cost. It is 0 when $$p_t = 1$$, and grows without limit as $$p_t$$ goes to 0:

| $$p_t$$ | 1 | 0.5 | 0.01 | 1 / 50257, an even guess |
|---|---:|---:|---:|---:|
| $$\ell = -\log p_t$$ | 0 | 0.6931472 | 4.6051702 | 10.8249051 |

*Table 1. The loss of one row.*

Written as it stands, $$\exp(z_j)$$ overflows: in a float, $$\exp(89)$$ is already infinity. So every
implementation first subtracts the row's largest score $$m$$ from every score. That changes no
probability, because $$\exp(z_j - m) = \exp(z_j) / \exp(m)$$, and the $$\exp(m)$$ cancels between top
and bottom:

$$
\frac{\exp(z_j - m)}{\sum_k \exp(z_k - m)}
= \frac{\exp(z_j) / \exp(m)}{\sum_k \exp(z_k) / \exp(m)}
= \frac{\exp(z_j)}{\sum_k \exp(z_k)} = p_j
$$

After the shift the largest score is 0, so every term is between 0 and $$\exp(0) = 1$$.

### 1.2 Five steps for one row

The classifier works on one row at a time: rows never mix. For one row of $$V$$ words with target $$t$$,
the formula is five steps:

$$
\begin{aligned}
&\text{F1:} \qquad m = \max_k z_k && \text{the row's largest score} \\
&\text{F2:} \qquad a_j = z_j - m && \text{shift every score: now the largest is 0} \\
&\text{F3:} \qquad e_j = \exp(a_j) && \text{every term between 0 and 1} \\
&\text{F4:} \qquad S = e_1 + e_2 + \cdots + e_V && \text{one sum over the row} \\
&\text{F5:} \qquad \ell = \log S - a_t && \text{the loss}
\end{aligned}
$$

F5 is $$-\log p_t$$ written out:

$$
-\log p_t = -\log \frac{e_t}{S} = -(\log e_t - \log S) = \log S - \log \exp(a_t) = \log S - a_t
$$

The probabilities are not a step of the forward. They come back by themselves in the backward (B3).

### 1.3 The names

| symbol | meaning | shape in GPT-2 small |
|---|---|---|
| $$j, k$$ | column: which word, $$1 \ldots V$$ | — |
| $$z$$ | logits: one score per word, per position; the head's output | `{4, 1024, 50304}` |
| $$t$$ | target: the right next word, the input shifted by one position | `{4, 1024}` ints |
| $$V$$ | vocabulary size: 50257 words (Section 6.3 explains the 50304) | — |
| $$m$$ | the row's largest score | one per row: `{4, 1024}` |
| $$a, e$$ | shifted scores and their exps | one per word |
| $$S$$ | the sum of $$e$$ over the row | one per row |
| $$\ell$$ | the row's loss | one per row: `{4, 1024}` |
| $$p$$ | probabilities, $$e_j / S$$ | one per word |
| $$[j = t]$$ | 1 if $$j$$ is the target's column, else 0 | — |
| $$R$$ | number of rows = batch size × sequence length | 4 × 1024 = 4096 |
| $$L$$ | the loss of the step: the mean of $$\ell$$ over the rows | 1 |
| $$\partial$$ | partial derivative: change one variable, hold all others fixed | — |
| $$da$$ | short for $$\partial L / \partial a$$, for any variable $$a$$; same shape as $$a$$ | — |

*Table 2. The job is to find $$dz$$. The classifier has no weight and no bias, so $$dz$$ is the only
gradient.*

$$d\ell$$ comes from the mean. $$L = (\ell_1 + \ell_2 + \cdots + \ell_R) / R$$, and the loss of row
$$r$$ is in one term only, so:

$$
d\ell_r = \frac{\partial L}{\partial \ell_r} = \frac{1}{R} = \frac{1}{4096}
$$

The steps below keep $$d\ell$$ as a letter, so they hold for any value of it.

### 1.4 The running example

One row of four words, $$z = [1,\ 2,\ 3,\ -1]$$, with target $$t = 1$$ (columns counted from 0 in the
example): the right word has only the second highest score, so the gradient has something to fix. Take
$$d\ell = 1$$; in training every number of $$dz$$ is then multiplied by $$1/4096$$.

| step | word 0 | word 1 (target) | word 2 | word 3 | per row |
|---|---:|---:|---:|---:|---:|
| $$z$$ | 1 | 2 | 3 | −1 | |
| F1: $$m = \max z$$ | | | | | 3 |
| F2: $$a = z - m$$ | −2 | −1 | 0 | −4 | |
| F3: $$e = \exp a$$ | 0.1353353 | 0.3678794 | 1 | 0.0183156 | |
| F4: $$S = \sum e$$ | | | | | 1.5215304 |
| F5: $$\ell = \log S - a_t$$ | | | | | $$0.4197166 - (-1) =$$ **1.4197166** |
| $$p = e / S$$ | 0.0889468 | 0.2417825 | 0.6572330 | 0.0120376 | sum 1 |

*Table 3. The example, forward. The right word got 24% of the probability, and
$$-\log 0.2417825 = 1.4197166$$, the loss of F5.*

### 1.5 Where each value is used

![The classifier's forward for one row](/images/classifier-backward/fig1-graph.svg)
*Figure 1. Each arrow is one use of a value. The backward pass walks the same arrows the other way, one
step per arrow. $$a$$ and $$z$$ each have two arrows going out, so each of their gradients has two
parts.*

Figure 1 is the whole plan. Seven arrows, so seven backward steps, from the loss back to the logits:

| step | arrow walked back | gives |
|---|---|---|
| B1 | F5: $$S \to \ell$$ | $$dS$$ |
| B2 | F4: $$e \to S$$ | $$de$$ |
| B3 | F3: $$a \to e$$ | $$da^{(1)}$$, the first part of $$da$$ |
| B4 | F5: $$a_t \to \ell$$ | $$da^{(2)}$$, the second part of $$da$$ |
| B5 | — | $$da = da^{(1)} + da^{(2)}$$ |
| B6 | F2: $$z \to a$$ and $$m \to a$$ | $$dz^{(1)}$$, the first part of $$dz$$, and $$dm$$ |
| B7 | F1: $$z \to m$$ | $$dz^{(2)}$$, and $$dz = dz^{(1)} + dz^{(2)}$$ |

*Table 4. The plan. Each step needs the gradient of the step before it: that is the chain rule.*

## 2. Three rules and three derivatives

Every step below uses the same three rules as the earlier notes [2, 3, 4], and nothing else.

| rule | when | what it says |
|---|---|---|
| **chain rule** | $$a$$ affects $$L$$ through $$b$$ | $$da = db \cdot \partial b / \partial a$$ |
| **shared value: sum** | $$a$$ is used in many places (all columns of a row) | add one chain-rule term per place: $$da = \sum_k db_k \cdot \partial b_k / \partial a$$ |
| **used twice: add the parts** | $$a$$ is used by two different forward steps | each step gives one part; $$da$$ = part 1 + part 2 |

*Table 5. The three rules. The classifier needs all three: $$m$$ is used by every column (B6), and
$$a_t$$ and $$z$$ are each used by two steps (B5, B7).*

And one habit: **each step looks at one input and holds the others fixed.** F5,
$$\ell = \log S - a_t$$, has two inputs: B1 takes the derivative with respect to $$S$$ and holds
$$a_t$$ fixed; B4 takes it with respect to $$a_t$$ and holds $$S$$ fixed, even though $$S$$ was made
from $$a_t$$. The path through $$S$$ is followed by B1 to B3, and B5 adds the two.

Whether a step has a sum, or two parts, comes only from where the variable is used in the forward pass,
Figure 1.

Three derivatives are needed beyond the rules:

| function | derivative | why |
|---|---|---|
| $$\exp a$$ | $$\exp a$$ | exp is its own slope |
| $$\log S$$ | $$1 / S$$ | log undoes exp |
| $$\max_k z_k$$ | 1 for the largest $$z_j$$, 0 for the others | only the largest moves the max |

*Table 6. The three derivatives. The max has no slope where two scores tie for largest, but B7 shows
that it never matters.*

**The derivative of exp.** exp turns a sum into a product, $$\exp(a + h) = \exp(a) \cdot \exp(h)$$, and
for a small $$h$$, $$\exp(h) \approx 1 + h$$. So nudging $$a$$ by $$h$$ gives:

$$
\begin{aligned}
\exp(a + h) - \exp(a) &= \exp(a) \cdot \exp(h) - \exp(a) && \exp \text{ turns a sum into a product} \\
                      &= \exp(a) \cdot \big(\exp(h) - 1\big) && \text{take } \exp(a) \text{ out} \\
                      &\approx \exp(a) \cdot h && \exp(h) \approx 1 + h \\
\frac{d}{da} \exp a   &= \exp a && \text{divide by } h
\end{aligned}
$$

At $$a = -1$$: $$e^{-1} = 0.3678794$$, the value and the slope both.

**The derivative of log.** log undoes exp: if $$y = \log S$$, then $$S = \exp(y)$$. By the line above,
$$dS/dy = \exp(y) = S$$. Turned round, $$y$$ changes $$1/S$$ times as fast as $$S$$:

$$
\frac{d}{dS} \log S = \frac{1}{dS/dy} = \frac{1}{S}
$$

At $$S = 1.5215304$$: $$1 / 1.5215304 = 0.6572330$$.

**The derivative of max.** In $$z = [1, 2, 3, -1]$$, $$m = 3$$ comes from word 2. Raise $$z_2$$ from 3 to
3.1, and $$m$$ becomes 3.1: it moves one for one. Raise $$z_0$$ from 1 to 1.1, and $$m$$ stays 3. So:

$$
\frac{\partial m}{\partial z_j} = \begin{cases} 1 & \text{if } z_j \text{ is the largest} \\ 0 & \text{otherwise} \end{cases}
$$

## 3. Breaking it down: B1 to B7

Each step is for one row. The whole tensor is the same steps for each of its 4096 rows, with nothing
added across rows: no value is shared by two rows.

### 3.1 B1: the sum, $$dS$$

From F5, $$\ell = \log S - a_t$$, with respect to $$S$$, holding $$a_t$$ fixed:

$$
\begin{aligned}
\frac{\partial \ell}{\partial S}
  &= \frac{\partial}{\partial S} \log S - \frac{\partial a_t}{\partial S}
  && \text{derivative of a difference = difference of derivatives} \\
  &= \frac{1}{S} - 0 = \frac{1}{S}
  && \text{Table 6; } a_t \text{ is held fixed, so it is a constant} \\
dS &= d\ell \cdot \frac{\partial \ell}{\partial S} = \frac{d\ell}{S}
  && \text{chain rule}
\end{aligned}
$$

$$S$$ is used once, by F5, so there is one path and nothing to add.

**Example.** $$dS = 1 / 1.5215304 = 0.6572330$$.

### 3.2 B2: the exps, $$de$$

From F4, with respect to $$e_j$$. Write the sum out; only one of its terms is $$e_j$$:

$$
\begin{aligned}
\frac{\partial S}{\partial e_j}
  &= \frac{\partial}{\partial e_j}\left(e_1 + \cdots + e_j + \cdots + e_V\right) \\
  &= 0 + \cdots + 1 + \cdots + 0 = 1
  && \text{every other term is a constant} \\
de_j &= dS \cdot \frac{\partial S}{\partial e_j} = dS \cdot 1 = dS
  && \text{chain rule}
\end{aligned}
$$

$$e_j$$ is used once, by F4, so there is one path.

**Example.** $$de = [0.6572330,\ 0.6572330,\ 0.6572330,\ 0.6572330]$$: every column gets the same, since
a sum treats all its terms the same.

### 3.3 B3: the shifted scores, part 1, $$da^{(1)}$$

From F3, $$e_j = \exp(a_j)$$, with respect to $$a_j$$:

$$
\begin{aligned}
\frac{\partial e_j}{\partial a_j} &= \exp(a_j) = e_j
  && \text{Table 6: exp is its own slope} \\
da^{(1)}_j &= de_j \cdot \frac{\partial e_j}{\partial a_j} = de_j \cdot e_j
  && \text{chain rule}
\end{aligned}
$$

$$a_j$$ goes into $$e_j$$ only: $$e_k$$ for $$k \ne j$$ does not hold $$a_j$$, so
$$\partial e_k / \partial a_j = 0$$ and those paths give nothing.

**Example.**

| | word 0 | word 1 (target) | word 2 | word 3 |
|---|---:|---:|---:|---:|
| $$de_j$$ (B2) | 0.6572330 | 0.6572330 | 0.6572330 | 0.6572330 |
| $$e_j$$ (F3) | 0.1353353 | 0.3678794 | 1 | 0.0183156 |
| $$da^{(1)}_j = de_j \cdot e_j$$ | 0.0889468 | 0.2417825 | 0.6572330 | 0.0120376 |

*Table 7. B3 in the example. The last row is $$p$$ of Table 3, exactly.*

The probabilities come back here, though the forward never worked them out: B1 to B3 multiplied each
$$e_j$$ by $$1/S$$, and $$e_j / S$$ is softmax. C1 says it in one line.

### 3.4 B4: the shifted scores, part 2, $$da^{(2)}$$

From F5 again, $$\ell = \log S - a_t$$, now with respect to $$a_j$$, holding $$S$$ fixed:

$$
\begin{aligned}
\frac{\partial \ell}{\partial a_j}
  &= \frac{\partial}{\partial a_j} \log S - \frac{\partial a_t}{\partial a_j}
  && \text{derivative of a difference} \\
  &= 0 - [j = t]
  && S \text{ is held fixed; } a_t \text{ holds } a_j \text{ only when } j = t \\
da^{(2)}_j &= d\ell \cdot \frac{\partial \ell}{\partial a_j} = -d\ell \cdot [j = t]
  && \text{chain rule}
\end{aligned}
$$

**Example.** $$da^{(2)} = [0,\ -1,\ 0,\ 0]$$.

This is the part that pushes the right word up: raise $$a_t$$, and the $$-a_t$$ of F5 brings $$\ell$$
down one for one.

### 3.5 B5: the shifted scores, complete, $$da$$

$$a_j$$ reaches the loss along two paths (Figure 1): through $$e_j$$ and $$S$$ for every column, and
straight into F5 for the target's column. By the used-twice rule, add the two. Written out as two
chains of the chain rule:

$$
\begin{aligned}
da_j &= \underbrace{\frac{\partial L}{\partial \ell} \cdot \frac{\partial \ell}{\partial S} \cdot \frac{\partial S}{\partial e_j} \cdot \frac{\partial e_j}{\partial a_j}}_{\text{path 1: } a_j \to e_j \to S \to \ell}
      + \underbrace{\frac{\partial L}{\partial \ell} \cdot \frac{\partial \ell}{\partial a_j}}_{\text{path 2: } a_j \to \ell} \\
     &= d\ell \cdot \frac{1}{S} \cdot 1 \cdot e_j \;+\; d\ell \cdot \big(-[j = t]\big)
     && \text{B1, B2, B3; and B4} \\
     &= da^{(1)}_j + da^{(2)}_j
\end{aligned}
$$

| | word 0 | word 1 (target) | word 2 | word 3 |
|---|---:|---:|---:|---:|
| $$da^{(1)}$$ (B3): path 1, through $$S$$ | 0.0889468 | 0.2417825 | 0.6572330 | 0.0120376 |
| $$da^{(2)}$$ (B4): path 2, the target | 0 | −1 | 0 | 0 |
| $$da = da^{(1)} + da^{(2)}$$ | **0.0889468** | **−0.7582175** | **0.6572330** | **0.0120376** |

*Table 8. The two parts of $$da$$. Only the target's column has two; the others have only the first.*

Read the column of word 1. Raise its score a little, and two things happen. Path 1: its $$e$$ grows, so
$$S$$ grows and the loss goes up by 0.2417825 per unit, its probability. Path 2: the $$-a_t$$ of F5
brings the loss down by 1 per unit. Path 2 wins: raising the right word's score lowers the loss, by
0.7582175 per unit.

### 3.6 B6: the logits, part 1, and the max, $$dz^{(1)}$$ and $$dm$$

F2, $$a_k = z_k - m$$, has two inputs, so two derivatives, each holding the other input fixed.

**With respect to $$z_j$$, holding $$m$$ fixed:**

$$
\begin{aligned}
\frac{\partial a_j}{\partial z_j} &= \frac{\partial z_j}{\partial z_j} - \frac{\partial m}{\partial z_j} = 1 - 0 = 1
  && m \text{ is held fixed} \\
dz^{(1)}_j &= da_j \cdot \frac{\partial a_j}{\partial z_j} = da_j
  && \text{chain rule}
\end{aligned}
$$

$$z_j$$ is in $$a_j$$ only, so there is one path through F2.

**With respect to $$m$$, holding $$z$$ fixed:**

$$
\frac{\partial a_k}{\partial m} = \frac{\partial z_k}{\partial m} - \frac{\partial m}{\partial m} = 0 - 1 = -1
\qquad \text{for every } k
$$

The same $$m$$ is subtracted in every column: $$V$$ paths to the loss, one per $$a_k$$. By the
shared-value rule:

$$
dm = da_1 \frac{\partial a_1}{\partial m} + \cdots + da_V \frac{\partial a_V}{\partial m}
   = \sum_{k=1}^{V} da_k \cdot (-1) = -\sum_{k=1}^{V} da_k
$$

**Example.** $$dz^{(1)} = da = [0.0889468,\ -0.7582175,\ 0.6572330,\ 0.0120376]$$, and

$$
dm = -(0.0889468 - 0.7582175 + 0.6572330 + 0.0120376) = 0
$$

($$-1 \times 10^{-7}$$ with the 7 digits shown, from rounding.)

**$$dm$$ is 0 in every row**, not only here. Put B5 into the sum:

$$
\begin{aligned}
\sum_k da_k &= \sum_k d\ell \cdot \frac{e_k}{S} - \sum_k d\ell \cdot [k = t]
  && \text{B5, summed over the row} \\
            &= d\ell \cdot \frac{e_1 + \cdots + e_V}{S} - d\ell \cdot 1
  && [k = t] \text{ is 1 in one column only} \\
            &= d\ell \cdot \frac{S}{S} - d\ell = 0
  && \text{F4} \\
dm &= -0 = 0
\end{aligned}
$$

### 3.7 B7: the logits, complete, $$dz$$

From F1, $$m = \max_k z_k$$, with respect to $$z_j$$, with Table 6:

$$
\begin{aligned}
\frac{\partial m}{\partial z_j} &= [j \text{ is the largest}]
  && \text{Table 6} \\
dz^{(2)}_j &= dm \cdot \frac{\partial m}{\partial z_j} = 0 \cdot [j \text{ is the largest}] = 0
  && \text{chain rule; } dm = 0 \text{ (B6)}
\end{aligned}
$$

$$z_j$$ is used by F1 and by F2, so by the used-twice rule:

$$
dz_j = dz^{(1)}_j + dz^{(2)}_j = da_j + 0 = da_j
$$

**Example.** $$dz = [0.0889468,\ -0.7582175,\ 0.6572330,\ 0.0120376]$$.

The path through the max carries nothing, because $$dm$$ is 0. That is why the max can be used without
worrying about its missing slope at ties: whatever its slope is, it is multiplied by 0.

### 3.8 All seven steps

| step | from | gradient of | formula | example, word 1 |
|---|---|---|---|---:|
| B1 | F5 | sum | $$dS = d\ell / S$$ | 0.6572330 (one per row) |
| B2 | F4 | exps | $$de_j = dS$$ | 0.6572330 |
| B3 | F3 | shifted score, part 1 | $$da^{(1)}_j = de_j \cdot e_j$$ | 0.2417825 |
| B4 | F5 | shifted score, part 2 | $$da^{(2)}_j = -d\ell \cdot [j = t]$$ | −1 |
| B5 | — | shifted score, complete | $$da_j = da^{(1)}_j + da^{(2)}_j$$ | −0.7582175 |
| B6 | F2 | logit, part 1; max | $$dz^{(1)}_j = da_j$$; $$dm = -\sum_k da_k$$ | −0.7582175; $$dm = 0$$ |
| B7 | F1 | logit, complete | $$dz_j = dz^{(1)}_j + dm \cdot [j \text{ is the largest}]$$ | −0.7582175 |

*Table 9. The backward pass, broken down. B1 needs only $$d\ell$$ and $$S$$; B2 to B7 each need the step
before. B4 needs only $$d\ell$$ and the target.*

Side by side with the three earlier notes:

| | LayerNorm [2] | linear [3] | GELU [4] | classifier |
|---|---|---|---|---|
| steps | 9 | 3 | 7 | 7 |
| a gradient summed over rows | $$d\gamma$$, $$d\beta$$ | $$dW$$, $$db$$ | none: no weights | none: no weights |
| a gradient summed over a row | $$dr$$, $$dv$$, $$d\mu$$ | $$dx$$, over the outputs | none: values never mix | $$dm$$, and it is 0 |
| a value used twice | $$x$$ and $$d$$ | none | $$x$$ | $$a_t$$ and $$z$$ |

*Table 10. Like LayerNorm, the classifier has one sum over the row ($$S$$) and one value taken from the
whole row ($$m$$). Unlike LayerNorm's, the gradient of that value is always 0.*

## 4. Rebuilding: C1 and C2

Table 9 has seven steps. This section folds them into one line.

**C1: one path, one product.** B1, B2 and B3 are one chain: each multiplies the one before by one
factor. Put them in a row:

$$
\begin{aligned}
da^{(1)}_j &= \underbrace{d\ell \cdot \frac{1}{S}}_{\text{B1: } dS} \cdot \underbrace{1}_{\text{B2}} \cdot \underbrace{e_j}_{\text{B3}}
           && \text{the chain rule, three factors} \\
           &= d\ell \cdot \frac{e_j}{S}
           && \text{put them together} \\
           &= d\ell \cdot p_j
           && \text{softmax: } p_j = e_j / S
\end{aligned}
$$

At word 1: $$1 \cdot 0.3678794 / 1.5215304 = 0.2417825$$, B3's number.

**C2: both parts, one line.** Add B4's part (B5), and pass $$da$$ through B6 and B7, which leave it
unchanged since $$dm = 0$$:

$$
\begin{aligned}
dz_j &= da_j
     && \text{B6, B7: } dz^{(1)}_j = da_j, \; dz^{(2)}_j = 0 \\
     &= da^{(1)}_j + da^{(2)}_j
     && \text{B5} \\
     &= d\ell \cdot p_j - d\ell \cdot [j = t]
     && \text{C1 and B4} \\
     &= d\ell \cdot \big(p_j - [j = t]\big)
     && \text{take } d\ell \text{ out}
\end{aligned}
$$

$$
\boxed{\;dz_j = d\ell \cdot \big(p_j - [j = t]\big), \qquad p_j = \frac{\exp(z_j - m)}{S}, \qquad S = \sum_k \exp(z_k - m)\;}
$$

The line at the top of this note. Every word is pushed down by as much probability as it got, and the
right word is pushed up by 1 on top of that.

![The example row: probabilities and gradient](/images/classifier-backward/fig2-probs-and-grad.svg)
*Figure 2. Left: the probabilities of the example. Right: $$dz$$, the same bars with 1 taken from the
target's. The optimizer moves each logit against its gradient, so word 1's score goes up and the
others go down.*

**Example**, straight from C2:

| | word 0 | word 1 (target) | word 2 | word 3 |
|---|---:|---:|---:|---:|
| $$p$$ | 0.0889468 | 0.2417825 | 0.6572330 | 0.0120376 |
| $$[j = t]$$ | 0 | 1 | 0 | 0 |
| $$dz = d\ell \cdot (p - [j = t])$$ | 0.0889468 | −0.7582175 | 0.6572330 | 0.0120376 |

*Table 11. The rebuilt line gives the same $$dz$$ as the seven steps (Table 8). In training,
$$d\ell = 1/4096$$, so word 1's gradient is $$-0.7582175 / 4096 = -0.0001851$$.*

## 5. Three checks

**Each row of $$dz$$ adds up to 0.** The probabilities add up to 1, and $$[j = t]$$ adds up to 1, so:

$$
\sum_j dz_j = d\ell \cdot \Big(\sum_j p_j - \sum_j [j = t]\Big) = d\ell \cdot (1 - 1) = 0
$$

It is B6's $$dm$$ with the sign turned, and it says the same as Section 1.1: adding one number to every
logit of a row changes no probability, so the gradient in that direction must be 0.

**Equal logits give $$\log V$$.** If every logit of a row is the same, every $$a_j$$ is 0, every
$$e_j$$ is 1, $$S = V$$, and

$$
\ell = \log V - 0 = \log V \qquad\qquad dz_j = d\ell \cdot \big(\tfrac{1}{V} - [j = t]\big)
$$

For GPT-2's vocabulary, $$\log 50257 = 10.8249051$$: the loss of an even guess, and close to where
training starts, when small random weights give logits near 0. With 4 words, $$\log 4 = 1.3862944$$
and $$dz = [0.25,\ -0.75,\ 0.25,\ 0.25]$$.

**The gradient stays between $$-d\ell$$ and $$d\ell$$.** Every $$p_j$$ is between 0 and 1, so the target's
$$dz_t = d\ell\,(p_t - 1)$$ is between $$-d\ell$$ and 0, and every other $$dz_j = d\ell\, p_j$$ between 0
and $$d\ell$$:

| row, target | $$p_t$$ | $$\ell$$ | $$dz / d\ell$$ |
|---|---:|---:|---|
| $$[0, 0, 20, 0]$$, target 2: sure, and right | $$\approx 1$$ | $$\approx 0$$ | $$\approx [0, 0, 0, 0]$$: nothing to learn |
| $$[0, 0, 20, 0]$$, target 0: sure, and wrong | $$\approx 2 \times 10^{-9}$$ | $$\approx 20$$ | $$\approx [-1, 0, 1, 0]$$: the largest push there is |

*Table 12. However wrong the model is, no number of $$dz$$ is larger than $$d\ell$$: the loss can grow
without limit (Table 1), but its gradient cannot.*

## 6. What the rebuilt formula gives the kernel

### 6.1 Two programs

The seven steps and the one line give the same numbers, but they are very different programs:

| | broken down (Table 9) | rebuilt (C2) |
|---|---|---|
| per word, besides $$dz$$ | $$a$$, $$e$$, $$de$$, $$da^{(1)}$$, $$da^{(2)}$$, $$da$$, $$dz^{(1)}$$, $$dz^{(2)}$$ | $$p$$: in registers, never stored |
| kept by the forward and needed | $$e$$ (or $$p$$) for every word | $$m$$ and $$S$$: two floats per row |
| GPU memory | many tensors of 206M values | read $$z$$, write $$dz$$ |

*Table 13. In GPT-2 small, the logits are 4 × 1024 × 50304 = 206M floats, 824 MB, by far the largest
tensor of the model. One more tensor of that size costs 824 MB; $$m$$ and $$S$$ for every row cost
32 KB.*

The kernels in my trainer [1] (`classifier.cuh`) are a forward, which works out F1 to F5 and keeps
$$m$$ and $$S$$, and a backward, which is C2. Both are limited by GPU memory, not by arithmetic.

### 6.2 The forward in one read: online softmax

F4 needs $$m$$ before it can start, so the plain way reads the row twice: once for the max, once for
the sum. The online softmax [5] finds both in one read. It keeps a running max $$m$$ and a running sum
$$s$$ measured against it. For each new value $$x$$:

$$
m' = \max(m, x) \qquad\qquad s \leftarrow s \cdot \exp(m - m') + \exp(x - m') \qquad\qquad m \leftarrow m'
$$

starting from $$m = -\infty$$, $$s = 0$$. Every term already in $$s$$ is $$\exp(x_k - m)$$, against the
old max. Multiplying by $$\exp(m - m')$$ moves all of them to the new max at once:

$$
\exp(x_k - m) \cdot \exp(m - m') = \exp(x_k - m + m - m') = \exp(x_k - m')
$$

**Example**, one thread over the example row:

| value | $$m$$ | $$s$$ |
|---:|---:|---|
| 1 | 1 | $$0 + e^{0} = 1$$ |
| 2 | 2 | $$1 \cdot e^{1-2} + e^{0} = 1.3678794$$ |
| 3 | 3 | $$1.3678794 \cdot e^{2-3} + e^{0} = 1.5032147$$ |
| −1 | 3 | $$1.5032147 \cdot e^{0} + e^{-1-3} = 1.5215304$$ |

*Table 14. The same $$m = 3$$ and $$S = 1.5215304$$ as F1 and F4 (Table 3), in one pass.*

The same rule puts two partial results together, which is what lets many threads share a row:

$$
m = \max(m_1, m_2) \qquad\qquad s = s_1 \exp(m_1 - m) + s_2 \exp(m_2 - m)
$$

Split the example between two threads: thread 1 takes $$[1, 2]$$ and gets $$(2,\ 1.3678794)$$, thread 2
takes $$[3, -1]$$ and gets $$(3,\ 1.0183156)$$. Merged: $$m = 3$$ and
$$s = 1.3678794 \cdot e^{-1} + 1.0183156 = 0.5032147 + 1.0183156 = 1.5215304$$. Any split and any order
give the same pair.

![One row, one read](/images/classifier-backward/fig3-online-softmax.svg)
*Figure 3. The forward kernel, one block of 1024 threads per row. Each thread runs the update rule over
its 12 or 13 float4s; the merge rule then puts the 1024 pairs together, across each warp with shuffles
and across the 32 warps in shared memory. Thread 0 writes the loss, $$m$$ and $$S$$.*

A row is 201 KB, and the logits 824 MB: one read instead of two halves the forward's memory traffic.

### 6.3 The padding: 50257 words in 50304 columns

GPT-2 has 50257 words, but the logits have 50304 columns. The head is a linear layer whose kernel reads
4 floats at a time and works in squares of 128 × 128, so its output width is rounded up to
$$393 \times 128 = 50304$$; nanoGPT does the same, for the same reason [7]. The 47 extra columns are
padding: the head works them out like any other column, but they are not words.

In the formulas, a padding column is a word with $$e_j = 0$$: it adds nothing to $$S$$, its $$p_j$$ is
0, and since a target is never padding, its $$dz_j = 0 - 0 = 0$$. The kernels get there as follows:

| | the forward | the backward |
|---|---|---|
| float4s read per row | $$(50257 + 3) / 4 = 12{,}565$$: those with at least one word | all 12,576 |
| float4 12,564: columns 50256 to 50259 | column 50256 is a word; the other three are set to the lowest float, so their exp is 0 | columns from 50257 get 0 |
| float4s 12,565 to 12,575 | padding only: not read | written as 0 |

*Table 15. Every float4 the forward reads starts with a word, so the max of its 4 values is always a real
score, and padding never counts as $$\exp(0) = 1$$.*

### 6.4 C2 in the kernel

The backward kernel is C2 written out, one thread per float4, 51,511,296 float4s in 201,216 blocks of
256 threads:

| in C2 | in the kernel | where |
|---|---|---|
| $$m$$, $$S$$ | `max_of_each_row`, `sum_of_each_row` | written by the forward's thread 0, one per row |
| $$\ell = \log S - (z_t - m)$$ | `losses` | `logf(row_sum) - (target_logit - row_max)`, F5 |
| $$d\ell$$ | `losses_grad` | $$1/4096$$ in every row, from the mean |
| $$p_j = \exp(z_j - m) / S$$ | `prob` | `expf(logit - row_max) * one_over_row_sum` |
| $$[j = t]$$ | `one_at_target` | `column == target ? 1.0f : 0.0f` |
| $$dz_j = d\ell\,(p_j - [j = t])$$ | `logits_grad` | `(prob - one_at_target) * row_loss_grad`, 0 for padding |

*Table 16. Each value of $$dz$$ needs its own logit and four numbers of its row: $$m$$, $$S$$, $$t$$ and
$$d\ell$$.*

**Working $$p$$ out again instead of keeping it.** Keeping $$p$$ would cost the forward one more 824 MB
write and the backward one more 824 MB read. Working it out again costs one `expf` and one multiply per
value, while the thread waits for memory anyway. LayerNorm's kernel makes the same choice for
$$\hat{x}$$ [2], and GELU's for $$\tanh$$ [4].

**`=` for $$dz$$**, from the rules of Section 2: the logits are used by the classifier alone, so $$dz$$
has one part and is written over, not added onto. And since each value of $$dz$$ needs only its own
logit and its row's four numbers, the backward may write $$dz$$ into the logits' own memory, without a
second 824 MB. llm.c's CUDA trainer goes one step further and does the forward and the backward in one
kernel, `fused_classifier` [6]. Mine keeps them apart, like every other kernel of the trainer, and pays
one more read of the logits for it.

## 7. What to remember

1. **One step per arrow.** The backward pass walks Figure 1's arrows backwards, each step with the
   chain rule and its other inputs held fixed.
2. **The sum gives the probabilities.** $$\log S$$, then the sum, then exp: B1 to B3 multiply
   $$\frac{1}{S} \cdot 1 \cdot e_j$$, which is softmax. Every word gets its own probability as its
   gradient.
3. **The target gets one more part.** $$a_t$$ is used twice, by F3 and by F5, so the right word's
   gradient is its probability minus 1: $$dz = d\ell \cdot (p - [j = t])$$.
4. **The max drops out.** $$dm = -\sum_k da_k = 0$$, because the probabilities add up to 1. The shift
   is there only to keep exp from overflowing.
5. **One read, two numbers kept.** The forward finds $$m$$ and $$S$$ in one read with the online
   softmax and keeps only those; the backward works every probability out again from them.

## References

1. D. B. Song. *gpt2-small*, branch `c-cuda-trainer`: GPT-2 small pre-training in C and CUDA.
   [github.com/ducbachsong/gpt2-small](https://github.com/ducbachsong/gpt2-small/tree/c-cuda-trainer)
2. D. B. Song. *LayerNorm backward, broken down and rebuilt: nine chain-rule steps and one formula.* 2026.
   [/posts/layer-norm-backward/](/posts/layer-norm-backward/)
3. D. B. Song. *Linear backward, broken down and rebuilt: three chain-rule steps and three matrix
   products.* 2026. [/posts/linear-backward/](/posts/linear-backward/)
4. D. B. Song. *GELU backward, broken down and rebuilt: seven chain-rule steps and one slope.* 2026.
   [/posts/gelu-backward/](/posts/gelu-backward/)
5. M. Milakov, N. Gimelshein. *Online normalizer calculation for softmax.* 2018.
   [arxiv.org/abs/1805.02867](https://arxiv.org/abs/1805.02867)
6. A. Karpathy. *llm.c*, `train_gpt2.cu`: `fused_classifier`, softmax, loss and gradient in one CUDA
   kernel. [github.com/karpathy/llm.c](https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu)
7. A. Karpathy. *nanoGPT*, `model.py`: `vocab_size = 50304`, GPT-2's 50257 padded up for efficiency.
   [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT/blob/master/model.py)
