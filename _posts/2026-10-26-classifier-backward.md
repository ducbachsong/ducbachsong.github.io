---
title: "Classifier backward, broken down and rebuilt: softmax, cross-entropy and prob minus one"
date: 2026-10-26
modified: 2026-10-26
version: 1
permalink: /posts/classifier-backward/
excerpt: "The last step of GPT-2 turns each row of 50257 scores into one loss, and its backward pass is usually given as one line: dz = p − 1 at the right word. This note takes the forward apart into five small steps, from the logits to a shift by the row's largest score, an exp, a sum and a log, and the backward into seven, each derived with nothing more than the chain rule, and works one row of four words through all of them. The shift drops out of the gradient, the sum turns every column's gradient into its probability, and the target's column gets one more part, −1. Then it puts the steps back together into the one line a GPU kernel runs, and shows how the kernel finds the row's largest score and its sum in one read of the row: the online softmax."
tags:
  - softmax
  - cross-entropy
  - backpropagation
  - gpt-2
  - notes
---

**What this is.** GPT-2 small ends every forward pass with a classifier: each of its 4096 rows of
logits, one score for every word of the vocabulary, becomes probabilities with softmax, and the loss is
how unlikely the right next word was. The backward pass of it is usually written as a single line:

$$
dz_j = d\ell \cdot \big(p_j - [j = t]\big) \qquad p_j = \text{softmax}(z)_j
$$

with $$[j = t]$$ equal to 1 in the right word's column and 0 in every other. That line is correct, and
it is what the kernel in my GPT-2 trainer [1] runs, but it is hard to see why a gradient should be a
probability minus one. This note **breaks it down** into seven small steps, B1 to B7, one for each arrow
of the forward pass, each derived with nothing more than the chain rule. Then it **rebuilds** the line
from those steps, in two moves, C1 and C2. One row of 4 words goes through every step, so each formula
has a number next to it. Section 6 then shows how the kernel works out the forward in one read of the
row.

**Sources.** The derivation is worked out by hand, with the same three rules as the notes on
[LayerNorm's backward](/posts/layer-norm-backward/) [2], [the linear layer's](/posts/linear-backward/) [3]
and [GELU's](/posts/gelu-backward/) [4], and every number is worked out from the formulas, to 7
digits. The one-read
forward is the online softmax of Milakov and Gimelshein [5]; llm.c's CUDA classifier uses it too [6].

* TOC
{:toc}

## 1. The forward pass

### 1.1 What the classifier does: scores to probabilities to one loss

The head, GPT-2's last linear layer, gives every position one score per word, the **logits** $$z$$. A
large $$z_j$$ means "word $$j$$ is a likely next word here". Softmax turns a row of scores into
probabilities, all between 0 and 1 and adding up to 1:

$$
p_j = \frac{\exp(z_j)}{\sum_k \exp(z_k)}
$$

Each part of the formula has a job:

| part | what it does |
|---|---|
| $$\exp(z_j)$$ | makes every score positive, even a negative one: $$\exp(-3) = 0.05$$. A probability cannot be negative. |
| $$\sum_k \exp(z_k)$$ | adds up all of them, so dividing by it makes the row add up to 1 |
| the two together | keep the order of the scores: a larger score always gets a larger probability |

*Table 1. Why softmax is built the way it is.*

exp also turns a **difference** of scores into a **ratio** of probabilities. Divide two of them, and the
sum cancels:

$$
\frac{p_i}{p_j} = \frac{\exp(z_i)}{\exp(z_j)} = \exp(z_i - z_j)
$$

A score 1 higher means $$e = 2.718$$ times more probability, 2 higher means $$e^2 = 7.389$$ times, whatever
the other scores are. Only the gaps between scores matter, never the scores themselves. The shift below uses
this.

The loss of the row is the **cross-entropy**: minus the log of the probability the right next word got.

$$
\ell = -\log p_t
$$

Why a log, and why minus? The model gives a whole text the probability of its first word, times that of
its second, and so on: a product of thousands of $$p_t$$, far too small for a float. The log turns the
product into a sum, $$\log(p_1 p_2 \cdots) = \log p_1 + \log p_2 + \cdots$$, so each position adds its
own term. Every $$p_t$$ is at most 1, so every $$\log p_t$$ is at most 0, and the minus turns it into a
cost: 0 is perfect, and larger is worse. The loss of the step is the mean of $$\ell$$ over the rows.

| $$p_t$$, the right word's probability | $$\ell$$ | meaning |
|---:|---:|---|
| 1 | 0 | sure, and right |
| 0.5 | 0.69 | |
| 0.01 | 4.61 | |
| 1 / 50257 | 10.82 | an even guess over GPT-2's vocabulary |

*Table 2. The loss of one row. It is 0 only when the right word gets all the probability, and it grows
without limit as that probability goes to 0.*

Written as it stands, $$\exp(z_j)$$ overflows: in a float, $$\exp(89)$$ is already infinity. So every
implementation first subtracts the row's largest score, $$m$$, from every score. Only the gaps matter, so
that changes no probability. Written out, $$\exp(z_j - m) = \exp(z_j) / \exp(m)$$, and the $$\exp(m)$$
cancels between top and bottom:

$$
\frac{\exp(z_j - m)}{\sum_k \exp(z_k - m)} = \frac{\exp(z_j) / \exp(m)}{\sum_k \exp(z_k) / \exp(m)} = \frac{\exp(z_j)}{\sum_k \exp(z_k)} = p_j
$$

After the shift the largest score is 0, so the largest term is $$\exp(0) = 1$$ and every other is
between 0 and 1: nothing can overflow.

### 1.2 Five steps for one row

The classifier works on one row at a time: rows never mix. For one row of $$V$$ words with target $$t$$,
the formula is five steps:

$$
\begin{aligned}
&\text{F1:} \qquad m = \max_k z_k && \text{the row's largest score} \\
&\text{F2:} \qquad a_j = z_j - m && \text{shift every score: now the largest is 0} \\
&\text{F3:} \qquad e_j = \exp(a_j) && \text{every term between 0 and 1} \\
&\text{F4:} \qquad S = \textstyle\sum_k e_k && \text{one sum over the row} \\
&\text{F5:} \qquad \ell = \log S - a_t && \text{the loss}
\end{aligned}
$$

F5 is $$-\log p_t$$ written out: $$p_t = e_t / S$$, so $$-\log p_t = \log S - \log e_t = \log S - a_t$$.
The probabilities themselves are not a step of the forward. They come back in the rebuild (C1).

### 1.3 The names

| symbol | meaning | shape in GPT-2 small |
|---|---|---|
| $$z$$ | logits: one score per word, per position; the head's output | `{4, 1024, 50304}` |
| $$t$$ | target: the right next word, the input shifted by one position | `{4, 1024}` ints |
| $$V$$ | vocabulary size: 50257 words (Section 6.3 explains the 50304) | — |
| $$m$$ | the row's largest score | one per row: `{4, 1024}` |
| $$a, e$$ | shifted scores and their exps | one per word |
| $$S$$ | the sum of $$e$$ over the row | one per row |
| $$\ell$$ | the row's loss | one per row: `{4, 1024}` |
| $$p$$ | probabilities, $$e / S$$ | one per word |
| $$[j = t]$$ | 1 if $$j$$ is the target's column, else 0 | — |
| $$L$$ | the loss of the step: the mean of $$\ell$$ over the $$R = 4096$$ rows | 1 |
| $$da$$ | short for $$\partial L / \partial a$$, for any variable $$a$$; same shape as $$a$$ | — |

*Table 3. $$d\ell$$ is given: it is what the mean over the rows sends back, $$1 / R = 1/4096$$ in every
row. The job is to find $$dz$$. The classifier has no weight and no bias, so $$dz$$ is the only
gradient.*

Where the classifier sits, after the last of GPT-2 small's 12 blocks:

```
{4, 1024, 768} ── final LayerNorm ──► {4, 1024, 768} ── head (· wteᵀ) ──► {4, 1024, 50304} ── classifier ──► {4, 1024}
```

### 1.4 The running example

One row of four words, $$z = [1,\ 2,\ 3,\ -1]$$, with target $$t = 1$$: the right word has only the
second highest score, so the gradient has something to fix. $$d\ell = 1$$, so $$dz$$ comes out as the
part that multiplies $$d\ell$$; in training every number of $$dz$$ is multiplied by $$1/4096$$.

| step | word 0 | word 1 (target) | word 2 | word 3 | per row |
|---|---:|---:|---:|---:|---:|
| $$z$$ | 1 | 2 | 3 | −1 | |
| F1: $$m = \max z$$ | | | | | 3 |
| F2: $$a = z - m$$ | −2 | −1 | 0 | −4 | |
| F3: $$e = \exp a$$ | 0.1353353 | 0.3678794 | 1 | 0.0183156 | |
| F4: $$S = \sum e$$ | | | | | 1.5215304 |
| F5: $$\ell = \log S - a_t$$ | | | | | 0.4197166 + 1 = **1.4197166** |
| $$p = e / S$$ | 0.0889468 | 0.2417825 | 0.6572330 | 0.0120376 | sum 1 |

*Table 4. The example, forward. The right word got 24% of the probability, and
$$-\log 0.2417825 = 1.4197166$$, the loss of F5.*

### 1.5 Where each value is used

![The classifier's forward for one row](/images/classifier-backward/fig1-graph.svg)
*Figure 1. Each arrow is one use of a value. The backward pass walks the same arrows the other way, one
step per arrow. $$a$$ and $$z$$ each have two arrows going out, so each of their gradients has two
parts.*

Figure 1 is the whole plan. Seven arrows, so seven backward steps, B1 to B7. B5 adds up the two parts
of $$da$$, and B7 the two parts of $$dz$$.

## 2. Three rules and three derivatives

Every step below uses the same three rules as the earlier notes [2, 3, 4]:

| rule | when | what it says |
|---|---|---|
| **chain rule** | $$a$$ affects $$L$$ through $$b$$ | $$da = db \cdot \partial b / \partial a$$ |
| **shared value: sum** | $$a$$ is used in many places (all rows, or all columns of a row) | add one chain-rule term per place |
| **used twice: add the parts** | $$a$$ is used by two different forward steps | each step gives one part; $$da$$ = part 1 + part 2 |

*Table 5. The three rules. The classifier needs all three: $$m$$ is used by every column (B6), and
$$a_t$$ and $$z$$ are each used by two steps (B5, B7).*

And one habit: **each step looks at one input and holds the others fixed.** F2,
$$a_j = z_j - m$$, has two inputs; B6 takes the derivative with respect to $$z_j$$ holding $$m$$ fixed,
even though $$m$$ was made from $$z$$. The path through $$m$$ is followed by B7.

Three derivatives are needed beyond the rules:

| function | derivative | why |
|---|---|---|
| $$\exp a$$ | $$\exp a$$ | exp is its own slope |
| $$\log S$$ | $$1 / S$$ | the inverse of exp |
| $$\max_k z_k$$ | 1 for the largest $$z_j$$, 0 for the others | only the largest moves the max |

*Table 6. The three derivatives of the classifier's steps. The others, of $$z_j - m$$ and of a sum, are 1
and −1. The max has no slope where two scores tie for largest, but B7 shows that it never matters.*

**The derivative of exp.** exp turns a sum into a product, $$\exp(a + h) = \exp(a) \cdot \exp(h)$$, and
for a small $$h$$, $$\exp(h) \approx 1 + h$$. So nudging $$a$$ by $$h$$ gives:

$$
\exp(a + h) - \exp(a) = \exp(a) \cdot \big(\exp(h) - 1\big) \approx \exp(a) \cdot h
\qquad\text{so}\qquad \frac{d}{da} \exp a = \exp a
$$

The slope of exp at any point is its own value there. In the example, at $$a = -1$$:
$$e^{-1} = 0.3678794$$, the value and the slope both.

**The derivative of log.** log undoes exp: if $$y = \log S$$, then $$S = \exp(y)$$. By the line above,
$$S$$ changes $$\exp(y) = S$$ times as fast as $$y$$. Turned round, $$y$$ changes $$1/S$$ times as fast
as $$S$$:

$$
\frac{dS}{dy} = S \qquad\text{so}\qquad \frac{d}{dS} \log S = \frac{1}{S}
$$

In the example, $$S = 1.5215304$$, so $$\log S$$ moves $$1 / 1.5215304 = 0.6572330$$ for each unit
$$S$$ moves.

**The derivative of max.** In $$z = [1, 2, 3, -1]$$, $$m = 3$$ comes from word 2. Raise $$z_2$$ from 3 to
3.1, and $$m$$ becomes 3.1: it moves one for one. Raise $$z_0$$ from 1 to 1.1, and $$m$$ stays 3: it
does not move at all. So:

$$
\frac{\partial m}{\partial z_j} = \begin{cases} 1 & \text{if } z_j \text{ is the largest} \\ 0 & \text{otherwise} \end{cases}
$$

## 3. Breaking it down: B1 to B7

Each step is for one row. The whole tensor is the same steps for each of its 4096 rows, with nothing
added across rows.

### 3.1 B1: the sum, $$dS$$

From F5, $$\ell = \log S - a_t$$, with respect to $$S$$, holding $$a_t$$ fixed:

$$
\frac{\partial \ell}{\partial S} = \frac{1}{S} \qquad\qquad dS = \frac{d\ell}{S}
$$

**Example.** $$dS = 1 / 1.5215304 = 0.6572330$$. It is also $$p_2$$, word 2's probability: word 2 has
the largest score, so its $$e$$ is exactly 1, and $$p_2 = 1/S$$.

### 3.2 B2: the exps, $$de$$

From F4, $$S = e_0 + e_1 + \dots$$: every $$e_j$$ is added once, so:

$$
\frac{\partial S}{\partial e_j} = 1 \qquad\qquad de_j = dS
$$

**Example.** $$de = [0.6572330,\ 0.6572330,\ 0.6572330,\ 0.6572330]$$: every column gets the same.

### 3.3 B3: the shifted scores, part 1, $$da^{(1)}$$

From F3, $$e_j = \exp a_j$$, with Table 6:

$$
\frac{\partial e_j}{\partial a_j} = e_j \qquad\qquad da^{(1)}_j = de_j \cdot e_j
$$

**Example.** $$da^{(1)} = [0.0889468,\ 0.2417825,\ 0.6572330,\ 0.0120376]$$: the probabilities of
Table 4, exactly. B1 to B3 multiplied each $$e_j$$ by $$1/S$$, and that is what softmax is. C1 says it
in one line.

### 3.4 B4: the shifted scores, part 2, $$da^{(2)}$$

From F5 again, with respect to $$a_t$$, holding $$S$$ fixed. Only the target's column is in F5:

$$
\frac{\partial \ell}{\partial a_t} = -1 \qquad\qquad da^{(2)}_j = -d\ell \cdot [j = t]
$$

**Example.** $$da^{(2)} = [0,\ -1,\ 0,\ 0]$$.

This is the part that pushes the right word up. Raise $$a_t$$, and $$\ell$$ falls one for one, before
counting what it does to $$S$$; B1 to B3 counted that.

### 3.5 B5: the shifted scores, complete

$$a_t$$ is used by F3 and by F5, so by the used-twice rule:

$$
da_j = da^{(1)}_j + da^{(2)}_j
$$

| | word 0 | word 1 (target) | word 2 | word 3 |
|---|---:|---:|---:|---:|
| $$da^{(1)}$$ (B3): through $$S$$ | 0.0889468 | 0.2417825 | 0.6572330 | 0.0120376 |
| $$da^{(2)}$$ (B4): the target | 0 | −1 | 0 | 0 |
| $$da$$ | **0.0889468** | **−0.7582175** | **0.6572330** | **0.0120376** |

*Table 7. The two parts of $$da$$. Only the target's column has two; the others have only the first.*

Read the column of word 1. Raise its score a little, and two things happen. Part 1: its $$e$$ grows, so
$$S$$ grows and the loss goes up by 0.242 per unit, its probability. Part 2: the $$-a_t$$ of F5 brings
the loss down by 1 per unit. Part 2 wins: raising the right word's score lowers the loss, by 0.758 per
unit.

### 3.6 B6: the logits, part 1, and the max, $$dz^{(1)}$$ and $$dm$$

From F2, $$a_j = z_j - m$$, with respect to $$z_j$$, holding $$m$$ fixed, and with respect to $$m$$,
holding $$z_j$$ fixed:

$$
\frac{\partial a_j}{\partial z_j} = 1 \qquad dz^{(1)}_j = da_j
\qquad\qquad
\frac{\partial a_j}{\partial m} = -1 \qquad dm = -\sum_j da_j
$$

$$m$$ is used by every column of F2, so $$dm$$ is a sum over the row: the shared-value rule.

**Example.** $$dz^{(1)} = da$$, and
$$dm = -(0.0889468 - 0.7582175 + 0.6572330 + 0.0120376) = 0$$ (it comes out as $$1.9 \times 10^{-16}$$, from
rounding).

It is 0 in every row, not only here. Add up B5's two parts: part 1 adds up to
$$d\ell \cdot \sum_j e_j / S = d\ell \cdot S/S = d\ell$$, and part 2 is $$-d\ell$$. So:

$$
dm = -\Big(d\ell - d\ell\Big) = 0
$$

### 3.7 B7: the logits, complete

From F1, $$m = \max_k z_k$$, with Table 6, and $$z$$ is used by F1 and F2, so by the used-twice rule:

$$
dz^{(2)}_j = dm \cdot [j \text{ is the largest}] = 0 \qquad\qquad dz_j = dz^{(1)}_j + dz^{(2)}_j = da_j
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
| B6 | F2 | logit, part 1; max | $$dz^{(1)}_j = da_j$$; $$dm = -\sum_j da_j$$ | −0.7582175; $$dm = 0$$ |
| B7 | F1 | logit, complete | $$dz_j = dz^{(1)}_j + dm \cdot [j \text{ is the largest}]$$ | −0.7582175 |

*Table 8. The backward pass, broken down. B1 needs only $$d\ell$$ and $$S$$; B2 to B7 each need the step
before. B4 needs only $$d\ell$$ and the target.*

Side by side with the three earlier notes:

| | LayerNorm [2] | linear [3] | GELU [4] | classifier |
|---|---|---|---|---|
| steps | 9 | 3 | 7 | 7 |
| a gradient summed over rows | $$d\gamma$$, $$d\beta$$ | $$dW$$, $$db$$ | none: no weights | none: no weights |
| a gradient summed over a row | $$dr$$, $$dv$$, $$d\mu$$ | $$dx$$, over the outputs | none: values never mix | $$dm$$, and it is 0 |
| a value used twice | $$x$$ and $$d$$ | none | $$x$$ | $$a_t$$ and $$z$$ |

*Table 9. Like LayerNorm, the classifier has one sum over the row ($$S$$) and one value taken from the
whole row ($$m$$). Unlike LayerNorm's, the gradient of that value is always 0.*

## 4. Rebuilding: C1 and C2

Table 8 has seven steps. This section folds them into one line.

**C1: one path, one product.** B1, B2 and B3 are a chain: each multiplies the one before by one factor.
Put them in a row:

$$
da^{(1)}_j = \underbrace{\frac{d\ell}{S}}_{\text{B1}} \cdot \underbrace{1}_{\text{B2}} \cdot \underbrace{e_j}_{\text{B3}}
           = d\ell \cdot \frac{e_j}{S} = d\ell \cdot p_j
$$

$$e_j / S$$ is the softmax: the probabilities come back here, in the backward, though the forward never
needed them. At word 1: $$1 \cdot 0.3678794 / 1.5215304 = 0.2417825$$, B3's number.

**C2: both parts, one line.** Add B4's part; B6 and B7 pass $$da$$ on unchanged, since $$dm = 0$$:

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

*Table 10. The rebuilt line gives the same $$dz$$ as the seven steps (Table 7).*

## 5. Three checks

**Each row of $$dz$$ adds up to 0.** The probabilities add up to 1, and $$[j = t]$$ adds up to 1, so:

$$
\sum_j dz_j = d\ell \cdot (1 - 1) = 0
$$

It is the same sum as B6's $$dm$$, with the sign turned. And it says the same thing as Section 1.1:
adding one number $$c$$ to every logit of a row changes no probability, so it changes no loss, and the
gradient in that direction, $$\sum_j dz_j$$, must be 0. In the example:
$$0.0889468 - 0.7582175 + 0.6572330 + 0.0120376 = 0$$ (to the 7 digits shown, $$-1 \times 10^{-7}$$ of
rounding in the last one).

**Equal logits give $$\log V$$.** If every logit of a row is the same, every $$e_j$$ is 1, $$S = V$$, and

$$
\ell = \log V - 0 = \log V \qquad\qquad dz_j = d\ell \cdot \big(\tfrac{1}{V} - [j = t]\big)
$$

For GPT-2's vocabulary, $$\log 50257 = 10.8249051$$: the loss of an even guess, and close to where
training starts, when small random weights give logits near 0. With 4 words, $$\log 4 = 1.3862944$$
and $$dz = [0.25,\ -0.75,\ 0.25,\ 0.25]$$, exactly; the kernel's tests check both.

**The gradient stays between $$-d\ell$$ and $$d\ell$$.** Every $$p_j$$ is between 0 and 1, so the target's
$$dz_t = d\ell (p_t - 1)$$ is between $$-d\ell$$ and 0, and every other $$dz_j$$ between 0 and
$$d\ell$$. At the two ends:

| row, target | $$p_t$$ | $$\ell$$ | $$dz / d\ell$$ |
|---|---:|---:|---|
| $$[0, 0, 20, 0]$$, target 2: sure, and right | $$\approx 1$$ | $$\approx 0$$ | $$\approx [0, 0, 0, 0]$$: nothing to learn |
| $$[0, 0, 20, 0]$$, target 0: sure, and wrong | $$\approx 2 \times 10^{-9}$$ | $$\approx 20$$ | $$\approx [-1, 0, 1, 0]$$: the largest push there is |

*Table 11. Each of the other words has $$e^{-20} = 2.1 \times 10^{-9}$$ of the probability. However wrong
the model is, no number of $$dz$$ is larger than $$d\ell$$: the loss can grow without limit (Table 2),
but its gradient cannot.*

## 6. What the rebuilt formula gives the kernel

### 6.1 Two programs

The seven steps and the one line give the same numbers, but they are very different programs:

| | broken down (Table 8) | rebuilt (C2) |
|---|---|---|
| per word, besides $$dz$$ | $$a$$, $$e$$, $$de$$, $$da^{(1)}$$, $$da^{(2)}$$, $$da$$, $$dz^{(1)}$$, $$dz^{(2)}$$ | $$p$$: in registers, never stored |
| kept by the forward and needed | $$e$$ (or $$p$$) for every word | $$m$$ and $$S$$: two floats per row |
| GPU memory | many tensors of 206M values | read $$z$$, write $$dz$$ |

*Table 12. In GPT-2 small, the logits are 4 × 1024 × 50304 = 206M floats, 824 MB, by far the largest
tensor of the model. One more tensor of that size costs 824 MB; $$m$$ and $$S$$ for every row cost
32 KB.*

The kernels in my trainer [1] (`classifier.cuh`) are a forward, which works out F1 to F5 and keeps
$$m$$ and $$S$$, and a backward, which is C2. Both are limited by GPU memory, not by arithmetic: the
logits are read once by each.

### 6.2 The forward in one read: online softmax

F4 needs $$m$$ before it can start, so the plain way reads the row twice: once for the max, once for
the sum. The online softmax [5] finds both in one read. It keeps a running max $$m$$ and a running sum
$$s$$ measured against it. For each new value $$x$$:

$$
m' = \max(m, x) \qquad\qquad s \leftarrow s \cdot \exp(m - m') + \exp(x - m') \qquad\qquad m \leftarrow m'
$$

starting from $$m = -\infty$$, $$s = 0$$. When the max grows, the sum so far was worked out against the
old max; multiplying by $$\exp(m - m')$$ moves every term of it to the new one at once, since
$$\exp(x - m) \cdot \exp(m - m') = \exp(x - m')$$. The values themselves are not kept: $$s$$ is the only
memory of them.

**Example**, one thread over the example row:

| value | $$m$$ | $$s$$ |
|---:|---:|---|
| 1 | 1 | $$0 + e^{0} = 1$$ |
| 2 | 2 | $$1 \cdot e^{1-2} + e^{0} = 1.3678794$$ |
| 3 | 3 | $$1.3678794 \cdot e^{2-3} + e^{0} = 1.5032147$$ |
| −1 | 3 | $$1.5032147 \cdot e^{0} + e^{-1-3} = 1.5215304$$ |

*Table 13. The same $$m = 3$$ and $$S = 1.5215304$$ as F1 and F4 (Table 4), in one pass.*

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

*Table 14. Every float4 the forward reads starts with a word, so the max of its 4 values is always a real
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

*Table 15. Each value of $$dz$$ needs its own logit and four numbers of its row: $$m$$, $$S$$, $$t$$ and
$$d\ell$$. The 256 threads of a block are all in the same row, so after the first thread those four come
from the cache.*

**Working $$p$$ out again instead of keeping it.** Keeping $$p$$ would cost the forward one more 824 MB
write and the backward one more 824 MB read. Working it out again costs one `expf` and one multiply per
value, while the thread waits for memory anyway. LayerNorm's kernel makes the same choice for
$$\hat{x}$$ [2], and GELU's for $$\tanh$$ [4].

**`=` for $$dz$$**, from the rules of Section 2:

| gradient | the variable is used | so |
|---|---|---|
| $$dz$$ | by the classifier alone: the logits go nowhere else | one gradient, nothing to add: written over |
| — | the classifier has no weights | nothing is added up over rows, so nothing to set to 0 per step |

*Table 16. And since each value of $$dz$$ needs only its own logit and its row's four numbers, the
backward may write $$dz$$ into the logits' own memory: the trainer can turn the 824 MB of logits into
their gradient without a second 824 MB.*

llm.c's CUDA trainer goes one step further and does the forward and the backward in one kernel,
`fused_classifier`, writing the gradient over the logits [6]. Mine keeps them apart, like every other
kernel of the trainer, and pays one more read of the logits for it.

The time is GPU memory: the forward reads 824 MB, the backward reads 824 MB and writes 824 MB. At a T4's
320 GB/s, that is at least 2.58 ms and 5.15 ms. How close the kernels get to that is for a later entry:
nothing here is timed.

## 7. What to remember

1. **One step per arrow.** The backward pass walks Figure 1's arrows backwards, each step with the
   chain rule and its other inputs held fixed.
2. **The sum gives the probabilities.** $$\log S$$, then the sum, then exp: B1 to B3 multiply each
   $$e_j$$ by $$1/S$$, which is softmax. Every word gets its own probability as its gradient.
3. **The target gets one more part.** $$a_t$$ is used twice, so the right word's gradient is its
   probability minus 1: $$dz = d\ell \cdot (p - [j = t])$$.
4. **The max drops out.** Shifting every logit changes nothing, so $$dm = 0$$ and every row of $$dz$$
   adds up to 0. The shift is there only to keep exp from overflowing.
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
