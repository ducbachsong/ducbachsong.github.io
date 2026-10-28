---
title: "GELU backward, broken down and rebuilt: seven chain-rule steps and one slope"
date: 2026-10-21
modified: 2026-10-28
version: 2
permalink: /posts/gelu-backward/
excerpt: "The backward pass of GELU is usually given as one line: dx = dy · gelu'(x), with a slope of two terms and a tanh in each. This note takes the forward apart into five small steps, from x to a gate between 0 and 1 and back to y = x times the gate, and the backward into seven, each derived with nothing more than the chain rule, and works four numbers through all of them. x is used twice, so its gradient has two parts: the gate as it is, and how fast the gate opens. Then it puts the steps back together into the one slope a GPU kernel runs. Two checks follow from the formula, and every number is checked against PyTorch."
tags:
  - gelu
  - backpropagation
  - gpt-2
  - notes
---

**What this is.** GPT-2 small runs GELU 12 times in every forward pass, once in the MLP of each
block, and the backward pass of each one is usually written as a single line:

$$
dx = dy \cdot \Big( \tfrac{1}{2}\big(1 + \tanh s\big) + \tfrac{1}{2}\, x \,\big(1 - \tanh^2 s\big)\, \beta\, \big(1 + 3\kappa x^2\big) \Big)
\qquad s = \beta\,(x + \kappa x^3)
$$

That line is correct, and it is what the kernel in my GPT-2 trainer [1] runs, but it is hard to see
where its two terms come from. This note **breaks it down** into seven small steps, B1 to B7, one for
each arrow of the forward pass, each derived with nothing more than the chain rule. Then it
**rebuilds** the line from those steps, in two moves, C1 and C2. One example of 4 numbers goes through
every step, so each formula has a number next to it.

**Sources.** The derivation is worked out by hand, with the same three rules as the notes on
[LayerNorm's backward](/posts/layer-norm-backward/) [2] and [the linear layer's](/posts/linear-backward/)
[3]. Every number in it was checked against PyTorch's `gelu` with `approximate="tanh"` and its autograd
in double precision [5]; Section 6 gives the results. GELU is from Hendrycks and Gimpel [4]; the tanh
form is the one GPT-2's code uses [6], and the one llm.c's C code runs [7].

* TOC
{:toc}

## 1. The forward pass

### 1.1 What GELU does: x times a gate

GELU multiplies each value by a **gate** $$p$$ between 0 and 1 that depends on the value itself:

$$
y = x \cdot p(x)
$$

For a large positive $$x$$ the gate is open, $$p \approx 1$$, and $$y \approx x$$. For a large negative
$$x$$ it is shut, $$p \approx 0$$, and $$y \approx 0$$. At $$x = 0$$ it is half open, $$p = 0.5$$. ReLU
does the same with a gate that jumps from 0 to 1 at $$x = 0$$; GELU's gate opens smoothly, so it has a
slope everywhere, and for small negative $$x$$ a little of $$x$$ gets through.

The exact gate is $$\Phi(x)$$, the chance that a value drawn from the standard normal distribution is
below $$x$$ [4]. $$\Phi$$ has no short formula, so GPT-2 uses the tanh form from the same paper:

$$
p(x) = \tfrac{1}{2}\Big(1 + \tanh\big(\beta\,(x + \kappa x^3)\big)\Big) \qquad \beta = \sqrt{2/\pi} = 0.7978846 \qquad \kappa = 0.044715
$$

It is a different function from $$x\,\Phi(x)$$, but close: the two differ by at most 0.00047, at
$$x \approx -2.70$$ (Section 6). PyTorch computes $$x\,\Phi(x)$$ unless asked for `approximate="tanh"`
[5].

### 1.2 Five steps for one value

GELU works on one value at a time: values never mix. For one value, the formula is five steps:

$$
\begin{aligned}
&\text{F1:} \qquad u = x + \kappa x^3 && \text{bend } x \text{ a little: } x^3 \text{ makes large values larger} \\
&\text{F2:} \qquad s = \beta\, u && \text{scale it} \\
&\text{F3:} \qquad h = \tanh(s) && \text{squash it between } -1 \text{ and } 1 \\
&\text{F4:} \qquad p = \tfrac{1}{2}(1 + h) && \text{move it between 0 and 1: the gate} \\
&\text{F5:} \qquad y = x \cdot p && \text{let that much of } x \text{ through}
\end{aligned}
$$

### 1.3 The names

| symbol | meaning | shape in GPT-2 small |
|---|---|---|
| $$x$$ | input: one per value; the output of the MLP's first linear layer | `{4, 1024, 3072}` |
| $$u, s, h$$ | the steps on the way to the gate: bent, scaled, squashed | one per value |
| $$p$$ | the gate, between 0 and 1 | one per value |
| $$y$$ | output: one per value; the input of the MLP's second linear layer | `{4, 1024, 3072}` |
| $$\kappa, \beta$$ | two constants, 0.044715 and $$\sqrt{2/\pi}$$; nothing is learned | — |
| $$L$$ | the loss: one number, how wrong the model is | 1 |
| $$\partial$$ | partial derivative: change one variable, hold all others fixed | — |
| $$da$$ | short for $$\partial L / \partial a$$, for any variable $$a$$; same shape as $$a$$ | — |

*Table 1. $$dy$$ is given: it is what the layer after GELU sends back. The job is to find $$dx$$. GELU has
no weight and no bias, so $$dx$$ is the only gradient.*

Where GELU sits in each of GPT-2 small's 12 blocks:

```
{4, 1024, 768} ── MLP up ──► {4, 1024, 3072} ── GELU ──► {4, 1024, 3072} ── MLP down ──► {4, 1024, 768}
```

### 1.4 The running example

Four values, $$x = [1,\ -1,\ 2,\ -2]$$: two pairs of the same size with opposite signs, so the gate can
be seen open and shut. $$dy = [1,\ 1,\ 1,\ 1]$$, so $$dx$$ comes out as the slope itself; any other
$$dy$$ multiplies each $$dx$$ by its own $$dy$$ (B1 and B2 both start with it).

| step | $$x = 1$$ | $$x = -1$$ | $$x = 2$$ | $$x = -2$$ |
|---|---:|---:|---:|---:|
| F1: $$u = x + \kappa x^3$$ | 1.0447150 | −1.0447150 | 2.3577200 | −2.3577200 |
| F2: $$s = \beta u$$ | 0.8335620 | −0.8335620 | 1.8811884 | −1.8811884 |
| F3: $$h = \tanh s$$ | 0.6823840 | −0.6823840 | 0.9545977 | −0.9545977 |
| F4: $$p = (1 + h) / 2$$ | 0.8411920 | 0.1588080 | 0.9772988 | 0.0227012 |
| F5: $$y = x\, p$$ | **0.8411920** | **−0.1588080** | **1.9545977** | **−0.0454023** |

*Table 2. The example, forward. At $$x = 2$$ the gate is 98% open; at $$x = -2$$, 2%. And
$$p(-x) = 1 - p(x)$$ in every column pair: Section 5 uses it.*

### 1.5 Where each value is used

![GELU's forward for one value](/images/gelu-backward/fig1-graph.svg)
*Figure 1. Each arrow is one use of a value. The backward pass walks the same arrows the other way, one
step per arrow. $$x$$ has two arrows going out, so its gradient has two parts.*

Figure 1 is the whole plan. Six arrows, so six backward steps, B1 to B6, and one more, B7, to add up
the two parts of $$dx$$.

## 2. Three rules and two derivatives

Every step below uses the same three rules as the LayerNorm and linear notes [2, 3]:

| rule | when | what it says |
|---|---|---|
| **chain rule** | $$a$$ affects $$L$$ only through $$b$$ | $$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial b} \cdot \frac{\partial b}{\partial a}$$, written short as $$da = db \cdot \partial b / \partial a$$ |
| **shared value: sum** | $$a$$ is used in many places (all rows, or all outputs of a row) | add one chain-rule term per place: $$\frac{\partial L}{\partial a} = \sum_k \frac{\partial L}{\partial b_k} \cdot \frac{\partial b_k}{\partial a}$$ |
| **used twice: add the parts** | $$a$$ is used by two different forward steps | each step gives one part; $$da$$ = part 1 + part 2 |

*Table 3. The three rules. GELU needs the first and the third. The second has nothing to do: no value
is shared with another value, so no gradient is a sum over rows or columns.*


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

And one habit: **each step looks at one input and holds the others fixed.** F5, $$y = x \cdot p$$, has
two inputs; B1 takes the derivative with respect to $$p$$ and holds $$x$$ fixed, and B2 the other way
round, even though $$p$$ was made from $$x$$. The path through $$p$$ is followed by B3 to B6.

Two derivatives are needed beyond the rules:

| function | derivative | why |
|---|---|---|
| $$x^3$$ | $$3x^2$$ | the power rule |
| $$\tanh s$$ | $$1 - \tanh^2 s$$ | below |

*Table 4. The two derivatives of GELU's steps. The others, of $$\beta u$$ and $$\tfrac{1}{2}(1 + h)$$, are
the constants $$\beta$$ and $$\tfrac{1}{2}$$.*

**The derivative of tanh.** $$\tanh s = \sinh s / \cosh s$$, with $$\sinh' = \cosh$$ and
$$\cosh' = \sinh$$. By the quotient rule:

$$
\frac{d}{ds} \tanh s = \frac{\cosh s \cdot \cosh s - \sinh s \cdot \sinh s}{\cosh^2 s}
                     = 1 - \frac{\sinh^2 s}{\cosh^2 s} = 1 - \tanh^2 s
$$

It is written with $$\tanh$$ itself, which F3 has already worked out. At $$s = 0$$ it is 1, the steepest;
for a large $$\lvert s \rvert$$, $$\tanh s$$ is near $$\pm 1$$ and the derivative near 0: tanh is flat there. In the
example, at $$x = 1$$: $$1 - 0.6823840^2 = 0.5343521$$.

## 3. Breaking it down: B1 to B7

Each step is for one value. The whole tensor is the same step for each of its 12.6M values, with
nothing added across them.

### 3.1 B1: the gate, $$dp$$

From F5, $$y = x \cdot p$$, with respect to $$p$$, holding $$x$$ fixed:

$$
\begin{aligned}
\frac{\partial y}{\partial p} &= x
  && x \text{ is held fixed: a constant factor} \\
\frac{\partial L}{\partial p} &= \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial p}
  && \text{chain rule: } p \text{ affects } L \text{ only through } y \\
dp &= dy \cdot x
  && \text{the same, written short}
\end{aligned}
$$

**Example.** $$dp = [1,\ -1,\ 2,\ -2]$$: $$x$$ itself, since $$dy = 1$$. Opening the gate a little helps
$$y$$ in proportion to how much $$x$$ there is to let through, and with $$x$$'s sign.

### 3.2 B2: the input, part 1, $$dx^{(1)}$$

From F5 again, with respect to $$x$$, holding $$p$$ fixed:

$$
\begin{aligned}
\frac{\partial y}{\partial x} &= p
  && p \text{ is held fixed: a constant factor} \\
\frac{\partial L}{\partial x}\Big\rvert_{\text{through F5}} &= \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}
  && \text{chain rule, along the arrow F5} \\
dx^{(1)} &= dy \cdot p
  && \text{the same, written short}
\end{aligned}
$$

**Example.** $$dx^{(1)} = [0.8411920,\ 0.1588080,\ 0.9772988,\ 0.0227012]$$: the gate.

This part alone would be the whole answer if the gate did not depend on $$x$$. ReLU is that case: its
gate is 0 or 1, flat on both sides, so ReLU's backward is only this part, $$dx = dy \cdot [x > 0]$$.
GELU's gate moves with $$x$$, and B3 to B6 follow that path.

### 3.3 B3: the squashed value, $$dh$$

From F4, $$p = \tfrac{1}{2} + \tfrac{1}{2} h$$:

$$
\begin{aligned}
\frac{\partial p}{\partial h} &= 0 + \tfrac{1}{2} = \tfrac{1}{2}
  && \tfrac{1}{2} \text{ is a constant} \\
\frac{\partial L}{\partial h} &= \frac{\partial L}{\partial p} \cdot \frac{\partial p}{\partial h}
  && \text{chain rule: } h \text{ affects } L \text{ only through } p \\
dh &= dp \cdot \tfrac{1}{2}
  && \text{the same, written short}
\end{aligned}
$$

**Example.** $$dh = [0.5,\ -0.5,\ 1,\ -1]$$.

### 3.4 B4: the scaled value, $$ds$$

From F3, $$h = \tanh s$$, with Table 4:

$$
\begin{aligned}
\frac{\partial h}{\partial s} &= 1 - h^2
  && \text{Table 4} \\
\frac{\partial L}{\partial s} &= \frac{\partial L}{\partial h} \cdot \frac{\partial h}{\partial s}
  && \text{chain rule: } s \text{ affects } L \text{ only through } h \\
ds &= dh \cdot (1 - h^2)
  && \text{the same, written short}
\end{aligned}
$$

**Example.**

| | $$x = 1$$ | $$x = -1$$ | $$x = 2$$ | $$x = -2$$ |
|---|---:|---:|---:|---:|
| $$1 - h^2$$ | 0.5343521 | 0.5343521 | 0.0887432 | 0.0887432 |
| $$ds = dh\,(1 - h^2)$$ | 0.2671761 | −0.2671761 | 0.0887432 | −0.0887432 |

*Table 5. At $$x = \pm 2$$, tanh is already flat: $$1 - h^2$$ is 6 times smaller than at $$\pm 1$$. The
gate is nearly fully open or shut and barely moves, so this path carries little.*

### 3.5 B5: the bent value, $$du$$

From F2, $$s = \beta u$$:

$$
\begin{aligned}
\frac{\partial s}{\partial u} &= \beta
  && \beta \text{ is a constant factor} \\
\frac{\partial L}{\partial u} &= \frac{\partial L}{\partial s} \cdot \frac{\partial s}{\partial u}
  && \text{chain rule: } u \text{ affects } L \text{ only through } s \\
du &= ds \cdot \beta
  && \text{the same, written short}
\end{aligned}
$$

**Example.** $$du = [0.2131756,\ -0.2131756,\ 0.0708069,\ -0.0708069]$$.

### 3.6 B6: the input, part 2, $$dx^{(2)}$$

From F1, $$u = x + \kappa x^3$$, with the power rule:

$$
\begin{aligned}
\frac{\partial u}{\partial x} &= 1 + 3\kappa x^2
  && \text{power rule, on } x \text{ and on } \kappa x^3 \\
\frac{\partial L}{\partial x}\Big\rvert_{\text{through F1}} &= \frac{\partial L}{\partial u} \cdot \frac{\partial u}{\partial x}
  && \text{chain rule, along the arrow F1} \\
dx^{(2)} &= du \cdot (1 + 3\kappa x^2)
  && \text{the same, written short}
\end{aligned}
$$

**Example.**

| | $$x = 1$$ | $$x = -1$$ | $$x = 2$$ | $$x = -2$$ |
|---|---:|---:|---:|---:|
| $$1 + 3\kappa x^2$$ | 1.134145 | 1.134145 | 1.536580 | 1.536580 |
| $$dx^{(2)} = du\,(1 + 3\kappa x^2)$$ | 0.2417721 | −0.2417721 | 0.1088004 | −0.1088004 |

*Table 6. The bend of F1 makes the gate open a little faster for large $$\lvert x \rvert$$: 1.54 at $$\pm 2$$.*

### 3.7 B7: the input, complete

$$x$$ is used by F1 and by F5, so by the used-twice rule:

$$
dx = dx^{(1)} + dx^{(2)}
$$

| | $$x = 1$$ | $$x = -1$$ | $$x = 2$$ | $$x = -2$$ |
|---|---:|---:|---:|---:|
| $$dx^{(1)}$$ (B2): the gate | 0.8411920 | 0.1588080 | 0.9772988 | 0.0227012 |
| $$dx^{(2)}$$ (B6): the gate opening | 0.2417721 | −0.2417721 | 0.1088004 | −0.1088004 |
| $$dx$$ | **1.0829641** | **−0.0829641** | **1.0860993** | **−0.0860993** |

*Table 7. The two parts of $$dx$$. For positive $$x$$ they add up; for negative $$x$$ they pull
opposite ways.*

Read the column $$x = -1$$. Raise $$x$$ a little, and two things happen. Part 1: more $$x$$ goes
through the gate, which is 16% open, so $$y$$ goes up by 0.159 per unit. Part 2: the gate opens wider,
and what it lets through more of is $$x = -1$$, a negative value, so $$y$$ goes down by 0.242 per unit.
Part 2 wins, and $$y$$ goes down: that is the dip of the GELU curve below 0 (Figure 2).

At $$x = 1$$ both parts push $$y$$ up, and $$dx = 1.083$$ is more than 1: GELU climbs faster there than
the line $$y = x$$ it approaches.

### 3.8 All seven steps

| step | from | gradient of | formula | example, $$x = 1$$ |
|---|---|---|---|---:|
| B1 | F5 | gate | $$dp = dy \cdot x$$ | 1 |
| B2 | F5 | input, part 1 | $$dx^{(1)} = dy \cdot p$$ | 0.8411920 |
| B3 | F4 | squashed value | $$dh = dp \cdot \tfrac{1}{2}$$ | 0.5 |
| B4 | F3 | scaled value | $$ds = dh \cdot (1 - h^2)$$ | 0.2671761 |
| B5 | F2 | bent value | $$du = ds \cdot \beta$$ | 0.2131756 |
| B6 | F1 | input, part 2 | $$dx^{(2)} = du \cdot (1 + 3\kappa x^2)$$ | 0.2417721 |
| B7 | — | input, complete | $$dx = dx^{(1)} + dx^{(2)}$$ | 1.0829641 |

*Table 8. The backward pass, broken down. B1 and B2 need only $$dy$$ and the forward's values; B3 to
B6 each need the step before; B7 needs both parts.*

Side by side with the two earlier notes:

| | LayerNorm [2] | linear [3] | GELU |
|---|---|---|---|
| steps | 9 | 3 | 7 |
| a gradient summed over rows | $$d\gamma$$, $$d\beta$$ | $$dW$$, $$db$$ | none: no weights |
| a gradient summed over a row | $$dr$$, $$dv$$, $$d\mu$$ | $$dx$$, over the outputs | none: values never mix |
| a value used twice | $$x$$ and $$d$$ | none | $$x$$ |

*Table 9. GELU is the only one of the three where every value is on its own, forward and backward.*

## 4. Rebuilding: C1 and C2

Table 8 has seven steps and nine values per input value. This section folds them into one slope.

**C1: one path, one product.** B1, B3, B4, B5 and B6 are a chain: each multiplies the one before by one
factor. Put them in a row:

$$
dx^{(2)} = \underbrace{dy \cdot x}_{\text{B1}} \cdot \underbrace{\tfrac{1}{2}}_{\text{B3}} \cdot \underbrace{(1 - h^2)}_{\text{B4}} \cdot \underbrace{\beta}_{\text{B5}} \cdot \underbrace{(1 + 3\kappa x^2)}_{\text{B6}}
$$

At $$x = 1$$: $$1 \cdot 1 \cdot 0.5 \cdot 0.5343521 \cdot 0.7978846 \cdot 1.134145 = 0.2417721$$, B6's number.

The last four factors are the slope of the gate: how fast $$p$$ changes with $$x$$, by the chain rule
through F1 to F4:

$$
p'(x) = \frac{\partial p}{\partial h}\,\frac{\partial h}{\partial s}\,\frac{\partial s}{\partial u}\,\frac{\partial u}{\partial x}
      = \tfrac{1}{2}\,(1 - h^2)\, \beta\, (1 + 3\kappa x^2)
\qquad\text{so}\qquad dx^{(2)} = dy \cdot x \cdot p'(x)
$$

**C2: both parts, one slope.** Add B2's part, $$dy \cdot p$$, and take $$dy$$ out:

$$
dx = dy \cdot \big(p + x\, p'(x)\big)
$$

This is the product rule on $$y = x \cdot p(x)$$, as it should be: the seven steps were the product rule
and the chain rule done one arrow at a time. With F4 and C1 written in:

$$
\boxed{\;dx = dy \cdot \text{gelu}'(x), \qquad \text{gelu}'(x) = \underbrace{\tfrac{1}{2}(1 + h)}_{\text{the gate, } p} + \underbrace{\tfrac{1}{2}\, x\,(1 - h^2)\, \beta\,(1 + 3\kappa x^2)}_{x \text{ times how fast the gate opens}}, \qquad h = \tanh\big(\beta(x + \kappa x^3)\big)\;}
$$

The line at the top of this note.

![GELU, its gate and its slope](/images/gelu-backward/fig2-curves.svg)
*Figure 2. Left: GELU next to ReLU. Right: the gate $$p$$ and the slope $$\text{gelu}'$$. The second
term, $$x$$ times how fast the gate opens, is 0 at $$x = 0$$ and far from it, and largest near
$$x = \pm 1.42$$: there it lifts the slope above 1 for positive $$x$$ and pulls it below 0 for negative.*

**Example**, straight from C2:

| | $$x = 1$$ | $$x = -1$$ | $$x = 2$$ | $$x = -2$$ |
|---|---:|---:|---:|---:|
| $$p$$ | 0.8411920 | 0.1588080 | 0.9772988 | 0.0227012 |
| $$x\, p'(x)$$ | 0.2417721 | −0.2417721 | 0.1088004 | −0.1088004 |
| $$\text{gelu}'(x)$$ | 1.0829641 | −0.0829641 | 1.0860993 | −0.0860993 |
| $$dx = dy \cdot \text{gelu}'(x)$$ | 1.0829641 | −0.0829641 | 1.0860993 | −0.0860993 |

*Table 10. The rebuilt slope gives the same $$dx$$ as the seven steps (Table 7).*

## 5. Two checks

**$$\text{gelu}(x) - \text{gelu}(-x) = x$$, and the slopes add up to 1.** tanh is odd, $$\tanh(-s) = -\tanh s$$,
and so is F1 to F2: $$s(-x) = -s(x)$$. So $$h(-x) = -h(x)$$ and the gate is mirrored:

$$
p(-x) = \tfrac{1}{2}(1 - h) = 1 - p(x)
$$

Then

$$
\text{gelu}(x) - \text{gelu}(-x) = x\, p(x) - (-x)\big(1 - p(x)\big) = x\, p(x) + x - x\, p(x) = x
$$

GELU is the line $$y = x$$ split into two halves: what the gate lets through at $$x$$, and what it holds
back at $$-x$$. Take the derivative of both sides with respect to $$x$$; $$\text{gelu}(-x)$$ gives
$$-\text{gelu}'(-x)$$ by the chain rule, so:

$$
\text{gelu}'(x) + \text{gelu}'(-x) = 1
$$

In the example, from Tables 2 and 10:

| pair | $$\text{gelu}(x) - \text{gelu}(-x)$$ | $$\text{gelu}'(x) + \text{gelu}'(-x)$$ |
|---|---|---|
| $$x = 1$$ | $$0.8411920 - (-0.1588080) = 1$$ | $$1.0829641 + (-0.0829641) = 1$$ |
| $$x = 2$$ | $$1.9545977 - (-0.0454023) = 2$$ | $$1.0860993 + (-0.0860993) = 1$$ |

*Table 11. The first check in the example. It is why the slope's highest point, 1.1290 at
$$x = 1.4185$$, and its lowest, $$-0.1290$$ at $$x = -1.4185$$, add up to 1.*

**The slope at 0 and far away.** At $$x = 0$$: $$h = 0$$, and the second term has the factor $$x = 0$$,
so $$\text{gelu}'(0) = p(0) = \tfrac{1}{2}$$, exactly. Far to the right, $$h \to 1$$: $$p \to 1$$ and
$$1 - h^2 \to 0$$, so $$\text{gelu}'(x) \to 1$$; far to the left, $$h \to -1$$ and
$$\text{gelu}'(x) \to 0$$. Far from 0, GELU's slope is ReLU's: 1 on the right and 0 on the left.

And the gate leaves 0 at the slope of the exact gate $$\Phi$$:

$$
p'(0) = \tfrac{1}{2} \cdot (1 - 0) \cdot \beta \cdot 1 = \tfrac{1}{2}\sqrt{2/\pi} = \frac{1}{\sqrt{2\pi}} = 0.3989423 = \Phi'(0)
$$

That is where $$\beta = \sqrt{2/\pi}$$ comes from: it makes the tanh gate match $$\Phi$$ at its steepest
point. $$\kappa$$ then bends it to stay close further out (Section 1.1).

## 6. Checked against PyTorch

| test | compared | largest difference |
|---|---|---|
| the example of Section 1.4 | every number of Tables 2, 5, 6, 7 and 10, with the five steps written out and autograd keeping each step's gradient (`retain_grad`) | none in the 7 digits shown; $$dx$$ within $$2.2 \times 10^{-16}$$ |
| the same | $$y$$ and $$dx$$ against `F.gelu(x, approximate="tanh")` and its autograd | $$5.6 \times 10^{-17}$$ and $$5.6 \times 10^{-17}$$ |
| 1,000,000 random values, $$x$$ from $$3\,\mathcal{N}(0, 1)$$, $$dy$$ from $$\mathcal{N}(0, 1)$$ | B1, B3, B4, B5 ($$dp$$, $$dh$$, $$ds$$, $$du$$), one value at a time, against autograd's gradients of the steps | 0 |
| the same | B7 and C2 against `F.gelu(x, approximate="tanh")`'s $$dx$$ | $$1.8 \times 10^{-14}$$ and $$1.8 \times 10^{-14}$$ |
| the same | the first check: $$\text{gelu}(x) - \text{gelu}(-x) - x$$ and $$\text{gelu}'(x) + \text{gelu}'(-x) - 1$$ | $$8.9 \times 10^{-16}$$ and $$2.2 \times 10^{-16}$$ |
| the same | $$1 - \tanh^2 s$$ against llm.c's $$1 / \cosh^2 s$$ [7] | $$3.3 \times 10^{-16}$$ |
| one value, $$x = 0.7$$ | C2 against a finite difference, $$(\text{gelu}(x + \epsilon) - \text{gelu}(x - \epsilon)) / 2\epsilon$$ with $$\epsilon = 10^{-6}$$ | 0.9763572187 both: $$2.9 \times 10^{-11}$$, the finite difference's own error |
| 160,001 values from −8 to 8 | the tanh form against the exact $$x\,\Phi(x)$$, PyTorch's default `F.gelu(x)` | $$4.7 \times 10^{-4}$$ at $$x = -2.70$$: two different functions |

*Table 12. PyTorch 2.12 [5], `torch.nn.functional.gelu` and `backward`, in double precision.
Differences near $$10^{-16}$$ are rounding. The last row is why the kernel's tests compare with
`approximate="tanh"`.*

## 7. What the rebuilt formula gives the kernel

The seven steps and the one slope give the same numbers, but they are very different programs:

| | broken down (Table 8) | rebuilt (C2) |
|---|---|---|
| per value, besides $$dx$$ | $$u$$, $$s$$, $$h$$, $$p$$, $$dp$$, $$dh$$, $$ds$$, $$du$$, $$dx^{(1)}$$, $$dx^{(2)}$$ | $$s$$, $$h$$ and the slope: in registers, never stored |
| kept by the forward and needed | $$x$$ and $$h$$ (or $$p$$) | $$x$$ only: $$h$$ is worked out again, one tanh |
| GPU memory read and written per value | many tensors of 12.6M values | read $$x$$ and $$dy$$, write $$dx$$: 12 bytes |

*Table 13. In GPT-2 small, one tensor of one value per number is 4096 × 3072 × 4 bytes = 50.3 MB, for
each of the 12 GELUs.*

The kernel in my trainer [1] (`gelu.cuh`) is C2 written out, one thread per 4 values (a `float4`):

| in C2 | in the kernel | where |
|---|---|---|
| $$s = \beta(x + \kappa x^3)$$ | `inner` | `GELU_SCALE * (x + GELU_CUBE_FACTOR * x * x * x)`, in `gelu_value` and `gelu_slope` |
| $$h = \tanh s$$ | `tanh_inner` | `tanhf(inner)` |
| $$\beta(1 + 3\kappa x^2)$$ | `inner_slope` | C1's last two factors, $$ds/du \cdot du/dx$$ |
| $$\text{gelu}'(x)$$ | `gelu_slope`'s result | `0.5f * (1.0f + tanh_inner) + 0.5f * x * (1.0f - tanh_inner * tanh_inner) * inner_slope` |
| $$y = x\, p$$ | `output` | `gelu_forward_kernel`, with `gelu_value` |
| $$dx = dy \cdot \text{gelu}'(x)$$ | `input_grad` | `gelu_backward_kernel`: `output_grad * gelu_slope(input)`, written over (`=`) |

*Table 14. `gelu_slope` and `gelu_value` each work out the tanh: the forward does not keep $$h$$ for the
backward.*

**Working $$h$$ out again instead of keeping it.** Keeping $$h$$ would cost the forward one more 50.3 MB
write and the backward one more 50.3 MB read. Working it out again costs one `tanhf` per value, a few
dozen instructions, while the thread waits for memory anyway. LayerNorm's kernel makes the same choice
for $$\hat{x}$$ [2].

**$$1 - h^2$$ instead of llm.c's $$1 / \cosh^2 s$$.** The same number (Section 6), but $$1 - h^2$$ reuses
$$h$$, which the gate term needs anyway; $$1 / \cosh^2 s$$ costs a `coshf` more.

**`=` for $$dx$$**, from the rules of Section 2:

| gradient | the variable is used | so |
|---|---|---|
| $$dx$$ | by GELU alone: MLP up's output goes nowhere else | one gradient, nothing to add: written over |
| — | GELU has no weights | nothing is added up over rows, so nothing to set to 0 per step |

*Table 15. And since each value of $$dx$$ needs only the same value of $$dy$$ and $$x$$, the kernel may
write $$dx$$ into $$dy$$'s own memory: the trainer can save one 50.3 MB tensor per GELU.*

Each value is on its own, so the kernel has nothing to share between threads: no shared memory, no
shuffles, no atomic adds. Its time is GPU memory: the forward reads 4 bytes and writes 4 per value, the
backward reads 8 and writes 4. For `{4, 1024, 3072}` that is 100.7 MB and 151 MB; at a T4's 320 GB/s, at
least 0.315 ms and 0.472 ms. How close the kernel gets to that is for a later entry: nothing here is
timed.

## 8. What to remember

1. **One step per arrow.** The backward pass walks Figure 1's arrows backwards, each step with the
   chain rule and its other inputs held fixed.
2. **Two uses, two parts.** $$x$$ is used by F1 and by F5, so $$dx$$ is two parts added: the gate as it
   is, and $$x$$ times how fast the gate opens. ReLU has only the first.
3. **No sums.** Every value is on its own, and GELU has no weights: nothing is added over rows or
   columns, forward or backward.
4. **Two derivatives:** the power rule, and $$\tanh' = 1 - \tanh^2$$, which reuses the tanh already
   worked out.
5. **The formula says what GELU is.** $$\text{gelu}(x) - \text{gelu}(-x) = x$$, so the slopes at $$x$$
   and $$-x$$ add up to 1; the slope is $$\tfrac{1}{2}$$ at 0, and ReLU's, 1 or 0, far from it.

## References

1. D. B. Song. *gpt2-small*, branch `c-cuda-trainer`: GPT-2 small pre-training in C and CUDA.
   [github.com/ducbachsong/gpt2-small](https://github.com/ducbachsong/gpt2-small/tree/c-cuda-trainer)
2. D. B. Song. *LayerNorm backward, broken down and rebuilt: nine chain-rule steps and one formula.* 2026.
   [/posts/layer-norm-backward/](/posts/layer-norm-backward/)
3. D. B. Song. *Linear backward, broken down and rebuilt: three chain-rule steps and three matrix
   products.* 2026. [/posts/linear-backward/](/posts/linear-backward/)
4. D. Hendrycks, K. Gimpel. *Gaussian Error Linear Units (GELUs).* 2016.
   [arxiv.org/abs/1606.08415](https://arxiv.org/abs/1606.08415)
5. PyTorch. `torch.nn.functional.gelu`.
   [docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html)
6. OpenAI. *gpt-2*, `src/model.py`: `gelu`, the tanh form.
   [github.com/openai/gpt-2](https://github.com/openai/gpt-2/blob/master/src/model.py)
7. A. Karpathy. *llm.c*, `train_gpt2.c`: `gelu_forward` and `gelu_backward`, GELU forward and backward
   in C. [github.com/karpathy/llm.c](https://github.com/karpathy/llm.c/blob/master/train_gpt2.c)
