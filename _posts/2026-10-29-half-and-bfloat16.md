---
title: "Half and bfloat16: two ways to fit a float into 16 bits"
date: 2026-10-29
permalink: /posts/half-and-bfloat16/
excerpt: "A float is 32 bits: a sign, an exponent and a fraction. Half and bfloat16 keep 16 of them, and they keep different ones: half keeps more fraction bits, so more digits, and bfloat16 keeps all the exponent bits, so float's whole range. This note builds a float from its bits with one formula, cuts it to 16 bits both ways, and works out from the formula everything that follows: the largest and smallest values, how far apart neighbours are, how a value is rounded and how far the rounding can move it. Then it shows the three things that go wrong when a model trains in 16 bits, too-small gradients, too-small updates and long sums, each with a number, and how my GPT-2 trainer keeps its tensors in bfloat16 and still does every sum in float."
tags:
  - floating-point
  - mixed-precision
  - cuda
  - notes
---

**What this is.** My GPT-2 trainer [1] can now keep its tensors in **bfloat16**, 2 bytes a value
instead of a float's 4, and its `Tensor` also holds **half**, the other 2-byte format. Both are a float
cut down to 16 bits, but they cut it in different places, and that choice decides everything: which
numbers they can hold, how close to the truth each one is, and what breaks when a model trains in them.

This note starts from one formula, the one that turns a float's 32 bits into a number (Section 1). It
cuts that formula to 16 bits the two ways (Section 2) and works out, from the formula alone, how a value
is rounded and how far the rounding can move it (Section 3). Then it shows the three things that go
wrong in 16 bits, each with a number (Section 4), and how the trainer's kernels avoid them (Section 5).
Two interactive figures let you try any number. Every number in the text was worked out exactly, by
hand from the bits and again in a script.

* TOC
{:toc}

## 1. A float, from its bits

### 1.1 Three fields and one formula

A float is 32 bits cut into three fields:

| field | bits | name here | what it holds |
|---|---|---|---|
| sign | 1 | $$s$$ | 0 for +, 1 for − |
| exponent | 8 | $$E$$ | a whole number 0 .. 255 |
| fraction | 23 | $$f$$ | a whole number 0 .. $$2^{23} - 1$$ |

*Table 1. The three fields of a float, from the left: bit 31 is the sign, bits 30 .. 23 the exponent,
bits 22 .. 0 the fraction.*

For every $$E$$ from 1 to 254, the number they stand for is

$$
x = (-1)^s \cdot 2^{E - 127} \cdot \left(1 + \frac{f}{2^{23}}\right)
$$

Each part has one job:

- $$(-1)^s$$ is the sign: $$(-1)^0 = 1$$, $$(-1)^1 = -1$$.
- $$2^{E - 127}$$ is the **scale**, a power of 2. The 127, called the **bias**, lets an exponent field
  from 1 to 254 stand for the powers $$2^{-126}$$ to $$2^{127}$$: small numbers and large ones. Write
  $$e = E - 127$$ for the power itself.
- $$1 + f / 2^{23}$$ is the **significand**, a number from 1 up to (not including) 2. $$f / 2^{23}$$ is
  the fraction field read as binary digits after the point: $$f = 2^{22}$$ (a 1 then 22 zeros) is
  $$0.1_2 = 0.5$$.

Why the "1 +": any number that is not 0 written in binary starts with a 1, as $$0.1$$ is
$$1.1001\ldots_2 \times 2^{-4}$$. A digit that is always 1 does not need to be stored, so a float keeps
**24 significant bits** with 23 stored. The 1 is called the hidden bit.

### 1.2 One example: 0.1

0.1 has no exact binary form (as 1/3 has none in decimal), so a float holds the nearest value it can.
Its 32 bits:

```
sign  exponent   fraction
0     01111011   10011001100110011001101
```

Read each field, then put them into the formula:

| field | bits | as a whole number | its part of the formula |
|---|---|---|---|
| $$s$$ | `0` | 0 | $$(-1)^0 = 1$$ |
| $$E$$ | `01111011` | 123 | $$e = 123 - 127 = -4$$, scale $$2^{-4} = 0.0625$$ |
| $$f$$ | `10011001100110011001101` | 5,033,165 | $$1 + 5{,}033{,}165 / 8{,}388{,}608 = 1.6000000238$$ |

*Table 2. The float nearest 0.1, field by field. $$2^{23} = 8{,}388{,}608$$.*

$$
x = 1 \cdot 2^{-4} \cdot 1.6000000238418579 = 0.10000000149011612
$$

The float is $$1.49 \times 10^{-9}$$ above 0.1: about 8 significant digits are right.

### 1.3 How far apart two neighbours are

Take all the floats with the same exponent $$e$$, from $$2^e$$ (fraction 0) up to just below $$2^{e+1}$$.
Each step of $$f$$ by 1 adds $$2^e \cdot 2^{-23}$$:

$$
\text{spacing} = 2^{e} \cdot \frac{1}{2^{23}} = 2^{e - 23}
$$

There are $$2^{23}$$ floats in each such band, evenly spaced, and the spacing doubles from one band to
the next:

| band | $$e$$ | spacing $$2^{e-23}$$ |
|---|---|---|
| 0.0625 .. 0.125 (0.1 is here) | −4 | $$2^{-27} = 7.45 \times 10^{-9}$$ |
| 1 .. 2 | 0 | $$2^{-23} = 1.19 \times 10^{-7}$$ |
| 1024 .. 2048 | 10 | $$2^{-13} = 1.22 \times 10^{-4}$$ |
| 16,777,216 .. 33,554,432 | 24 | $$2^{1} = 2$$: the odd numbers are missing |

*Table 3. The spacing of floats grows with the numbers: always the same fraction of them, about
$$2^{-23}$$.*

So the spacing is never a fixed amount, it is a fixed **fraction** of the number. That is what a
floating point is for: the same number of digits for a weight of 0.02 and for a loss of 10.

### 1.4 The two ends of the exponent

The exponent fields 0 and 255 are kept for special values:

| $$E$$ | $$f$$ | stands for |
|---|---|---|
| 0 | 0 | zero (+0 or −0) |
| 0 | not 0 | a **subnormal**: $$(-1)^s \cdot 2^{-126} \cdot f / 2^{23}$$, no hidden 1, so tinier numbers than $$2^{-126}$$, with fewer digits |
| 255 | 0 | infinity, the answer for anything too large |
| 255 | not 0 | NaN, "not a number", as in $$0 / 0$$ |

*Table 4. The two reserved exponent fields.*

## 2. Two ways to cut it to 16 bits

### 2.1 The three formats

A 16-bit format still has a sign, an exponent and a fraction, and the same formula with its own field
sizes. Write $$m$$ for the fraction's bits and $$k$$ for the exponent's:

$$
x = (-1)^s \cdot 2^{E - \text{bias}} \cdot \left(1 + \frac{f}{2^{m}}\right), \qquad \text{bias} = 2^{k-1} - 1
$$

```
float      s eeeeeeee fffffffffffffffffffffff     1 + 8 + 23 = 32 bits
half       s eeeee ffffffffff                     1 + 5 + 10 = 16 bits
bfloat16   s eeeeeeee fffffff                     1 + 8 +  7 = 16 bits
```

| format | sign | exponent $$k$$ | fraction $$m$$ | bias | significant bits $$p = m + 1$$ |
|---|---|---|---|---|---|
| float (fp32) | 1 | 8 | 23 | 127 | 24 |
| half (fp16) | 1 | 5 | 10 | 15 | 11 |
| bfloat16 | 1 | 8 | 7 | 127 | 8 |

*Table 5. Half and bfloat16 both give up 16 bits. Half takes 3 from the exponent and 13 from the fraction;
bfloat16 takes all 16 from the fraction and keeps float's exponent whole.*

**Half** is the IEEE standard's 16-bit format [2]. **bfloat16**, "brain float", came from Google Brain for
training on TPUs [7]: the idea was that a network minds losing digits less than losing range [6].

### 2.2 What each can hold, from the formula

Every limit comes out of the formula with the largest or smallest fields:

- **Largest value.** The largest normal exponent field is $$2^k - 2$$ (all 1s is taken by infinity), and
  the largest fraction all 1s, $$f = 2^m - 1$$, which makes the significand $$2 - 2^{-m}$$:

  $$
  x_{\max} = (2 - 2^{-m}) \cdot 2^{(2^k - 2) - \text{bias}}
  $$

  half: $$(2 - 2^{-10}) \cdot 2^{30 - 15} = 1.9990234 \times 32{,}768 = 65{,}504$$.
  bfloat16: $$(2 - 2^{-7}) \cdot 2^{254 - 127} = 3.39 \times 10^{38}$$, nearly float's
  $$3.40 \times 10^{38}$$.
- **Smallest normal value.** $$E = 1$$, $$f = 0$$: $$x = 2^{1 - \text{bias}}$$. Half:
  $$2^{-14} = 6.10 \times 10^{-5}$$. bfloat16: $$2^{-126} = 1.18 \times 10^{-38}$$, float's.
- **Smallest subnormal.** $$E = 0$$, $$f = 1$$: $$2^{1 - \text{bias}} \cdot 2^{-m}$$. Half:
  $$2^{-24} = 5.96 \times 10^{-8}$$. bfloat16: $$2^{-133} = 9.18 \times 10^{-41}$$.
- **Spacing just above 1** ($$e = 0$$): $$2^{-m}$$. Half $$2^{-10} = 0.00098$$, bfloat16
  $$2^{-7} = 0.0078$$. The next bfloat16 after 1 is 1.0078125.

| | float | half | bfloat16 |
|---|---|---|---|
| largest | $$3.40 \times 10^{38}$$ | **65,504** | $$3.39 \times 10^{38}$$ |
| smallest normal | $$1.18 \times 10^{-38}$$ | $$6.10 \times 10^{-5}$$ | $$1.18 \times 10^{-38}$$ |
| smallest subnormal | $$1.40 \times 10^{-45}$$ | $$5.96 \times 10^{-8}$$ | $$9.18 \times 10^{-41}$$ |
| spacing above 1, $$2^{-m}$$ | $$1.19 \times 10^{-7}$$ | $$9.77 \times 10^{-4}$$ | $$7.81 \times 10^{-3}$$ |
| the most rounding moves a value, $$u = 2^{-p}$$ (Section 3.2) | $$5.96 \times 10^{-8}$$ | $$4.88 \times 10^{-4}$$ | $$3.91 \times 10^{-3}$$ |
| about this many digits | 7 | 3 | 2 |

*Table 6. The two 16-bit formats, from the formula. Half has 8 times finer steps than bfloat16; bfloat16
reaches about $$5 \times 10^{33}$$ times further in both directions.*

### 2.3 bfloat16 is a float's first 16 bits

bfloat16 has float's sign field and float's exponent field, with the same bias, so the first 16 bits of
a float **are** a bfloat16. Going from bfloat16 to float is appending 16 zero bits to the fraction:
exact, every bfloat16 is a float. Going the other way is dropping the last 16 bits, and rounding (next
section). For 0.1:

```
float      0 01111011 1001100 1100110011001101
bfloat16   0 01111011 1001101                     the last 16 bits dropped, the kept ones rounded up
```

Half has to move the exponent: the same $$e$$ is the field $$E_{\text{half}} = e + 15$$ instead of
$$e + 127$$. For 0.1, $$e = -4$$, so $$E_{\text{half}} = 11 = 01011_2$$. A float above 65,504 has no half
(it becomes inf), and one below $$2^{-25}$$ neither (it becomes 0): Table 10.

<script src="/assets/js/float16-widgets.js" defer></script>

<div class="fw" data-widget="converter">This picture needs JavaScript.</div>

*Interactive 1. One number in three formats. Each card shows the stored bits and puts them into the
formula. For bfloat16 the grey bits are the float's last 16, the ones it drops; their size decides the
rounding (Section 3.1).*

## 3. Rounding

### 3.1 To the nearest value, a tie to the even one

A float that falls between two values of the shorter format goes to the **nearest** of the two. If it
sits exactly halfway, a **tie**, it goes to the one whose last bit is 0, the "even" one. This rule,
round to nearest even, is the IEEE default [2], and what CUDA's `__float2bfloat16` and `__float2half`
do.

To round, look at the bits that do not fit, read as a fraction of one step of the last kept bit:
more than one half, round up; less, keep; exactly one half, go to the even one.

**0.1 to bfloat16.** The fraction `1001100 | 1100110011001101` keeps 7 bits; the 16 that do not fit,
`1100110011001101` = 52,429, are $$52{,}429 / 65{,}536 = 0.80$$ of a step: more than one half, so the kept
bits go up by one, $$1001100_2 + 1 = 1001101_2 = 77$$:

$$
\hat{x} = 2^{-4} \cdot \left(1 + \frac{77}{128}\right) = 0.0625 \times 1.6015625 = 0.10009765625
$$

**0.1 to half.** $$e = -4$$ is the same; the fraction keeps 10 bits, `1001100110 | 0110011001101`. The 13
that do not fit, `0110011001101` = 3277, are $$3277 / 8192 = 0.40$$ of a step: less than one half, so the
kept bits stay, $$1001100110_2 = 614$$:

$$
\hat{x} = 2^{-4} \cdot \left(1 + \frac{614}{1024}\right) = 0.0625 \times 1.599609375 = 0.0999755859375
$$

### 3.2 How far rounding can move a value

Take a value $$x > 0$$ in the band of exponent $$e$$: $$2^e \le x < 2^{e+1}$$. Its two neighbours in a
format with $$m$$ fraction bits are $$2^{e - m}$$ apart (Section 1.3, with $$m$$ for 23). The nearest of
the two is at most half that away:

$$
|x - \hat{x}| \le \tfrac{1}{2} \cdot 2^{e - m} = 2^{e - m - 1}
$$

and since $$x \ge 2^e$$, dividing by $$x$$ makes the right side no larger than with $$2^e$$ below it:

$$
\frac{|x - \hat{x}|}{|x|} \le \frac{2^{e - m - 1}}{2^{e}} = 2^{-(m+1)} = 2^{-p} = u
$$

$$u$$ is called the **unit roundoff**. It does not depend on $$e$$: every normal value is rounded by at
most the same fraction of itself, $$u$$. The worst case is a value just above a power of 2, where the
spacing is largest next to it.

| | $$\hat{x}$$ for 0.1 | off by | as a fraction of 0.1 | at most $$u$$ |
|---|---|---|---|---|
| float | 0.10000000149011612 | $$1.49 \times 10^{-9}$$ | $$1.5 \times 10^{-8}$$ | $$5.96 \times 10^{-8}$$ |
| half | 0.0999755859375 | $$2.44 \times 10^{-5}$$ | $$2.4 \times 10^{-4}$$ | $$4.88 \times 10^{-4}$$ |
| bfloat16 | 0.10009765625 | $$9.77 \times 10^{-5}$$ | $$9.8 \times 10^{-4}$$ | $$3.91 \times 10^{-3}$$ |

*Table 7. 0.1 in each format, against the bound. The half and the bfloat16 are rounded from the float,
as a kernel rounds them; the "off by" is against 0.1 itself.*

For a subnormal the bound does not hold: the spacing stays $$2^{-24}$$ (half) however small the value,
so a half subnormal near $$10^{-7}$$ is off by up to 25% of itself, and below $$2.98 \times 10^{-8}$$, half
the smallest subnormal, a value becomes 0.

### 3.3 Why ties go to the even one

From 256 to 512 the bfloat16s are 2 apart (Table 3's rule: $$2^{8 - 7} = 2$$), so the odd numbers are
exact ties:

| float | the two bfloat16s around it | last bits | goes to |
|---|---|---|---|
| 257 | 256 and 258 | `0000000`, `0000001` | **256**, down |
| 259 | 258 and 260 | `0000001`, `0000010` | **260**, up |

*Table 8. Two ties. Neither always up nor always down: half the ties go each way.*

Always rounding a tie up would push every sum of many ties upward; going to the even one sends half
of them down, so on average rounding adds nothing.

### 3.4 The same rule in three integer operations

Since a bfloat16 is a float's first 16 bits, rounding is integer work on the float's 32 bits $$b$$, as
PyTorch writes it [8]:

$$
\text{bfloat16 bits} = \left\lfloor \frac{b + \texttt{0x7FFF} + \text{lsb}}{2^{16}} \right\rfloor,
\qquad \text{lsb} = \text{bit 16 of } b \text{, the last bit kept}
$$

Why it works: write the last 16 bits of $$b$$ as $$r$$, from 0 to $$\texttt{0xFFFF}$$. Adding
$$\texttt{0x7FFF} + \text{lsb}$$ carries a 1 into the kept bits exactly when rounding up is right:

| the dropped bits $$r$$ | $$r + \texttt{0x7FFF} + \text{lsb}$$ | carries into the kept bits? | which is |
|---|---|---|---|
| above $$\texttt{0x8000}$$ (more than one half) | at least $$\texttt{0x10000}$$ | yes | round up |
| below $$\texttt{0x8000}$$ (less than one half) | at most $$\texttt{0xFFFF}$$ | no | keep |
| exactly $$\texttt{0x8000}$$, lsb 0 (a tie, kept bits even) | $$\texttt{0xFFFF}$$ | no | stay even |
| exactly $$\texttt{0x8000}$$, lsb 1 (a tie, kept bits odd) | $$\texttt{0x10000}$$ | yes | up, to even |

*Table 9. The three cases of round to nearest even, as one add.*

0.1 is $$b = \texttt{0x3DCCCCCD}$$, lsb 0: $$\texttt{0x3DCCCCCD} + \texttt{0x7FFF} = \texttt{0x3DCD4CCC}$$,
and dropping the last 16 bits leaves $$\texttt{0x3DCD}$$, Section 3.1's `0 01111011 1001101`.

### 3.5 Too large and too small

A value past the largest is rounded to infinity; one below half the smallest subnormal, to 0:

| float | half | bfloat16 |
|---|---|---|
| 65,504 | 65,504, the largest | 65,536: the nearest, its band's spacing is 256 |
| 70,000 | **inf**: from $$65{,}504 + 16$$ on, half a step past the largest | 70,144 |
| $$1 \times 10^{-8}$$ | **0**: below $$2.98 \times 10^{-8}$$ | $$1.0012 \times 10^{-8}$$ |

*Table 10. Where half runs out and bfloat16 does not.*

## 4. What goes wrong when a model trains in 16 bits

Three things, each from Section 3: a number too small for the format, an add too small for the number,
and a sum too long for the digits.

### 4.1 Too small: gradients that become 0 in half

The classifier's gradient [11] for a word of the vocabulary is the mean's $$1/R$$ over $$R$$ rows times
that word's probability. GPT-2 has $$R = 4 \times 1024 = 4096$$ rows and 50,257 words; a model at the
start gives each word about $$1/50{,}257$$:

$$
\frac{\partial L}{\partial z} \approx \frac{1}{4096} \cdot \frac{1}{50{,}257} = 4.86 \times 10^{-9}
$$

In half that is 0 (Table 10): almost the whole gradient of the logits is gone. In bfloat16 it is
$$4.860 \times 10^{-9}$$, off by 0.05%.

Half's way out is **loss scaling** [5]. Multiply the loss by a constant $$S$$ before the backward pass.
Every gradient is a chain-rule product that starts at $$\partial L / \partial L = 1$$; for $$S L$$ it
starts at $$S$$, and a constant factor goes through every step unchanged (the derivative of $$c \cdot y$$
is $$c$$ times that of $$y$$):

$$
\frac{\partial (S L)}{\partial w} = S \cdot \frac{\partial L}{\partial w}
$$

So every gradient is $$S$$ times larger, the same digits shifted up into half's range, and the update
divides by $$S$$ again in float. With $$S = 1024$$ the gradient above is $$4.97 \times 10^{-6}$$: in half
still a subnormal, off by 0.5%. With $$S = 65{,}536$$ it is $$3.18 \times 10^{-4}$$, a normal half, off by
0.02%. Too large an $$S$$ sends the largest gradients past 65,504 to infinity, so trainers lower $$S$$
whenever an infinity shows up and skip that step [5].

bfloat16 needs none of this: its smallest normal value is float's. This is the main reason it is the
format of training on newer GPUs.

### 4.2 Too close: updates that are lost

An update $$w \leftarrow w - \Delta$$ changes the stored $$w$$ only if $$w - \Delta$$ rounds to another
value than $$w$$: only if $$|\Delta|$$ is more than half the spacing at $$w$$. Section 3.2 makes that
spacing about $$2u$$ times $$|w|$$, so roughly:

$$
\text{the update is lost when} \quad |\Delta| < \tfrac{1}{2} \cdot 2^{e - m}, \quad \text{about } u \cdot |w|
$$

A weight of 0.07, a size GPT-2's weights have ($$e = -4$$, the band 0.0625 .. 0.125):

| | spacing at 0.07, $$2^{-4-m}$$ | half of it | an update of $$10^{-3}$$ | an update of $$10^{-4}$$ |
|---|---|---|---|---|
| float | $$7.5 \times 10^{-9}$$ | $$3.7 \times 10^{-9}$$ | kept | kept: 100 steps give 0.0600 |
| half | $$6.1 \times 10^{-5}$$ | $$3.1 \times 10^{-5}$$ | kept | kept, rounded each time: 100 steps give 0.0590 |
| bfloat16 | $$4.9 \times 10^{-4}$$ | $$2.4 \times 10^{-4}$$ | kept | **lost**: 100 steps leave 0.0698, unchanged |

*Table 11. 100 steps of $$w \leftarrow w - 10^{-4}$$ from 0.07. bfloat16's nearest to 0.07 is
0.06982421875, and every step rounds back to it.*

$$10^{-4}$$ is not an odd size: AdamW moves each weight by about the learning rate, $$6 \times 10^{-4}$$
at GPT-2's peak and less as the schedule lowers it. The usual fix [5, 6] is to keep a **float copy of the
weights**, the master weights: the update is made on the float copy, and only a bfloat16 copy of it goes
into the forward pass. llm.c instead rounds the update **stochastically** [10]: up with probability
equal to how far the value is towards the upper neighbour. For $$x$$ between neighbours $$a$$ and
$$a + \delta$$, rounding up with probability $$(x - a) / \delta$$ gives on average

$$
a \cdot \left(1 - \frac{x - a}{\delta}\right) + (a + \delta) \cdot \frac{x - a}{\delta} = a + (x - a) = x
$$

exactly $$x$$: each update is kept on average, even one much smaller than the spacing.

The same holds for AdamW's moving averages. $$m \leftarrow 0.9\, m + 0.1\, g$$ from $$m = 0.5$$ with
$$g = 0.515$$ gives 0.5015, which in bfloat16 rounds back to 0.5: $$m$$ would never move towards $$g$$.
So the trainer keeps $$m$$ and $$v$$ in float.

### 4.3 Too long: sums that stop growing

Adding numbers one at a time rounds after every add, and the roundings pile up. Add 1 to itself:

| adds | float | half | bfloat16 |
|---|---|---|---|
| 256 | 256 | 256 | 256 |
| 257 | 257 | 257 | **256** |
| 1024 | 1024 | 1024 | 256 |
| 2049 | 2049 | **2048** | 256 |
| 4096 | 4096 | 2048 | 256 |

*Table 12. The sum of 1, 1, 1, ... kept in each format.*

At 256 the bfloat16s are 2 apart, so $$256 + 1 = 257$$ is a tie between 256 and 258, and it goes to the
even one, 256 (Table 8). The sum stops there for good: every later add is the same tie. Half stops the
same way at 2048, where its spacing reaches 2 ($$2^{11 - 10}$$).

The general bound [4] for adding $$n$$ numbers one at a time, each add rounded:

$$
\left| \hat{S} - S \right| \le (n - 1)\, u \sum_{i=1}^{n} |x_i|
$$

For one output of the qkv linear, a dot product of 768 terms: in float
$$767 \times 5.96 \times 10^{-8} = 4.6 \times 10^{-5}$$ of the sum of their sizes; in bfloat16
$$767 \times 3.91 \times 10^{-3} = 3.0$$, a bound that promises nothing. The fix is to **add in float and
round once**: then the error is the float sum's, plus one rounding, at most $$u$$.

<div class="fw" data-widget="running-sum">This picture needs JavaScript.</div>

*Interactive 2. Three sums kept in each format: the ones of Table 12, Table 11's weight update and 1 plus
many small adds. The table says when each format stops moving and why: the spacing at the sum against
the size of the add.*

## 5. How the trainer uses them

### 5.1 Store 16 bits, work in float

Every kernel of the trainer [1] is now a template on the dtype of its tensors, float or bfloat16, and
the rule is the same in all of them:

1. **Read**: `load4` reads 4 neighbouring values, 8 bytes for bfloat16, and turns each into a float by
   appending 16 zero bits (Section 2.3): exact.
2. **Work in float**: every multiply, add, exp and sum, the 64 sums of the linear kernel's threads and the
   softmax's row sums among them.
3. **Write**: `store4` rounds each result to the nearest bfloat16, once (Section 3.1).

So each value a kernel writes is the float result rounded once: off by at most $$u = 0.39\%$$ of itself,
plus the float math's own, far smaller, error. Section 4.3's piling up never happens inside a kernel.
cuBLAS is told the same: its products take bfloat16 matrices and sum in float (`CUBLAS_COMPUTE_32F`).

Some values stay float whatever the tensors are:

| value | why float |
|---|---|
| AdamW's $$m$$ and $$v$$ | moving averages: each step changes them by a little (Section 4.2) |
| LayerNorm's mean and rstd, the classifier's losses, row maxes and sums | one per row: few, and the backward builds on them |
| the gradient norm | a sum of 124 million squares (Section 4.3) |

*Table 13. What the trainer keeps in float, next to bfloat16 tensors.*

### 5.2 Adding onto a bfloat16 from many threads

Two kernels add onto values that other threads may add onto at the same moment: the encoder's
`wte_grad` (a token seen twice) and LayerNorm's `weight_grad` and `bias_grad`. For floats, `atomicAdd`
does the read-add-write as one step. A T4 has no `atomicAdd` for bfloat16 (it came with compute
capability 8.0), so `atomic_add4` does it for 4 values, 8 bytes, with a compare-and-swap:

1. read the 8 bytes;
2. add the 4 values in float, and round each to bfloat16;
3. write the 8 bytes back, **only if** they still hold what step 1 read; `atomicCAS` does this check and
   the write as one step.

If another thread wrote in between, step 3 writes nothing, and the thread goes round again from what is
there now. Each add rounds once, so a value $$n$$ threads add onto is rounded $$n$$ times: the small piling
up of Section 4.3, on a few adds.

### 5.3 What 16 bits save

| | float | bfloat16 |
|---|---|---|
| GPT-2 small's 124,439,808 parameters | 498 MB | 249 MB |
| one layer's attention weights, $$4 \times 12 \times 1024 \times 1024$$ | 201 MB | 101 MB |
| bytes GELU reads and writes per value, forward and backward | 8 and 12 | 4 and 6 |

*Table 14. Half the memory, and half the bytes moved by the kernels whose time is memory.*

On a T4 the formats are not equal for the matrix products: its tensor cores multiply half, at 65 TFLOPS
against 8.1 for float [9], and have no bfloat16; tensor cores for bfloat16 came with the A100. So on a T4
half with loss scaling is the fast way, and bfloat16 the simple way, and the one newer GPUs run fast.
The trainer's `Tensor` holds both; its kernels take bfloat16.

## 6. What to remember

1. **One formula.** $$x = (-1)^s \cdot 2^{E - \text{bias}} \cdot (1 + f / 2^m)$$, with a hidden 1 that
   gives $$p = m + 1$$ significant bits. The exponent's bits decide the range, the fraction's the digits.
2. **Two cuts.** Half: 5 exponent bits, 10 fraction bits, so 3 digits up to 65,504. bfloat16: float's 8
   exponent bits, 7 fraction bits, so 2 digits and float's whole range; a float's first 16 bits.
3. **One bound.** Rounding to the nearest moves a normal value by at most $$u = 2^{-p}$$ of itself:
   $$4.9 \times 10^{-4}$$ for half, $$3.9 \times 10^{-3}$$ for bfloat16. Ties go to the even one.
4. **Three failures.** Gradients below half's range become 0 (loss scaling multiplies them up); updates
   below half the spacing at a weight are lost (float master weights, or stochastic rounding); long sums
   stop growing (add in float).
5. **The trainer's rule.** Tensors in bfloat16, every sum in float, each result rounded once as it is
   stored; AdamW's $$m$$ and $$v$$, means, losses and the norm in float.

## References

1. D. B. Song. *gpt2-small*, branch `c-cuda-trainer`: GPT-2 small pre-training in C and CUDA;
   `llmc/tensor.cuh` (the dtypes, `load4`, `store4`, `atomic_add4`) and the kernels in `llmc/kernels/`.
   [github.com/ducbachsong/gpt2-small](https://github.com/ducbachsong/gpt2-small/tree/c-cuda-trainer)
2. IEEE. *IEEE Standard for Floating-Point Arithmetic*, IEEE 754-2019. The binary16 format, and round to
   nearest, ties to even, as the default.
3. D. Goldberg. *What Every Computer Scientist Should Know About Floating-Point Arithmetic.* ACM Computing
   Surveys 23(1), 1991. Section 1.2: relative error and the unit in the last place.
4. N. J. Higham. *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM, 2002. Chapter 4: the error
   of a sum added one term at a time.
5. P. Micikevicius et al. *Mixed Precision Training.* 2017. Float master weights, loss scaling, and
   gradients that half rounds to 0. [arxiv.org/abs/1710.03740](https://arxiv.org/abs/1710.03740)
6. D. Kalamkar et al. *A Study of BFLOAT16 for Deep Learning Training.* 2019.
   [arxiv.org/abs/1905.12322](https://arxiv.org/abs/1905.12322)
7. S. Wang, P. Kanwar. *BFloat16: The secret to high performance on Cloud TPUs.* Google Cloud blog, 2019.
   [cloud.google.com](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)
8. PyTorch. `c10/util/BFloat16.h`: `round_to_nearest_even`, the rounding of Section 3.4.
   [github.com/pytorch/pytorch](https://github.com/pytorch/pytorch/blob/main/c10/util/BFloat16.h)
9. NVIDIA. *Tesla T4 Tensor Core GPU* datasheet: 65 TFLOPS in mixed precision (half), 8.1 in float.
   [nvidia.com](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)
10. A. Karpathy. *llm.c*: `floatX`, and the AdamW kernel's stochastic rounding of bfloat16 parameters.
    [github.com/karpathy/llm.c](https://github.com/karpathy/llm.c)
11. D. B. Song. *Classifier backward, broken down and rebuilt: softmax, cross-entropy and prob minus
    one.* 2026. The gradient of Section 4.1. [/posts/classifier-backward/](/posts/classifier-backward/)
