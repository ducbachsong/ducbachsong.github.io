Raw logs for the post "How many threads? Launch sizes for GPT-2's encoder and LayerNorm kernels on a T4"
(published 2026-10-07; part 2, LayerNorm, added 2026-10-14 in version 4).
Code: https://github.com/ducbachsong/gpt2-small (branch c-cuda-trainer). All runs on Google Colab, Tesla T4,
built with the command in the post (batches A-C: no -arch flag, Colab's nvcc built sm_75 machine code
directly; batch D: -arch=native, which on a T4 is sm_75 as well).

a-run1.log .. a-run6.log
    Batch A, commit a1f1b7b. Every check, the two 20-step timings (a Tensor each, one TensorBuffer) and
    part 4: the 3 kernels at 32 launch sizes, the wte backward loading one float at a time. No warm-up.
    a-run1.log is the log the first version of the post was written from.

b-summary.txt
    Batch B, commit a1f1b7b plus a first version of part 5 (the float-load wte backward against a copy with
    float4 loads; the forward with 256-1024 blocks of 128 threads). Only the 5 runs' summary (with the SM
    clock from nvidia-smi) and run 1's part 5 output were kept.

c-run1.log .. c-run5.log
    Batch C, commit e84ce51: the wte backward loads float4s, every timing loop starts with a one-second
    warm-up, and part 5 compares the old float-load kernel (kept in the test) with it. Every check and every
    timing, as printed. In c-run5.log the float4 wte backward's reference at 512 x 1024 in part 5 came out
    at 0.188 ms (0.163-0.167 in the other runs); the post leaves that run out of its float4 ratios.

c-clocks-run1.csv .. c-clocks-run5.csv
    Batch C: nvidia-smi every 200 ms during each run: timestamp, SM clock (MHz), memory clock (MHz),
    temperature (C), power (W).

sass-loads-and-atomics.txt
    The loads and atomics of each kernel of commit e84ce51's test program, from cuobjdump, and the list of
    machine code in it (sm_75).

An earlier execution of batch C's code, whose logs were not kept, printed the same picture: float loads
2.10-2.47x and float4 loads 1.02-1.04x at 5,120 threads, and 768 x 128 at 0.878-0.885 of 512 x 1024.

d-run1.log .. d-run3.log  (part 2, LayerNorm)
    Batch D, tests/test_layer_norm.cu. The kernels (llmc/kernels/layer_norm.cuh) are those of commit c730f66
    in all three runs; the test grew between them:
    d-run1.log  20-step timing averaged over steps 2..20; stops at a hand-worked test whose bias was mistyped
                as 1 instead of 0 (the test's error: the kernel gave norm + 1, as asked). The post recomputes
                its 20-step averages over steps 3..20 from the table.
    d-run2.log  the bias fixed, averages over steps 3..20; no part 4. Everything passes.
    d-run3.log  commit c730f66 (only part 4's comment differs): everything, with part 4, the 2 kernels at 40
                launch sizes after a one-second warm-up. Everything passes.

Copied from the notebook output.
