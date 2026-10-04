Raw logs for the post "Adding 124 million squares: a one-kernel gradient norm for GPT-2 in CUDA" (2026-10-05).
Code: https://github.com/ducbachsong/gpt2-small/tree/ea4b2ab (branch c-cuda-trainer)

test-grad-norm-t4-float.log        Run 1. tests/test_grad_norm.cu on Google Colab, Tesla T4, with the first
                                   version of the kernel: one float per load. Every check and the timing run
                                   (2 tensors, PyTorch per tensor), as printed. This version is not in the
                                   repository's history: it was replaced by the float4 version before the commit.

test-grad-norm-t4-float4.log       Run 2. The committed kernel (float4 loads), commit b7081af: every check and
                                   the timing run (2 tensors, PyTorch per tensor), as printed. The two block
                                   tests use 4x more values than in run 1, and one test was added.

test-grad-norm-t4-148-tensors.log  Run 3. The same kernel with the timing extended: 2 tensors and GPT-2 small's
                                   real 148 tensors, each against PyTorch per tensor (C++ clip_grad_norm_) and
                                   PyTorch foreach (Python clip_grad_norm_). Every check, as printed.

All copied from the notebook output.
