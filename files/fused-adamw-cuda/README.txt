Raw logs for the post "One launch: a fused AdamW kernel for GPT-2 in CUDA" (2026-10-02).
Code: https://github.com/ducbachsong/gpt2-small/tree/f336dc8 (branch c-cuda-trainer)

test-adamw-t4-timing.log   tests/test_adamw.cu on Google Colab, Tesla T4: the two timing runs and the
                           check that ends each of them, as printed, then the run's last line.
                           Copied from the notebook output.

The full output of that run (test_tensor, test_tensor_buffer and test_adamw; every check printed,
about a thousand lines) is not kept as a file. Its largest differences from PyTorch are listed in
the post's Table 2.
