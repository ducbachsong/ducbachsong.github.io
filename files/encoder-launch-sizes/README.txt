Raw log for the post "How many threads? Launch sizes for GPT-2's encoder kernels on a T4" (2026-10-07).
Code: https://github.com/ducbachsong/gpt2-small/tree/a1f1b7b (branch c-cuda-trainer)

test-encoder-t4.log  tests/test_encoder.cu on Google Colab, Tesla T4: every check, the two 20-step timing runs
                     (each weight its own Tensor, then wte and wpe in one TensorBuffer), and the 32 launch sizes
                     of time_each_kernel_with_other_block_and_thread_counts, as printed. The test file of that
                     run is the committed one except for one comment, changed afterwards.

Copied from the notebook output.
