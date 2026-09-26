Raw logs for the post "Flat buffers: a fast AdamW for GPT-2 in C#" (2026-09-26).
Code: https://github.com/ducbachsong/gpt2-small/tree/csharp-trainer

test-run-cpu.log                   all 30 tests (dotnet test), Intel Core i7-13620H, LibTorch 2.10.0 CPU
benchmark-csharp-cpu.log           BenchmarkOursAgainstTheBuiltInAdamW, same CPU
benchmark-pytorch-cpu.log          src/test/adamw-benchmark-torch.py, same CPU, PyTorch 2.12.0+cpu
benchmark-csharp-t4.log            BenchmarkOursAgainstTheBuiltInAdamW, Google Colab, Tesla T4 (confirmed by nvidia-smi)
benchmark-pytorch-t4.log           src/test/adamw-benchmark-torch.py, same Colab session, PyTorch 2.11.0+cu128
benchmark-csharp-gpu-first-run.log the first GPU run, on Colab; the GPU model was not recorded for it,
                                   so the post reports benchmark-csharp-t4.log instead (see the post's Threats to validity)

The Colab logs were copied from the notebook output; terminal control characters were removed.
Earlier CPU runs of the C# benchmark (not kept as logs): ours 20.93 / built-in 53.07 ms and
ours 18.26 / built-in 49.68 ms.
