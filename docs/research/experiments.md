# Experiments

## E002 — Optimization Step Scaling

### Purpose

The first Gatys run established that the pipeline was working end to end. E002 was designed to look at what changes when the number of optimization steps is increased.

The same content image, style image, model configuration, and hardware were used for all three runs. Only the number of optimization steps was changed.

### Setup

- Method: Gatys-style optimization
- Content layer: `conv4_2`
- Style layers: `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`
- Image size: 256 px maximum dimension
- Optimization steps: 100, 300, 500
- Device: NVIDIA GeForce GTX 1650
- GPU memory: 4 GB
- PyTorch: 2.14.0+cu130
- CUDA runtime: 13.0

The benchmark was run through `scripts/benchmark.py` so that the three configurations were measured using the same procedure.

### Measurements

| Steps | Runtime (s) | Peak GPU Memory (MB) | Final Content Loss | Final Style Loss |
|---:|---:|---:|---:|---:|
| 100 | 5.0686 | 264.57 | 10.795232 | 7.127049e-06 |
| 300 | 13.5208 | 263.63 | 9.615572 | 5.248773e-06 |
| 500 | 22.5941 | 263.63 | 9.292706 | 4.966042e-06 |

### What the experiment shows

Increasing the step count increased runtime in this setup. The measured content loss also decreased as more optimization steps were allowed.

The change from 100 to 300 steps was larger than the change from 300 to 500 steps in terms of the measured content loss. This is useful for understanding the practical trade-off in optimization-based NST: additional iterations continue to improve the measured objective, but each additional block of iterations costs more compute.

Peak GPU memory stayed almost unchanged across these runs, at roughly 264 MB. For this particular 256 px configuration, the optimization was therefore not memory-limited on the GTX 1650.

These observations apply to this experiment and image pair. They should not be treated as universal performance characteristics of Gatys NST.

### Reproducibility

Run the benchmark with:

```powershell
python scripts/benchmark.py --content assets/content/content1.jpg --style assets/styles/style1.jpg --max-size 256
```

The benchmark writes the individual outputs and measurements under:

```text
outputs/e002_benchmark/
```

The main result table is stored in:

```text
outputs/e002_benchmark/benchmark_results.csv
```

The run configuration and hardware information are stored in:

```text
outputs/e002_benchmark/benchmark_metadata.txt
```
