# Benchmark Report

## Purpose

Measure practical engineering trade-offs among Studio NST methods.

## Required controls

Use the same:

- content/style pairs,
- output dimensions,
- device,
- measurement protocol.

## Metrics

### Performance

- wall-clock inference time,
- throughput where applicable.

### Resource use

- peak allocated GPU memory,
- peak reserved GPU memory,
- model parameter count.

### Output

Store generated images alongside machine-readable metadata.

### Reproducibility

Each run should identify:

- commit hash,
- Python version,
- PyTorch version,
- GPU,
- method configuration.

## Interpretation

Benchmark numbers are descriptive measurements. Avoid declaring a universal winner because different applications may prioritize different constraints.
