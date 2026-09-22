# Methodology

## Experimental factors

The initial benchmark matrix should vary:

- method,
- image resolution,
- optimization steps where applicable,
- device,
- elapsed inference time,
- peak GPU memory where measurable,
- output artifact.

## Methods

### Baseline A — Optimization-based NST

Optimize the pixels of the generated image against content and style objectives.

### Baseline B — Feed-forward NST

A transformation network is trained to map an input image to a stylized image in one forward pass.

### Baseline C — AdaIN

Encode content and style, align feature statistics, and decode the transformed representation.

## Reproducibility

Every experiment should record:

- source image identifiers,
- method,
- configuration,
- software versions,
- hardware,
- random seed where relevant,
- runtime,
- memory measurements where available.

No benchmark value should be manually entered before the corresponding experiment has been executed.
