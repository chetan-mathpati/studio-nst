# Literature Review

## 1. Gatys et al. — A Neural Algorithm of Artistic Style

Gatys et al. established the optimization-based formulation used as the conceptual starting point for this project.

The central idea is to represent:

- **content** through activations of a convolutional neural network;
- **style** through feature correlations, commonly represented with Gram matrices.

A generated image is optimized so that its deep representation matches the content image while its feature statistics match the style image.

### Engineering implication

The method is highly flexible because the output image itself is optimized for each content/style pair. The trade-off is repeated optimization at inference time.

## 2. Johnson et al. — Perceptual Losses for Real-Time Style Transfer

Johnson et al. explored training a feed-forward transformation network using perceptual losses derived from a pretrained network.

The key engineering shift is:

> move expensive optimization into training so inference can become a forward pass.

### Engineering implication

This creates a natural comparison against optimization-based NST: training cost versus per-image inference cost.

## 3. Huang & Belongie — Arbitrary Style Transfer in Real-Time

AdaIN introduced a mechanism for aligning channel-wise feature statistics of content and style representations.

The approach enables arbitrary style images without training a separate transformation network for every style.

### Engineering implication

AdaIN changes the user workflow from selecting among trained style-specific models to providing arbitrary style references.

## 4. VGG

VGG-style convolutional representations are commonly used as perceptual feature spaces in NST.

Studio NST uses a pretrained VGG19 feature extractor for the optimization-based pipeline.

## Research gap for Studio NST

The purpose of Studio NST is not to claim a new NST algorithm. Its research contribution is an implementation-and-evaluation study that unifies multiple NST paradigms and measures their practical trade-offs under a reproducible experimental setup.
