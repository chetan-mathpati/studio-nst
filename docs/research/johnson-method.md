# Johnson-Style Feed-Forward NST

## Why this method exists

The Gatys pipeline optimizes the pixels of a new image for every transfer. That makes it flexible, but the same optimization process has to be repeated whenever a new content image is supplied.

Johnson, Alahi, and Fei-Fei proposed a different setup: train a feed-forward transformation network using perceptual losses, then use the trained network for inference. Their paper describes this as learning to solve the Gatys-style optimization problem with a feed-forward network. citeturn0academia0

Studio NST implements this idea as the second major method in the project.

## Model

The transformation network uses:

- an initial 9x9 convolution
- two downsampling convolutions
- five residual blocks
- two learned upsampling stages
- a final 9x9 convolution
- instance normalization
- reflection padding

The network takes an RGB image and produces an RGB image of the same spatial size.

## Perceptual supervision

The transformation network is trained without a pixel-level target stylized image.

A frozen VGG19 network provides the supervision:

- `conv4_2` is used for content representation
- `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, and `conv5_1` are used for style representation
- style targets are represented with Gram matrices

The training objective combines content and style losses.

## Important distinction from Gatys

Gatys optimizes the output image directly.

Johnson-style NST optimizes the parameters of a transformation network during training. Once training is finished, the network can transform new content images without running the same iterative pixel optimization again.

The paper reports a large speed advantage over optimization-based transfer. Studio NST will measure its own implementation rather than copying the paper's reported timing. citeturn0academia0

## Current implementation scope

The first implementation is intentionally focused on a single fixed style. Arbitrary style transfer is a later stage and will be implemented separately through AdaIN.

The current training pipeline supports either:

- an image-folder dataset supplied under `assets/training`
- CIFAR-10 as a small automatic dataset option for pipeline validation

CIFAR-10 is useful for checking that the training loop works, but it should not be presented as the final research training dataset for a high-quality style-transfer model.
