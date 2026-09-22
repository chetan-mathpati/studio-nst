# Model Architecture

## Optimization-based NST

```text
Content image ──> VGG19 ──> content features
                              \
                               +── loss ──> optimize generated image
                              /
Style image ────> VGG19 ──> Gram matrices
```

## Feed-forward NST

```text
Input image ──> Transformation network ──> Stylized image
                      ^
                      |
               perceptual training
```

## AdaIN

```text
Content ──> Encoder ──> Content features ──┐
                                           ├─> AdaIN ──> Decoder ──> Output
Style ────> Encoder ──> Style features ───┘
```
