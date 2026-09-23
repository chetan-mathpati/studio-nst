# Experimental Results

Studio NST evaluates three neural style transfer approaches: optimization-based Neural Style Transfer, feed-forward transformation, and Adaptive Instance Normalization.

All measurements below are from experiments actually executed in the project environment.

## Hardware

- GPU: NVIDIA GeForce GTX 1650
- GPU memory: 4 GB
- CUDA runtime: 13.0
- PyTorch: 2.14.0+cu130
- Python: 3.14
- Operating system: Windows

## Gatys — Step Scaling

The same content/style pair was processed at 256 pixels using different optimization step counts.

| Steps | Runtime | Peak GPU Memory | Final Content Loss | Final Style Loss |
|---:|---:|---:|---:|---:|
| 100 | 5.0686 s | 264.57 MB | 10.795232 | 7.127049e-06 |
| 300 | 13.5208 s | 263.63 MB | 9.615572 | 5.248773e-06 |
| 500 | 22.5941 s | 263.63 MB | 9.292706 | 4.966042e-06 |

Runtime increased substantially as the optimization budget increased. The measured content loss continued to decrease, although the improvement became smaller as additional optimization steps were added.

These values describe the specific experiment and should not be interpreted as universal performance characteristics.

## Gatys — Resolution Scaling

The same content/style pair was processed for 300 optimization steps at three resolutions.

| Resolution | Runtime | Peak GPU Memory | Content Loss | Style Loss |
|---:|---:|---:|---:|---:|
| 256 px | 14.1449 s | 264.57 MB | 9.591365 | 5.511013e-06 |
| 384 px | 29.0578 s | 451.32 MB | 6.550163 | 2.810368e-06 |
| 512 px | 53.6524 s | 729.36 MB | 5.367336 | 2.035976e-06 |

Increasing resolution increased both runtime and peak GPU memory. The lower measured losses at higher resolutions are reported as optimization metrics only and are not treated as direct perceptual quality scores.

## Johnson — Training Validation

The feed-forward transformation network was trained on a limited 200-image CIFAR-10 subset.

| Metric | Measured Result |
|---|---:|
| Training images | 200 |
| Optimization steps | 50 |
| Runtime | 6.15 s |
| Initial total loss | 325.259888 |
| Final total loss | 101.421913 |
| Initial content loss | 9.104837 |
| Final content loss | 8.196975 |
| Initial style loss | 3.161550e-03 |
| Final style loss | 9.322494e-04 |

The checkpoint was successfully saved and loaded for inference.

However, the generated result did not preserve the content structure sufficiently and did not provide satisfactory style transfer quality. The experiment is therefore treated as a training and inference pipeline validation rather than a successful final Johnson model.

The primary limitation was the extremely small training dataset.

## AdaIN — End-to-End Inference

AdaIN was evaluated using the same project content/style pair at a maximum resolution of 512 pixels.

| Metric | Measured Result |
|---|---:|
| Resolution | 512 px |
| Alpha | 1.0 |
| Runtime | 0.8758 s |
| Peak GPU Memory | 147.68 MB |
| Device | CUDA |
| GPU | NVIDIA GeForce GTX 1650 |

The generated image preserved the main structure of the content scene while transferring the visual statistics and painterly appearance of the style image.

## Observations

The three implementations demonstrate different computational approaches.

Gatys performs iterative optimization directly on the generated image. This provides flexibility but requires repeated forward and backward passes. The resolution and step-scaling experiments show the associated computational cost.

Johnson replaces per-image optimization with a learned feed-forward transformation network. Its inference architecture is substantially different from Gatys, but the quality of the trained model depends on the training process and dataset. The limited validation experiment in Studio NST was sufficient to verify the complete pipeline but not sufficient to establish high-quality transfer.

AdaIN performs style transfer by aligning channel-wise feature statistics between content and style representations and reconstructing the result with a trained decoder. In the measured Studio NST run, inference completed in under one second at the tested 512-pixel setting.

The measurements are hardware-, software-, configuration-, and image-dependent. They are intended to document the behavior of this implementation rather than provide universal rankings of the three methods.
