# Johnson Experiments

## J001 — Training Pipeline Validation

### Goal

Verify that the feed-forward transformation network can be trained end to end with perceptual content and style losses.

### Initial configuration

- Model: TransformerNet
- Residual blocks: 5
- Input size: 128 px
- Batch size: 4
- Optimizer: Adam
- Learning rate: 1e-3
- Content layer: `conv4_2`
- Style layers: `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`
- Style: `assets/styles/style1.jpg`

The first run should be treated as a pipeline validation run rather than a final quality benchmark.

No performance or quality numbers should be added to the report until they are measured from an actual run.

## J002 — Feed-Forward Inference Benchmark

After a trained checkpoint exists, inference will be measured on the same content image used by the Gatys experiments.

The benchmark should record:

- inference runtime
- peak GPU memory
- output resolution
- checkpoint size
- whether the model was run with CUDA

The same content image and resolution should then be passed through the Gatys pipeline and the feed-forward model so that the two approaches can be compared under clearly stated conditions.

## J003 — Training Data Scale

The training dataset size can later be varied while keeping the style and training configuration fixed.

The purpose is to observe how training-set size affects the learned transformation rather than to claim a universal relationship.

All measurements will be recorded from actual runs.

## J004 — Training and Inference Validation

The Johnson-style feed-forward pipeline was trained using a limited CIFAR-10 subset of 200 images to validate the complete training and inference workflow.

The experiment used CUDA on an NVIDIA GeForce GTX 1650.

Training completed for 50 optimization steps in 6.15 seconds. The recorded total loss decreased from 325.259888 at the first step to 101.421913 at the final recorded step. Content loss decreased from 9.104837 to 8.196975, while style loss decreased from 3.161550e-03 to 9.322494e-04.

The resulting checkpoint was successfully loaded by the inference pipeline and applied to the project content image at a maximum resolution of 512 pixels.

The generated image did not produce satisfactory preservation of the original scene structure. The result showed substantial distortion and did not provide sufficient evidence of successful style transfer.

This experiment is therefore treated as a pipeline validation rather than a final Johnson style-transfer quality result.

The main limitation was the extremely small training dataset. The experiment confirms that the implementation can train, save, reload, and execute a feed-forward transformation network, but it does not establish that the model has learned a high-quality representation of the target style.

A larger and more appropriate training corpus would be required for a meaningful Johnson-style quality evaluation.
