\# Studio NST System Design



\## 1. System Overview



Studio NST is a research-oriented neural style transfer system that implements and evaluates three approaches:



\- Optimization-based neural style transfer based on Gatys et al.

\- Feed-forward style transfer based on the Johnson et al. approach.

\- Arbitrary style transfer based on Adaptive Instance Normalization (AdaIN).



The system is organized so that the three methods can be executed independently while sharing common image preprocessing, device management, output handling, and experimental measurement utilities.



\## 2. High-Level Pipeline



The system follows this general pipeline:



1\. Load content and style images.

2\. Resize images according to the selected experiment.

3\. Normalize images for the neural network.

4\. Select the style transfer method.

5\. Execute the selected model or optimization procedure.

6\. Measure runtime and GPU memory where supported.

7\. Convert the generated tensor back into an image.

8\. Save the stylized output.

9\. Record experiment configuration and results.



\## 3. System Components



\### 3.1 Image Utilities



`core/image\_utils.py` provides shared image processing functionality.



Responsibilities include:



\- Loading images from disk.

\- Resizing images.

\- Converting images into tensors.

\- Applying ImageNet normalization.

\- Converting model outputs back into image format.

\- Saving generated images.



\### 3.2 VGG Feature Extractor



`core/vgg.py` provides the pretrained VGG19 feature representation used by the research methods.



VGG19 is used as a fixed feature extractor rather than as a trainable model during style transfer optimization.



The feature representation provides intermediate activations used to describe image content and style.



\### 3.3 Gatys Implementation



`core/gatys.py` implements optimization-based neural style transfer.



The generated image is initialized from the content image and optimized directly.



Content similarity is measured using a VGG feature representation.



Style similarity is measured using Gram matrices derived from multiple VGG layers.



The implementation exposes configurable parameters including:



\- Number of optimization steps.

\- Content weight.

\- Style weight.

\- Learning rate.



This method provides the most direct optimization-based implementation in Studio NST.



\### 3.4 Johnson Implementation



`core/johnson.py` implements a feed-forward transformation network.



The architecture contains:



\- Initial convolutional feature extraction.

\- Downsampling layers.

\- Residual blocks.

\- Upsampling layers.

\- Final image reconstruction.



The training objective uses perceptual losses computed using a pretrained VGG19 network.



The trained transformation network can then generate a stylized image with a single forward pass.



The current project uses a limited CIFAR-10 training subset as a pipeline validation experiment rather than as a production-quality training dataset.



\### 3.5 AdaIN Implementation



`core/adain.py` implements Adaptive Instance Normalization.



The method aligns the channel-wise mean and standard deviation of content features with the corresponding statistics of style features.



`core/adain\_model.py` combines:



\- VGG-based feature extraction.

\- Adaptive instance normalization.

\- A pretrained decoder.



The method supports arbitrary style images without training a separate transformation network for every style.



\## 4. Experimental Scripts



The `scripts/` directory provides reproducible entry points for experiments.



\### Gatys



`scripts/run\_gatys.py`



Runs a single optimization-based style transfer experiment.



\### Gatys Benchmark



`scripts/benchmark.py`



Evaluates multiple optimization step counts and records runtime, memory usage, and final losses.



\### Resolution Benchmark



`scripts/benchmark\_resolution.py`



Evaluates the effect of input resolution on runtime, GPU memory, and recorded losses.



\### Johnson Training



`scripts/train\_johnson.py`



Trains the feed-forward transformation network.



\### Johnson Inference



`scripts/run\_johnson.py`



Runs inference using a trained Johnson transformation network.



\### Training Data Preparation



`scripts/prepare\_training\_data.py`



Prepares the image dataset used by the Johnson training pipeline.



\### AdaIN Inference



`scripts/run\_adain.py`



Runs arbitrary style transfer using the pretrained AdaIN components and records runtime and GPU memory usage.



\## 5. Data Flow



\### Gatys



```text

Content Image ────────────────┐

&#x20;                             ▼

&#x20;                        VGG19 Features

&#x20;                             │

&#x20;                             ├── Content Representation

&#x20;                             │

Style Image ────────────────┐ │

&#x20;                           ▼ ▼

&#x20;                        VGG19 Features

&#x20;                             │

&#x20;                             ▼

&#x20;                        Gram Matrices

&#x20;                             │

&#x20;                             ▼

&#x20;                   Optimization Objective

&#x20;                             │

&#x20;                             ▼

&#x20;                      Generated Image



\### Johnson



Training:



Training Images

&#x20;     │

&#x20;     ▼

Transformer Network

&#x20;     │

&#x20;     ▼

Generated Training Images

&#x20;     │

&#x20;     ▼

VGG19 Feature Extractor

&#x20;     │

&#x20;     ▼

Content + Style Perceptual Loss

&#x20;     │

&#x20;     ▼

Backpropagation

&#x20;     │

&#x20;     ▼

Updated Transformer Network



\### Inference:



Content Image

&#x20;     │

&#x20;     ▼

Trained Transformer Network

&#x20;     │

&#x20;     ▼

Stylized Image



\### AdaIN

Content Image ──► VGG Encoder ──► Content Features ──┐

&#x20;                                                      │

&#x20;                                                      ▼

&#x20;                                               AdaIN Operation

&#x20;                                                      ▲

&#x20;                                                      │

Style Image ────► VGG Encoder ──► Style Features ─────┘

&#x20;                                                      │

&#x20;                                                      ▼

&#x20;                                                 Decoder

&#x20;                                                      │

&#x20;                                                      ▼

&#x20;                                             Stylized Image



\### 6. Evaluation Layer



Studio NST evaluates the methods using several measurements.



Runtime



Runtime is measured using Python's high-resolution performance counter.



CUDA synchronization is performed around GPU operations where required so that asynchronous GPU execution does not produce misleading timing measurements.



GPU Memory



Peak GPU memory is measured using PyTorch CUDA memory statistics.



The reported value represents the peak allocated GPU memory observed during the experiment.



Optimization Loss



For Gatys experiments, final content and style losses are recorded.



These values describe the optimization objective and should not be interpreted as direct measures of human-perceived image quality.



Visual Evaluation



Generated images are inspected to determine whether:



Content structure is preserved.

Style characteristics are transferred.

Major artifacts are present.

The resulting image remains visually coherent.



Visual evaluation is qualitative and is therefore reported separately from numerical measurements.



\### 7. Hardware and Runtime Environment



The experiments were executed on:



GPU: NVIDIA GeForce GTX 1650

GPU memory: 4 GB

CUDA runtime: 13.0

PyTorch: 2.14.0+cu130

Python: 3.14

Operating system: Windows



The hardware configuration is recorded because runtime and memory behavior depend on the execution environment.



\### 8. Repository Structure

Studio NST

├── assets/

│   ├── content/

│   ├── styles/

│   └── training/

├── checkpoints/

│   └── johnson/

├── core/

│   ├── adain.py

│   ├── adain\_model.py

│   ├── gatys.py

│   ├── image\_utils.py

│   ├── johnson.py

│   └── vgg.py

├── docs/

│   ├── architecture/

│   ├── benchmarks/

│   └── research/

├── outputs/

│   ├── adain/

│   ├── gatys\_e001/

│   ├── johnson/

│   └── ...

├── scripts/

│   ├── benchmark.py

│   ├── benchmark\_resolution.py

│   ├── prepare\_training\_data.py

│   ├── run\_adain.py

│   ├── run\_gatys.py

│   ├── run\_johnson.py

│   └── train\_johnson.py

├── weights/

│   └── adain/

├── .gitignore

└── README.md



\### 9. Design Principles



Studio NST follows several implementation principles.



Reproducibility



Experiments are executed through explicit scripts with configurable parameters.



Separation of Methods



Each style transfer approach is implemented as a separate module so that its behavior can be evaluated independently.



Shared Infrastructure



Image processing, feature extraction, output handling, and measurement utilities are reused where appropriate.



Measured Results



Experimental claims are based on recorded runs rather than estimated performance.



Explicit Limitations



Limitations of the datasets, training procedure, evaluation metrics, and hardware are documented alongside the results.



\### 10. Current Scope



The current implementation focuses on research experimentation rather than production deployment.



The system currently provides:



Three neural style transfer approaches.

Reproducible experiment scripts.

Runtime measurements.

GPU memory measurements.

Optimization loss measurements.

Recorded benchmark results.

Research documentation.

Pretrained AdaIN inference.

A validated Johnson training and inference pipeline.



Production deployment, large-scale training, video consistency, interactive UI, and comprehensive perceptual evaluation remain future work.

