\# Methodology



\## Objective



Studio NST implements and evaluates three approaches to neural style transfer:



1\. Optimization-based Neural Style Transfer based on the Gatys formulation.

2\. Feed-forward style transfer based on the Johnson architecture.

3\. Arbitrary style transfer using Adaptive Instance Normalization.



The objective is to understand the computational and architectural differences between these approaches through a common implementation and controlled experiments.



\## Experimental Setup



All experiments were executed locally using:



\- GPU: NVIDIA GeForce GTX 1650

\- GPU memory: 4 GB

\- CUDA runtime: 13.0

\- PyTorch: 2.14.0+cu130

\- Python: 3.14

\- Operating system: Windows



The same content and style images were reused where applicable so that differences between methods were not caused by changing image inputs.



\## Content and Style Images



The project uses a city skyline photograph as the primary content image and an impressionist landscape painting as the primary style image.



The images are stored under:



\- `assets/content/content1.jpg`

\- `assets/styles/style1.jpg`



\## Gatys Method



The Gatys implementation represents content using intermediate VGG19 activations and represents style using Gram matrices of selected feature maps.



The generated image is initialized from the content image and optimized directly using Adam.



The main variables investigated were:



\- optimization step count

\- input resolution

\- runtime

\- peak GPU memory

\- content loss

\- style loss



Step scaling was evaluated at 100, 300, and 500 optimization steps.



Resolution scaling was evaluated at 256, 384, and 512 pixels using 300 optimization steps.



\## Johnson Method



The Johnson implementation uses a feed-forward transformation network containing convolutional layers, downsampling, residual blocks, upsampling, and a final reconstruction layer.



The network is trained using perceptual losses derived from VGG19 features.



A limited 200-image CIFAR-10 subset was used to validate the complete training pipeline. This experiment was intentionally treated as a pipeline validation rather than a final quality benchmark.



The validation covered:



\- dataset preparation

\- model training

\- CUDA execution

\- checkpoint creation

\- checkpoint loading

\- inference on the project content image



\## AdaIN Method



The AdaIN implementation uses a VGG-based feature encoder and a pretrained decoder.



The content and style images are encoded into feature representations. The AdaIN operation aligns the channel-wise mean and standard deviation of the content features with those of the style features.



The transformed representation is then passed through the decoder to reconstruct the stylized image.



The implementation exposes an `alpha` parameter that controls the interpolation between the original content representation and the transformed representation.



The primary evaluation used:



\- 512-pixel maximum image size

\- alpha = 1.0

\- CUDA inference

\- runtime measurement

\- peak GPU memory measurement

\- visual inspection of the resulting image



\## Measurement



Runtime is measured using a high-resolution performance counter around the model execution.



CUDA synchronization is performed before recording the final runtime so that asynchronous GPU execution does not result in an incomplete measurement.



Peak GPU memory is obtained from PyTorch CUDA memory statistics.



Loss values are recorded directly from the corresponding experiments.



\## Reproducibility



The main experiments can be reproduced using the project scripts.



Gatys:



```text

python scripts/run\_gatys.py --content assets/content/content1.jpg --style assets/styles/style1.jpg --steps 500 --max-size 256 --output outputs/gatys/

