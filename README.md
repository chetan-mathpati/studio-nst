\# Studio NST



Studio NST is a research-oriented implementation of neural style transfer.



The project implements and compares three different approaches to neural style transfer:



\- Gatys — optimization-based style transfer

\- Johnson — feed-forward style transfer

\- AdaIN — arbitrary style transfer using Adaptive Instance Normalization



The goal was not just to build a style-transfer demo, but to understand how these methods differ in their architecture, computation, speed, memory usage, and practical behavior.



\## What is Neural Style Transfer?



Neural style transfer combines the content of one image with the visual characteristics of another.



For example, a photograph of a city can be combined with the visual style of a painting to produce a new image that preserves the scene while taking on the appearance of the artwork.



Studio NST approaches this problem in three different ways.



\## The Three Methods



\### Gatys



Gatys-style transfer optimizes the output image directly.



A pretrained VGG19 network is used to extract representations of the content and style images. Content is represented through feature activations, while style is represented using correlations between feature channels through Gram matrices.



The generated image is then iteratively optimized to reduce the difference between these representations.



This makes Gatys useful for understanding the optimization process, but each output requires an iterative optimization run.



\### Johnson



Johnson-style transfer uses a feed-forward transformation network.



Instead of optimizing every output image from scratch, a neural network learns a transformation that can generate stylized images in a single forward pass.



The project uses perceptual losses derived from VGG19 during training.



The Johnson implementation in this project was primarily used to validate the complete training and inference pipeline. The training experiment used a limited 200-image CIFAR-10 subset, so the resulting model is not treated as a meaningful visual-quality benchmark.



\### AdaIN



AdaIN uses Adaptive Instance Normalization to transfer style statistics from the style image to the content representation.



The method aligns the channel-wise mean and variance of the content and style features before decoding the result.



Unlike a transformation network trained for one particular style, AdaIN can transfer arbitrary styles at inference time.



\## Experiments



The experiments were designed around a few practical questions:



1\. How does Gatys behave as optimization steps increase?

2\. How does image resolution affect runtime and GPU memory?

3\. How does AdaIN style strength affect the resulting transfer?

4\. How do optimization-based and feed-forward approaches differ in practice?

5\. What can be run comfortably on a 4 GB GPU?



All reported measurements below come from runs performed as part of this project.



\## Experimental Setup



The main experiments were performed on:



\- GPU: NVIDIA GeForce GTX 1650

\- GPU memory: 4 GB

\- CUDA: 13.0

\- PyTorch: 2.14.0+cu130

\- Python: 3.14

\- OS: Windows



Runtime and GPU memory were measured during inference or optimization runs.



\## Results



\### Gatys: Optimization Step Scaling



| Steps | Runtime | Peak GPU Memory |

|---:|---:|---:|

| 100 | 5.0686 s | 264.57 MB |

| 300 | 13.5208 s | 263.63 MB |

| 500 | 22.5941 s | 263.63 MB |



Increasing the number of optimization steps increased runtime.



The recorded content loss also continued to decrease, although the improvement became smaller as more steps were added.



These measurements describe this experiment and should not be interpreted as universal performance characteristics.



\### Gatys: Resolution Scaling



300 optimization steps were used for each resolution.



| Resolution | Runtime | Peak GPU Memory |

|---:|---:|---:|

| 256 px | 14.1449 s | 264.57 MB |

| 384 px | 29.0578 s | 451.32 MB |

| 512 px | 53.6524 s | 729.36 MB |



Higher resolutions required substantially more runtime and GPU memory in the recorded experiment.



The measured optimization losses also changed with resolution, but these values are optimization metrics rather than direct measures of perceptual image quality.



\### AdaIN



A 512 px AdaIN inference run completed in:



\- Runtime: 0.8758 s

\- Peak GPU memory: 147.68 MB



The resulting image preserved the main structure of the content image while transferring the visual characteristics of the selected style image.



\### AdaIN: Style Strength



AdaIN was also tested at five style-strength values using the same content and style images.



| Alpha | Runtime | Peak GPU Memory |

|---:|---:|---:|

| 0.00 | 0.4021 s | 147.68 MB |

| 0.25 | 0.2863 s | 147.68 MB |

| 0.50 | 0.2627 s | 147.68 MB |

| 0.75 | 0.2944 s | 147.68 MB |

| 1.00 | 0.2566 s | 147.68 MB |



The main purpose of this experiment was to observe the visual interpolation between the original content representation and the transferred style.



Peak GPU memory remained at 147.68 MB across the tested values.



Only one run was performed for each alpha value, so the runtime differences should not be interpreted as evidence that alpha itself changes inference speed.



\### Johnson



The Johnson pipeline was successfully trained and executed on the local GPU.



The experiment used 200 CIFAR-10 images to validate the training and inference pipeline.



The resulting model was not sufficiently trained for a meaningful visual-quality comparison, so it is documented as a pipeline validation experiment rather than presented as a successful style-transfer benchmark.



\## Project Structure



```text

Studio-NST/

├── api/

├── assets/

│   ├── content/

│   └── styles/

├── checkpoints/

├── core/

├── docs/

│   ├── architecture/

│   └── research/

├── outputs/

├── scripts/

├── tests/

├── web/

├── weights/

├── generate\_report.py

├── requirements.txt

├── run\_studio.ps1

└── README.md



\### Core



Contains the implementations used by the three style-transfer approaches and supporting image/model utilities.



\### Scripts



Contains experiment and inference entry points.



\### Docs



Contains the research notes, experimental methodology, results, limitations, future work, and system architecture.



\### Web



Contains the local Studio NST interface used to run the models interactively.



\## Running the Experiments



Create and activate the project environment, install the required dependencies, and make sure the required model weights are available.



Example Gatys run:



```powershell

python scripts/run\_gatys.py --content assets/content/content1.jpg --style assets/styles/style1.jpg --steps 500 --output outputs/gatys/



\## Gatys step-scaling benchmark:

python scripts/benchmark.py --content assets/content/content1.jpg --style assets/styles/style1.jpg --max-size 256



\## Gatys resolution benchmark:

python scripts/benchmark\_resolution.py --content assets/content/content1.jpg --style assets/styles/style1.jpg



\## AdaIN inference:

python scripts/run\_adain.py --content assets/content/content1.jpg --style assets/styles/style1.jpg --output outputs/adain/stylized.png --max-size 512



\### Local Studio



The project also includes a small local web interface for experimenting with the three methods.



Start it with:



.\\run\_studio.ps1



Then open:



http://127.0.0.1:8000



The interface allows a user to:



choose content and style images

select Gatys, Johnson, or AdaIN

adjust method-specific parameters

generate a result

inspect runtime and GPU memory

compare the original and generated images



The interface is intended as an experimental front end rather than a production service.



\### Research Documentation



More detailed notes are available in:



docs/research/literature-review.md

docs/research/methodology.md

docs/research/experiments.md

docs/research/experiments-adain.md

docs/research/experiments-adain-alpha.md

docs/research/results.md

docs/research/limitations.md

docs/research/future-work.md

docs/architecture/system-design.md



\## A technical report is also included at:



docs/Studio-NST-Technical-Report.pdf



\### Limitations



This project is an experimental implementation rather than a production-ready style-transfer system.



\## Important limitations include:



The primary visual evaluation uses one content image and one style image.

The Johnson experiment uses only 200 CIFAR-10 images for pipeline validation.

Optimization losses are not direct measures of human-perceived image quality.

Runtime and memory measurements were collected on one GTX 1650 4 GB GPU.

Results can vary depending on hardware, software versions, and system conditions.

Video style transfer and temporal consistency were not evaluated.

The AdaIN alpha experiment uses one image pair and one run per alpha value.

Future Work



\## Possible extensions include:



training Johnson on a larger and more suitable dataset

evaluating multiple content and style image pairs

adding perceptual evaluation metrics

expanding AdaIN experiments

profiling inference across different resolutions and hardware

investigating video style transfer and temporal consistency

improving reproducibility across environments



\### References



\## Gatys, L. A., Ecker, A. S., \& Bethge, M.



A Neural Algorithm of Artistic Style.



https://arxiv.org/abs/1508.06576



\## Johnson, J., Alahi, A., \& Fei-Fei, L.



Perceptual Losses for Real-Time Style Transfer and Super-Resolution.



https://arxiv.org/abs/1603.08155



\## Huang, X., \& Belongie, S.



Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization.



https://arxiv.org/abs/1703.06868



\## Simonyan, K., \& Zisserman, A.



Very Deep Convolutional Networks for Large-Scale Image Recognition.



https://arxiv.org/abs/1409.1556





