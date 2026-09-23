\# Studio NST



A research-oriented implementation and experimental evaluation of neural style transfer.



Studio NST implements three approaches:



\- Gatys optimization-based neural style transfer

\- Johnson feed-forward style transfer

\- Adaptive Instance Normalization (AdaIN)



The project focuses on understanding the computational and practical differences between these approaches through reproducible experiments.



\## Overview



Neural style transfer separates visual content from artistic style and combines them to generate a new image.



Studio NST studies three different approaches:



\- \*\*Gatys:\*\* directly optimizes the generated image using content and style representations.

\- \*\*Johnson:\*\* learns a feed-forward transformation network for faster inference.

\- \*\*AdaIN:\*\* aligns feature statistics between content and style representations to enable arbitrary style transfer.



The implementations are evaluated using runtime, GPU memory, optimization losses, and visual inspection.



\## Research Questions



The project investigates:



1\. How does optimization-based style transfer behave as the number of optimization steps increases?

2\. How does input resolution affect runtime and GPU memory usage?

3\. What practical differences exist between iterative optimization and feed-forward transformation?

4\. How does AdaIN enable arbitrary style transfer?

5\. What are the practical constraints of neural style transfer on a 4 GB GPU?



\## Methods



\### Gatys



The generated image is optimized directly using VGG19 content features and Gram-matrix-based style representations.



\### Johnson



A feed-forward transformation network is trained using perceptual losses derived from VGG19 features.



\### AdaIN



Content and style feature statistics are aligned using Adaptive Instance Normalization, followed by decoding into the output image.



\## Experimental Setup



Experiments were performed on:



\- GPU: NVIDIA GeForce GTX 1650

\- GPU Memory: 4 GB

\- CUDA: 13.0

\- PyTorch: 2.14.0+cu130

\- Python: 3.14

\- Operating System: Windows



The experiments measure runtime, peak GPU memory, optimization losses where applicable, and visual output characteristics.



\## Results



\### Gatys Step Scaling



| Steps | Runtime | Peak GPU Memory |

|---:|---:|---:|

| 100 | 5.0686 s | 264.57 MB |

| 300 | 13.5208 s | 263.63 MB |

| 500 | 22.5941 s | 263.63 MB |



Increasing optimization steps increased runtime while continuing to reduce the recorded optimization losses.



\### Gatys Resolution Scaling



| Resolution | Runtime | Peak GPU Memory |

|---:|---:|---:|

| 256 px | 14.1449 s | 264.57 MB |

| 384 px | 29.0578 s | 451.32 MB |

| 512 px | 53.6524 s | 729.36 MB |



Higher resolutions required more runtime and GPU memory in the recorded experiment.



\### AdaIN



AdaIN inference at 512 px completed in 0.8758 seconds with 147.68 MB peak GPU memory.



\### Johnson



The Johnson experiment successfully validated the training and inference pipeline using a limited 200-image CIFAR-10 subset. The resulting model was not sufficiently trained for a meaningful quality comparison.



\## Reproducibility



Run the main experiments using:



```powershell

python scripts/run\_gatys.py --content assets/content/content1.jpg --style assets/styles/style1.jpg --steps 500 --output outputs/gatys/



python scripts/benchmark.py --content assets/content/content1.jpg --style assets/styles/style1.jpg --max-size 256



python scripts/benchmark\_resolution.py --content assets/content/content1.jpg --style assets/styles/style1.jpg



python scripts/run\_adain.py --content assets/content/content1.jpg --style assets/styles/style1.jpg --output outputs/adain/stylized.png --max-size 512



\### Step 9 — Limitations



\## Limitations



\- The primary visual evaluation uses one content image and one style image.

\- The Johnson experiment uses only 200 CIFAR-10 images for pipeline validation.

\- Optimization losses are not direct measures of human-perceived image quality.

\- Experiments were performed on a single GTX 1650 4 GB GPU.

\- Runtime and memory measurements can vary across hardware and system conditions.

\- Video style transfer and temporal consistency were not evaluated.



\## Project Structure



Studio NST

├── assets/

├── checkpoints/

├── core/

├── docs/

├── outputs/

├── scripts/

├── weights/

├── .gitignore

└── README.md



\### Step 11 — Documentation



Detailed research and architecture documentation is available in:



\- `docs/research/literature-review.md`

\- `docs/research/methodology.md`

\- `docs/research/experiments.md`

\- `docs/research/results.md`

\- `docs/research/limitations.md`

\- `docs/research/future-work.md`

\- `docs/architecture/system-design.md`



\## References



\- Gatys et al. — A Neural Algorithm of Artistic Style

&#x20; https://arxiv.org/abs/1508.06576



\- Johnson et al. — Perceptual Losses for Real-Time Style Transfer and Super-Resolution

&#x20; https://arxiv.org/abs/1603.08155



\- Huang \& Belongie — Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization

&#x20; https://arxiv.org/abs/1703.06868



\- Simonyan \& Zisserman — Very Deep Convolutional Networks for Large-Scale Image Recognition

&#x20; https://arxiv.org/pdf/1409.1556
