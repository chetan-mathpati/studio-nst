\# Limitations



Studio NST is an experimental implementation rather than a production style-transfer system.



\## Dataset Limitations



The primary visual experiments use one content image and one style image. This makes the experiments controlled and reproducible, but it does not establish how the methods behave across a broad range of scenes and artistic styles.



The Johnson training validation uses only 200 CIFAR-10 images. This was sufficient to validate the training and inference pipeline but insufficient for establishing a high-quality learned style-transfer model.



\## Hardware Limitations



All experiments were performed on an NVIDIA GeForce GTX 1650 with 4 GB of GPU memory.



The recorded runtime and memory measurements are therefore specific to this hardware and software environment.



\## Metric Limitations



Content and style losses are optimization objectives rather than direct perceptual quality measurements.



Lower loss values should not automatically be interpreted as visually superior results.



The project does not currently use a human perceptual study or a dedicated image-quality metric to evaluate visual quality across methods.



\## Benchmark Limitations



The Gatys experiments vary optimization steps and resolution, while the AdaIN experiment measures a single inference configuration.



The Johnson experiment is primarily a pipeline validation because of its limited training dataset.



Therefore, the measurements should be interpreted as observations from controlled project experiments rather than a comprehensive benchmark of the three methods.



\## Model Limitations



The Johnson model has not been trained on a sufficiently large dataset for a strong final quality evaluation.



AdaIN uses pretrained VGG and decoder weights rather than training the complete model from scratch within Studio NST.



The current project also focuses on single-image style transfer and does not evaluate video consistency or temporal style transfer.



\## Reproducibility Limitations



Although the main scripts and configurations are included, exact runtime and memory values can vary between executions because of GPU state, system load, library versions, and other environmental factors.

