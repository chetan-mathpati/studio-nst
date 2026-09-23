# A001 — AdaIN End-to-End Inference

The AdaIN implementation was evaluated using the same content and style images used for the Gatys experiments.

The pipeline uses a VGG-based feature encoder, adaptive instance normalization, and a pretrained decoder. The AdaIN transformation aligns the channel-wise feature mean and standard deviation of the content representation with those of the style representation before reconstruction.

Inference was performed on an NVIDIA GeForce GTX 1650 using CUDA at a maximum image size of 512 pixels.

Measured inference runtime was 0.8758 seconds with a peak GPU memory allocation of 147.68 MB.

The generated result preserved the main structure of the city scene while transferring the painterly appearance of the style image. Unlike the limited Johnson training validation, the AdaIN output provided visually meaningful style transfer.

The experiment demonstrates the practical advantage of AdaIN for arbitrary style transfer: the style image can be changed at inference time without retraining the transformation network for each new style.

The measured runtime and memory values are specific to this image pair, resolution, implementation, hardware, and software environment.
