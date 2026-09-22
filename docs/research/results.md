# Results

## E002 — Step Scaling

E002 measured the effect of increasing the optimization budget from 100 to 300 to 500 steps.

At 100 steps, the run took 5.0686 seconds and finished with a content loss of 10.795232. At 300 steps, runtime increased to 13.5208 seconds while the content loss fell to 9.615572. At 500 steps, runtime reached 22.5941 seconds and the content loss fell further to 9.292706.

The measured style loss followed the same downward direction:

- 100 steps: `7.127049e-06`
- 300 steps: `5.248773e-06`
- 500 steps: `4.966042e-06`

The main pattern is that more optimization steps continued to reduce the measured losses, while the size of the content-loss improvement became smaller between successive settings. Runtime, on the other hand, continued to increase with the step count.

Peak GPU memory remained close to 264 MB for all three runs. This suggests that, at the tested 256 px resolution, increasing the number of optimization steps affected compute time much more than peak memory usage.

### Interpretation

This experiment gives a useful baseline for the later experiments in Studio NST. It also shows why optimization-based style transfer can become expensive when the user wants more iterations: the additional computation is paid for every new image.

The results are specific to the current implementation, images, resolution, and GTX 1650 hardware. They are not intended as a general benchmark for all Gatys implementations or GPUs.

### Next experiment

The next experiment will change the image resolution while keeping the optimization budget controlled. The goal is to measure how image size affects runtime and GPU memory in the same pipeline.
