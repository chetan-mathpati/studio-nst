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

The results are specific to the current implementation, images, resolution, and GTX 1650 hardware.

## E003 — Resolution Scaling

E003 kept the optimization budget fixed at 300 steps and changed the maximum image resolution from 256 px to 384 px and then 512 px.

Runtime increased from 14.1449 seconds at 256 px to 29.0578 seconds at 384 px and 53.6524 seconds at 512 px. Peak GPU memory increased at the same time, from 264.57 MB to 451.32 MB and then 729.36 MB.

The recorded content loss decreased across the three runs:

```text
256 px  9.591365
384 px  6.550163
512 px  5.367336
```

The recorded style loss also decreased:

```text
256 px  5.511013e-06
384 px  2.810368e-06
512 px  2.035976e-06
```

The most direct engineering observation is the cost of increasing resolution. Moving from 256 px to 512 px increased runtime by about 3.8 times and peak GPU memory by about 2.8 times in this setup.

The lower losses at higher resolution are useful measurements, but they should not be treated as evidence of better visual quality without a separate image-quality evaluation. The next stages of Studio NST will therefore consider both computational measurements and visual results rather than relying on a single loss value.

### Experimental note

The 256 px E003 runtime was 14.1449 seconds, compared with 13.5208 seconds for the earlier 300-step 256 px E002 run. This small difference comes from separate executions of the same pipeline and does not change the resolution-scaling observation. E003 results are reported using the values recorded by the E003 benchmark itself.

### Next experiment

The next major stage is to implement a feed-forward style-transfer approach and compare it with the optimization-based Gatys pipeline. This changes the question from how much computation is needed for iterative optimization to how a trained transformation network behaves at inference time.
