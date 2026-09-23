\# AdaIN Style Strength Experiment



\## Objective



Evaluate how the AdaIN style-strength parameter, alpha, affects inference behavior while keeping the content image, style image, resolution, model weights, and hardware constant.



\## Experimental Setup



\- Method: AdaIN

\- Content image: `assets/content/content1.jpg`

\- Style image: `assets/styles/style1.jpg`

\- Maximum resolution: 512 px

\- Device: NVIDIA GeForce GTX 1650

\- CUDA: Enabled

\- Alpha values: 0.00, 0.25, 0.50, 0.75, 1.00

\- Measurements: runtime and peak GPU memory

\- Runs per alpha value: 1



\## Results



| Alpha | Runtime (s) | Peak GPU Memory (MB) |

|---:|---:|---:|

| 0.00 | 0.4021 | 147.68 |

| 0.25 | 0.2863 | 147.68 |

| 0.50 | 0.2627 | 147.68 |

| 0.75 | 0.2944 | 147.68 |

| 1.00 | 0.2566 | 147.68 |



\## Observations



Peak GPU memory remained constant at 147.68 MB across all tested alpha values.



Measured runtime ranged from 0.2566 seconds to 0.4021 seconds. The observed variation should not be interpreted as alpha making inference faster or slower because each alpha value was measured only once.



The main experimental variable is the visual strength of the style transfer. Lower alpha values interpolate more toward the original content representation, while higher alpha values apply more of the transformed style representation.



\## Interpretation



The experiment demonstrates that AdaIN exposes a continuous style-strength parameter without requiring a separate model or retraining for each tested strength.



Within this experiment, changing alpha did not change peak GPU memory usage. Runtime also remained within a relatively narrow range, suggesting that the alpha parameter primarily changes the output interpolation rather than introducing a substantially different computational path.



\## Limitations



This experiment uses one content image and one style image.



Each alpha value was measured once, so runtime variation cannot be separated from normal execution variability.



The experiment evaluates runtime and memory directly. Visual quality was not assigned a numerical perceptual score.



The results describe this local experiment and should not be treated as universal performance characteristics of AdaIN.



\## Reproducibility



The experiment can be reproduced with the existing AdaIN runner:



```powershell

python scripts/run\_adain.py --content assets/content/content1.jpg --style assets/styles/style1.jpg --output outputs/adain/alpha\_0.00.png --max-size 512 --alpha 0.00

