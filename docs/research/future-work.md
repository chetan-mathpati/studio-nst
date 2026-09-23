\# Future Work



Studio NST establishes a reproducible local implementation of three neural style transfer approaches, but several extensions would make the evaluation more complete.



\## Larger Johnson Training



The current Johnson experiment uses only 200 CIFAR-10 images as a pipeline validation dataset. A larger and more appropriate image corpus should be used to train a production-quality feed-forward style transfer model.



Future experiments should evaluate how training dataset size affects convergence, transfer quality, and inference behavior.



\## Broader Evaluation Dataset



The current visual evaluation uses one primary content image and one primary style image.



A larger evaluation set should include:



\- urban scenes

\- natural landscapes

\- portraits

\- architecture

\- photographs with fine structures

\- multiple artistic styles



This would make it possible to evaluate whether observations generalize across different image categories.



\## Perceptual Evaluation



The current experiments rely mainly on optimization losses, runtime, memory usage, and visual inspection.



Future work could introduce perceptual evaluation methods and a human evaluation protocol to study the relationship between numerical metrics and perceived image quality.



\## Additional AdaIN Experiments



The current AdaIN experiment uses alpha = 1.0.



Future experiments could evaluate different alpha values to measure the trade-off between content preservation and stylization strength.



Multiple style images could also be evaluated to study arbitrary style transfer behavior more systematically.



\## Performance Evaluation



Future benchmarking could evaluate:



\- CPU versus GPU execution

\- additional GPU hardware

\- batch inference

\- larger resolutions

\- repeated inference stability

\- model loading time

\- end-to-end latency



\## Video Style Transfer



A natural extension would be video style transfer.



This would introduce additional challenges because independently stylizing video frames can produce temporal inconsistencies and flickering.



\## Model Compression



The feed-forward and AdaIN models could be investigated using model compression techniques such as reduced precision or architectural optimization to reduce memory usage and inference latency.



\## Interactive Application



The research implementation could eventually be exposed through a lightweight interface allowing users to:



1\. upload a content image

2\. upload a style image

3\. select a method

4\. adjust style strength

5\. generate the result

6\. compare outputs and performance



\## Reproducible Experiment Suite



A larger automated experiment runner could execute all configurations and produce standardized tables and plots from raw experiment outputs.



This would make future comparisons easier to reproduce and reduce manual reporting.

