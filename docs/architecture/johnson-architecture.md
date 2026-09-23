# Johnson-Style Architecture

```text
Content image
     |
     v
TransformerNet
     |
     +--> Input convolution
     |
     +--> Downsample
     |
     +--> Downsample
     |
     +--> Residual blocks
     |
     +--> Upsample
     |
     +--> Upsample
     |
     +--> Output convolution
     |
     v
Stylized image
```

During training, the output is evaluated by a frozen VGG19 network.

```text
Content image ----------------------+
                                     |
                                     v
                              Frozen VGG19
                                     |
                                     +--> Content features
                                     |
                                     +--> Style features
                                     |
                                     v
                              Perceptual losses
                                     |
                                     v
TransformerNet <---------------- Backpropagation
```

The VGG network is not trained in this stage. Only the transformation network parameters are updated.
