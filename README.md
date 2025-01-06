# ColorGANvas

## Introduction

ColorGANvas is a machine learning model that colourizes black-and-white images using data from online repositories like ImageNet and self-captured grayscale images. 

## Technical Overview

We employ a modified Generative Adversarial Network (GAN) architecture, where grayscale images are input into a UNET-based generator that incorporates a Recurrent Neural Network (RNN) at the smallest kernel size to enhance temporal coherence. This RNN bottleneck helps maintain temporal consistency across images, ensuring realistic color transitions by leveraging contextual feature learning. In the training phase, the generator encodes and decodes the data, while the discriminator provides feedback, enhancing the generator’s accuracy. During testing, we retain only the generator, omitting the evaluation step. This hybrid approach balances spatial and sequential dependencies, aiming to improve colorization quality by preserving contextual details.

## Packages Used

- Numpy
- PyTorch

## Images:

**Model Architecture**:
<img src=./img/finalArchitecture.png alt="Custom Size 3"/>



**Result 1**:
<img src=./img/result1.png alt="Custom Size 3"/>

**Result 2**:
<img src=./img/result2.png alt="Custom Size 3"/>

