# Densely Connected Convolutional Networks

## Summary

In order to design the DenseNet architecture, its creators got inspired in the 
results of previous works. It was already shown that convolutional networks 
could be deeper, more accurate and efficient if they contained shorter 
connections between layers close to the input and layers close to the output. 
DenseNet architecture takes into account this observation, and connects each 
layer to every other layer in a feed-forward way.

In the case of DenseNets, each layer takes as input all the feature maps from 
preceding layers. As main advantages, DenseNets alleviate the vanishing 
gradient problem, strengthen feature propagation, encourage feature reuse, and 
reduce the number of parameters.

### The vanishing gradient problem

At the time DenseNet was designed, it was already known that, as information 
passed  through many layers, it can “vanish” and “dissapear” by the 
time it is supposed to reach the end of the net. 

As a solution, DenseNet seeks to ensure maximum connection between layers by 
connecting all layers in the net with each other. Each layer obtains inputs 
from all the previous layers, and passes its output to all the following 
layers. This introduces L(L+1)/2 connections in an L-layer network, instead of 
just L, as in traditional architectures.

### Number of parameters

Even though DenseNets has much more connections between layers, they require 
fewer parameters than traditional convolutional networks. 

## Result

As a combination of the previous observations, DenseNets are more efficient, 
have better flow of information, and are easier to train. Also, authors 
observed that dense connections have a regularizing effect that helps reducing 
overfitting.

### Inspiration (state of the art at the moment):

The authors were inspired by previous results such as:

- Stochastic depth improves the training of deep residual networks by dropping 
layers randomly during training. This showed that not all layers may be needed 
and that there is a great amount of redundancy in deep (residual) networks. 
- ResNet architecture was performing very well.
- An orthogonal approach to making networks deeper was to increase the network 
width.

#### DenseNet

The main idea behind DenseNet is to increase the efficiency of the net through 
feature reuse. Concatenating feature-maps learned by different layers increases 
variation in the input of subsequent layers and improves efficiency.

Traditional convolutional feed-forward networks connect the output of the Lth 
layer as input to the (L + 1) th layer. As a result, the transition is:

<a href="" target="_blank">
<img src="https://latex.codecogs.com/gif.latex?x_L%20%3DH_L%20%28x_L%20-%201%29" />
</a>

ResNets also adds directly the input:

<a href="" target="_blank">
<img src="https://latex.codecogs.com/gif.latex?x_L%20%3D%20H_L%20%28x_L%20-%201%29%20+%20x_L%20-%201" />
</a>

DenseNets introduce direct connections from any layer to all subsequent layers. 
Each layer receives the feature maps of all preceding layers as input:

<a href="" target="_blank">
<img src="https://latex.codecogs.com/gif.latex?x_L%20%3D%20H_L%20%28x_0%2C%20x_1%2C%20...%2C%20x_%7BL-1%7D%29" />
</a> 

In the particular case of DenseNets, they define the function <img src="https://latex.codecogs.com/gif.latex?H_L" /> as the 
combination of batch normalization (BN), followed by a rectified linear unit 
(ReLU) and a 3 × 3 convolution (Conv). The previous explanation only applies 
to images with the same size.

On the other hand, the Net also needs to downsample the sets of images: for 
this goal they add “transition layers” that do convolution and pooling.

Some other important concepts implemented in the DenseNets are the Growth rate 
(how many feature maps are produced by each function <img src="https://latex.codecogs.com/gif.latex?H_L" />), Bottleneck layers 
(1×1 convolutions introduced before each 3×3 convolution to reduce the number 
of input feature-maps, and thus improve computational efficiency), Compression 
(reduction of the number of feature-maps at transition layers)

## Experiments

In the paper, the authors evaluate the performance of DenseNets on different 
datasets (the two CIFAR datasets (CIFAR-10 and CIFAR-100), the Street View 
House Numbers (SVHN) dataset and the ILSVRC 2012 classification dataset 
(ImageNet)), and compare its results against other networks (mainly ResNets).

When tuning DenseNet for the different datasets, different combinations of 
hyperparameters were chosen in each case. The details can be seen in section 
4.2 of the original paper.

## Results

The most noticeable results are the following: 

- The authors found a particular hyperparameter combination by which DenseNet 
outperformed the existing state-of-the-art consistently on all the CIFAR 
datasets.
- On C10 and C100, both results were close to 30% lower than FractalNet with 
drop-path regularization. 
- On SVHN, with dropout, the DenseNet also surpassed the best result achieved 
by wide ResNet at the time.
- Without compression or bottleneck layers, DenseNets performed better as the 
number of layers and the number of feature maps increased. They attributed this 
to the corresponding growth in model capacity.
- DenseNets utilize parameters more efficiently than alternative architectures 
(in particular, ResNets).
- DenseNets are less sensitive to overfitting.
- On the ImageNet dataset, DenseNets perform on par with the state-of-the-art 
ResNets, whilst requiring significantly fewer parameters and computation to 
achieve comparable performance.
- DenseNet-BC was consistently the most parameter efficient variant of DenseNet

## Discussion 

The most relevant insights that the authors discuss in the paper can be 
summarized as follows:

By performing some experiments, the authors found that features extracted by 
very early layers were directly used by deep layers throughout the same dense 
block.
They discovered the existence of information flow from the first to the last 
layers of the DenseNet through few indirections.
They verified that the transition layers outputs many redundant features

## Conclusions

The main change proposed by DenseNets is the existence of direct connections 
between any two layers with the same feature-map size.

After performing several experiments, the authors found that DenseNets showed 
improvements in accuracy with growing number of parameters, without any signs
of performance degradation or overfitting.





