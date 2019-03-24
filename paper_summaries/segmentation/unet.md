# U-Net: Convolutional Networks for Biomedical Image Segmentation

## Summary

The authors of the paper implement an FCN to segment medical images. In this 
field, it is not common to have thousands of training images to train the 
models, so they use data augmentation to simulate the different variations that 
could be found in this type of images (mainly tissue deformation).

Prior to this work, state-of-the-art networks consisted of a sliding-window 
setup to predict the class label of each pixel by providing a local region 
(patch) around that pixel. In this way the network could locate different 
regions and, in addition, the training data in terms of patches was much larger 
than the number of training images.

However, this strategy has two mein drawbacks: First, it is quite slow because 
the network must be run separately for each patch, and there is a lot of 
redundancy due to overlapping patches. Secondly, there is a trade-off between 
localization accuracy and the use of context. Larger patches require more 
max-pooling layers that reduce the localization accuracy, while small patches 
allow the network to see only little context.

The main idea of the authors is to supplement a usual contracting network by 
successive layers, where pooling operators are replaced by upsampling 
operators. In order to localize, high resolution features from the contracting 
path are combined with the upsampled output.

As we said before, in biomedicine there is very little training data available, 
so they perform an intense data augmentation step by applying elastic 
deformations to the available training images. This allows the network to learn 
invariance to such deformations, without the need to see these transformations 
in the annotated image corpus, whichs is particularly important in biomedical 
segmentation.

Another challenge that they face in many cell segmentation tasks is the 
separation of touching objects of the same class. To this end, they propose the 
use of a weighted loss, where the separating background labels between touching 
cells obtain a large weight in the loss function.

## Network architecture

The network architecture consists of a contracting path followed by an 
expansive path. The contracting path follows the typical architecture of a 
convolutional network. It consists of the repeated application of two 3x3 
convolutions (unpadded convolutions), each followed by a rectified linear unit 
(ReLU) and a 2x2 max pooling operation with stride 2 for downsampling. Every 
step in the expansive path consists of an upsampling of the feature map 
followed by a 2x2 convolution (“up-convolution”) that halves the number of 
feature channels, a concatenation with the correspondingly cropped feature map 
from the contracting path, and two 3x3 convolutions, each followed by a ReLU. 
At the final layer a 1x1 convolution is used to map each 64-component feature 
vector to the desired number of classes.

## Training

For the training stage the try several things:
- Favor large input tiles over a large batch size and hence reduce the batch to 
a single image, in order to minimize the overhead and make maximum use of the 
GPU memory
- High momentum (0.99) such that a large number of the previously seen training 
samples determine the update in the current optimization step
- The energy function is computed by a pixel-wise soft-max over the final 
feature map combined with the cross entropy loss function
- Weight map pre-computed for each ground truth segmentation to compensate the 
different frequency of pixels from a certain class in the training data set, 
and to force the network to learn the small separation borders that we 
introduce between touching cells
- Initial weights from a Gaussian distribution with a standard deviation of p 
2/N, where N denotes the number of incoming nodes of one neuron

Due to the unpadded convolutions, the output image is smaller than the input by 
a constant border width

## Data augmentation

They primarily need shift and rotation invariance as well as robustness to 
deformations and gray value variations. For this, they generate smooth 
deformations using random displacement vectors on a coarse 3 by 3 grid. The 
displacements were sampled from a Gaussian distribution with 10 pixels standard 
deviation. Per-pixel displacements were then computed using bicubic 
interpolation.

## Conclusions

The U-Net architecture achieves very good performance on very different 
biomedical segmentation applications. Thanks to data augmentation with elastic 
deformations, it only needs very few annotated images and has a very reasonable 
training time of only 10 hours on a NVidia Titan GPU (6 GB). 
