# Fully Convolutional Networks for Semantic Segmentation

## Main idea of the paper

They describe a new architecture for spatially dense prediction tasks based on 
fully convolutional networks. They adapt existing classification networks (such 
as AlexNet, VGG, and GoogLeNet) into fully convolutional networks and transfer 
their learned representations by fine-tuning to the segmentation task.


## Introduction

The idea of the paper is inspired in the fact that Convnets were not only  
improving for whole-image classification, but also making progress on local 
tasks with  structured output. Nevertheless, these nets were only able to 
generate coarse predictions. They see this gap and state that “the natural 
next step in the progression from coarse to fine inference is to make a 
prediction at every pixel”.

Basically, their model transfers the success in classification to dense 
prediction by reinterpreting classification nets as fully convolutional and 
fine-tuning from their learned representations. In order to do this they 
managed to extend a convnet to arbitrary-sized inputs.


## Related work

Fully convolutional networks: The idea of extending a convnet to 
arbitrary-sized inputs and training fully convolutional nets for detection 
already existed. The previous work was simpler, and the number of dimensions of 
the output was reduced (1 or 2 dimensions).

Several previous works had already applied convnets to dense prediction 
problems. These existent methods had common elements such as:

- They used small models, restricting capacity and receptive fields
- Patchwise training
- Post-processing by superpixel projection, random field regularization, 
filtering, or local classification
- Input shifting and output interlacing for dense output
- Multi-scale pyramid processing
- Saturating tanh nonlinearities
- Ensembles

An important breakthrough of this work is that they manage to perform this task 
without this additional processing (except for patchwise training).Unlike 
existing methods, they adapt and extend deep classification architectures, 
using image classification as supervised pre-training, and fine-tune fully 
convolutionally to learn simply and efficiently from whole image inputs and 
whole image ground thruths. They fuse features across layers to define a 
nonlinear local to-global representation that they tune end-to-end.


## Fully convolutional networks

Convnets are built on translation invariance. Their basic components 
(convolution, pooling, and activation functions) operate on local input 
regions, and depend only on relative spatial coordinates. In Convnets we find 
layers that perform matrix multiplications for convolution or average pooling, 
spatial max for max pooling, or an elementwise nonlinearity for an activation 
function.

While a general deep net computes a general nonlinear function, a net with only 
layers of this form computes a nonlinear filter. An FCN naturally operates on 
an input of any size, and produces an output of corresponding spatial 
dimensions.


## Adapting classifiers for dense prediction

In typical recognition nets, the fully connected layers have fixed dimensions 
and throw away spatial coordinates. A major breakthrough of this work is that 
the authors find out that:

**“However, these fully connected layers can also be viewed as convolutions 
with kernels that cover their entire input regions. Doing so casts them into 
fully convolutional networks that take input of any size and output 
classification maps”.**

As a consequence, they state that the spatial output maps of these 
convolutionalized models make them a natural choice for dense problems like 
semantic segmentation.

In order to perform this task, they implement the “Shift-and-stitch” 
technique: they find that dense predictions can be obtained from coarse outputs 
by stitching together output from shifted versions of the input.

Also, another possible way to connect coarse outputs to dense pixels 
isupsampling by interpolation. Upsampling with a factor f is performing a 
convolution with a fractional input stride of 1/f . So as long as f is 
integral, a natural way to upsample is therefore backwards convolution (or 
deconvolution). An interesting observation is that the deconvolution filter in 
such a layer need not be fixed, but can be learned. This approach results 
particularly interesting given that in-network upsampling turns to be fast and 
effective for learning dense prediction.

Finally, they explore patchwise training. They discover that sampling with this 
technique can correct class imbalance and mitigate the spatial correlation of 
dense patches.


## Segmentation Architecture

The main characteristics of the implemented architecture are the following:

- They cast ILSVRC classifiers into FCNs and augment them for dense prediction 
with in-network upsampling and a pixelwise loss.
- They train for segmentation by fine-tuning.
- They add skips between layers to fuse coarse, semantic and local, appearance 
information (this skip architecture is learned end-to-end to refine the 
semantics and spatial precision of the output).
- They train and validate on the PASCAL VOC 2011 segmentation challenge.
- They train with a per-pixel multinomial logistic loss and validate with the 
standard metric of mean pixel intersection over union, with the mean taken over 
all classes, including background.


## From classifier to dense FCN

They adapted existing classifiers to dense FCNs. With this goal, they begin by 
convolutionalizing AlexNet 3 , VGG and GoogLeNet. ​**They decapitate each net 
by discarding the final classifier layer, and convert all fully connected 
layers to convolutions. They append a 1 × 1 convolution with channel dimension 
21 to predict scores for each of the PASCAL classes (including background) at 
each of the coarse output locations, followed by a deconvolution layer to 
bilinearly upsample the coarse outputs to pixel-dense outputs.**

As a simple first approach, they discovered that fine-tuning from 
classification to segmentation already gave reasonable predictions for each net.

They built a more complex model by defining a new FCN for segmentation that 
combined layers of the feature hierarchy and refined the spatial precision of 
the output.

In existing segmentation nets, the 32 pixel stride at the final prediction 
layer limited the scale of detail in the upsampled output. The authors of this 
paper address this problem by adding skips that combine the final prediction 
layer with lower layers with finer strides. They find out that combining fine 
layers and coarse layers lets the model make local predictions that respect 
global structure.


## Model

The implementation of the model is described in the following paragraph of the 
paper:“We first divide the output stride in half by predicting from a 16 
pixel stride layer. We add a 1x1 convolution layer on top of pool4 to produce 
additional class predictions. We fuse this output with the predictions computed 
on top of conv7 (convolutionalized fc7) at stride 32 by adding a 2x upsampling 
layer and summing 6 both predictions. We initialize the 2x upsampling to 
bilinear interpolation, but allow the parameters to be learned. Finally, the 
stride 16 predictions are upsampled back to the image. We call this net 
FCN-16s. FCN-16s is learned end-to-end, initialized with the parameters of the 
last, coarser net, which we now call FCN-32s”.

For optimization, they train the net by SGD with momentum. They fine-tune all 
layers by back-propagation through the whole net. The full image training 
batches each image into a regular grid of large, overlapping patches. In order 
to perform the dense prediction, the scores are upsampled to the input 
dimensions by deconvolution layers within the net.


## Results

They test their FCN on semantic segmentation and scene parsing, exploring 
PASCAL VOC, NYUDv2, and SIFT Flow. They obtain results that beats the state of 
the art algorithms.


## Conclusion

The major breakthrough of this work, is that the authors use the fact that 
fully convolutional networks are a rich class of models, of which modern 
classification convnets are a special case.

By recognizing this, they extend these classification nets to segmentation, and 
improve the architecture with multi-resolution layer combinations. As a result, 
they obtained a network that dramatically improves the state-of-the-art, while 
simultaneously simplifying and speeding up learning and inference.