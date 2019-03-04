# VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE -SCALE IMAGE RECOGNITION


## Summary

The VGG neural network was the winner of the ImageNet Challenge 2014. As the 
authors state in the paper, “a number of attempts have been made to improve 
the original architecture of Krizhevsky et al. (2012) in order to achieve 
better accuracy”. 

The main contribution of VGG was to increase the depth of the original ConvNet 
architecture. To this end, the authors “fixed other parameters of the 
architecture, and steadily increased the depth of the network by adding more 
convolutional layers, which is feasible due to the use of very small (3 × 3) 
convolution filters in all layers”. As a result, they obtained significantly 
better results than other ConvNet architectures, that were also applicable to 
other image recognition datasets.

## Architecture of the Net

Overall, all the ConvNet layer configurations they designed were based in the 
same principles, inspired by Ciresan et al. (2011); Krizhevsky et al. (2012). 
The most important variation they tried was using 3 × 3 receptive fields 
throughout the whole net, which were convolved with the input at every pixel 
(instead of using large receptive fields in the first convolutional layers).

They observed some improvements when using a stack of three 3 × 3 
convolutional layers instead of a single 7 × 7 layer: they managed to 
incorporate three non-linear rectification layers instead of a single one, 
while also decreasing the number of parameters. 

Finally, they found that the incorporation of 1 × 1 convolutional layers was a 
good way to increase the non-linearity of the decision function without 
affecting the receptive fields of the convolutional layers. Also, they added an 
additional non-linearity by implementing the rectification function.

## Training

Basically, the training of the net was performed in a similar way as Krizhevsky 
et al. (2012), except from applying scale variations on the images. 

The literal description of the training described in the paper is the 
following: “Namely, the training is carried out by optimising the multinomial 
logistic regression objective using mini-batch gradient descent (based on 
back-propagation (LeCun et al., 1989)) with momentum. The batch size was set to 
256, momentum to 0.9. The training was regularised by weight decay and dropout 
regularisation for the first two fully-connected layers (dropout ratio set to 
0.5). The learning rate was initially set to 10 −2 , and then decreased by a 
factor of 10 when the validation set accuracy stopped improving. In total, the 
learning rate was decreased 3 times, and the learning was stopped after 370K 
iterations (74 epochs). “

They observed that VGG required less epochs than Krizhevsky et al., 2012 in 
spite of having more parameters and larger depth. They think this might be due 
to an implicit regularization or implementing the pre-initialisation of certain 
layers.

They began training a shallow net with random initialisation. Then, they 
trained deeper architectures initialising the first four convolutional layers 
and the last three fully-connected layers with the layers of the previous net. 
They did not decrease the learning rate for the pre-initialised layers, 
allowing them to change during learning

Finally, they also tried two different approaches regarding the image sizes: 
first, they tried different fixed images sizes. Secondly, they tried a 
multi-scale training, where each training image was individually rescaled by 
randomly sampling S from a certain range.

## Testing

Testing was performed following a specific technique that exploits image 
reescaling: first, the image is resized to a pre-defined smallest image side. 
Then, the network is applied densely over the rescaled test image. The 
resulting net is then applied to the whole (uncropped) image. The result is a 
class score map with the number of channels equal to the number of classes, and 
a variable spatial resolution, dependent on the input image size. Finally, to 
obtain a fixed-size vector of class scores for the image, the class score map 
is spatially averaged (sum-pooled).

They also increased the test set by applying data augmentation (horizontal 
flipping). The soft-max class posteriors of the original and flipped images 
were averaged to obtain the final scores for the image.

## Data parallelism

They worked with Multi-GPU training in order to exploit data parallelism: they 
split each batch of training images into several GPU batches, processed in 
parallel on each GPU. The different gradients obtained in each GPU were 
averaged to obtain the gradient of the full batch. Gradient computation was 
synchronous across the GPUs, so the result was exactly the same as when 
training on a single GPU.

## Experiments

The most important conclusions they achieved while performing experiments were 
the following:

- While adding additional non-linearity helps, it is also important to capture 
spatial context by using convolutional filters with non-trivial receptive 
fields.
- Training set augmentation by scale jittering was helpful for capturing 
multi-scale image statistics.
- Scale jittering at test time leaded to better performance
- Using multiple crops performed slightly better than dense evaluation, and the 
two approaches were indeed complementary, as their combination outperformed 
each of them separately. 
- ConvNet fusion: they combined the outputs of several models by averaging 
their soft-max class posteriors. This improved the performance due to 
complementarity of the models.
- Comparison with state of the art: VGG significantly outperformed the previous 
generation of models. In terms of the single-net performance, they achieved the 
best result, outperforming a single GoogLeNet by 0.9%. They did not depart from 
the classical ConvNet architecture of LeCun et al. (1989), but improved it by 
substantially increasing the depth.

## Conclusions

The most important conclusion of this paper, is that they confirm the 
importance of depth in visual representations.

They demonstrated that the representation depth is beneficial for the 
classification accuracy, and that state-of-the-art performance on the ImageNet 
challenge dataset could be achieved using a conventional ConvNet architecture 
(LeCun et al., 1989; Krizhevsky et al., 2012) with substantially increased 
depth. 


