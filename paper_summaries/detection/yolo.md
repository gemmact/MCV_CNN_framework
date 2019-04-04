# You Only Look Once: Unified, Real-Time Object Detection


## Main idea of the paper

The major difference in YOLO network is that it frames object detection as a 
regression problem to spatially separated bounding boxes and associated class 
probabilities. The architecture consists of a single neural network that 
predicts bounding boxes and class probabilities directly from full images in 
one evaluation.

An important consequence is that, since the whole detection pipeline is a 
single network, it can be optimized end-to-end directly on detection 
performance. As a result, this unified architecture turns out to be extremely 
fast (real-time image processing).

Compared to state-of-the-art detection systems, YOLO makes more localization 
errors but is less likely to predict false positives on background. Also, YOLO 
is able to generalize better than other systems.


## Introduction

The name of the network comes from the idea that “you only look once” at an 
image in order to predict the class and localization of different objects. The 
authors reframe object detection as a single regression problem, straight from 
image pixels to bounding box coordinates and class probabilities.

A single convolutional network simultaneously predicts multiple bounding boxes 
and class probabilities for those boxes. YOLO trains on full images and 
directly optimizes detection performance. YOLO presents a unified architecture 
that results in several benefits:

- It is extremely fast.
- It reasons globally about the image when making predictions (YOLO sees the 
entire image during training and test time so it implicitly encodes contextual 
information about classes as well as their appearance).
- It learns generalizable representations of objects. 

Nevertheless, YOLO still lags behind state-of-the-art detection systems in 
accuracy (it struggles to precisely localize some objects, especially small 
ones).


## Unified Detection

The architecture unifies the separate components of object detection into a 
single neural network, which uses features from the entire image to predict 
each bounding box. Also, all bounding boxes across all classes for an image are 
predicted simultaneously. The unified design enables end-to-end training and 
real time speeds while maintaining high average precision.

The system works by implementing the following steps:

- The input image is divided into an S × S grid.
- If the center of an object falls into a grid cell, that grid cell is 
responsible for detecting that object.
- Each grid cell predicts B bounding boxes and confidence scores for those 
boxes.
- The confidence prediction represents the IOU between the predicted box and 
any ground truth box.


## Network Design

The model is implemented by a convolutional neural network. The initial 
convolutional layers extract features from the image, while the fully connected 
layers predict the output probabilities and coordinates.

Regular YOLO has 24 convolutional layers followed by 2 fully connected layers. 
They also use 1 × 1 reduction layers followed by 3 × 3 convolutional layers. 
Fast YOLO uses a neural network with fewer convolutional layers (9 instead of 
24) and fewer filters in those layers.


## Training

For the work explained in the paper, they pretrained the convolutional layers 
on the ImageNet 1000-class dataset. For pretraining they use the first 20 
convolutional layers followed by an average-pooling layer and a fully connected 
layer.

In order to perform detection, they add four convolutional layers and two fully 
connected layers with randomly initialized weights. The final layer predicts 
both class probabilities and bounding box coordinates.

They use a linear activation function for the final layer, and all other layers 
use a leaky rectified linear activation. They optimize for sum-squared error in 
the output. However, it does not perfectly align with their goal of maximizing 
average precision since It weights localization error equally with 
classification error. To remedy this, they increase the loss from bounding box 
coordinate predictions and decrease the loss from confidence predictions for 
boxes that don’t contain objects.

The model training leads to a specialization between the bounding box 
predictors. Each predictor gets better at predicting certain sizes, aspect 
ratios, or classes of object, improving overall recall.


### Training configuration

- 135 epochs
- Training and validation data sets from P ASCAL VOC 2007 and 2012.
- Batch size = 64
- Momentum = 0.9
- Decay = 0.0005.
- Learning rate schedule: for the first epochs they slowly raise the learning 
rate from 10-3 to 10-2 . They continue training with 10 −2 for 75 epochs, 
then 10 −3 for 30 epochs, and finally 10 −4 for 30 epochs.
- They use dropout and extensive data augmentation.
- Dropout layer with rate = .5 after the first connected layer
- Data augmentation = random scaling and translations of up to 20% of the 
original image size. They also randomly adjust the exposure and saturation of 
the image by up to a factor of 1.5 in the HSV color space.


## Inference

Just like in training, predicting detections for a test image only requires one 
network evaluation. YOLO is extremely fast at test time since it only requires 
a single network evaluation. Also, the grid design enforces spatial diversity 
in the bounding box predictions.


## Limitations

- The model struggles with small objects that appear in groups, such as flocks 
of birds.
- It struggles to generalize to objects in new or unusual aspect ratios or 
configurations.
- The model uses relatively coarse features for predicting bounding boxes.
- The loss function treats errors the same in small bounding boxes versus large 
bounding boxes. A small error in a large box is generally benign but a small 
error in a small box has a much greater effect on IOU. The main source of error 
is incorrect localizations.


## Comparison to other models

- _Deformable parts models_: YOLO unified architecture leads to a faster, more 
accurate model
than DPM.
- _R-CNN_: YOLO shares some similarities with R-CNN. Each grid cell proposes 
potential bounding boxes and scores those boxes using convolutional features. 
However, YOLO puts spatial constraints on the grid cell proposals which helps 
mitigate multiple detections of the same object. YOLO also proposes fewer 
bounding boxes, and combines these individual components into a single, jointly 
optimized model.
- _Other Fast Detectors_: While they offer speed and accuracy improvements over 
R-CNN, both still fall short of real-time performance. YOLO is a real-time 
general purpose detector that learns to detect a variety of objects 
simultaneously.
- _Deep MultiBox_:​ MultiBox can also perform single object detection by 
replacing the confidence prediction with a single class prediction. However, it 
cannot perform general object detection and is still just a piece in a larger 
detection pipeline, requiring further image patch classification. Both YOLO and 
MultiBox use a convolutional network to predict bounding boxes in an image, but 
YOLO is a complete detection system.
- _Over-Feat_: It optimizes for localization, not detection performance. The 
localizer only sees local information when making a prediction. OverFeat cannot 
reason about global context and thus requires significant post-processing to 
produce coherent detections.
- _MultiGrasp_: It only needs to predict a single graspable region for an image 
containing one object. It doesn’t have to estimate the size, location, or 
boundaries of the object or predict it’s class, only find a region suitable 
for grasping. YOLO predicts both bounding boxes and class probabilities for 
multiple objects of multiple classes in an image.


## Experiments

### Comparison to Other Real-Time Systems

- Fast YOLO is the fastest object detection method on PASCAL. It is more than 
twice as accurate as prior work on real-time detection.
- They also trained YOLO using VGG-16: this model is more accurate but also 
significantly slower than YOLO.
- Different adaptations of R-CNN improve their time performance, but none of 
them achieves real-time image processing.


### Combining Fast R-CNN and YOLO

YOLO struggles to localize objects correctly. Localization errors account for 
more of YOLO’s errors than all other sources combined. Fast R-CNN makes much 
fewer localization errors but far more background errors

By using YOLO to eliminate background detections from Fast R-CNN they got a 
significant boost in performance. It is precisely because YOLO makes different 
kinds of mistakes at test time that it is so effective at boosting Fast 
R-CNN’s performance. Nevertheless, this combination doesn’t benefit from 
the speed of YOLO since they run each model separately and then combine the 
results.


### Real time detection

While YOLO processes images individually, when attached to a webcam it 
functions like a tracking system, detecting objects as they move around and 
change in appearance.

## Conclusion

This paper introduces YOLO, a unified model for object detection. The model is 
simple and can be trained directly on full images. Unlike classifier-based 
approaches, the entire model is trained jointly.

Fast YOLO is the fastest general-purpose object detector in the literature and 
YOLO pushes the state-of-the-art in real-time object detection. YOLO also 
generalizes well to new domains making it ideal for applications that rely on 
fast, robust object detection.
