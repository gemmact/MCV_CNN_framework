# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks


## Introduction

Nowadays, state-of-the-art object detection networks hypothesize object 
locations based on region proposal algorithms. Some of them have reduced the 
running time of these detection networks, encountering the region proposal 
computation as a bottleneck. In this work, the authors introduce a Region 
Proposal Network (RPN) that shares full-image convolutional features with the 
detection network, which enables nearly cost-free region proposals.An RPN is a 
fully convolutional network that simultaneously predicts object  bounds and 
objectness scores at each position. This network is trained end-to-end to 
generate high-quality region proposals, which are used by Fast R-CNN for 
detection.

They also merge RPN and Fast R-CNN into a single network by sharing their 
convolutional features. For this, they use the recently popular terminology of 
neural networks with “attention” mechanisms, where the RPN component tells 
the unified network where to look To put this in perspective, for the very deep 
VGG-16 model, their detection  system has a frame rate of 5fps (including all 
steps) on a GPU.


## Region proposals

Although region-based CNNs were computationally expensive as originally 
developed, their cost has been drastically reduced thanks to sharing 
convolutions across proposals. For example, Fast R-CNN achieves near real-time 
rates using very deep networks, when ignoring the time spent on region 
proposals.

Region proposal methods typically rely on inexpensive features and economical 
inference schemes, such as:

- _Selective Search_, one of the most popular methods, which greedily merges 
superpixels based on engineered low-level features. Yet when compared to 
efficient detection networks, _Selective Search_ is an order of magnitude 
slower, at 2 seconds per image in a CPU implementation.
- _EdgeBoxes_ currently provides the best tradeoff between proposal quality and 
speed, at 0.2 seconds per image.

The RPN is thus a kind of fully convolutional network (FCN) and can be trained 
end-to-end specifically for the task for generating detection proposals.

In contrast to prevalent methods that use pyramids of images or pyramids of 
filters, the authors introduce novel “anchor” boxes that serve as 
references at multiple scales and aspect ratios.

To unify RPNs with Fast R-CNN object detection networks, they propose a 
training scheme that alternates between fine-tuning for the region proposal 
task and then fine-tuning for object detection, while keeping the proposals 
fixed. This scheme converges quickly and produces a unified network with 
convolutional features that are shared between both tasks.


## Some related work

### Object Proposals
Widely used object proposal methods include those based on grouping 
super-pixels (e.g., Selective Search) and those based on sliding windows (e.g., 
EdgeBoxes)

### Deep Networks for Object Detection
The R-CNN method trains CNNs end-to-end to classify the proposal regions into 
object categories or background. R-CNN mainly plays as a classifier, and it 
does not predict object bounds (except for refining by bounding box 
regression). Its accuracy, however, depends on the performance of the region 
proposal module


## Faster R-CNN

The Faster R-CNN is composed of two modules. The first module is a deep fully 
convolutional network that proposes regions, and the second module is the Fast 
R-CNN detector that uses the proposed regions single, unified network for 
object detection.

### Region Proposal Networks
A _Region Proposal Network_ (RPN) is a FCNN that takes an image (of any size) 
as input and outputs a set of rectangular object proposals, each with an 
objectness score.

To generate region proposals, they slide a small network over the convolutional 
feature map output by the last shared convolutional layer. This small network 
takes as input an n × n spatial window of the input convolutional feature map. 
Each sliding window is mapped to a lower-dimensional feature. Finally, this 
feature is fed into two sibling fullyconnected layers: a _box-regression layer_ 
(_reg_) and a _box-classification layer_ (_cls_).

This mini-network operates in a sliding-window fashion and the fully-connected 
layers are shared across all spatial locations. This architecture is naturally 
implemented with an n×n convolutional layer followed by two sibling 1 × 1 
convolutional layers (for _reg_ and _cls_, respectively)

### Anchors
At each sliding-window location, they simultaneously predict multiple region 
proposals, where the number of maximum possible proposals for each location is 
denoted as k. So the _reg_ layer has 4k outputs encoding the coordinates of k 
boxes, and the cls layer outputs 2k scores that estimate probability of object 
or not object for each proposal The k proposals are parameterized relative to k 
reference boxes, which they call _anchors_. An anchor is then centered at the 
sliding window in question, and is associated with a scale and aspect ratio


#### Translation-Invariant Anchors
An important property of this approach is that it is translation invariant, 
both in terms of the anchors and the functions that compute proposals relative 
to the anchors. If one translates an object in an image, the proposal should 
translate and the same function should be able to predict the proposal in 
either location. This translation-invariant property is guaranteed by this 
method. As a comparison, the MultiBox method uses k-means to generate 800 
anchors, which are not translation invariant. So MultiBox does not guarantee 
that the same proposal is generated if an object is translated. The 
translation-invariant property also reduces the model size, which seems to have 
less risk of overfitting on small datasets, like PASCAL VOC.

#### Multi-Scale Anchors as Regression References
To get scale-invariance, one way is to use on image/feature pyramids. The 
images are resized at multiple scales, and feature maps (HOG or DNN) are 
computed for each scale. This way is often useful but is time-consuming. The 
second way is to use sliding windows of multiple scales (and/or aspect ratios) 
on the feature maps.

As a comparison, the anchor-based method that the authors propose is built on a 
pyramid of anchors, which is more cost-efficient. This method classifies and 
regresses bounding boxes with reference to anchor boxes of multiple scales and 
aspect ratios.


## Training RPNs

The RPN can be trained end-to-end by backpropagation and stochastic gradient 
descent (SGD). It is possible to optimize for the loss functions of all 
anchors, but this will bias towards negative samples as they are dominate. 
Instead, they randomly sample 256 anchors in an image to compute the loss 
function of a mini-batch, where the sampled positive and negative anchors have 
a ratio of up to 1:1. If there are fewer than 128 positive samples in an image, 
they pad the mini-batch with negative ones

## Sharing Features for RPN and Fast R-CNN

Both RPN and Fast R-CNN, trained independently, will modify their convolutional 
layers in different ways. Because of this, the authors discuss three ways for 
training networks with features shared:

1. _Alternating training_: In this solution, the first train RPN, and use the 
proposals to train Fast R-CNN. The network tuned by Fast R-CNN is then used to 
initialize RPN, and this process is iterated. This is the solution that is used 
in all experiments in this paper.
2. _Approximate joint training_: In this solution, the RPN and Fast R-CNN 
networks are merged into one network during training.
3. _Non-approximate joint training_

Some RPN proposals highly overlap with each other. To reduce redundancy, they 
adopt non-maximum suppression (NMS) on the proposal regions based on their cls 
scores.

## Experiments

They test their results on PASCAL VOC and MS COCO datasets. They outperforms 
the previous methods and regarding proposals, somewhat surprisingly, the RPN 
still leads to a competitive result (55.1%) when using the top-ranked 100 
proposals at test-time, indicating that the top ranked RPN proposals are 
accurate. On the other extreme, using the top-ranked 6000 RPN proposals 
(without NMS) has a comparable mAP (55.2%), suggesting NMS does not harm the 
detection mAP and may reduce false alarms

## Conclusions

They have presented RPNs for efficient and accurate region proposal generation. 
By sharing convolutional features with the down-stream detection network, the 
region proposal step is nearly cost-free. This method enables a unified, 
deep-learning-based object detection system to run at near real-time frame 
rates. The learned RPN also improves region proposal quality and thus the 
overall object detection accuracy.
