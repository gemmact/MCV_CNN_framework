# CNN framework for PyTorch

## M5 Project: Scene Understanding for Autonomous Driving

### Abstract

#### Week 1
The summaries of the VGG and DenseNet papers can be found in the 
`paper_summaries` directory within this project.

#### Week 2
In this week we implement object recognition techniques for identify objects in 
images. For this, we use the provided framework built of top of PyTorch to 
implement and train different CNN architectures. We use known architectures, 
such as VGG, DenseNet or ResNet to get used to the framework, and finally we 
implement our own CNN architecture called CanNET.

#### Week 3
In this week we train and implement semantic segmentation networks for segment 
objects in different datasets. For this, we use the provided framework train a 
FCN and implement SegNet, a state-of-the-art network for semantic segmentation. 
We train both networks from scratch and from pretrained weights, over `camvid` 
and `kitti` datasets and boost their performance using different techniques.

#### Week 4
This week's main task was to implement an object detection network and evaluate 
performance in different datasets. Since the framework does not yet support this 
type of network, we used the repository of an already implemented YOLO network. 
You can find this repository in the `external' folder or in this 
[link](https://github.com/AlexeyAB/darknet). 

As every two weeks, we offer the summary of two articles on this topic that you 
can find inside the folder `paper_summaries`.

### Code

In the `config` folder we define some configuration files for the different 
architectures.

In the`job` folder we define the configuration files for the jobs to run the 
script with the architectures listed in `config` folder.

`paper_summaries` folder contain the summary of two papers for every type of
network that we used.

And in the `models` folder we update some files to add the configuration of our 
custom CNN, CaNet. 

### Setup

To run the file, you need to install the dependencies listed in the 
`requirements.txt` file:


```
$ pip install -r requirements.txt
```

Or you could create a virtual environment and install them on it:

```
$ mkvirtualenv -p python2.7 m5
(m5) $ pip install -r requirements.txt
```

To use the code, you need to create a new `yml` configuration file and run it 
with:

```
$ CUDA_VISIBLE_DEVICES=1 python main.py --exp_name <exp_name> --exp_folder 
<exp_folder> --config_file <config_file>
```

In case you are using SLURM as a job scheduler for a cluster, you need to 
configure a job file and run it using:

```
$ sbatch <job_file>
```


## Week 4

### Completeness of the tasks

The status of the tasks for this week are:

- [X] Train an existing object detection network 
- [X] Read and summarize two papers
- [X] Train the network for another dataset
- [X] Boost the performance of your network
- [X] Report + slides showing the achieved results 

### Presentation

In the following links you can find the 
[report](https://www.overleaf.com/read/qcfvbbgcrjfq) and 
[presentation](https://docs.google.com/presentation/d/1cgN1IeviTxtlPYSLyiDImsOqivyF8VM8kW_wCJrUytM/edit#slide=id.g558f7df010_0_256) 
of this week.


## Week 3

### Completeness of the tasks

The status of the tasks for this week are:

- [X] Run the provided code 
- [X] Read and summarize two papers
- [X] Implement a new network
- [X] Train the network on a different dataset
- [X] Boost the performance of your network
- [X] Report + slides showing the achieved results 

### Presentation

In the following links you can find the 
[report](https://www.overleaf.com/7622391142xnsbbbqtwhcd) and 
[presentation](https://docs.google.com/presentation/d/1fQPXDX5Zbv6USf3_KzVRJtGdBBPlMBIELjMuLztm_iE/edit?usp=sharing) 
of this week.


## Week 2

### Completeness of the tasks

The status of the tasks for this week are:

- [X] Run the provided code 
- [X] Train a network for other dataset
- [X] Implement a new Network
- [X] Using an existing PyTorch implementation.
- [X] Writing our own implementation.
- [X] Boost the performance of your networks 
- [X] Report + slides showing the achieved results 

### Presentation

In the following links you can find the 
[report](https://www.overleaf.com/read/vkbfjtmdyzhx) and 
[presentation](https://docs.google.com/presentation/d/1N-bBVhSZk_i_b0ar4wIW-FT0C_mkbmTev4YtfNO8FMQ/edit?usp=sharing) of this week.

### Weights of the models

In the following link you can find the 
[weights](https://drive.google.com/open?id=1DQszuEbTh61MMo1Z0tWf-z2ceZ1uiON6) 
for the CaNet architecture. 


## Team Members:

- Gemma Canet Tarrés (gemmacanettarres@gmail.com)
- Sara Cela Alfonso (saracelalfonso@gmail.com)
- Facundo Ferrin (facundo.ferrin@gmail.com)
- Agustina Pose (aguupose@gmail.com)
