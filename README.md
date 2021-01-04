# Mini-project4 - COMP551 - Applied Machine Learning

## Overview
Project aiming to reproduce the results of the following published paper and extend its results: [SQUEEZENET: ALEXNET-LEVEL ACCURACY WITH
50X FEWER PARAMETERS AND <0.5MB MODEL SIZE](https://arxiv.org/pdf/1602.07360.pdf).


## Prerequisites
The packages used are torch, torchvision, numpy, matplotlib and pandas.
It will download the dataset (CIFAR10) into a folder called 'data'.

## Authors
John Flores

Luis Pinto

John McGowan

## Scripts

**alexnet_cifar10.py**
AlexNet pretrained model (on ImageNet)
Outputs: hyperparameters, total number of parameters, architecture of the model and the parameters it needs to learn.
It needs about 13 epochs to reach its minimum test loss (0.511) / maximum accuracy (82.7%).

**squeezenet_cifar10.py**
SqueezeNet pretrained model (on ImageNet)
Outputs: hyperparameters, total number of parameters, architecture of the model and the parameters it needs to learn.
It needs about 15 epochs to reach its minimum test loss (0.49) / maximum accuracy (83.1%).

**Sq-1.py**
Sq-1 model: Uses SqueezeNet's pretrained parameters on Imagenet for the first layers. This variation adds 2 linear layers and 2 dropouts after the original SqueezeNet CNN.
Outputs: hyperparameters, total number of parameters, architecture of the model and the parameters it needs to learn.
It needs about 34 epochs to reach its minimum test loss (0.403) / maximum accuracy (86.7%).

**Fire-Next.py**
Fire-Next model:  Uses SqueezeNet's pretrained parameters on Imagenet for the first layers. This variation changes the last Fire module (SqueezeNet) to a SqNxt block (SqueezeNext).
Outputs: hyperparameters, total number of parameters, architecture of the model and the parameters it needs to learn.
It needs about 9 epochs to reach its minimum test loss (0.404) / maximum accuracy (86.5%).

