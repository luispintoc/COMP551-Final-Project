'''
Mini-Project4 - Group 95 - COMP 551 - Winter 2019
John Flores
Luis Pinto
John McGowan
'''

print("Mini-Project4 - Group 95 - COMP 551 - Winter 2019")

import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time, copy, os
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

data_dir = "./data"
model_name = "squeezenet"
num_classes = 10            #Classes for CIFAR10
batch_size = 128
epochs = 150
learning_rate = 0.001

print('Squeeze pretrained model - train only last conv layer for CIFAR10:')
print('batch size = ', batch_size)
print('Epochs = ', epochs)
print('Learning rate = ', learning_rate)

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True
print('Use pretrained weights: ', feature_extract)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(num_classes, feature_extract, use_pretrained=True):
    model_ft = models.squeezenet1_1(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))

    model_ft.num_classes = num_classes
    input_size = 224
    return model_ft, input_size

model_ft, input_size = initialize_model(num_classes, feature_extract, use_pretrained=True)


#GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 0:
  print(torch.cuda.device_count(), "GPUs available")
  model_ft = nn.DataParallel(model_ft)

model_ft.to(device)


#LOAD DATA
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )
trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


#OPTIMIZATION
params_to_update = model_ft.parameters()
total_params = sum(p.numel() for p in model_ft.parameters())
print("Total number of parameters = ", total_params)

print("")
print(" ****** SqueezeNet Architecture for CIFAR10 ******")
print(model_ft)

print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params_to_update,lr=learning_rate)

train_losses, test_losses = [] ,[]
for epoch in range(epochs):
    running_loss = 0
    for images,labels in train_loader:
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()
        train = Variable(images.view(-1,3,224,224))
        labels = Variable(labels)
        
        optimizer.zero_grad()
        
        output = model_ft(train)
        # print(output.size())
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        test_loss = 0
        accuracy = 0


        with torch.no_grad(): #Turning off gradients to speed up
            model_ft.eval()
            for images,labels in test_loader:
                if use_cuda:
                    images, labels = images.cuda(), labels.cuda()
                test = Variable(images.view(-1,3,224,224))
                labels = Variable(labels)
                
                log_ps = model_ft(test)
                test_loss += criterion(log_ps,labels)
            
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim = 1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        model_ft.train()        
        train_losses.append(running_loss/len(train_loader))
        test_losses.append(test_loss/len(test_loader))

        print("Epoch: {}/{}.. ".format(epoch+1, epochs),
            "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
            "Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
            "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))

plt.figure(1)
plt.plot(train_losses, label='Training loss')
plt.figure(2)
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()