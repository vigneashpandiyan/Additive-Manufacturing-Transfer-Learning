# -*- coding: utf-8 -*-
"""
Created on Sat Feb 8 22:10:18 2020
---------------------------------------------------------------------
-- Author: Vigneashwara Pandiyan
---------------------------------------------------------------------
Utils file for visualization/ Plots
"""

#%%

import torchvision.transforms as transforms 
import torchvision 
import torch
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import StepLR
from mlxtend.plotting import plot_confusion_matrix
import seaborn as sns
from torchvision import datasets
from Heatmap import heatmap , annotate_heatmap
#torch.cuda.empty_cache()
from torch import optim, cuda
import os
from PIL import Image
import pandas as pd
import torchvision.models as models
from torch import nn
from collections import OrderedDict
# Whether to train on a gpu
train_on_gpu = cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')
from Utils import *
#%%

classes = ('Balling',  'LoF', 'Nopores','Keyhole')

PATH = './VGG16-Pytorch.pth'
Trained_model = torch.load(PATH)
print(Trained_model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# setting the root directories and categories of the images
# Data--> https://polybox.ethz.ch/index.php/s/7tAitrlpVuUAxWJ 
#%%

#%%

datadir = 'Bronze_dataset/'
traindir = datadir + 'Train/'
testdir = datadir + 'Test/'



summary(Trained_model, (3 ,32 ,32))   

#%%         


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        print('Learning rate =')
        print(param_group['lr'])
        return param_group['lr']

    
    
transform = transforms.Compose([transforms.Resize((512,512)),
                                transforms.ToTensor()])
#transform = transforms.Compose([transforms.ToTensor()])

trainload = datasets.ImageFolder(root=traindir, transform=transform)
trainset = torch.utils.data.DataLoader(trainload, batch_size=10,
                                          shuffle=True, num_workers=0)

testload = datasets.ImageFolder(root=testdir, transform=transform)
testset = torch.utils.data.DataLoader(testload, batch_size=10,
                                          shuffle=True, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)


net= Trained_model
net.to(device)

y_pred = []
y_true = []
correctHits=0
total=0
for batches in testset:
    data,output = batches
    data,output = data.to(device),output.to(device)
    prediction = net(data)
    # _,prediction = torch.max(prediction.data,1)  #returns max as well as its index
    prediction = torch.argmax(prediction, dim=1)
    
    total += output.size(0)
    correctHits += (prediction==output).sum().item()
    
    prediction=prediction.data.cpu().numpy()
    output=output.data.cpu().numpy()
    y_true.extend(output) # Save Truth 
    y_pred.extend(prediction)
    
print('Accuracy = '+str((correctHits/total)*100))

    #Trained_model = torch.load(PATH)
#%%
    


count_parameters(net)

plotname= 'VGG16'+'_Cross_Bronze'+'_confusion_matrix'+'.png'
plot_confusion_matrix(y_true, y_pred,classes,plotname)  
