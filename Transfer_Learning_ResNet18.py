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
PATH = './ResNet18-Pytorch.pth'
Trained_model = torch.load(PATH)
# setting the root directories and categories of the images
# Data--> https://polybox.ethz.ch/index.php/s/7tAitrlpVuUAxWJ 
#%%
datadir = 'Bronze_dataset/'
traindir = datadir + 'Train/'
testdir = datadir + 'Test/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
Trained_model.to(device)
summary(Trained_model, (3 ,32 ,32))

#%%
classes = ('Balling',  'LoF', 'Nopores','Keyhole')
for name, child in Trained_model.named_children():
    for name2, params in child.named_parameters():
        print(name, name2)
        
#%%
        
ct = 0
for name, child in Trained_model.named_children():
    ct += 1
    if ct < 8:
        for name2, params in child.named_parameters():
            #print(name2, params)
            params.requires_grad = False

#%%
            
Trained_model.to(device)
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
trainset = torch.utils.data.DataLoader(trainload, batch_size=40,
                                          shuffle=True, num_workers=0)

testload = datasets.ImageFolder(root=testdir, transform=transform)
testset = torch.utils.data.DataLoader(testload, batch_size=40,
                                         shuffle=True, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)


net=Trained_model
    
    
net.to(device)

costFunc = torch.nn.CrossEntropyLoss()
optimizer =  torch.optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

scheduler = StepLR(optimizer, step_size = 25, gamma= 0.5 )

Loss_value =[]
Iteration_count=1
iteration=[]
Epoch_count=0
Total_Epoch =[]
Accuracy=[]
Learning_rate=[]

for epoch in range(2):
    learingrate_value = get_lr(optimizer)
    Learning_rate.append(learingrate_value)
    closs = 0
    scheduler.step()
    
    for i,batch in enumerate(trainset,0):
        data,output = batch
        data,output = data.to(device),output.to(device)
        prediction = net(data)
        loss = costFunc(prediction,output)
        
        # print("loss",loss)
        closs += loss
        # print("closs",closs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Iteration_count = Iteration_count + i
        iteration.append(Iteration_count)
        
        
        
        # closs = loss.item()
        #print every 1000th time
        if i%100 == 0:
            print('[%d  %d] loss: %.4f'% (epoch+1,i+1,loss))
            
        
        
    loss_train = closs / i
    
    print('Loss on epoch: [%d] is %.4f'% (epoch+1,loss_train))
    Loss_value.append(loss_train)
    
    
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
        
    
    Epoch_accuracy = (correctHits/total)*100
    Accuracy.append(Epoch_accuracy)
    print('Accuracy on epoch ',epoch+1,'= ',str((correctHits/total)*100))
    
    Epoch_count = epoch+1
    Total_Epoch.append (Epoch_count)



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

print('Finished Training')
PATH = './ResNet18_Transfer_Bronze-Pytorch.pth'
torch.save(net.state_dict(), PATH)
torch.save(net, PATH)
    #Trained_model = torch.load(PATH)
#%%


Loss_value= torch.stack(Loss_value)   
Loss_value=Loss_value.cpu().detach().numpy()     

plots(iteration,Loss_value,Total_Epoch,Accuracy,Learning_rate,'ResNet18_Transfer_Inconel')
count_parameters(net)

plotname= 'ResNet18'+'_Transfer_Bronze'+'_confusion_matrix'+'.png'
plot_confusion_matrix(y_true, y_pred,classes,plotname)  
