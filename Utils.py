# -*- coding: utf-8 -*-
"""
Created on Sat Feb 8 22:10:18 2020
---------------------------------------------------------------------
-- Author: Vigneashwara Pandiyan
---------------------------------------------------------------------
Utils file for visualization/ Plots
"""

#%%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

#%%

def plot_confusion_matrix(y_true, y_pred,classes,plotname):
            
    # Build confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Normalise
    cmn = cm.astype('float')  / cm.sum(axis=1)[:, np.newaxis]
    cmn=cmn*100
    
    fig, ax = plt.subplots(figsize=(12,9))
    sns.set(font_scale=3) 
    b=sns.heatmap(cmn, annot=True, fmt='.1f', xticklabels=classes, yticklabels=classes,cmap="coolwarm",linewidths=0.1,annot_kws={"size": 25},cbar_kws={'label': 'Classification Accuracy %'})
    for b in ax.texts: b.set_text(b.get_text() + " %")
    plt.ylabel('Actual',fontsize=25)
    plt.xlabel('Predicted',fontsize=25)
    plt.margins(0.2)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90, va="center", fontsize= 20)
    ax.set_xticklabels(ax.get_xticklabels(), va="center",fontsize= 20)
    # plt.setp(ax.get_yticklabels(), rotation='vertical')
    plotname=str(plotname)
    plt.savefig(plotname,bbox_inches='tight',dpi=100)
    plt.show()
    plt.clf()

#%%

# def plots(iteration,Loss_value,Total_Epoch,Accuracy,Learning_rate,Training_loss_mean,Training_loss_std):
def plots(iteration,Loss_value,Total_Epoch,Accuracy,Learning_rate,model_name):    
    
    Accuracyfile = str(model_name)+'_Accuracy'+'.npy'
    Lossfile = str(model_name)+'_Loss_value'+'.npy'

    np.save(Accuracyfile,Accuracy,allow_pickle=True)
    np.save(Lossfile,Loss_value, allow_pickle=True)
    
    
    fig, ax = plt.subplots()
    plt.plot(Loss_value,'r',linewidth =2.0)
    # ax.fill_between(Loss_value, Training_loss_mean - Training_loss_std, Training_loss_mean + Training_loss_std, alpha=0.9)
    plt.title('Iteration vs Loss_Value')
    plt.xlabel('Iteration')
    plt.ylabel('Loss_Value')
    plot_1=  str(model_name)+'_Loss_value_'+ '.png'
    plt.savefig(plot_1, dpi=600,bbox_inches='tight')
    plt.show()
    plt.clf()
    
    plt.figure(2)
    plt.plot(Total_Epoch,Accuracy,'g',linewidth =2.0)
    plt.title('Total_Epoch vs Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plot_2=  str(model_name)+'_Accuracy_'+'.png'
    plt.savefig(plot_2, dpi=600,bbox_inches='tight')
    plt.show()
    
    plt.figure(3)
    plt.plot(Total_Epoch,Learning_rate,'b',linewidth =2.0)
    plt.title('Total_Epoch vs Learning_Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning_Rate')
    plot_3=  str(model_name)+'_Learning_rate_'+ '.png'
    plt.savefig(plot_3, dpi=600,bbox_inches='tight')
    plt.show()


#%%
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params