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
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import seaborn as sns
from scipy.stats import norm
import joypy
import pandas as pd
from matplotlib import cm
from scipy import signal
import pywt
import matplotlib.patches as mpatches
import os
from PIL import Image

import torchvision.transforms as transforms 
import torchvision 
import torch
from torchsummary import summary
from torchvision import datasets
#%%
datadir = 'Bronze dataset/'
traindir = datadir + 'Train/'
testdir = datadir + 'Test/'

#%%
data_transform = transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    ])
trainset = datasets.ImageFolder(root=traindir,
                                            transform=data_transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=4, shuffle=True,
                                              num_workers=0)

testset = datasets.ImageFolder(root=testdir,
                                            transform=data_transform)
testloader = torch.utils.data.DataLoader(testset,
                                              batch_size=4, shuffle=False,
                                              num_workers=0)

#%%

classes = ('Balling', 'Keyhole', 'LoF', 'Nopores')

#%%

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    img=np.transpose(npimg, (1, 2, 0))
    return img

    
   
#%%
    
# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
fig, ax = plt.subplots(figsize=(12,5))
img=imshow(torchvision.utils.make_grid(images))
img=np.flip(img, 0)
plt.imshow(img)
plt.savefig('Class Images.png',bbox_inches='tight',dpi=800)
plt.show()
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

#%%

categories = []
img_categories = []
n_train = []
n_valid = []
n_test = []
hs = []
ws = []

# Iterate through each category
for d in os.listdir(traindir):
    categories.append(d)

    # Number of each image
    train_imgs = os.listdir(traindir + d)
    #valid_imgs = os.listdir(validdir + d)
    test_imgs = os.listdir(testdir + d)
    n_train.append(len(train_imgs))
    # n_valid.append(len(valid_imgs))
    n_test.append(len(test_imgs))

    # Find stats for train images
    for i in train_imgs:
        img_categories.append(d)
        img = Image.open(traindir + d + '/' + i)
        img_array = np.array(img)
        # Shape
        hs.append(img_array.shape[0])
        ws.append(img_array.shape[1])

# Dataframe of categories
cat_df = pd.DataFrame({'category': categories,
                        'n_train': n_train,
                        'n_test': n_test}).\
    sort_values('category')

# Dataframe of training images
image_df = pd.DataFrame({
    'category': img_categories,
    'height': hs,
    'width': ws
})

cat_df.sort_values('n_train', ascending=False, inplace=True)
cat_df.head()
cat_df.tail()


fig, ax = plt.subplots(figsize=(12,5))
cat_df.set_index('category')['n_train'].plot.bar(
    color=plt.cm.Paired(np.arange(len(cat_df))))
plt.xticks(rotation=25,fontsize= 20)
plt.ylabel('Total count',fontsize= 20)
plt.title('Training Images by Category',fontsize= 20)
plt.savefig('Training Images.png',bbox_inches='tight',dpi=800)

img_dsc = image_df.groupby('category').describe()
img_dsc.head()

