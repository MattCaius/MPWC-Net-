#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 11:26:40 2020

@author: ali
"""
import torch
import torch.nn as nn
import torch.nn.functional as tf
import sys
import matplotlib.pyplot as plt
import copy
import numpy as np
import onnx

from scipy.ndimage import gaussian_filter
sys.path.insert(1, './irr-master') # path of irr-master
sys.path.insert(1, './irr-master/models/') # path of models folder dowloaded from irr-pwcnet git-hub
model_path='./MPWCNet_irr-20201118-234043/checkpoint_best.ckpt' # the path of model
from scipy import signal
import sys
print(sys.executable)

import scipy.io as sio
import matplotlib.pyplot as plt
import os
import torch.nn as nn
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
torch.cuda.set_device(0)
print(torch.cuda.get_device_name(0))
from models import *
from scipy import signal

from scipy.ndimage import gaussian_filter

def Strain_Calc(ax_dis):
    ss=np.shape(ax_dis)
    sigma1=(ss[0]/9)/7
    strain = gaussian_filter(ax_dis, sigma=[sigma1,0.5],truncate=3)
    strain2=signal.convolve2d(strain, np.array([[1],[0],[-1]]))
    #strain2=strain2[75:-75,15:-15]
    return strain2
#%%

        

checkpoint=torch.load(model_path)
pwcnet=pwcnet_irr.MPWCNet(None).cuda()
Adict={}
for i in checkpoint['state_dict']:
    Adict[i[7:]]=checkpoint['state_dict'][i]
pwcnet.load_state_dict(Adict)

A=sio.loadmat('Phantom_Data_Demo.mat')
in_dic={}
in_dic['input1']=torch.from_numpy(A['Im1']).cuda()
in_dic['input2']=torch.from_numpy(A['Im2']).cuda()

pwcnet.eval()

with torch.no_grad():
    print("Running")
    tmp=pwcnet(in_dic)['flow']

from pynvml import *
nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(h)
print(f'total    : {info.total // 1024 ** 2}')
print(f'free     : {info.free // 1024 ** 2}')
print(f'used     : {info.used // 1024 ** 2}')

flo = tmp[0,1,:,:].cpu().data.numpy()
flo2 = tmp[0,0,:,:].cpu().data.numpy()

plt.imshow(flo2[:,:],aspect='auto')
plt.title('Lateral Displacement')
plt.colorbar()
st=Strain_Calc(flo2)
plt.figure()
plt.imshow(st[150:-150,5:-5],aspect='auto',cmap='hot')
plt.colorbar()
plt.title('Strain')
plt.show()

plt.imshow(flo[:,:],aspect='auto')
plt.title('Axial Displacement')
plt.colorbar()
st=Strain_Calc(flo)
plt.figure()
plt.imshow(st[150:-150,5:-5],aspect='auto',cmap='hot')
plt.colorbar()
plt.title('Strain')
plt.show()
