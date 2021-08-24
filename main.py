#main

!wget www.di.ens.fr/~lelarge/MNIST.tar.gz
!tar -zxvf MNIST.tar.gz

# import utils
# import AUQADMM
import math
import numpy as np
import pywt
import torch
import torchvision
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.datasets import MNIST
from six.moves import urllib  
import torch.nn as nn
import torch.nn.functional as F
import copy
import time

#PREPARATIONS
#Initial x's, M=28*28 here
M = 28*28
##Upload TrainLoaders.pth, then load it
TrainLoaders = torch.load('TrainLoaders.pth')

##Load target trainloader

#if MNIST:
trainloader_min = TrainLoaders['MNIST'][0]
#if CIFAR10:
trainloader_cifar10 = TrainLoaders['CIFAR'][0]

trainloader_svhn = torch.load('trainloader_svhn.pth')


#Use the function above to generate trainsets
#if MNIST:
trainsets = Generate_and_Classify_Trainsets(1000, 'MNIST', trainloader_min)

trainset1 = trainsets[0]
trainset2 = trainsets[1]
trainset3 = trainsets[2]
trainset4 = trainsets[3]
trainset5 = trainsets[4]
trainset6 = trainsets[5]
trainset7 = trainsets[6]
trainset8 = trainsets[7]
trainset9 = trainsets[8]
trainset10 = trainsets[9]

trainsets = [trainset1, trainset2, trainset3, trainset4, trainset5, trainset6, trainset7, trainset8, trainset9, trainset10]


# 0 and 1 are the coefficients of regularizers, rho1=0 and rho2=1. rho1 is the coeff of |x| and rho2 is the coeff of 0.5*||x||^2. 
#REGULARIZER g
def g(z):
    return 0.5*((z*z).sum())

x1 = torch.randn(M,1,requires_grad=True)
x2 = torch.randn(M,1,requires_grad=True)
x3 = torch.randn(M,1,requires_grad=True)
x4 = torch.randn(M,1,requires_grad=True)
x5 = torch.randn(M,1,requires_grad=True)
x6 = torch.randn(M,1,requires_grad=True)
x7 = torch.randn(M,1,requires_grad=True)
x8 = torch.randn(M,1,requires_grad=True)
x9 = torch.randn(M,1,requires_grad=True)
x10 = torch.randn(M,1,requires_grad=True)

initial_x = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]

#AUQADMM SETUP
auqadmm = AUQADMM(initial_x, g , 0.0, 1.0, trainsets)

#Initial Restriction interval
a = 0.1; b = 1.0

#OPTIMIZATION
result = auqadmm.fit(maxiter=250, function=nn.MSELoss(), X_EPOCHS=30, abs_tol=1e-4, rel_tol=1e-5, rank=10, a=a, b=b, wopt=wopt, Nesterov=False, K=1)
