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

#PREPARATIONS (Trainsets, Initial u)
SAMPLE_NUM_EACH_WORKER = 1000
DATASET_NAME = 'SVHN'
LOSS_NAME = 'Elastic_Net'
wopt_str = 'wopt.pth'
[params, trainsets, wopt] = Initialization(SAMPLE_NUM_EACH_WORKER, DATASET_NAME, LOSS_NAME, wopt_str)

#REGULARIZER g
def g(z):
    return 0.5*((z*z).sum())

#AUQADMM SETUP
auqadmm = AUQADMM(params, g , 0.0, 1.0, trainsets, LOSS_NAME)
a = 0.1; b = 1.0 #Initial Restriction interval

#OPTIMIZATION
result = auqadmm.fit(maxiter=250, X_EPOCHS=30, abs_tol=1e-4, rel_tol=1e-5, rank=10, a=a, b=b, wopt=wopt, Nesterov=False, K=1)
