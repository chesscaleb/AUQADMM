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

##PREPARATIONS (Trainsets, Initial U)
SAMPLE_NUM_EACH_WORKER = 1000
DATASET_NAME = 'MNIST'
LOSS_NAME = 'ElasticNet' #or 'Multinomial' or 'SmoothedSVM', 
                          #but if they are not the desired loss functions, 
                          #please directly input the loss function, e.g. nn.MSELoss()

#Generate Initial Values for U
[params, trainsets] = Initialization(SAMPLE_NUM_EACH_WORKER, DATASET_NAME, LOSS_NAME, n=None, num_workers=None)
#n: number of columns of u_j for worker j
#num_workers: total number of workers
#n and num_workers are required on when LOSS_NAME is the loss function other than
#'Multinomial', 'Elastic_Net' and 'Smoothed_SVM'.

##REGULARIZER g
def g(z):
    return torch.norm(z,p=1)+0.5*((z*z).sum()) #this g is particularly for elastic net regression (rho1=1, rho2=1)

##AUQADMM SETUP
auqadmm = AUQADMM(params, g , 0.0, 1.0, trainsets, LOSS_NAME)
a = 0.1; b = 1.0 #Initial Restriction interval

##OPTIMIZATION
#AUQADMM
result_auqadmm = auqadmm.fit(maxiter=250, X_EPOCHS=30, abs_tol=1e-4, rel_tol=1e-5, rank=10, a=a, b=b, K=1)

#Other ADMM Algs: CADMM or RBADMM or ACADMM
admm = ADMM(params, g , 0.0, 1.0, trainsets, LOSS_NAME)
print('%s   %s    %s          %s          %s'
          % ('Iter',    'f(z) + g(z)',    'p_res',     'd_res',     'time used'))
result_admm = admm.do(maxiter=250, X_EPOCHS=30, epsilon=0.2, start_tau=1.0, T_f=2, C_cg=1e10, method='CADMM')
