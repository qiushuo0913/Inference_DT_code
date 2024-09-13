#%%
from sklearn.model_selection import train_test_split

import numpy as np
import torch
import torch.nn as nn
import random
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

bw_mhz = 5
Nf = 10 # Keep consistent with n_prb in config_environment.json
Tmax = 20
# Adjust the shape of the sigmoid function in app selection probability 
# (the larger the T, the more independent the algorithms are)
temp = 1
num_user = 8


def cqi_to_upper_spectral_efficiency(min_cqi):
    # As per Table 7.2.3-1 in TS 36.213 Rel-11
    if min_cqi == 0:
        spectral_efficiency = 0.1523
    if min_cqi == 1:
        spectral_efficiency = 0.2344
    if min_cqi == 2:
        spectral_efficiency = 0.3770
    if min_cqi == 3:
        spectral_efficiency = 0.6016
    if min_cqi == 4:
        spectral_efficiency = 0.8770
    if min_cqi == 5:
        spectral_efficiency = 1.1758
    if min_cqi == 6:
        spectral_efficiency = 1.4766
    if min_cqi == 7:
        spectral_efficiency = 1.9141
    if min_cqi == 8:
        spectral_efficiency = 2.4063
    if min_cqi == 9:
        spectral_efficiency = 2.7305
    if min_cqi == 10:
        spectral_efficiency = 3.3223
    if min_cqi == 11:
        spectral_efficiency = 3.9023
    if min_cqi == 12:
        spectral_efficiency = 4.5234
    if min_cqi == 13:
        spectral_efficiency = 5.1152
    if min_cqi == 14:
        spectral_efficiency = 5.5547
    if min_cqi == 15:
        spectral_efficiency = 9.6
    
    return spectral_efficiency

# compute the probability of running algorithm 1 - propotional fair channel aware
def calculate_ex(min_CQI, s):
    
    # measure with time
    tx_data_bits = min_CQI
    
    input = (s- tx_data_bits*Tmax/num_user)/(tx_data_bits)
    
    
    input = torch.max(input)
    
    # out is the probability of running PFCA
    out = 1/(1+np.exp(input/temp))
    out = 1-out
    out = out.item()
    
    return out
    
    
    
        
        
        
        
