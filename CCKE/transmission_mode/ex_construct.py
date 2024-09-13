#%%
from sklearn.model_selection import train_test_split

import numpy as np
import math
import torch
import torch.nn as nn
import random
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR


temp = 1

# compute the probability of running different transmission mode
def calculate_ex(gamma, Ns):
    # input:
    # gamma - SNR
    # Ns - number of scatters
    
    # output:
    # probability of running BPSK alamouti, BPSK multiplexing, QPSK alamouti, and QPSK multiplexing
    
    # correlate factor
    k=0.5
    # modulation
    M_list = np.array([2, 4])
    # factor in the SER formula related with modulation
    b = 2*(1-1/np.sqrt(M_list))
    c = 3/(2*(M_list-1))
    
    # calculate the Multipplexing SER according to paper “Unifying Performance Analysis of Linear MIMO 
    # Receivers in Correlated Rayleigh Fading Environments”
    if Ns ==1:
        ps_mu = np.array([0.25,0.5])
    else:  
        ps_mu = b/(4*c*(1-k**2/(Ns.item()**2))**2*gamma.item()) # BPSK QPSK multiplexing
    
    
    # calculate the Alamouti SER according to paper “Symbol Error Probability for Space-Time Block Codes Over 
    # Spatially Correlated Rayleigh Fading Channels”
    t = k/Ns.item()
    v = gamma.item()*(np.sin(np.pi/M_list))**2
    ps_al = 1/(2*t)*((1+t)*(1-1/np.sqrt(1+1/((1+t)*v))) - (1-t)*(1-1/np.sqrt(1+1/((1-t)*v))))
    
    
    ps = np.hstack((ps_mu, ps_al))
    # inverse SER
    ps = 0.05/ps
    
    sum = np.exp(ps[0]/temp) + np.exp(ps[1]/temp) + np.exp(ps[2]/temp) + np.exp(ps[3]/temp)
    
    out_BPSK_MU = np.exp(ps[0]/temp)/sum
    out_QPSK_MU = np.exp(ps[1]/temp)/sum
    out_BPSK_Al = np.exp(ps[2]/temp)/sum
    out_QPSK_Al = np.exp(ps[3]/temp)/sum
    
   

    return out_BPSK_Al.item(), out_BPSK_MU.item(), out_QPSK_Al.item(), out_QPSK_MU.item()

    
    
    
        
        
        
        
