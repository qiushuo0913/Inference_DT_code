
#%%
import sys
sys.path.insert(0, '../..')
from sklearn.model_selection import train_test_split
import torch.nn.init as init
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import ex_construct as ex

from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter()


# ensure randomness
def reset_randomness(random_seed):
    torch.manual_seed(random_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)


# load data
Gamma = np.load('./data/SNR.npy')
Ns_list = np.load('./data/Num_scatter.npy')
re_time_BPSK_Al = np.load('./data/re_time_BPSK_Al.npy')
re_time_BPSK_MU = np.load('./data/re_time_BPSK_MU.npy')
re_time_QPSK_Al = np.load('./data/re_time_QPSK_Al.npy')
re_time_QPSK_MU = np.load('./data/re_time_QPSK_MU.npy')




Gamma = Gamma.astype(np.float32)
Gamma = torch.from_numpy(Gamma)

Ns_list = Ns_list.astype(np.float32)
Ns_list = torch.from_numpy(Ns_list)


re_time_BPSK_Al = re_time_BPSK_Al.astype(np.float32)
re_time_BPSK_Al = torch.from_numpy(re_time_BPSK_Al)

re_time_BPSK_MU = re_time_BPSK_MU.astype(np.float32)
re_time_BPSK_MU = torch.from_numpy(re_time_BPSK_MU)

re_time_QPSK_Al = re_time_QPSK_Al.astype(np.float32)
re_time_QPSK_Al = torch.from_numpy(re_time_QPSK_Al)

re_time_QPSK_MU = re_time_QPSK_MU.astype(np.float32)
re_time_QPSK_MU = torch.from_numpy(re_time_QPSK_MU)



X = torch.stack((Gamma, Ns_list), dim=1)


# Defining Datasets
class SchedulingDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x, y


# MLP
class MyMLPQuantile(nn.Module):
    def __init__(self, num_quantiles):
        super(MyMLPQuantile, self).__init__()
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 5)
        self.fc4 = nn.Linear(5, num_quantiles)

    def forward(self, x):
        x = self.fc1(x)
        x = self.elu(x)
        
        x = self.fc2(x)
        x = self.elu(x)

        x = self.fc3(x)
        x = self.elu(x)
        
        x = self.fc4(x)
        return x.view(-1,2)






class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles
        
        
    def forward(self, preds, target, mode='train'):
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target.view(-1,1) - preds[:, i].view(-1,1)

            losses.append(torch.mean(torch.max((q - 1) * errors, q * errors)))
             
        
        upper_size = (preds[:,1] -  preds[:,0])
        upper_size = torch.mean(torch.abs(upper_size))
             
        loss = torch.sum(torch.stack(losses))
        if mode == 'train':
            return loss
        elif mode == 'test':
            return upper_size
        else:
            raise NotImplementedError


# system parameters
learning_rate = 0.005
num_epochs = 342
batch_size = 64


alpha = 0.2
quantiles = [alpha/2, 1-alpha/2]
num_quantiles = len(quantiles)



# train
def train(epochs, model, train_loader, test_loader, criterion, optimizer, scheduler, device):
    
    for epoch in range(epochs):
        model.train()
        
        train_loss = 0
        
        for features, targets in train_loader:
            
            features = features.to(device)
            targets = targets.to(device)

           
            
            
            outputs = model(features)
            loss = criterion(outputs, targets)

           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
       
        if (epoch + 1) % 2 == 0:
            model.eval()
           
            test_loss = 0
            with torch.no_grad():
                in_interval_count = 0
                for features, targets in test_loader:
                    features = features.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(features)
                    test_loss += criterion(outputs, targets, mode='test').item()
                    for i in range(outputs.size(0)):
                        lower_bound = outputs[i,0].detach().cpu().numpy()
                        upper_bound = outputs[i,1].detach().cpu().numpy()
                        if (lower_bound <= targets[i].detach().cpu().numpy()) and (targets[i].detach().cpu().numpy() <= upper_bound):
                            in_interval_count += 1
            test_loss /= len(test_loader)
            coverage = in_interval_count/((len(test_loader)-1)*batch_size+outputs.size(0))
            
            
            # writer.add_scalars('Loss', {'Train': train_loss, 'Test': test_loss}, epoch+1)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test coverage: {coverage:.4f}")
    
    # save model
    torch.save(model.state_dict(), './model/pre_model_re_time_BPSK_MU.pth')
    print("Model saved.")



def overall_train():

    reset_randomness(random_seed=9)

    
    X_train = []
    y_train = []

   
    
    X_0_train, _, y_0_train, _ = train_test_split(X, re_time_BPSK_MU, test_size=0.4, random_state=0)
    
    
    X_train = X_0_train
    y_train = y_0_train
    


    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=0)
    
    
    train_dataset = SchedulingDataset(X_train, y_train)
    test_dataset = SchedulingDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)


    
    model = MyMLPQuantile(num_quantiles=num_quantiles)
    
    

    criterion = QuantileLoss(quantiles)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

   
    scheduler = StepLR(optimizer, step_size=200, gamma=0.8)


    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    

    train(num_epochs, model, train_loader, test_loader, criterion, optimizer, scheduler, device)   


# overall_train()
  
          