
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

writer = SummaryWriter()


# ensure the randomness
def reset_randomness(random_seed):
    torch.manual_seed(random_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)


# load the initial backlog and channel information
CQI_0 = np.load('../data/CQI_0_init.npy')
s_0 = np.load('../data/s_0_init.npy')


# load the true value of remaining bits of different methods
Y_0_true = np.load('../data/Y_0_true_sep.npy')
Y_1_false = np.load('../data/Y_1_false_sep.npy')


# spectral efficiency
se = np.zeros((CQI_0.shape[0],CQI_0.shape[1]))

def cqi_to_se(min_cqi):
    # As per Table 7.2.3-1 in TS 36.213 Rel-11
    if min_cqi == 0:
        se_value = 0.1523
    if min_cqi == 1:
        se_value = 0.2344
    if min_cqi == 2:
        se_value = 0.3770
    if min_cqi == 3:
        se_value = 0.6016
    if min_cqi == 4:
        se_value = 0.8770
    if min_cqi == 5:
        se_value = 1.1758
    if min_cqi == 6:
        se_value = 1.4766
    if min_cqi == 7:
        se_value = 1.9141
    if min_cqi == 8:
        se_value = 2.4063
    if min_cqi == 9:
        se_value = 2.7305
    if min_cqi == 10:
        se_value = 3.3223
    if min_cqi == 11:
        se_value = 3.9023
    if min_cqi == 12:
        se_value = 4.5234
    if min_cqi == 13:
        se_value = 5.1152
    if min_cqi == 14:
        se_value = 5.5547
    if min_cqi == 15:
        se_value = 9.6
    
    return se_value


#change CQI to se
for i in range(len(CQI_0)):
    for j in range(len(CQI_0[i])):
        se[i][j] = cqi_to_se(CQI_0[i][j])
# calculate the transmission rate
tx_data_bits = np.floor(se * 5 / 10 * 1E3)
CQI_0 = tx_data_bits


#change format from array to tensor
CQI_0 = CQI_0.astype(np.float32)
CQI_0 = torch.from_numpy(CQI_0)


s_0 = s_0.astype(np.float32)
s_0 = torch.from_numpy(s_0)


Y_0_true = Y_0_true.astype(np.float32)
Y_0_true = torch.from_numpy(Y_0_true).unsqueeze(1)

Y_1_false = Y_1_false.astype(np.float32)
Y_1_false = torch.from_numpy(Y_1_false).unsqueeze(1)

X_0 = torch.stack((s_0, CQI_0), dim=2)


# define the dataset
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



# MLP with attention mechanism to ensure permutation equivariance
class AttentionMLP(nn.Module):
    def __init__(self, input_dim, q_dim, v_dim, hidden_dim, output_dim, input_dim1, q_dim1, v_dim1, hidden_dim1, output_dim1):
        super(AttentionMLP, self).__init__()
        self.input_dim = input_dim  # 2
        self.q_dim = q_dim  # 10
        self.v_dim = v_dim # 10
        self.hidden_dim = hidden_dim # hidden layer of the first shared MLP
        self.output_dim = output_dim # the input size of the second attention mechanism
        
        self.input_dim1 = input_dim1  # equal to output_dim
        self.q_dim1 = q_dim1  # 10
        self.v_dim1 = v_dim1 # 10
        self.hidden_dim1 = hidden_dim1 # hidden layer of the second shared MLP
        self.output_dim1 = output_dim1 # quantile number
        
       
        
        # define Q, K, V
        self.q = nn.Parameter(torch.empty(input_dim, q_dim))  # (2, 10)
        self.k = nn.Parameter(torch.empty(input_dim, q_dim))  # (2, 10)
        self.v = nn.Parameter(torch.empty(input_dim, v_dim))  # (2, 10)

        # Initialized using Kaiming normal distribution
        init.kaiming_normal_(self.q, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.k, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.v, mode='fan_out', nonlinearity='relu')
        
        
        # define Q, K, V
        self.q1 = nn.Parameter(torch.empty(input_dim1, q_dim1))  # (10, 10)
        self.k1 = nn.Parameter(torch.empty(input_dim1, q_dim1))  # (10, 10)
        self.v1 = nn.Parameter(torch.empty(input_dim1, v_dim1))  # (10, 10)
        
        # Initialized using Kaiming normal distribution
        init.kaiming_normal_(self.q1, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.k1, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.v1, mode='fan_out', nonlinearity='relu')
    

        # first Shared MLP layer
        self.fc1 = nn.Linear(v_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 输出变为10维
        
        # second Shared MLP layer
        self.fc11 = nn.Linear(v_dim1, hidden_dim1)
        self.fc22 = nn.Linear(hidden_dim1, output_dim1)  # 输出变为2维
        

    def attention(self, x):
        # # x shape: (batch_size, seq_len(number of users), input_dim) = (32, 8, 2)

        # Calculating attention weights
        q = torch.matmul(x, self.q)  # (32, 8, 10)
        k = torch.matmul(x, self.k)  # (32, 8, 10)
        v = torch.matmul(x, self.v)  # (32, 8, 10)

        
        attn_weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / (self.q_dim ** 0.5), dim=-1)  # (32, 8, 8)
        attn_output = torch.matmul(attn_weights, v)  # (32, 8, 10)
        return attn_output
    
    
    def attention1(self, x):
        # # x shape: (batch_size, seq_len (number of users), input_dim) = (32, 8, 10)

        # Calculating attention weights
        q1 = torch.matmul(x, self.q1)  # (32, 8, 10)
        k1 = torch.matmul(x, self.k1)  # (32, 8, 10)
        v1 = torch.matmul(x, self.v1)  # (32, 8, 10)

        
        attn_weights = F.softmax(torch.matmul(q1, k1.transpose(-2, -1)) / (self.q_dim1 ** 0.5), dim=-1)  # (32, 8, 8)
        attn_output = torch.matmul(attn_weights, v1)  # (32, 8, 10)
        return attn_output

    def forward(self, x):
        
        # attention
        attn_output = self.attention(x)  # (32, 8, 10)

        # Reshape to process all users in batches
        reshaped_output = attn_output.view(-1, self.v_dim)  # (32*8, v_dim)

        # MLP
        output = F.elu(self.fc1(reshaped_output))
        output = self.fc2(output)  # (32*8, 10)

        # Reshape back to original shape
        output = output.view(x.size(0), x.size(1), -1)  # (32, 8, 10)
        
        
        attn_output1 = self.attention1(output)  # (32, 8, 10)

    
        reshaped_output1 = attn_output1.view(-1, self.v_dim1)  # (32*8, 10)

        
        output1 = F.elu(self.fc11(reshaped_output1))
        output1 = self.fc22(output1)  # (32*8, 2)

        
        output1 = output1.view(x.size(0), x.size(1), -1)  # (32, 8, 2)

        return output1



class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles
        
        
    def forward(self, preds, target, mode='train'):
        losses = []
        for i, q in enumerate(self.quantiles):
            
            errors = target.squeeze()-preds[:,:,i]
            losses.append(torch.mean(torch.max((q - 1) * errors, q * errors)))
            

        upper_size = torch.max((preds[:,:,1] -  preds[:,:,0]), dim=1)[0]
        upper_size = torch.mean(torch.abs(upper_size))
             
        loss = torch.sum(torch.stack(losses))
        if mode == 'train':
            return loss
        elif mode == 'test':
            return upper_size
        else:
            raise NotImplementedError


# training parameters
learning_rate = 0.001
num_epochs = 300
batch_size = 64

# set the level of quantile
alpha = 0.2
num_user= 8
quantiles = [alpha/2, 1-alpha/2]
num_quantiles = num_user * len(quantiles)



# training
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
        
        # evaluate
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
                        lower_bound = outputs[i,:,0].detach().cpu().numpy()
                        upper_bound = outputs[i,:,1].detach().cpu().numpy()
                        if np.all(lower_bound <= targets.squeeze(1)[i,:].detach().cpu().numpy()) and np.all(targets.squeeze(1)[i,:].detach().cpu().numpy() <= upper_bound):
                            in_interval_count += 1
            test_loss /= len(test_loader)
            coverage = in_interval_count/((len(test_loader)-1)*batch_size+outputs.size(0))
            
            
            # writer.add_scalars('Loss', {'Train': train_loss, 'Test': test_loss}, epoch+1)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test coverage: {coverage:.4f}")
    
    # save model
    torch.save(model.state_dict(), 'pre_train_model.pth')
    print("Model saved.")



def overall_train():

    reset_randomness(random_seed=9)

    # Split the dataset into training and testing sets
    X_train = []
    y_train = []

    ex.temp=1
    
    X_0_train, _, y_0_train, _ = train_test_split(X_0, Y_0_true, test_size=0.4, random_state=0)
    
    X_train = X_0_train
    y_train = y_0_train
  
    X_train_1 = torch.clone(X_train)
    
    for i in range(len(y_train)):
        X_train[i,:,:] = X_train[i,:,:]/torch.max(X_train_1[i,:,0])
        y_train[i,0,:] = y_train[i,0,:]/torch.max(X_train_1[i,:,0])
        
       
        

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=0)
    
    train_dataset = SchedulingDataset(X_train, y_train)
    test_dataset = SchedulingDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)


   
    model = AttentionMLP(input_dim=2, q_dim=10, v_dim=10, hidden_dim=10, output_dim=10, input_dim1=10, q_dim1=10, v_dim1=10, hidden_dim1=10, output_dim1=2)
    
    
    criterion = QuantileLoss(quantiles)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    
    scheduler = StepLR(optimizer, step_size=200, gamma=0.8)


   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
   

    train(num_epochs, model, train_loader, test_loader, criterion, optimizer, scheduler, device)   


# overall_train()
  
          