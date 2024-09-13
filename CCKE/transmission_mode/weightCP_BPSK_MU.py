
#%%
from sklearn.model_selection import train_test_split

import numpy as np
import torch


import regression_model as renew
import ex_construct as ex
import random



confidence_level = 0.8


# Extract the common elements of two lists for visualization
def common_elements_ordered(list1, list2):
    set2 = set(list2)
    return [x for x in list1 if x in set2]

def extract_elements(full_list, index_list):
    return [full_list[i] for i in index_list]

# ensure the randomness
def reset_randomness(random_seed):
    torch.manual_seed(random_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

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
re_time_BPSK_Al = re_time_BPSK_Al.view(-1,1)

re_time_BPSK_MU = re_time_BPSK_MU.astype(np.float32)
re_time_BPSK_MU = torch.from_numpy(re_time_BPSK_MU)
re_time_BPSK_MU = re_time_BPSK_MU.view(-1,1)

re_time_QPSK_Al = re_time_QPSK_Al.astype(np.float32)
re_time_QPSK_Al = torch.from_numpy(re_time_QPSK_Al)
re_time_QPSK_Al = re_time_QPSK_Al.view(-1,1)

re_time_QPSK_MU = re_time_QPSK_MU.astype(np.float32)
re_time_QPSK_MU = torch.from_numpy(re_time_QPSK_MU)
re_time_QPSK_MU = re_time_QPSK_MU.view(-1,1)



X = torch.stack((Gamma, Ns_list), dim=1)


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')



#1. test data QPSK multiplexing; waht if analyisi: BPSK multiplexing

def construct_interval_match(seed,T):
    # input:
    # seed - the random seed index
    # T - the parameter in app selection probability
    
    # output:
    # nonconformity_scores - NC scores
    # X_1_test - test data
    # y_0_test_false - label of test data
    # Out_cal_app_a - e(x) of calibration data on app a
    # Out_cal_app_aa - e(x) of calibration data on app a'
    # Out_test_app_a - e(x) of tets data on app a
    # Out_test_app_aa - e(x) of test data on app a'
    # pre_model - saved model
    # len(X_calibration) - size of calibration data set
    
    reset_randomness(random_seed=9)

    # Split the dataset into training and testing sets
    _, X_0_test_init, _, y_0_test_init = train_test_split(X, re_time_BPSK_MU, test_size=0.4, random_state=0)
    

    reset_randomness(random_seed=seed)
    X_0_calibration, X_0_test_init, y_0_calibration, y_0_test_init = train_test_split(X_0_test_init, y_0_test_init, test_size=0.6, random_state=seed)
    
    
    
    
    # Construct calibration dataset and test data based on app selection probability
    X_calibration = []
    y_calibration = []
    
    
    X_1_test = []
    y_0_test_false = []
    
    
    # Store the value of e(x) of calibration data and test data, respectively
    # a is the actual app
    # aa is the counterfactual app
    Out_cal_app_a = []
    Out_cal_app_aa = []
    
    Out_test_app_a = []
    Out_test_app_aa = []
    

    
    # Re-divide and allocate 0 (app a = QPSK multiplexing) and 1 (app aa = BPSK multiplexing) data sets 
    # as the calibration data set according to the score of e(x)
    for i in range(X_0_calibration.shape[0]):

        # calculate e(x)
        gamma = X_0_calibration[i,0]
        m = X_0_calibration[i,1]
        
        out_BPSK_Al, out_BPSK_MU, out_QPSK_Al, out_QPSK_MU = ex.calculate_ex(gamma, m)
        
        noise = np.random.random(1)
        
        
        if (noise <= out_BPSK_MU):
        
            Out_cal_app_aa.append(out_BPSK_MU)
            Out_cal_app_a.append(out_QPSK_MU)
            X_calibration.append(X_0_calibration[i])
            y_calibration.append(y_0_calibration[i])
        
        

    X_calibration = torch.stack(X_calibration)
    y_calibration = torch.stack(y_calibration)
    
    
   
    # the size of calibration dataset
    value_cal = 50
    value_cal_index  = np.random.choice(len(X_calibration), value_cal)
    
    
    X_calibration = X_calibration[value_cal_index]
    y_calibration = y_calibration[value_cal_index]
    
    
    
    Out_cal_app_aa = [Out_cal_app_aa[i] for i in value_cal_index]
    Out_cal_app_a = [Out_cal_app_a[i] for i in value_cal_index]
    
    
    
    # Re-divide and allocate 0 (app a = QPSK multiplexing) and 1 (app aa = BPSK multiplexing) data sets 
    # as the test data set according to the score of e(x)
    for i in range(X_0_test_init.shape[0]):
        
        # calculate e(x)
        gamma = X_0_test_init[i,0]
        m = X_0_test_init[i,1]
        
        out_BPSK_Al, out_BPSK_MU, out_QPSK_Al, out_QPSK_MU = ex.calculate_ex(gamma, m)
          
        noise = np.random.random(1)
        
        if (noise <= out_QPSK_MU):
        
            Out_test_app_a.append(out_QPSK_MU)
            Out_test_app_aa.append(out_BPSK_MU)
            X_1_test.append(X_0_test_init[i])
            y_0_test_false.append(y_0_test_init[i])
        
            
    
    
    X_1_test = torch.stack(X_1_test)
    y_0_test_false = torch.stack(y_0_test_false)
    
    
    # test size
    if T == 0.5:
        value = 20
    elif T == 1:
        value = 40
    else:
        value = 100
    

 
    X_1_test = X_1_test[0:value]
    y_0_test_false = y_0_test_false[0:value]
    
  
    
    Out_test_app_aa = Out_test_app_aa[0:value]
    Out_test_app_a = Out_test_app_a[0:value]
    
    
  
    pre_model = renew.MyMLPQuantile(num_quantiles=2) 
    

    pre_model.load_state_dict(torch.load('./model/pre_model_re_time_BPSK_MU.pth', map_location=torch.device(device)))
    pre_model = pre_model.to(device)
    pre_model.eval()  


    
    
    X_calibration = X_calibration.to(device)
    
    
    calibration_predictions = pre_model(X_calibration)
    
    nonconformity_scores = np.maximum((calibration_predictions.detach().cpu()[:,0].view(-1,1) - y_calibration), (y_calibration-calibration_predictions.detach().cpu()[:,1].view(-1,1)))
    
   
    return nonconformity_scores, X_1_test, y_0_test_false, Out_cal_app_a, Out_cal_app_aa, Out_test_app_a, Out_test_app_aa, pre_model, len(X_calibration)

# construct_interval_match(seed=0,T=1)

def construct_interval_mismatch(nonconformity_scores, X_1, Y_0_false,  Out_cal_app_a, Out_cal_app_aa, Out_test_app_a, Out_test_app_aa, pre_model, cal_size, epoch):
    
    # input:
    # nonconformity_scores - NC scores
    # X_1 - test data
    # Y_0_false - label of test data
    # Out_cal_app_a - e(x) of calibration data on app a
    # Out_cal_app_aa - e(x) of calibration data on app a'
    # Out_test_app_a - e(x) of tets data on app a
    # Out_test_app_aa - e(x) of test data on app a'
    # pre_model - saved model
    # cal_size - size of calibration data set
    # epoch - index of random independent experiment
    
    # output:
    # coverage - coverage performance of CKE
    # coverage_ratio_weight - coverage performance of CCKE
    # coverage_ratio_mismatch - coverage performance of NCCKE
    # Size - inefficiency performance of CKE
    # Size_weight - inefficiency performance of CCKE
    # Size_mismatch - inefficiency performance of NCCKE
    
    
    
    
    pre_model.eval()  
    
   
    X_1_test = X_1.to(device)
    test_predictions = pre_model(X_1_test)
    prediction_intervals = []
    
    # CKE
    Size = []
    # CCKE
    Size_weight = []
    # NCCKE
    Size_mismatch = []
    
    in_interval_count_test = 0
    I_naive = []
    
    for i in range(len(test_predictions)):
        
        if (test_predictions[i,0].detach().cpu().numpy() <= Y_0_false[i].numpy()) and (Y_0_false[i].numpy() <= test_predictions[i,1].detach().cpu().numpy()):
            in_interval_count_test += 1
        else:
            I_naive.append(i)
        
        # cap the upper quantile by 10
        if test_predictions[i,0] < 0:
            test_predictions[i,0] = torch.tensor(0).to(device=device)
        
        if test_predictions[i,1] > 10:
            test_predictions[i,0] = torch.tensor(10).to(device=device)
        
        
        size = test_predictions[i,1].detach().cpu().numpy()-test_predictions[i,0].detach().cpu().numpy()
        Size.append(abs(size)) 
    
    
        
    coverage = in_interval_count_test/len(test_predictions)
    
        
    
    # CCKE
    
    w_cal = (np.array(Out_cal_app_a))/np.array(Out_cal_app_aa)
    w_test = (np.array(Out_test_app_a))/np.array(Out_test_app_aa)
    
    
    
    sort_nonconformity_scores = np.sort(nonconformity_scores, axis=0)
    sorted_indices = np.array(np.argsort(nonconformity_scores.view(-1)))
    
    w_cal = w_cal[sorted_indices]
    
    in_interval_count = 0
    I_weight = []
    
    
    # Calculate quantile for each test data
    for i in range(len(test_predictions)):
        cdf_sum = 0
        p_cal = w_cal/(np.sum(w_cal)+w_test[i])
        p_test = w_test[i]/(np.sum(w_cal)+w_test[i])
        for j in range(len(p_cal)):
            if cdf_sum < confidence_level:
                cdf_sum = cdf_sum + p_cal[j]
            else:
                break
        if j == len(p_cal)-1:
            quantile = np.inf
        else:
            quantile = sort_nonconformity_scores[j][0]
        
        
        
        lower_bound = test_predictions[i,0].detach().cpu().numpy()-quantile
        upper_bound = test_predictions[i,1].detach().cpu().numpy()+quantile
        
       
        if (lower_bound <= Y_0_false[i].numpy()) and (Y_0_false[i].numpy() <= upper_bound):
            in_interval_count += 1
            
            I_weight.append(i)
        
        # cap the upper quantile by 10
        if upper_bound > 10:
            upper_bound = 10
        if lower_bound < 0:
            lower_bound = 0
        
        prediction_intervals.append((lower_bound, upper_bound))
        size_weight = upper_bound - lower_bound
        
        
        
        Size_weight.append(abs(size_weight))
        
        
        
    coverage_ratio_weight = in_interval_count/len(Y_0_false)
   
           
        
        
        
   
    # NCCKE   
    weight_nonconformity_scores = nonconformity_scores
    
    
    sort_nonconformity_scores = np.sort(weight_nonconformity_scores, axis=0)
    index = np.ceil(((cal_size+1)*confidence_level)).astype(int)-1
    quantile = sort_nonconformity_scores[index][0]
    
    in_interval_count = 0
    I = []
    
    
    for i in range(len(test_predictions)):
        
        
       
        lower_bound = test_predictions[i,0].detach().cpu().numpy()-quantile
        upper_bound = test_predictions[i,1].detach().cpu().numpy()+quantile
        
       
        if (lower_bound <= Y_0_false[i].numpy()) and (Y_0_false[i].numpy() <= upper_bound):
            in_interval_count += 1
        else:
            I.append(i)
        
        # cap the upper quantile by 10    
        if upper_bound > 10:
            upper_bound = 10
        if lower_bound < 0:
            lower_bound = 0   
            
        prediction_intervals.append((lower_bound, upper_bound))
        size_mismatch = upper_bound - lower_bound
        
        
        
        Size_mismatch.append(abs(size_mismatch))
        
    
            
            
        
    coverage_ratio_mismatch = in_interval_count/len(Y_0_false)
    
    
    return coverage, coverage_ratio_weight, coverage_ratio_mismatch, Size, Size_weight, Size_mismatch
    


def test_coverage(Value_array, Epoch):
    
    # input:
    # Value_array - array of parameters T
    # Epoch - The number of random independent experiments
    
    
    
    for i in range(len(Value_array)):
        value = Value_array[i]
        
        ex.temp = value
        
        # CKE
        Coverage = []
        # CCKE
        Coverage_weight = []
        # NCCKE
        Coverage_mismatch = []
        
        
        # CKE
        Final_size = []
        # CCKE
        Final_size_weight = []
        # NCCKE
        Final_size_mismatch = []
      
    
        for epoch in range (Epoch):
            nc_score, X_1_test, y_0_test_false, P_score_cal, P_score_test, P_score_cal_inverse, P_score_test_inverse, pre_model, cal_size = construct_interval_match(seed=epoch, T = value)
            coverage_ratio, coverage_ratio_weight, coverage_ratio_mismatch, Size, Size_weight, Size_mismatch = construct_interval_mismatch(nc_score, X_1_test, y_0_test_false, P_score_cal, P_score_test, P_score_cal_inverse, P_score_test_inverse, pre_model, cal_size, epoch)
            
            
            
            Coverage.append(coverage_ratio)
            Coverage_weight.append(coverage_ratio_weight)
            Coverage_mismatch.append(coverage_ratio_mismatch)
            
            
            
            Final_size.append(Size)
            Final_size_weight.append(Size_weight)
            Final_size_mismatch.append(Size_mismatch)
            
            
            # print('finish %d' % (epoch+1))
            print('epoch %d coverage CKE: %f coverage NCCKE: %f coverage CCKE: %f' % (epoch+1, coverage_ratio, coverage_ratio_mismatch, coverage_ratio_weight))
            
            
        print('T: %f average coverage ratio CKE: %f' % (ex.temp, np.mean(Coverage)))
        print('T: %f average coverage ratio NCCKE: %f' % (ex.temp, np.mean(Coverage_mismatch)))
        print('T: %f average coverage ratio CCKE: %f' %(ex.temp, np.mean(Coverage_weight)))
        

    
# Value_array = [500, 1000, 1200, 1500, 2000, 2500]
T_array = [0.5, 1, 5, 10, 100, 1000]
test_coverage(T_array, Epoch=200)

