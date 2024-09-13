
#%%
from sklearn.model_selection import train_test_split

import numpy as np
import torch

# from regression_new import MyCNNQuantile
import regression_new as renew
import ex_construct as ex
import random



confidence_level = 0.8

num_user = 8



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


CQI_0 = np.load('../data/CQI_0_init.npy')
s_0 = np.load('../data/s_0_init.npy')

# load the true value of remaining bits of different methods 
Y_0_true = np.load('../data/Y_0_true_sep.npy')
Y_1_false = np.load('../data/Y_1_false_sep.npy')


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



for i in range(len(CQI_0)):
    for j in range(len(CQI_0[i])):
        se[i][j] = cqi_to_se(CQI_0[i][j])

tx_data_bits = np.floor(se * 5 / 10 * 1E3)
CQI_0 = tx_data_bits



CQI_0 = CQI_0.astype(np.float32)
CQI_0 = torch.from_numpy(CQI_0)

s_0 = s_0.astype(np.float32)
s_0 = torch.from_numpy(s_0)

Y_0_true = Y_0_true.astype(np.float32)
Y_0_true = torch.from_numpy(Y_0_true).unsqueeze(1)

Y_1_false = Y_1_false.astype(np.float32)
Y_1_false = torch.from_numpy(Y_1_false).unsqueeze(1)

X_0 = torch.stack((s_0, CQI_0), dim=2)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# out the necessary elemnts for the CCKE
def construct_interval_match(seed,T):
    # input:
    # seed - the random seed index
    # T - the parameter in app selection probability
    
    # output:
    # nonconformity_scores - NC scores of calibration datasets
    # X_1_test - test data
    # y_0_test_false - lable of test data
    # P_score_cal - e(x) of calibration data
    # P_score_test - e(x) of test data
    # pre_model - pre-trained quantile regressor
    # len(X_calibration) - the size of calibration dataset
    # Max_bit - the maximum size of the initial backlog among users
    
    

    # control data is the same as that used for training.
    reset_randomness(random_seed=9)

    # Split the dataset into training and testing sets
    _, X_0_test_init, _, y_0_test_init = train_test_split(X_0, Y_0_true, test_size=0.4, random_state=0)
    
    
    reset_randomness(random_seed=seed)
    X_0_calibration, X_0_test_init, y_0_calibration, y_0_test_init = train_test_split(X_0_test_init, y_0_test_init, test_size=0.6, random_state=seed)
    
    
    
    # Construct the real calibration dataset and testing data according to e(x)
    X_calibration = []
    y_calibration = []
    
    # testing data likely to run PFCA algorithm
    X_1_test = []
    y_0_test_false = []
    
    
    
    # Store the e(x) value of calibration data and testing data, respectively
    P_score_cal = []
    P_score_test = []
    
    
    # Re-divide and allocate 0 (app a = RR) and 1 (app a = PFCA) data sets 
    # as the calibration data set according to the score of e(x)
    for i in range(X_0_calibration.shape[0]):
       
        min_CQI = X_0_calibration[i,:,1]
        s = X_0_calibration[i,:,0]
        p_score_cal = ex.calculate_ex(min_CQI, s)
        
        noise = np.random.random(1)
        
        if (noise >= p_score_cal) and (~np.isnan(p_score_cal)):
            P_score_cal.append(p_score_cal)
            X_calibration.append(X_0_calibration[i])
            y_calibration.append(y_0_calibration[i])
    

    X_calibration = torch.stack(X_calibration)
    y_calibration = torch.stack(y_calibration)
    
    y_calibration = torch.squeeze(y_calibration)
    
    
    
    # the size of calibration dataset
    value_cal = 50
    value_cal_index  = np.random.choice(len(X_calibration), value_cal)
    
    
    
    X_calibration = X_calibration[value_cal_index]
    y_calibration = y_calibration[value_cal_index]
    P_score_cal = [P_score_cal[i] for i in value_cal_index]
    
   
    # normalization
    X_calibration_1 = torch.clone(X_calibration)
    
    
    for i in range(len(y_calibration)):
        X_calibration[i,:,:] = X_calibration[i,:,:]/torch.max(X_calibration_1[i,:,0])
        y_calibration[i,:] = y_calibration[i,0:]/torch.max(X_calibration_1[i,:,0])

    
    
    
    
    # Re-divide and allocate 0 (app a = RR) and 1 (app a = PFCA) data sets 
    # as the test data set according to the score of e(x)
    for i in range(X_0_test_init.shape[0]):
        
        min_CQI = X_0_test_init[i,:,1]
        s = X_0_test_init[i,:,0]
        p_score_test = ex.calculate_ex(min_CQI, s)
        
        noise = np.random.random(1)
        if (~np.isnan(p_score_test)):
            if noise < p_score_test:
                P_score_test.append(p_score_test)
                X_1_test.append(X_0_test_init[i])
                y_0_test_false.append(y_0_test_init[i])

    
    X_1_test = torch.stack(X_1_test)
    y_0_test_false = torch.stack(y_0_test_false)
    
    y_0_test_false = torch.squeeze(y_0_test_false)
    
    
    # test size
    if T == 0.5:
        value = 60
    else:
        value = 100
    
    
   
    
    X_1_test = X_1_test[0:value]
    y_0_test_false = y_0_test_false[0:value]
    
    P_score_test = P_score_test[0:value]
    
    
    # normalization
    
    X_1_test_1 = torch.clone(X_1_test)
    
    Max_bit = []
    for i in range(len(y_0_test_false)):
        X_1_test[i,:,:] = X_1_test[i,:,:]/torch.max(X_1_test_1[i,:,0])
        y_0_test_false[i,:] = y_0_test_false[i,0:]/torch.max(X_1_test_1[i,:,0])
        # save the maximum size of the initial backlog amoong users
        max_bit = torch.max(X_1_test_1[i,:,0])
        Max_bit.append(max_bit.detach().numpy())
        
        
    

    # load saved model
    
    pre_model = renew.AttentionMLP(input_dim=2, q_dim=10, v_dim=10, hidden_dim=10, output_dim=10, input_dim1=10, q_dim1=10, v_dim1=10, hidden_dim1=10, output_dim1=2)
    
    pre_model.load_state_dict(torch.load('pre_train_model.pth', map_location=torch.device(device)))
    pre_model = pre_model.to(device)
    pre_model.eval() 


    
    
    X_calibration = X_calibration.to(device)
    
    
    calibration_predictions = pre_model(X_calibration)
    
    nonconformity_scores = torch.max(np.maximum((calibration_predictions.detach().cpu()[:,:,0] - y_calibration), (y_calibration-calibration_predictions.detach().cpu()[:,:,1])), dim=1)[0]
    
    nonconformity_scores = nonconformity_scores.numpy()
    

    
    return nonconformity_scores, X_1_test, y_0_test_false, P_score_cal, P_score_test, pre_model, len(X_calibration), Max_bit


def construct_interval_mismatch(nonconformity_scores, X_1, Y_0_false, P_score_cal, P_score_test, pre_model, cal_size, epoch, Max_bit):
   
    # intput:
    # nonconformity_scores - NC scores of calibration datasets
    # X_1 - test data
    # Y_0_test_false - lable of test data
    # P_score_cal - e(x) of calibration data
    # P_score_test - e(x) of test data
    # pre_model - pre-trained quantile regressor
    # cal_size - the size of calibration dataset
    # epoch - index of random independent experiment
    # Max_bit - the maximum size of the initial backlog among users
    
    
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
        
        if np.all(test_predictions[i,:,0].detach().cpu().numpy() <= Y_0_false[i,:].numpy()) and np.all(Y_0_false[i,:].numpy() <= test_predictions[i,:,1].detach().cpu().numpy()):
            in_interval_count_test += 1
        
        # Count the number of users who are not in the range
        in_interval_count = 0
        in_range = (test_predictions[i,:,0].detach().cpu().numpy() <= Y_0_false[i,:].numpy()) & (Y_0_false[i,:].numpy() <= test_predictions[i,:,1].detach().cpu().numpy())
        
        in_interval_count += np.sum(in_range)
        out_interval_count = num_user-in_interval_count
        if out_interval_count >=3:
            I_naive.append(i)
        
        size = test_predictions[i,:,1].detach().cpu().numpy()-test_predictions[i,:,0].detach().cpu().numpy()
        Size.append(np.mean(size)) 
        
        
    coverage = in_interval_count_test/len(test_predictions)
    
        
    
    # CCKE
    w_cal = np.array(P_score_cal)/(1-np.array(P_score_cal))
    w_test = np.array(P_score_test)/(1-np.array(P_score_test))
    
    sort_nonconformity_scores = np.sort(nonconformity_scores, axis=0)
    sorted_indices = np.argsort(nonconformity_scores)
    
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
            quantile = sort_nonconformity_scores[j]
        
        
       
        lower_bound = test_predictions[i,:,0].detach().cpu().numpy()-quantile
        upper_bound = test_predictions[i,:,1].detach().cpu().numpy()+quantile
        
            
        if np.all(lower_bound <= Y_0_false[i,:].numpy()) and np.all(Y_0_false[i,:].numpy() <= upper_bound):
            in_interval_count += 1
            
            I_weight.append(i)
        
        
        prediction_intervals.append((lower_bound, upper_bound))
        size_weight = upper_bound - lower_bound
        
        
        Size_weight.append(np.mean(size_weight))
        
            
    coverage_ratio_weight = in_interval_count/len(Y_0_false)
   
           
        
      
    # NCCKE  
    weight_nonconformity_scores = nonconformity_scores
    
    
    sort_nonconformity_scores = np.sort(weight_nonconformity_scores, axis=0)
    index = np.ceil(((cal_size+1)*confidence_level)).astype(int)-1
    quantile = sort_nonconformity_scores[index]
    
    in_interval_count = 0
    I = []
    for i in range(len(test_predictions)):
        
        
       
        lower_bound = test_predictions[i,:,0].detach().cpu().numpy()-quantile
        upper_bound = test_predictions[i,:,1].detach().cpu().numpy()+quantile
        
       
        if np.all(lower_bound <= Y_0_false[i,:].numpy()) and np.all(Y_0_false[i,:].numpy() <= upper_bound):
            in_interval_count += 1
        else:
            I.append(i)
            
            
        prediction_intervals.append((lower_bound, upper_bound))
        size_mismatch = upper_bound - lower_bound
        
       
        
        Size_mismatch.append(np.mean(size_mismatch))
       
       
    coverage_ratio_mismatch = in_interval_count/len(Y_0_false)
    
    
    return coverage, coverage_ratio_weight, coverage_ratio_mismatch, Size, Size_weight, Size_mismatch


# main function
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
            nc_score, X_1_test, y_0_test_false, P_score_cal, P_score_test, pre_model, cal_size, Max_bit = construct_interval_match(seed=epoch, T = value)
            coverage_ratio, coverage_ratio_weight, coverage_ratio_mismatch, Size, Size_weight, Size_mismatch = construct_interval_mismatch(nc_score, X_1_test, y_0_test_false, P_score_cal, P_score_test, pre_model, cal_size, epoch, Max_bit)
            
            
            
            Coverage.append(coverage_ratio)
            Coverage_weight.append(coverage_ratio_weight)
            Coverage_mismatch.append(coverage_ratio_mismatch)
            
            
           
            Final_size.append(Size)
            
            Final_size_weight.append(Size_weight)
            Final_size_mismatch.append(Size_mismatch)
            
            
            
            print('epoch %d coverage CKE: %f coverage NCCKE: %f coverage CCKE: %f' % (epoch+1, coverage_ratio, coverage_ratio_mismatch, coverage_ratio_weight))
            
            
        print('T: %f average coverage ratio CKE: %f' % (ex.temp, np.mean(Coverage)))
        print('T: %f average coverage ratio NCCKE: %f' % (ex.temp, np.mean(Coverage_mismatch)))
        print('T: %f average coverage ratio CCKE: %f' %(ex.temp, np.mean(Coverage_weight)))
        


T_array = [0.5, 1, 5, 10, 100, 1000]
test_coverage(T_array, Epoch=200)



