# What If We Had Used a Different Algorithm? Reliable Counterfactual KPI Analysis in Wireless Systems
This repository contains code for "[What If We Had Used a Different Algorithm? Reliable Counterfactual KPI Analysis in Wireless Systems](https://arxiv.org/pdf/2410.00150)" -- Qiushuo Hou, Sangwoo Park, Matteo Zecchin, Yunlong Cai, Guanding Yu, and Osvaldo Simeone

![O-RAN](https://github.com/qiushuo0913/Inference_DT_code/blob/master/illustration_00.png)  
*Figure 1: In the wireless system under study, the radio access network (RAN) is managed by a non-real-time controller. The controller collects data from the RAN about key performance indicators (KPIs) attained by AI “apps” implemented by the RAN. Accordingly, the controller logs data in the form $(x,a,y)$ as the data set, where $x$ is the context, $a$ is the app identifier, and $y$ is the KPIs. The controller implements the counterfactual analysis by answering a \emph{what-if} question: Given that app $a$ has obtained KPIs $y$ for context $x$, what would the KPIs have been for the same context $x$ had some other app $a'\neq a$ been selected by the non-real-time controller?*
## Dependencies
Python 3.9.18  
Pytorch 1.12.1    
## How to use
### Example 1
**transmission_mode/mimo_transmission_time.py** --- run the scenario of example1-- *python mimo_transmission_time.py*  
**transmission_mode/ex_construct.py** --- construct the app selection probability-- *python ex_construct.py*  
**transmission_mode/regression_model.py** --- train the quantile regressor-- *python regression_model.py*  
**transmission_mode/weightCP_BPSK_MU.py** --- run CCKE-- *python weightCP_BPSK_MU.py*  

### Example 2
**wireless/scripts/launch_agent.py** --- run the scenario of example2-- *python launch_agent.py*  
**wireless/envs/time_freq_resource_allocation_v112.py** --- change the setting of the scenario of example2*  
**wireless/agent/** --- change the scheduling algorithms in example2 (now is PFCA and RR)*  
**config/config_agent.json** --- setting of agent, i.e., the scheduling apps*  
**config/config_environment.json** --- setting of scenario*  


**model/ex_construct.py** --- construct the app selection probability-- *python ex_construct.py*    
**model/regression_new.py** --- train the quantile regressor-- *python regression_new.py*   
**model/weightCP_wo.py** --- run CCKE-- *python weightCP_wo.py*  

```

Inference_DT_code
└─ CCKE
   ├─ config - configuration of the example2
   │  ├─ config_agent.json
   │  ├─ config_environment.json
   │  └─ config_sacred.json

   ├─ data - saved data of the example2 (initial backlog, CQI, and the remaining baclog of different algorithms)
   │  ├─ CQI_0_init.npy
   │  ├─ Y_0_true_sep.npy
   │  ├─ Y_1_false_sep.npy
   │  └─ s_0_init.npy

   ├─ model - saved pre-trained model and run the CCKE for example2
   │  ├─ ex_construct.py - app selection probability
   │  ├─ pre_train_model.pth
   │  ├─ regression_new.py - train the quantile regressor
   │  └─ weightCP_wo.py - run the CCKE
   
   ├─ transmission_mode - code on example1

   │  ├─ data - saved data of the example1 (SNR, the number of scatters, and the retransmission times of different apps)
   │  │  ├─ Num_scatter.npy
   │  │  ├─ SNR.npy
   │  │  ├─ re_time_BPSK_Al.npy
   │  │  ├─ re_time_BPSK_MU.npy
   │  │  ├─ re_time_QPSK_Al.npy
   │  │  └─ re_time_QPSK_MU.npy

   │  ├─ ex_construct.py - app selection probability
   │  ├─ mimo_transmission_time.py - run the scenario of example1

   │  ├─ model - saved pre-trained model
   │  │  └─ pre_model_re_time_BPSK_MU.pth

   │  ├─ regression_model.py - train the quantile regressor
   │  └─ weightCP_BPSK_MU.py - run the CCKE

   └─ wireless - setting of example2
      ├─ agents - execute the app (PFCA and RR)
      │  └─ time_freq_resource_allocation_v0
      │     ├─ proportional_fair.py
      │     └─ round_robin_agent.py
      
      ├─ envs - scenario of example2
      │  ├─ time_freq_resource_allocation_v112.py

      ├─ scripts - run the scenario of example2
      │  ├─ launch_agent.py

      └─ utils

```
