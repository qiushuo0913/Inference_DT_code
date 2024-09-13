
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
      │  ├─ launch_agent_false.py
      │  ├─ launch_q_learn_umts_olpc.py

      └─ utils

```