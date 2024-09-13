"""
© 2020 Nokia
Licensed under the BSD 3 Clause license
SPDX-License-Identifier: BSD-3-Clause
"""
#%%
import sys
sys.path.insert(0, '../..')
import gym
import json
import importlib
import os

from sacred import Experiment

from wireless.agents.bosch_agent import BoschAgent
from wireless.agents.time_freq_resource_allocation_v0.round_robin_agent import *
from wireless.agents.time_freq_resource_allocation_v0.proportional_fair import *
from wireless.agents.noma_ul_time_freq_resource_allocation_v0.noma_ul_proportional_fair import *

import wireless
import random

#importlib.reload(wireless)



print(os.path.abspath(wireless.__file__))

#from wireless.envs.time_freq_resource_allocation_v88 import TimeFreqResourceAllocationV87

# 重新加载自定义环境模块
#importlib.reload(wireless.envs.time_freq_resource_allocation_v0)

# 注册自定义环境
from gym.envs.registration import register

register(
    id='TimeFreqResourceAllocation-v1012',
    entry_point='wireless.envs.time_freq_resource_allocation_v112:TimeFreqResourceAllocationV87',
)

#控制每次仿真初始种子一样
def keep_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# Load agent parameters
with open('../../config/config_agent.json') as f:
    ac = json.load(f)

# Configure experiment
with open('../../config/config_sacred.json') as f:
    sc = json.load(f)   # Sacred Configuration
    ns = sc["sacred"]["n_metrics_points"]  # Number of points per episode to log in Sacred
    ex = Experiment(ac["agent"]["agent_type"], save_git_info=False)
    ex.add_config(sc)
    ex.add_config(ac)
mongo_db_url = f'mongodb://{sc["sacred"]["sacred_user"]}:{sc["sacred"]["sacred_pwd"]}@' +\
               f'{sc["sacred"]["sacred_host"]}:{sc["sacred"]["sacred_port"]}/{sc["sacred"]["sacred_db"]}'
# ex.observers.append(MongoObserver(url=mongo_db_url, db_name=sc["sacred"]["sacred_db"]))  # Uncomment to save to DB

# Load environment parameters
with open('../../config/config_environment.json') as f:
    ec = json.load(f)
    ex.add_config(ec)


def change_algorithm(_run, num, threshold=0.5):
    # keep_seed(_run.config['seed'] + num)
    # noise = random.random() 
    #固定算法为round robin
    noise=0.4
    if noise > threshold:
        # ac["agent"]["agent_type"] = "proportional fair channel aware"
        ac["agent"]["agent_type"] = "round robin"
    else:
        # ac["agent"]["agent_type"] = "round robin"
        ac["agent"]["agent_type"] = "proportional fair channel aware"


# @ex.automain
@ex.main
def main(_run):
    n_eps = _run.config["agent"]["n_episodes"]
    t_max = _run.config['agent']['t_max']
    n_sf = t_max//_run.config['env']['n_prbs']  # Number of complete subframes to run per episode
    log_period_t = max(1, (n_sf//ns)*_run.config['env']['n_prbs'])  # Only log rwd on last step of each subframe

    rwd = np.zeros((n_eps, t_max))  # Memory allocation

    # 存储真实的协变量X，对应的reward以及算法选择用户的动作集合
    # 默认“round robin”为0， "proportional fair channel aware"为1
    CQI_0 = []
    CQI_1 = []
    
    s_0 = []
    s_1 = []
    
    Y_0_True = []
    Y_1_True = []
    
    A_0_True = []
    A_1_True = []
    
    
    # Simulate
    for ep in range(n_eps):  # Run episodes
        #存储每个随机种子下的算法动作集合
        A = np.zeros(t_max)
        #随机选择不同算法
        change_algorithm(_run, ep, threshold=0.5)
        
        if _run.config['env']['env'] == 'TimeFreqResourceAllocation-v0':
            env = gym.make('TimeFreqResourceAllocation-v1012', n_ues=_run.config['env']['n_ues'],
                           n_prbs=_run.config['env']['n_prbs'], buffer_max_size=_run.config['env']['buffer_max_size'],
                           eirp_dbm=_run.config['env']['eirp_dbm'], f_carrier_mhz=_run.config['env']['f_carrier_mhz'],
                           max_pkt_size_bits=_run.config['env']['max_pkt_size_bits'],
                           it=_run.config['env']['non_gbr_traffic_mean_interarrival_time_ttis'])  # Init environment
            env.seed(seed=_run.config['seed'] + ep)
            
            # Init agent
            if ac["agent"]["agent_type"] == "random":
                agent = RandomAgent(env.action_space)
                agent.seed(seed=_run.config['seed'] + ep)
            elif ac["agent"]["agent_type"] == "round robin":
                agent = RoundRobinAgent(env.action_space, env.K, env.L)
            elif ac["agent"]["agent_type"] == "round robin iftraffic":
                agent = RoundRobinIfTrafficAgent(env.action_space, env.K, env.L)
            elif ac["agent"]["agent_type"] == "proportional fair":
                agent = ProportionalFairAgent(env.action_space, env.K, env.L)
            elif ac["agent"]["agent_type"] == "proportional fair channel aware":
                agent = ProportionalFairChannelAwareAgent(env.action_space, env.K, env.L)
            elif ac["agent"]["agent_type"] == "knapsack":
                agent = Knapsackagent(env.action_space, env.K, env.L, env.Nf)
            elif ac["agent"]["agent_type"] == "Bosch":
                agent = BoschAgent(env.action_space, env.K, env.L, env.max_pkt_size_bits)
            else:
                raise NotImplemented
                
        elif _run.config['env']['env'] == 'NomaULTimeFreqResourceAllocation-v0':
            env = gym.make('NomaULTimeFreqResourceAllocation-v0', n_ues=_run.config['env']['n_ues'],
                           n_prbs=_run.config['env']['n_prbs'], n_ues_per_prb=_run.config['env']['n_ues_per_prb'], buffer_max_size=_run.config['env']['buffer_max_size'],
                           eirp_dbm=_run.config['env']['eirp_dbm'], f_carrier_mhz=_run.config['env']['f_carrier_mhz'],
                           max_pkt_size_bits=_run.config['env']['max_pkt_size_bits'],
                           it=_run.config['env']['non_gbr_traffic_mean_interarrival_time_ttis'])  # Init environment
            env.seed(seed=_run.config['seed'] + ep)
            
            # Init agent
            if ac["agent"]["agent_type"] == "random":
                agent = RandomAgent(env.action_space)
                agent.seed(seed=_run.config['seed'] + ep)
            elif ac["agent"]["agent_type"] == "proportional fair channel aware":
                agent = NomaULProportionalFairChannelAwareAgent(env.action_space, env.K, env.M, env.L, env.n_mw, env.SINR_COEFF)
            else:
                raise NotImplemented
        else:
            raise NotImplemented

        reward = 0
        done = False
        state = env.reset()
        #存储场景的初始状态x
        # covariate = np.concatenate((env.cqi, env.s.flatten()))
        s = np.copy(env.s.reshape(-1))
        cqi = np.copy(env.cqi)
        if ac["agent"]["agent_type"] == "round robin":
            CQI_0.append(cqi)
            s_0.append(s)
        else:
            CQI_1.append(cqi)
            s_1.append(s)
            
        # print('env state', env.qi)
        for t in range(t_max):  # Run one episode
            # Collect progress
            if t_max < ns or (t > 0 and (t+1) % log_period_t == 0):  # If it's time to log
                s = np.reshape(state[env.K:env.K * (1 + env.L)], (env.K, env.L))
                qi_ohe = np.reshape(state[env.K+2*env.K*env.L:5*env.K + 2*env.K*env.L], (env.K, 4))
                qi = [np.where(r == 1)[0][0] for r in qi_ohe]  # Decode One-Hot-Encoded QIs
                for u in range(0, env.K, env.K//2):  # Log KPIs for some UEs
                    _run.log_scalar(f"Episode {ep}. UE {u}. CQI vs time step", state[u], t)
                    _run.log_scalar(f"Episode {ep}. UE {u}. Buffer occupancy [bits] vs time step", np.sum(s[u, :]), t)
                    _run.log_scalar(f"Episode {ep}. UE {u}. QoS Identifier vs time step", qi[u], t)

            action = agent.act(state, reward, done)
            state, reward, reward_sep, done, _ = env.step(action)
            # print('t', t, 'qi', np.reshape(state[env.K+2*env.K*env.L:5*env.K + 2*env.K*env.L], (env.K, 4)))
            
            # Collect progress
            if t_max < ns or (t > 0 and (t+1) % log_period_t == 0):
                _run.log_scalar(f"Episode {ep}. Rwd vs time step", reward, t)

            A[t] = action
            rwd[ep, t] = reward
            if done:
                break
            # if (ep*t_max + t) % log_period_t == 0:
            #     print(f"{(ep*t_max + t)*100/(n_eps*t_max):3.0f}% completed.")

        # 存储最终的reward和action
        if ac["agent"]["agent_type"] == "round robin":
            A_0_True.append(A)
            # Y_0_True.append(-rwd[ep,-1])
            Y_0_True.append(-reward_sep)
        else:
            A_1_True.append(A)
            # Y_1_True.append(-rwd[ep,-1])
            Y_1_True.append(-reward_sep)
        
        print(f"{(ep+1)*100/(n_eps):3.0f}% completed.")
            
        env.close()
        
    # np.save('../../data/CQI_0_init_change.npy', CQI_0)   
    # np.save('../../data/CQI_1.npy', CQI_1)
    # np.save('../../data/s_0_init_change.npy', s_0)   
    # np.save('../../data/s_1.npy', s_1)
    
    np.save('../../data/Y_1_false_sep_T20.npy', Y_1_True)   
    # np.save('../../data/Y_0_true_sep_T20.npy', Y_0_True) 
    
    # np.save('../../data/A_0_true.npy', A_0_True)   
    # np.save('../../data/A_1_true.npy', A_1_True)
    

    # if n_eps > 1:
    #     rwd_avg = np.mean(rwd, axis=0)
    #     for t in range(t_max):
    #         if t_max < ns or (t > 0 and (t+1) % log_period_t == 0):  # If it's time to log
    #             _run.log_scalar(f"Mean rwd vs time step", rwd_avg[t], t)

    # result = np.mean(rwd)  # Save experiment result
    # print(f"Result: {result}")
    # return result

ex.run()

# %%
