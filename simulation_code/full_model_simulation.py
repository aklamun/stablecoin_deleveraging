# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 12:11:50 2019

@author: aklamun
"""

import simple_agent_model
import numpy as np
import threading

def initial_step(speculator, stblc_holder, ETH, stblc, D):
    ETH.p_0 = ETH.p_1
    stblc.p_0 = stblc.p_1
    
    stblc_holder.decide_portfolio((ETH,stblc))
    #set up system so that stablecoin demand is D, set stblc_holder value to give this
    stblc_holder.n_stblc = D
    stblc_holder.port_val = D/stblc_holder.wts[1]
    stblc_holder.n_eth = stblc_holder.wts[0]*stblc_holder.port_val
    stblc.eta = stblc_holder.n_stblc
    speculator.L = stblc.eta
    stblc.update_collat(speculator.n_eth, ETH)
    return

def run_time_step(speculator, stblc_holder, ETH, stblc, t_sample=None, const_r=None, eth_distr='tdistribution'):
    if t_sample == None:
        ETH.next_return()
    else:
        next_ETH_ret(ETH, t_sample, typ=eth_distr)
    stblc.p_0 = stblc.p_1
    speculator.update_expectations_ETH(ETH)
    if const_r != None:
        speculator.rets[0] = const_r
    stblc_holder.update_port_val_ETH(ETH)
    stblc_holder.update_expectations_ETH(ETH)
    stblc_holder.decide_portfolio((ETH,stblc))
    
    #update stblc market
    stblc.update_market(stblc_holder, ETH)
    delta_L = speculator.decide_leverage(ETH,stblc)
    
    #clear stblc market
    stblc.clear_stblc_market(delta_L, ETH)    
    speculator.L += delta_L
    speculator.n_eth += delta_L*stblc.p_1/ETH.p_1
    stblc_holder.update_port_val_stblc(stblc)
    stblc_holder.n_eth = stblc_holder.wts[0]*stblc_holder.port_val/ETH.p_1
    stblc_holder.n_stblc = stblc_holder.wts[1]*stblc_holder.port_val/stblc.p_1
    
    speculator.update_expectations_stblc(ETH,stblc)
    stblc_holder.update_expectations_stblc(ETH,stblc)
    stblc.update_collat(speculator.n_eth, ETH)    
    return

def next_ETH_ret(ETH, t_sample, typ='tdistribution'):
    if typ == 'tdistribution':
        assert ETH.df>2
        scale = ETH.stdv/np.sqrt(ETH.df/(ETH.df-2))
    elif typ == 'normal':
        scale = ETH.stdv
    ret = np.exp(scale*t_sample + ETH.drift)
    ETH.p_0 = ETH.p_1
    ETH.p_1 = ret*ETH.p_0
    return

def log_ret(stblc):
    return np.log(stblc.p_1/float(stblc.p_0))

def simulate(speculator, stblc_holder, ETH, stblc, t_samples, return_dict, const_r=None, eth_distr='tdistribution'):
    num = len(t_samples)
    D = 100 #initial size of stblc market
    
    initial_step(speculator, stblc_holder, ETH, stblc, D)
    for i in range(num):
        try:
            if const_r == None:
                run_time_step(speculator, stblc_holder, ETH, stblc, t_sample=t_samples[i], const_r=None, eth_distr=eth_distr)
            else:
                run_time_step(speculator, stblc_holder, ETH, stblc, t_sample=t_samples[i], const_r=const_r, eth_distr=eth_distr)
        except:
            break
        ret = log_ret(stblc)
        print('n_stblc', stblc.eta)
        print('stblc expectations', stblc_holder.rets[1], stblc_holder.cov[1,1])
        print('stblc wt', stblc_holder.wts[1])
        print('stblc p', stblc.p_1)
        print('')
        if speculator.constraint_active == 0:
            return_dict['rets_constraint_inactive'] += [ret]
            if np.abs(stblc.p_0-1) < 0.5:
                return_dict['rets_inactive_normal'] += [ret]
        elif speculator.constraint_active == 1:
            return_dict['rets_constraint_active'] += [ret]
            if speculator.recovery_mode == 0:
                return_dict['rets_active_not_recovery'] += [ret]
            if stblc.p_0 > 0.5:
                return_dict['rets_active_normal'] += [ret]
        if speculator.recovery_mode == 1:
            return_dict['rets_recovery_mode'] += [ret]
        
        if stblc.p_1 < 0.5 and stblc.p_0 < 0.5: #suppose stopping time for global settlement
            break
    return_dict['i'] = i
    return

def get_ETHrets_array(eth_distr):
    if eth_distr == 'tdistribution':
        t_samples = np.random.standard_t(df, max_time)
    elif eth_distr == 'normal':
        t_samples = np.random.normal(size=max_time)
    return t_samples
    

#################################################################################################################################
#################################################################################################################################
'''simulations ...'''

'''
#from daily ETH data 2017-2018, from log returns
ETH_drift = 0.00162
ETH_vol = 0.027925
max_time = 1000
#num_sims = 10000
num_sims = 1
eth_distr = 'tdistribution'
df = 3

init_cov = np.array([[ETH_vol**2,0],[0,0.00001]])

const_r = 0.00162
const_active = []
const_inactive = []

for i in range(num_sims):
    t_samples = get_ETHrets_array(eth_distr)
    
    speculator = Speculator(rets=np.array([ETH_drift,0]), cov=init_cov, n_eth=400., L=0., a=0.1, sigma0=0, b=0.5)
    stblc_holder = StblcHolder(port_val=100., rets=np.array([ETH_drift,0]), cov=init_cov, gamma=0.1, decision_method='below_target_var', var_target=0.0001)
    ETH = Cryptocurrency(p_1=1., df=df, stdv=ETH_vol, drift=ETH_drift)
    stblc = DStablecoin(p_1=1., eta=0., beta=1.5)
    return_dict = {'i':0,
                   'rets_constraint_inactive':[],
                   'rets_inactive_normal':[],
                   'rets_constraint_active':[],
                   'rets_active_not_recovery':[],
                   'rets_active_normal':[],
                   'rets_recovery_mode':[]
                   }
    simulate(speculator, stblc_holder, ETH, stblc, t_samples, return_dict, eth_distr=eth_distr)
    const_active += return_dict['rets_constraint_active']
    const_inactive += return_dict['rets_constraint_inactive']
   '''
