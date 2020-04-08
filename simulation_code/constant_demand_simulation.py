# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 18:43:58 2019

@author: aklamun
"""

import simple_agent_model
import numpy as np
import threading


def initial_step(speculator, ETH, stblc, D):
    '''first exchange of stablecoins'''
    ETH.p_0 = ETH.p_1
    stblc.p_0 = stblc.p_1
    
    stblc.eta = D
    speculator.L = stblc.eta
    stblc.update_collat(speculator.n_eth, ETH)
    return    

def run_time_step(speculator, ETH, stblc, D, t_sample=None, const_r=None, eth_distr='tdistribution'):
    if t_sample == None:
        ETH.next_return()
    else:
        next_ETH_ret(ETH, t_sample, typ=eth_distr)
    stblc.p_0 = stblc.p_1
    speculator.update_expectations_ETH(ETH)
    if const_r != None:
        speculator.rets[0] = const_r
    
    #update stblc market
    stblc.x = D
    stblc.y = -stblc.eta
    delta_L = speculator.decide_leverage(ETH,stblc)
    
    stblc.clear_stblc_market(delta_L, ETH)    
    speculator.L += delta_L
    speculator.n_eth += delta_L*stblc.p_1/ETH.p_1
    
    speculator.update_expectations_stblc(ETH,stblc)
    stblc.update_collat(speculator.n_eth, ETH)
    return

def vol_contribution(speculator, stblc):
    log_ret = np.log(stblc.p_1/float(stblc.p_0))
    return (log_ret - speculator.mu[1])**2

def log_ret(stblc):
    return np.log(stblc.p_1/float(stblc.p_0))

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

def simulate(speculator, ETH, stblc, t_samples, const_r=None, eth_distr='tdistribution'):
    num = len(t_samples)
    D = 100. #constant stablecoin demand
    i = 0
    rets_constraint_inactive = []
    rets_inactive_normal = []
    rets_constraint_active = []
    rets_active_not_recovery = []
    rets_active_normal = []
    rets_recovery_mode = []
    
    initial_step(speculator, ETH, stblc, D)
    for i in range(num):
        try:
            if const_r == None:
                run_time_step(speculator, ETH, stblc, D, t_sample=t_samples[i], eth_distr=eth_distr)
            else:
                run_time_step(speculator, ETH, stblc, D, t_sample=t_samples[i], const_r=const_r, eth_distr=eth_distr)
        except:
            break
        ret = log_ret(stblc)
        if speculator.constraint_active == 0:
            rets_constraint_inactive += [ret]
            if np.abs(stblc.p_0-1) < 0.5:
                rets_inactive_normal += [ret]
        elif speculator.constraint_active == 1:
            rets_constraint_active += [ret]
            if speculator.recovery_mode == 0:
                rets_active_not_recovery += [ret]
            if stblc.p_0 > 0.5:
                rets_active_normal += [ret]
        if speculator.recovery_mode == 1:
            rets_recovery_mode += [ret]
        
        if stblc.p_1 < 0.5 and stblc.p_0 < 0.5: #suppose stopping time for global settlement
            break
    return i, rets_constraint_inactive, rets_inactive_normal, rets_constraint_active, rets_active_not_recovery, rets_active_normal, rets_recovery_mode

def simulate_no_return(speculator, ETH, stblc, t_samples, return_dict, const_r=None, eth_distr='tdistribution'):
    i, rci, rin, rca, rar, ran, rrm = simulate(speculator, ETH, stblc, t_samples, const_r, eth_distr)
    return_dict['stopping'] += [i]
    return_dict['vol'] += [np.std(rci+rca)]
    return
    






#################################################################################################################################
#################################################################################################################################
'''simulations to compare speculator risk management -- record limited data'''

'''
#from daily ETH data 2017-2018, from log returns
ETH_drift = 0.00162*0
ETH_vol = 0.027925
max_time = 1000
num_sims = 10000
eth_distr = 'tdistribution'
df = 3

init_cov = np.array([[ETH_vol**2,0],[0,0.00001]])

const_r = 0.1
results_g1 = {'stopping':[],'vol':[]}
results_g2 = {'stopping':[],'vol':[]}
results_g3 = {'stopping':[],'vol':[]}
results_g4 = {'stopping':[],'vol':[]}
results_g5 = {'stopping':[],'vol':[]}
results_g6 = {'stopping':[],'vol':[]}
results_g7 = {'stopping':[],'vol':[]}

for i in range(num_sims):
    if eth_distr == 'tdistribution':
        t_samples = np.random.standard_t(df, max_time)
    elif eth_distr == 'normal':
        t_samples = np.random.normal(size=max_time)
    
    #group 1: speculator VaR using normal assumption, a=0.1
    speculator = Speculator(rets=np.array([ETH_drift,0]), cov=init_cov, n_eth=400., L=0., a=0.1, sigma0=0, b=0.5, alpha='normal')
    ETH = Cryptocurrency(p_1=1., df=df, stdv=ETH_vol, drift=ETH_drift)
    stblc = DStablecoin(p_1=1., eta=0., beta=1.5)
    i, rci, rin, rca, rar, ran, rrm = simulate(speculator, ETH, stblc, t_samples)
    results_g1['stopping'] += [i]
    results_g1['vol'] += [np.std(rci+rca)]
    
    #group 2: speculator VaR using normal assumption, a=0.01
    speculator = Speculator(rets=np.array([ETH_drift,0]), cov=init_cov, n_eth=400., L=0., a=0.01, sigma0=0, b=0.5, alpha='normal')
    ETH = Cryptocurrency(p_1=1., df=df, stdv=ETH_vol, drift=ETH_drift)
    stblc = DStablecoin(p_1=1., eta=0., beta=1.5)
    i, rci, rin, rca, rar, ran, rrm = simulate(speculator, ETH, stblc, t_samples)
    results_g2['stopping'] += [i]
    results_g2['vol'] += [np.std(rci+rca)]
    
    #group 3: speculator VaR using max_heavy assumption, a=0.1
    speculator = Speculator(rets=np.array([ETH_drift,0]), cov=init_cov, n_eth=400., L=0., a=0.1, sigma0=0, b=0.5, alpha='max_heavy')
    ETH = Cryptocurrency(p_1=1., df=df, stdv=ETH_vol, drift=ETH_drift)
    stblc = DStablecoin(p_1=1., eta=0., beta=1.5)
    i, rci, rin, rca, rar, ran, rrm = simulate(speculator, ETH, stblc, t_samples)
    results_g3['stopping'] += [i]
    results_g3['vol'] += [np.std(rci+rca)]
    
    #group 4: speculator VaR using max_heavy assumption, a=0.01
    speculator = Speculator(rets=np.array([ETH_drift,0]), cov=init_cov, n_eth=400., L=0., a=0.01, sigma0=0, b=0.5, alpha='max_heavy')
    ETH = Cryptocurrency(p_1=1., df=df, stdv=ETH_vol, drift=ETH_drift)
    stblc = DStablecoin(p_1=1., eta=0., beta=1.5)
    i, rci, rin, rca, rar, ran, rrm = simulate(speculator, ETH, stblc, t_samples)
    results_g4['stopping'] += [i]
    results_g4['vol'] += [np.std(rci+rca)]
    
    #group 5: speculator anti-cyclic
    speculator = Speculator(rets=np.array([ETH_drift,0]), cov=init_cov, n_eth=400., L=0., sigma0=0, b=-0.25, alpha=1/100.)
    ETH = Cryptocurrency(p_1=1., df=df, stdv=ETH_vol, drift=ETH_drift)
    stblc = DStablecoin(p_1=1., eta=0., beta=1.5)
    i, rci, rin, rca, rar, ran, rrm = simulate(speculator, ETH, stblc, t_samples)
    results_g5['stopping'] += [i]
    results_g5['vol'] += [np.std(rci+rca)]
    
    #group 6: speculator anti-cyclic
    speculator = Speculator(rets=np.array([ETH_drift,0]), cov=init_cov, n_eth=400., L=0., sigma0=0, b=-0.25, alpha=1/50.)
    ETH = Cryptocurrency(p_1=1., df=df, stdv=ETH_vol, drift=ETH_drift)
    stblc = DStablecoin(p_1=1., eta=0., beta=1.5)
    i, rci, rin, rca, rar, ran, rrm = simulate(speculator, ETH, stblc, t_samples)
    results_g6['stopping'] += [i]
    results_g6['vol'] += [np.std(rci+rca)]
    
    #group 7: speculator is risk neutral
    speculator = Speculator(rets=np.array([ETH_drift,0]), cov=init_cov, n_eth=400., L=0., alpha='risk_neutral')
    ETH = Cryptocurrency(p_1=1., df=df, stdv=ETH_vol, drift=ETH_drift)
    stblc = DStablecoin(p_1=1., eta=0., beta=1.5)
    i, rci, rin, rca, rar, ran, rrm = simulate(speculator, ETH, stblc, t_samples)
    results_g7['stopping'] += [i]
    results_g7['vol'] += [np.std(rci+rca)]
'''