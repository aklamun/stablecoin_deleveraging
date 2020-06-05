# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 18:17:38 2020

@author: Ariah
"""

import numpy as np
import pickle


#path = "..."

def calc_msd(vec1, vec2):
    '''Calculate relative mean-squared difference between the vectors'''
    v = np.divide(np.array(vec1) - np.array(vec2), np.array(vec1))
    return np.sum(np.multiply(v, v))/len(v)

def calc_msd_fails(vec1, vec2, x=999):
    '''Calculate relative mean-squared difference between the vectors, excluding non-failure scenarios
    x = value for non-failure scenario'''
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    v = np.divide(v1-v2, v1)
    num = (v1+v2 < 2*x).sum()
    return np.sum(np.multiply(v, v))/num

with open(path+'fs_simplified_risk_compare_eth_drift_nz', 'rb') as f:
    results_g1, results_g2, results_g3, results_g4, results_g5, results_g6, results_g7 = pickle.load(f)

stopping_msd_wrt_g7 = []
stopping_msd_wrt_g7_fails = []
vol_msd_wrt_g7 = []

stopping_msd_wrt_g7.append( calc_msd(results_g7['stopping'], results_g1['stopping']) )
stopping_msd_wrt_g7_fails.append( calc_msd_fails(results_g7['stopping'], results_g1['stopping']) )
vol_msd_wrt_g7.append( calc_msd(results_g7['vol'], results_g1['vol']) )
stopping_msd_wrt_g7.append( calc_msd(results_g7['stopping'], results_g2['stopping']) )
stopping_msd_wrt_g7_fails.append( calc_msd_fails(results_g7['stopping'], results_g2['stopping']) )
vol_msd_wrt_g7.append( calc_msd(results_g7['vol'], results_g2['vol']) )
stopping_msd_wrt_g7.append( calc_msd(results_g7['stopping'], results_g3['stopping']) )
stopping_msd_wrt_g7_fails.append( calc_msd_fails(results_g7['stopping'], results_g3['stopping']) )
vol_msd_wrt_g7.append( calc_msd(results_g7['vol'], results_g3['vol']) )
stopping_msd_wrt_g7.append( calc_msd(results_g7['stopping'], results_g4['stopping']) )
stopping_msd_wrt_g7_fails.append( calc_msd_fails(results_g7['stopping'], results_g4['stopping']) )
vol_msd_wrt_g7.append( calc_msd(results_g7['vol'], results_g4['vol']) )
stopping_msd_wrt_g7.append( calc_msd(results_g7['stopping'], results_g5['stopping']) )
stopping_msd_wrt_g7_fails.append( calc_msd_fails(results_g7['stopping'], results_g5['stopping']) )
vol_msd_wrt_g7.append( calc_msd(results_g7['vol'], results_g5['vol']) )
stopping_msd_wrt_g7.append( calc_msd(results_g7['stopping'], results_g6['stopping']) )
stopping_msd_wrt_g7_fails.append( calc_msd_fails(results_g7['stopping'], results_g6['stopping']) )
vol_msd_wrt_g7.append( calc_msd(results_g7['vol'], results_g6['vol']) )



with open(path+'fs_simplified_risk_compare_normal', 'rb') as f:
    results_g1, results_g2, results_g3, results_g4, results_g5, results_g6, results_g7 = pickle.load(f)



with open(path+'fs_simplified_risk_compare_tdist', 'rb') as f:
    results_g1, results_g2, results_g3, results_g4, results_g5, results_g6, results_g7 = pickle.load(f)