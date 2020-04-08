# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 14:21:02 2019

@author: aklamun
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import pickle

def Freedman_Diaconis_h(data, n=None):
    '''rule for histogram bin width'''
    if n == None:
        n = len(data)
    #IQR = np.percentile(data, 75) - np.percentile(data,25)
    IQR = np.percentile(data, 95) - np.percentile(data,5)
    return 2.*IQR/n**(1/3.)

#plot histogram of change in returns
rets1 = results_const_r['const_inactive']
rets1p = [r for r in rets1 if r!=0]
rets2 = results_const_r['const_active']
(mn1, mx1, mn2, mx2) = (np.percentile(rets1p, 0.1),np.percentile(rets1p,99.9),np.percentile(rets2,0.1),np.percentile(rets2,99.9))
(h1, h2) = (Freedman_Diaconis_h(rets1p, n=len(rets1p)), Freedman_Diaconis_h(rets2, n=len(rets2)))
(num_bins1, num_bins2) = (int(np.ceil((mx1-mn1)/h1)), int(np.ceil((mx2-mn2)/h2)))
hist2 = plt.hist(rets2, bins=num_bins2*5, range=(mn2,mx2), density=True, log=True, label='Active')
hist1 = plt.hist(rets1, bins=num_bins1, range=(mn1,mx1), density=True, log=True, label='Inactive')
plt.xlim(min(np.percentile(rets1p,0.1),np.percentile(rets2,0.1)),max(np.percentile(rets1p,99.9),np.percentile(rets2,99.9)))
plt.title('DStablecoin Returns by Constraint Activity', fontsize=14)
plt.legend(loc='upper right', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.xlabel('Log return', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig('figures/hist_constraint_returns.eps')
plt.show()

################################################################################
################################################################################
#plot volatility from different learning rates

with open(path+'fs_simplified_compare_learning', 'rb') as f:
    results_g1, results_g2, results_g3 = pickle.load(f)

#plot heat maps
plt.hist2d(np.zeros(10000)+0.1, np.array(results_g1['vol']), range=[[0.05,0.35],[0,0.055]], bins=[9,100] , norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.2, np.array(results_g2['vol']), range=[[0.05,0.35],[0,0.055]], bins=[9,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.3, np.array(results_g3['vol']), range=[[0.05,0.35],[0,0.055]], bins=[9,100], norm=LogNorm())

#plot percentiles
pct = 70
pct1 = [np.percentile(results_g1['vol'],pct),np.percentile(results_g2['vol'],pct), np.percentile(results_g3['vol'],pct)]
plt.scatter([0.1,0.2,0.3],pct1, color='b', label='70 percentile')
plt.plot([0.1,0.2,0.3],pct1, color='b')
pct = 95
pct2 = [np.percentile(results_g1['vol'],pct),np.percentile(results_g2['vol'],pct), np.percentile(results_g3['vol'],pct)]
plt.scatter([0.1,0.2,0.3],pct2,color='r', label='95 percentile')
plt.plot([0.1,0.2,0.3],pct2, color='r')
plt.plot([0.05,0.35],[ETH_vol, ETH_vol], color='k', linestyle='--', label='Ether volatility')
plt.legend(loc='upper left', fontsize=14)
plt.title('DStablecoin Volatility vs. Memory Parameter', fontsize=14)
plt.ylabel('Volatility (Daily)', fontsize=14)
plt.xlabel('Memory Parameter', fontsize=14)
plt.xticks([0.1,0.2,0.3])
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig('figures/hist_vol_learning_rate.pdf')
plt.show()


################################################################################
################################################################################
#plot stopping time from different learning rates

with open(path+'fs_simplified_compare_learning', 'rb') as f:
    results_g1, results_g2, results_g3 = pickle.load(f)

#plot heat maps
plt.hist2d(np.zeros(10000)+0.1, np.array(results_g1['stopping']), range=[[0.05,0.35],[0,1000]], bins=[9,100] , norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.2, np.array(results_g2['stopping']), range=[[0.05,0.35],[0,1000]], bins=[9,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.3, np.array(results_g3['stopping']), range=[[0.05,0.35],[0,1000]], bins=[9,100], norm=LogNorm())

#plot percentiles
pct = 20
pct1 = [np.percentile(results_g1['stopping'],pct),np.percentile(results_g2['stopping'],pct), np.percentile(results_g3['stopping'],pct)]
plt.scatter([0.1,0.2,0.3],pct1, color='b', label='20 percentile')
plt.plot([0.1,0.2,0.3],pct1, color='b')
pct = 5
pct2 = [np.percentile(results_g1['stopping'],pct),np.percentile(results_g2['stopping'],pct), np.percentile(results_g3['stopping'],pct)]
plt.scatter([0.1,0.2,0.3],pct2,color='r', label='5 percentile')
plt.plot([0.1,0.2,0.3],pct2, color='r')
plt.legend(loc='center left', fontsize=14)
plt.title('DStablecoin Stopping Time vs. Memory Parameter', fontsize=14)
plt.ylabel('Stopping Time (Days)', fontsize=14)
plt.xlabel('Memory Parameter', fontsize=14)
plt.xticks([0.1,0.2,0.3])
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig('figures/hist_stopping_learning_rate.pdf')
plt.show()


################################################################################
################################################################################
#plot volatility from different risk management

with open(path+'fs_simplified_risk_compare_eth_drift_nz', 'rb') as f:
    results_g1, results_g2, results_g3, results_g4, results_g5, results_g6, results_g7 = pickle.load(f)

#plot heat maps
plt.hist2d(np.zeros(10000)+0.1, np.array(results_g1['vol']), range=[[0.05,0.75],[0,0.08]], bins=[15,100] , norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.2, np.array(results_g2['vol']), range=[[0.05,0.75],[0,0.08]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.3, np.array(results_g3['vol']), range=[[0.05,0.75],[0,0.08]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.4, np.array(results_g4['vol']), range=[[0.05,0.75],[0,0.08]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.5, np.array(results_g5['vol']), range=[[0.05,0.75],[0,0.08]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.6, np.array(results_g6['vol']), range=[[0.05,0.75],[0,0.08]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.7, np.array(results_g7['vol']), range=[[0.05,0.75],[0,0.08]], bins=[15,100], norm=LogNorm())
plt.xticks((0.1,0.2,0.3,0.4,0.5,0.6,0.7),('VaRN.1','VaRN.01','VaRM.1','VaRM.01','AC1','AC2','RN'))

#plot percentiles
pct = 70
pct1 = [np.percentile(results_g1['vol'],pct),np.percentile(results_g2['vol'],pct), np.percentile(results_g3['vol'],pct), np.percentile(results_g4['vol'],pct), np.percentile(results_g5['vol'],pct), np.percentile(results_g6['vol'],pct), np.percentile(results_g7['vol'],pct)]
plt.scatter([0.1,0.2,0.3,0.4,0.5,0.6,0.7],pct1, color='b', label='70 percentile')
plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7],pct1, color='b')
pct = 99
pct2 = [np.percentile(results_g1['vol'],pct),np.percentile(results_g2['vol'],pct), np.percentile(results_g3['vol'],pct), np.percentile(results_g4['vol'],pct), np.percentile(results_g5['vol'],pct), np.percentile(results_g6['vol'],pct), np.percentile(results_g7['vol'],pct)]
plt.scatter([0.1,0.2,0.3,0.4,0.5,0.6,0.7],pct2, color='r', label='99 percentile')
plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7],pct2, color='r')
plt.plot([0,0.8],[ETH_vol, ETH_vol], color='k', linestyle='--', label='Ether volatility')
plt.legend(loc='upper right', fontsize=14)
plt.title('DStablecoin Volatility vs. Risk Management', fontsize=14)
plt.ylabel('Volatility (Daily)', fontsize=14)
plt.xlabel('Speculator Risk Management', fontsize=14)
plt.tight_layout()
plt.savefig('figures/hist_vol_risk_mgmt_drift_nz.pdf')
plt.show()

################################################################################
################################################################################
#plot volatility from different risk management

with open(path+'fs_simplified_risk_compare_normal', 'rb') as f:
    results_g1, results_g2, results_g3, results_g4, results_g5, results_g6, results_g7 = pickle.load(f)

#plot heat maps
plt.hist2d(np.zeros(10000)+0.1, np.array(results_g1['vol']), range=[[0.05,0.75],[0,0.08]], bins=[15,100] , norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.2, np.array(results_g2['vol']), range=[[0.05,0.75],[0,0.08]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.3, np.array(results_g3['vol']), range=[[0.05,0.75],[0,0.08]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.4, np.array(results_g4['vol']), range=[[0.05,0.75],[0,0.08]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.5, np.array(results_g5['vol']), range=[[0.05,0.75],[0,0.08]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.6, np.array(results_g6['vol']), range=[[0.05,0.75],[0,0.08]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.7, np.array(results_g7['vol']), range=[[0.05,0.75],[0,0.08]], bins=[15,100], norm=LogNorm())
plt.xticks((0.1,0.2,0.3,0.4,0.5,0.6,0.7),('VaRN.1','VaRN.01','VaRM.1','VaRM.01','AC1','AC2','RN'))

#plot percentiles
pct = 70
pct1 = [np.percentile(results_g1['vol'],pct),np.percentile(results_g2['vol'],pct), np.percentile(results_g3['vol'],pct), np.percentile(results_g4['vol'],pct), np.percentile(results_g5['vol'],pct), np.percentile(results_g6['vol'],pct), np.percentile(results_g7['vol'],pct)]
plt.scatter([0.1,0.2,0.3,0.4,0.5,0.6,0.7],pct1, color='b', label='70 percentile')
plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7],pct1, color='b')
pct = 95
pct2 = [np.percentile(results_g1['vol'],pct),np.percentile(results_g2['vol'],pct), np.percentile(results_g3['vol'],pct), np.percentile(results_g4['vol'],pct), np.percentile(results_g5['vol'],pct), np.percentile(results_g6['vol'],pct), np.percentile(results_g7['vol'],pct)]
plt.scatter([0.1,0.2,0.3,0.4,0.5,0.6,0.7],pct2, color='r', label='95 percentile')
plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7],pct2, color='r')
plt.plot([0,0.8],[ETH_vol, ETH_vol], color='k', linestyle='--', label='Ether volatility')
plt.legend(loc='upper right', fontsize=14)
plt.title('DStablecoin Volatility vs. Risk Management', fontsize=14)
plt.ylabel('Volatility (Daily)', fontsize=14)
plt.xlabel('Speculator Risk Management', fontsize=14)
plt.tight_layout()
plt.savefig('figures/hist_vol_risk_mgmt_normal.pdf')
plt.show()



################################################################################
################################################################################
#plot volatility from different risk management

with open(path+'fs_simplified_risk_compare_tdist', 'rb') as f:
    results_g1, results_g2, results_g3, results_g4, results_g5, results_g6, results_g7 = pickle.load(f)

#plot heat maps
plt.hist2d(np.zeros(10000)+0.1, np.array(results_g1['vol']), range=[[0.05,0.75],[0,0.08]], bins=[15,100] , norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.2, np.array(results_g2['vol']), range=[[0.05,0.75],[0,0.08]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.3, np.array(results_g3['vol']), range=[[0.05,0.75],[0,0.08]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.4, np.array(results_g4['vol']), range=[[0.05,0.75],[0,0.08]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.5, np.array(results_g5['vol']), range=[[0.05,0.75],[0,0.08]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.6, np.array(results_g6['vol']), range=[[0.05,0.75],[0,0.08]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.7, np.array(results_g7['vol']), range=[[0.05,0.75],[0,0.08]], bins=[15,100], norm=LogNorm())
plt.xticks((0.1,0.2,0.3,0.4,0.5,0.6,0.7),('VaRN.1','VaRN.01','VaRM.1','VaRM.01','AC1','AC2','RN'))

#plot percentiles
pct = 70
pct1 = [np.percentile(results_g1['vol'],pct),np.percentile(results_g2['vol'],pct), np.percentile(results_g3['vol'],pct), np.percentile(results_g4['vol'],pct), np.percentile(results_g5['vol'],pct), np.percentile(results_g6['vol'],pct), np.percentile(results_g7['vol'],pct)]
plt.scatter([0.1,0.2,0.3,0.4,0.5,0.6,0.7],pct1, color='b', label='70 percentile')
plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7],pct1, color='b')
pct = 95
pct2 = [np.percentile(results_g1['vol'],pct),np.percentile(results_g2['vol'],pct), np.percentile(results_g3['vol'],pct), np.percentile(results_g4['vol'],pct), np.percentile(results_g5['vol'],pct), np.percentile(results_g6['vol'],pct), np.percentile(results_g7['vol'],pct)]
plt.scatter([0.1,0.2,0.3,0.4,0.5,0.6,0.7],pct2, color='r', label='95 percentile')
plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7],pct2, color='r')
plt.plot([0,0.8],[ETH_vol, ETH_vol], color='k', linestyle='--', label='Ether volatility')
plt.legend(loc='upper left', fontsize=14)
plt.title('DStablecoin Volatility vs. Risk Management', fontsize=14)
plt.ylabel('Volatility (Daily)', fontsize=14)
plt.xlabel('Speculator Risk Management', fontsize=14)
plt.tight_layout()
plt.savefig('figures/hist_vol_risk_mgmt_tdist.pdf')
plt.show()


################################################################################
################################################################################
#plot stopping time from different risk management

with open(path+'fs_simplified_risk_compare_tdist', 'rb') as f:
    results_g1, results_g2, results_g3, results_g4, results_g5, results_g6, results_g7 = pickle.load(f)

plt.hist2d(np.zeros(10000)+0.1, np.array(results_g1['stopping']), range=[[0.05,0.75],[0,1000]], bins=[15,100] , norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.2, np.array(results_g2['stopping']), range=[[0.05,0.75],[0,1000]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.3, np.array(results_g3['stopping']), range=[[0.05,0.75],[0,1000]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.4, np.array(results_g4['stopping']), range=[[0.05,0.75],[0,1000]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.5, np.array(results_g5['stopping']), range=[[0.05,0.75],[0,1000]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.6, np.array(results_g6['stopping']), range=[[0.05,0.75],[0,1000]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.7, np.array(results_g7['stopping']), range=[[0.05,0.75],[0,1000]], bins=[15,100], norm=LogNorm())
plt.xticks((0.1,0.2,0.3,0.4,0.5,0.6,0.7),('VaRN.1','VaRN.01','VaRM.1','VaRM.01','AC1','AC2','RN'))

#plot percentiles
pct = 20
pct1 = [np.percentile(results_g1['stopping'],pct),np.percentile(results_g2['stopping'],pct), np.percentile(results_g3['stopping'],pct), np.percentile(results_g4['stopping'],pct), np.percentile(results_g5['stopping'],pct), np.percentile(results_g6['stopping'],pct), np.percentile(results_g7['stopping'],pct)]
plt.scatter([0.1,0.2,0.3,0.4,0.5,0.6,0.7],pct1, color='b', label='20 percentile')
plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7],pct1, color='b')
pct = 5
pct2 = [np.percentile(results_g1['stopping'],pct),np.percentile(results_g2['stopping'],pct), np.percentile(results_g3['stopping'],pct), np.percentile(results_g4['stopping'],pct), np.percentile(results_g5['stopping'],pct), np.percentile(results_g6['stopping'],pct), np.percentile(results_g7['stopping'],pct)]
plt.scatter([0.1,0.2,0.3,0.4,0.5,0.6,0.7],pct2, color='r', label='5 percentile')
plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7],pct2, color='r')
plt.legend(loc='best', fontsize=14)
plt.title('DStablecoin Stopping Time vs. Risk Management', fontsize=14)
plt.ylabel('Stopping Time (Days)', fontsize=14)
plt.xlabel('Speculator Risk Management', fontsize=14)
plt.tight_layout()
plt.savefig('figures/hist_stopping_risk_mgmt_tdist.pdf')
plt.show()

################################################################################
################################################################################
#plot stopping time from different risk management

with open(path+'fs_simplified_risk_compare_normal', 'rb') as f:
    results_g1, results_g2, results_g3, results_g4, results_g5, results_g6, results_g7 = pickle.load(f)

plt.hist2d(np.zeros(10000)+0.1, np.array(results_g1['stopping']), range=[[0.05,0.75],[0,1000]], bins=[15,100] , norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.2, np.array(results_g2['stopping']), range=[[0.05,0.75],[0,1000]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.3, np.array(results_g3['stopping']), range=[[0.05,0.75],[0,1000]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.4, np.array(results_g4['stopping']), range=[[0.05,0.75],[0,1000]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.5, np.array(results_g5['stopping']), range=[[0.05,0.75],[0,1000]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.6, np.array(results_g6['stopping']), range=[[0.05,0.75],[0,1000]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.7, np.array(results_g7['stopping']), range=[[0.05,0.75],[0,1000]], bins=[15,100], norm=LogNorm())
plt.xticks((0.1,0.2,0.3,0.4,0.5,0.6,0.7),('VaRN.1','VaRN.01','VaRM.1','VaRM.01','AC1','AC2','RN'))

#plot percentiles
pct = 5
pct1 = [np.percentile(results_g1['stopping'],pct),np.percentile(results_g2['stopping'],pct), np.percentile(results_g3['stopping'],pct), np.percentile(results_g4['stopping'],pct), np.percentile(results_g5['stopping'],pct), np.percentile(results_g6['stopping'],pct), np.percentile(results_g7['stopping'],pct)]
plt.scatter([0.1,0.2,0.3,0.4,0.5,0.6,0.7],pct1, color='m', label='5 percentile')
plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7],pct1, color='m')
pct = 1
pct2 = [np.percentile(results_g1['stopping'],pct),np.percentile(results_g2['stopping'],pct), np.percentile(results_g3['stopping'],pct), np.percentile(results_g4['stopping'],pct), np.percentile(results_g5['stopping'],pct), np.percentile(results_g6['stopping'],pct), np.percentile(results_g7['stopping'],pct)]
plt.scatter([0.1,0.2,0.3,0.4,0.5,0.6,0.7],pct2, color='r', label='1 percentile')
plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7],pct2, color='r')
plt.legend(loc='lower left', fontsize=14)
plt.title('DStablecoin Stopping Time vs. Risk Management', fontsize=14)
plt.ylabel('Stopping Time (Days)', fontsize=14)
plt.xlabel('Speculator Risk Management', fontsize=14)
plt.tight_layout()
plt.savefig('figures/hist_stopping_risk_mgmt_normal.pdf')
plt.show()

################################################################################
################################################################################
#plot stopping time from different risk management

with open(path+'fs_simplified_risk_compare_eth_drift_nz', 'rb') as f:
    results_g1, results_g2, results_g3, results_g4, results_g5, results_g6, results_g7 = pickle.load(f)

plt.hist2d(np.zeros(10000)+0.1, np.array(results_g1['stopping']), range=[[0.05,0.75],[0,1000]], bins=[15,100] , norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.2, np.array(results_g2['stopping']), range=[[0.05,0.75],[0,1000]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.3, np.array(results_g3['stopping']), range=[[0.05,0.75],[0,1000]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.4, np.array(results_g4['stopping']), range=[[0.05,0.75],[0,1000]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.5, np.array(results_g5['stopping']), range=[[0.05,0.75],[0,1000]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.6, np.array(results_g6['stopping']), range=[[0.05,0.75],[0,1000]], bins=[15,100], norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.7, np.array(results_g7['stopping']), range=[[0.05,0.75],[0,1000]], bins=[15,100], norm=LogNorm())
plt.xticks((0.1,0.2,0.3,0.4,0.5,0.6,0.7),('VaRN.1','VaRN.01','VaRM.1','VaRM.01','AC1','AC2','RN'))

#plot percentiles
pct = 5
pct1 = [np.percentile(results_g1['stopping'],pct),np.percentile(results_g2['stopping'],pct), np.percentile(results_g3['stopping'],pct), np.percentile(results_g4['stopping'],pct), np.percentile(results_g5['stopping'],pct), np.percentile(results_g6['stopping'],pct), np.percentile(results_g7['stopping'],pct)]
plt.scatter([0.1,0.2,0.3,0.4,0.5,0.6,0.7],pct1, color='m', label='5 percentile')
plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7],pct1, color='m')
pct = 1
pct2 = [np.percentile(results_g1['stopping'],pct),np.percentile(results_g2['stopping'],pct), np.percentile(results_g3['stopping'],pct), np.percentile(results_g4['stopping'],pct), np.percentile(results_g5['stopping'],pct), np.percentile(results_g6['stopping'],pct), np.percentile(results_g7['stopping'],pct)]
plt.scatter([0.1,0.2,0.3,0.4,0.5,0.6,0.7],pct2, color='r', label='1 percentile')
plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7],pct2, color='r')
plt.legend(loc='lower left', fontsize=14)
plt.title('DStablecoin Stopping Time vs. Risk Management', fontsize=14)
plt.ylabel('Stopping Time (Days)', fontsize=14)
plt.xlabel('Speculator Risk Management', fontsize=14)
plt.tight_layout()
plt.savefig('figures/hist_stopping_risk_mgmt_eth_drift_nz.pdf')
plt.show()

################################################################################
################################################################################
#plot stopping time from const v var r

with open(path+'fs_simplified_const_var_r_new', 'rb') as f:
    results_const_r, results_var_r = pickle.load(f)

plt.hist2d(np.zeros(10000)+0.1, np.array(results_const_r['stopping']), range=[[0.05,0.25],[0,1000]], bins=[5,100] , norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.2, np.array(results_var_r['stopping']), range=[[0.05,0.25],[0,1000]], bins=[5,100], norm=LogNorm())
plt.xticks((0.1,0.2),('Constant','Learned'))

#plot percentiles
pct = 20
pct1 = [np.percentile(results_const_r['stopping'],pct),np.percentile(results_var_r['stopping'],pct)]
plt.scatter([0.1,0.2],pct1, color='m', label='20 percentile')
plt.plot([0.1,0.2],pct1, color='m')
pct = 5
pct2 = [np.percentile(results_const_r['stopping'],pct),np.percentile(results_var_r['stopping'],pct)]
plt.scatter([0.1,0.2],pct2, color='r', label='5 percentile')
plt.plot([0.1,0.2],pct2, color='r')
plt.legend(loc='lower left', fontsize=14)
plt.title('DStablecoin Stopping Time vs. Return Assumptions', fontsize=14)
plt.ylabel('Stopping Time (Days)', fontsize=14)
plt.xlabel('Speculator Return Assumptions', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig('figures/hist_stopping_const_var_r.pdf')
plt.show()

################################################################################
################################################################################
#plot volatility from const v var r

with open(path+'fs_simplified_const_var_r_new', 'rb') as f:
    results_const_r, results_var_r = pickle.load(f)

plt.hist2d(np.zeros(10000)+0.1, np.array(results_const_r['vol']), range=[[0.05,0.25],[0,0.08]], bins=[5,100] , norm=LogNorm())
plt.hist2d(np.zeros(10000)+0.2, np.array(results_var_r['vol']), range=[[0.05,0.25],[0,0.08]], bins=[5,100], norm=LogNorm())
plt.xticks((0.1,0.2),('Constant','Learned'))

#plot percentiles
pct = 70
pct1 = [np.percentile(results_const_r['vol'],pct),np.percentile(results_var_r['vol'],pct)]
plt.scatter([0.1,0.2],pct1, color='m', label='70 percentile')
plt.plot([0.1,0.2],pct1, color='m')
pct = 95
pct2 = [np.percentile(results_const_r['vol'],pct),np.percentile(results_var_r['vol'],pct)]
plt.scatter([0.1,0.2],pct2, color='r', label='95 percentile')
plt.plot([0.1,0.2],pct2, color='r')
plt.plot([0.05,0.25],[ETH_vol, ETH_vol], color='k', linestyle='--', label='Ether volatility')
plt.legend(loc='upper right', fontsize=14)
plt.title('DStablecoin Volatility vs Return Assumptions', fontsize=14)
plt.ylabel('Volatility (Daily)', fontsize=14)
plt.xlabel('Speculator Return Assumptions', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig('figures/hist_vol_const_var_r.pdf')
plt.show()






#histogram of survival times
stopping = results_const_r['stopping']
(mn, mx) = (min(stopping),max(stopping))
h = Freedman_Diaconis_h(stopping)
num_bins = int(np.ceil((mx-mn)/h))
plt.hist(stopping, density=False, log=False)
plt.show()


