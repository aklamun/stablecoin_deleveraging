# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 2018

@author: aklamun
"""

import numpy as np
from cvxopt import matrix, solvers
import pandas as pd
solvers.options['show_progress'] = False
import optimization_algos as opt
from scipy import special


###############################################################################
###############################################################################
'''Define the asset classes'''

class Asset(object):
    ''' Attributes:
    p_0 = price at t-1
    p_1 = price at t
    '''
    
    def __init__(self, p_0=None, p_1=None, eta=None, collat=0, beta=0):
        self.p_0 = p_0
        self.p_1 = p_1

class Cryptocurrency(Asset):
    '''Attributes:
    df = degrees of freedom for t-distribution
    stdv = standard deviation
    drift
    '''
    
    def __init__(self, p_0=None, p_1=None, df=3, stdv=1., drift=0.):
        super().__init__(p_0, p_1)
        self.df = df
        self.stdv = stdv
        self.drift = drift
    
    def next_return(self):
        '''simulate multiplicative return on asset price'''
        assert self.df>2
        scale = self.stdv/np.sqrt(self.df/(self.df-2))
        ret = np.exp(scale*np.random.standard_t(self.df) + self.drift)
        self.p_0 = self.p_1
        self.p_1 = ret*self.p_0
        return

class DStablecoin(Asset):
    '''Attributes:
    eta = supply size
    collat = total collateral value
    beta = over-collateralization threshold
    x = stablecoin demand measure
    y = negative of free supply measure
    collat_pct = collateral/liabilities
    '''
    
    def __init__(self, p_0=None, p_1=None, eta=None, collat=0, beta=0, x=0, y=0):
        super().__init__(p_0, p_1)
        self.eta = eta
        self.collat = collat
        self.beta = beta
        self.x = x
        self.y = y
        self.collat_pct = 0.
    
    def update_market(self, cons_inv, ETH):
        self.x = cons_inv.wts[1]*cons_inv.n_eth*ETH.p_1
        self.y = cons_inv.wts[1]*cons_inv.n_stblc - self.eta
        return
    
    def update_collat(self, ETH_collat, ETH):
        self.collat = ETH_collat*ETH.p_1
        self.collat_pct = self.collat/self.eta
        return
    
    def clear_stblc_market(self, delta_L, ETH):
        '''return market price of stablecoin, L is previous stablecoin supply, delta_L is decided change to that supply'''
        assert delta_L - self.y > 0
        self.p_1 = self.x/(delta_L - self.y)
        self.eta += delta_L
        return
        

#later add Cstablecoin class for centralized stablecoins

###############################################################################
###############################################################################
'''Define the agent classes'''

class Agent(object):
    '''Attributes:
    port_val = portfolio value
    rets = expected returns, non-discounted for stablecoin
    cov = covariance estimation
    gamma = learning rate for returns
    delta = learning rate for covariance
    n_eth = ETH held
    
    Subclasses of Agent will be StblcHolder and Speculator
    '''
    
    def __init__(self, rets, cov, gamma=0.1, delta=0.1, n_eth=0):
        self.rets = np.array(rets)
        self.cov = np.array(cov)
        self.mu = np.log(np.array(rets)+np.ones_like(np.array(rets)))
        self.gamma = gamma
        self.delta = delta
        self.n_eth = n_eth
    
    def update_expectations_ETH(self, ETH):
        #update expected ETH log return
        log_ret = np.log(ETH.p_1/float(ETH.p_0))
        self.rets[0] = (1-self.gamma)*self.rets[0] + self.gamma*log_ret
        
        #update expected ETH variance
        self.mu[0] = self.delta*log_ret + (1-self.delta)*self.mu[0]
        log_ret_diff = log_ret - self.mu[0]
        self.cov[0,0] = self.delta*(log_ret_diff)**2 + (1-self.delta)*self.cov[0,0]
        return
    
    def update_expectations_stblc(self, ETH, stblc):
        #update expected log return
        log_rets = [np.log(a.p_1/float(a.p_0)) for a in (ETH, stblc)]
        self.rets[1] = (1-self.gamma)*self.rets[1] + self.gamma*log_rets[1]
        
        #update expected covariance (except for ETH variance entry, which was already updated)
        self.mu[1] = self.delta*log_rets[1] + (1-self.delta)*self.mu[1]
        log_ret_diff = np.zeros((len(log_rets),1))
        log_ret_diff[:,0] = log_rets - self.mu
        
        new_cov = np.dot( log_ret_diff, np.transpose(log_ret_diff) )
        ETH_var = self.cov[0,0]
        self.cov = self.delta*new_cov + (1-self.delta)*self.cov
        self.cov[0,0] = ETH_var
        return
    
###############################################################################
class StblcHolder(Agent):
    ''' Attributes:
    port_val = portfolio value
    rets = expected returns, non-discounted for stablecoin
    cov = covariance estimation
    gamma = learning rate
    n_stblc = stablecoins held
    decision_method = portfolio decision method
    target_var = target variance threshold (if applicable for method)
    '''
    
    def __init__(self, port_val, rets, cov, gamma=0.1, delta=0.1, n_eth=0, n_stblc=0, decision_method='below_target_var', var_target=0.01):
        super().__init__(rets, cov, gamma, delta, n_eth)
        self.port_val = port_val
        self.wts = np.zeros(len(rets))
        self.n_stblc=n_stblc
        self.decision_method = decision_method
        if self.decision_method == 'below_target_var':
            self.var_target = var_target
    
    def update_port_val_ETH(self, ETH):
        self.port_val += self.n_eth*(ETH.p_1 - ETH.p_0)
        return
    
    def update_port_val_stblc(self, stblc):
        self.port_val += self.n_stblc*(stblc.p_1 - stblc.p_0)
        return
    
    def discount_stblc_ret(self, stblc):
        #implement this in future
        return self.rets[1]
    
    def decide_portfolio(self, assets):
        if self.decision_method == 'sharpe_portfolio':
            rets = np.array([self.rets[0], self.discount_stblc_ret(assets[1])])
            sol, x = opt.mean_var_sharpe_opt(rets, self.cov, rf=-0.5)
        elif self.decision_method == 'below_target_var':
            rets = np.array([self.rets[0], self.discount_stblc_ret(assets[1])])
            sol, x = opt.mean_var_target_opt(rets, self.cov, self.var_target)
        else:
            raise Exception('Invalid stablecoin holder decision method')
        
        if sol['primal infeasibility'] == None:
            x = np.array([[0.],[1.]])
            #raise Exception('Primal infeasible in stablecoin holder decision')
        #else:
        if np.sum(self.wts) > 0.9999:
            self.wts = 0.9*self.wts + 0.1*x[:,0]
        else:
            self.wts = x[:,0]
        return
        


###############################################################################
class Speculator(Agent):
    ''' Attributes:
    n_eth = # of ethers owned and in collateral
    L = # of stablecoins issued against collateral
    a = VaR quantile for decision / scale of leverage/riskiness
    sigma0 = constant added to perceived risk
    b = cyclicality parameter
    lam_bar = leverage bound
    recovery_mode = indicates if in recovery mode (lam_bar unachievable) in the time period
    constraint_active = indicates whether risk constraint was active in the time period
    alpha = inverse measure of riskiness; can be numeric, 'normal', or 'max_heavy' (for normal vs. maximally heavy tailed/finite var distributions),
        or 'risk_neutral' for no risk constraint (lam_bar=1)
    '''
    
    def __init__(self, rets, cov, gamma=0.1, delta=0.1, n_eth=0, L=0, a=0.1, sigma0=0, b=0.5, alpha='normal'):
        super().__init__(rets, cov, gamma, delta, n_eth)
        self.L = L
        self.a = a
        self.sigma0 = sigma0
        self.b = b
        self.alpha = alpha
        self.recovery_mode = 0
        self.constraint_active = 0
    
    def set_leverage_bound(self):
        if self.alpha == 'normal':
            #default alpha is value based on VaR (b=-0.5), speculator assumes normally distributed ETH log returns
            alpha = -np.sqrt(2)*special.erfinv(2*self.a-1)
        elif self.alpha == 'max_heavy':
            #speculator assumes maximally heavy-tailed, finite variance ETH log returns
            alpha = np.sqrt(1/(2*self.a))
        elif self.alpha == 'risk_neutral':
            #speculator has no risk constraint, only constraint from forced liquidation
            self.lam_bar = 1.
            return
        else:
            alpha = self.alpha
        sig = (self.cov[0,0] + self.sigma0**2)**self.b
        '''later update next line to look at effect of drift term on leverage, for now assume speculators treat as 0'''
        lnlam = -alpha*sig
        assert np.exp(lnlam) < 1
        self.lam_bar = np.exp(lnlam)
        return
    
    def solve_quadratic_leverage(self, ETH, stblc, lam=-1):
        #assert self.n_eth*ETH.p_1 - stblc.beta*self.L >= 0
        if lam == -1: #default lamda
            lam = self.lam_bar
        z = self.n_eth*ETH.p_1
        #solve quadratic equation:
        a = -stblc.beta
        b = lam*(z+stblc.x) - stblc.beta*(self.L-stblc.y)
        c = -lam*z*stblc.y + stblc.beta*self.L*stblc.y
        (x1,x2) = quadratic_formula(a,b,c)
        if lam == self.lam_bar:
            #print('VaR-leverage: ', x1,x2)
            self.recovery_mode = 0
        else:
            #print('forced liquidation: ', x1,x2)
            self.recovery_mode = 1
        sols = []
        for xx in [x1, x2]:
            if xx!=np.nan and self.L + xx > 0 and xx - stblc.y > 0:
                sols.append(xx)
        return sols
    
    def maximize_exp_equity(self, ETH, stblc, constraints):
        '''len(constraints) in [1,2], gives boundary we need to check for maximization'''
        z = self.n_eth*ETH.p_1
        r = np.exp(self.rets[0])
        
        a = 1
        b = -2*stblc.y
        c = stblc.y*(r*stblc.x + stblc.y)
        (x1,x2) = quadratic_formula(a,b,c)
        if stblc.y in [x1,x2]:
            raise Exception("handle this later")
        delta_L = np.max([x1,x2]) #this one will be >y, other will not be
        #print('dL: ', delta_L)
        
        check = [i for i in constraints if i > stblc.y]
        if delta_L <= np.max(constraints) and delta_L >= np.min(constraints) and delta_L > stblc.y:
            check.append(delta_L) #if this is called, delta_L will be the optimal
            self.constraint_active = 0
        else:
            self.constraint_active = 1
        eps = lambda d : r*(z+d*stblc.x/(d-stblc.y)) - self.L-d
        equities = [eps(i) for i in check]
        return check[np.argmax(equities)]
    
    def minimize_achievable_leverage(self, ETH, stblc):
        '''work this out later'''
        x = stblc.x
        y = stblc.y
        z = self.n_eth*ETH.p_1
        
        lam_num1 = -2*np.sqrt(-self.L*x^2*y-3*self.L*x*y*z-2*self.L*y*z^2+x*y^2*z+2*y^2*z^2) + self.L*(x+z) - x*y - 3*y*z
        lam_num2 = 2*np.sqrt(-self.L*x^2*y-3*self.L*x*y*z-2*self.L*y*z^2+x*y^2*z+2*y^2*z^2) + self.L*(x+z) - x*y - 3*y*z
        lam_den = (x^2 + 2*x*z + z^2)/stblc.beta
        sols = [lam_num1/lam_den, lam_num2/lam_den]
        sols = [i for i in sols if 0<=i and i<1]
        return sols
    
    def decide_leverage(self, ETH, stblc):
        '''returns delta_L, the change in speculator liabilities/stablecoins issued'''
        z = self.n_eth*ETH.p_1
        self.set_leverage_bound()
        
        constraints = self.solve_quadratic_leverage(ETH, stblc)
        if len(constraints)==0:
            #print('leverage bound unachievable')
            constraints = self.solve_quadratic_leverage(ETH, stblc, 1.)
            #maybe handle this better later: calculate min attainable leverage
        if len(constraints)==0:
            raise Exception('Beta collateralization unachievable')
            
        delta_L = self.maximize_exp_equity(ETH, stblc, constraints)
        #print('max dL: ', delta_L)
        assert z + delta_L*stblc.x/(delta_L - stblc.y) >= stblc.beta*(self.L+delta_L) - 1e-10
        return delta_L

###############################################################################
###############################################################################

def quadratic_formula(a,b,c):
    '''solve ax^2 + bx + c = 0 for x'''
    if b**2 - 4*a*c >= 0:
        x1 = (-b + np.sqrt(b**2 - 4*a*c))/(2*float(a))
        x2 = (-b - np.sqrt(b**2 - 4*a*c))/(2*float(a))
        return (x1, x2)
    else:
        return (np.nan, np.nan)

def clear_stblc_market(cons_inv, speculator, ETH, stblc, delta_L):
    '''return market price of stablecoin, L is previous stablecoin supply, delta_L is decided change to that supply'''
    stblc.clear_stblc_market(delta_L, ETH)
    
    speculator.L += delta_L
    speculator.n_eth += delta_L*stblc.p_1/ETH.p_1
    cons_inv.update_port_val_stblc(stblc)
    cons_inv.n_eth = cons_inv.wts[0]*cons_inv.port_val/ETH.p_1
    cons_inv.n_stblc = cons_inv.wts[1]*cons_inv.port_val/stblc.p_1
    return

###############################################################################

def initial_step(cons_inv, speculator, ETH, stblc):
    '''first exchange of stablecoins'''
    ETH.p_0 = ETH.p_1
    stblc.p_0 = stblc.p_1
    
    cons_inv.decide_portfolio((ETH,stblc))
    cons_inv.n_stblc = cons_inv.wts[1]*cons_inv.port_val
    cons_inv.n_eth = cons_inv.wts[0]*cons_inv.port_val
    speculator.L = cons_inv.n_stblc
    stblc.eta = cons_inv.n_stblc
    stblc.update_collat(speculator.n_eth, ETH)
    
    #initially holds only ETH, but price doesn't change yet
    #cons_inv.n_eth = cons_inv.port_val/ETH.p_1
    
    #cons_inv.decide_portfolio((ETH,stblc))
    #delta_L = speculator.decide_leverage(cons_inv, (ETH,stblc))
    #clear_stblc_market(cons_inv, speculator, ETH, stblc, delta_L)
    return    

def run_time_step(cons_inv, speculator, ETH, stblc):
    ETH.next_return()
    stblc.p_0 = stblc.p_1
    
    #do we need this anymore?
    cons_inv.update_port_val_ETH(ETH)
    
    cons_inv.update_expectations_ETH(ETH)
    speculator.update_expectations_ETH(ETH)
    
    cons_inv.decide_portfolio((ETH,stblc))
    stblc.update_market(cons_inv, ETH)
    delta_L = speculator.decide_leverage(ETH,stblc)
    
    clear_stblc_market(cons_inv, speculator, ETH, stblc, delta_L)
    cons_inv.update_expectations_stblc(ETH,stblc)
    speculator.update_expectations_stblc(ETH,stblc)
    stblc.update_collat(speculator.n_eth, ETH)
    return

'''
cons_inv = StblcHolder(port_val=100., rets=np.array([0.2,0]), cov=np.array([[0.2,0],[0,0.01]]))
speculator = Speculator(rets=np.array([0.2,0]), cov=np.array([[0.2,0],[0,0.01]]), n_eth=500., L=0., a=0.1, sigma0=0, b=0.5)
ETH = Cryptocurrency(p_1=1., df=3, stdv=0.2, drift=0)
stblc = DStablecoin(p_1=1., eta=0., beta=2.)

i = 0
initial_step(cons_inv, speculator, ETH, stblc)
eth_hist = [ETH.p_1]
stblc_hist = [stblc.p_1]
collat_hist = [stblc.collat_pct]
while i < 100:
    i += 1
    print(i)
    print()
    run_time_step(cons_inv, speculator, ETH, stblc)
    eth_hist.append(ETH.p_1)
    stblc_hist.append(stblc.p_1)
    collat_hist.append(stblc.collat_pct)
'''


