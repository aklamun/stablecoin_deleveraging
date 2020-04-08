# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 19:08:52 2018

@author: aklamun
"""

import numpy as np
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

###############################################################################
def convert_constraint(A,b,c,d):
    '''Convert SOCP constraint of form ||Ax+b||_2 \leq c^T x + d to cvxopt form Gx + s = h'''
    A = np.array(A)
    b = np.array(b)
    c = np.array(c)
    d = np.array(d)
    
    G = np.zeros((np.shape(A)[0]+1, np.shape(A)[1]))
    G[0,:] = -np.transpose(c)
    G[1:,:] = -A
    h = np.zeros((np.shape(A)[0]+1,1))
    h[0,0] = d[0,0]
    h[1:,0] = b[:,0]
    return matrix(G,tc='d'), matrix(h,tc='d')

def convert_SOCP(f,A,b,c,d,F,g):
    '''convert SOCP of form     min   f^T x
                                s.t.  ||A_i x + b_i ||_2 \leq c_i^T x + d_i   i=1,...m
                                      Fx = g
    to the form needed for cvxopt solver
    A is list of A_i, b is list of b_i, c is list of c_i, d is list of d_i'''
    assert len(A) == len(b) == len(c) == len(d)
    G = []
    h = []
    for i in range(len(A)):
        Gi, hi = convert_constraint(A[i],b[i],c[i],d[i])
        G.append(Gi)
        h.append(hi)
    return matrix(f,tc='d'), G, h, matrix(F,tc='d'), matrix(g,tc='d')

def solve_SOCP(f,A,b,c,d,F,g):
    '''solve SOCP of form     min   f^T x
                                s.t.  ||A_i x + b_i ||_2 \leq c_i^T x + d_i   i=1,...m
                                      Fx = g
    using cvxopt'''
    c, G, h, A, b = convert_SOCP(f,A,b,c,d,F,g)
    sol = solvers.socp(c=c, Gq=G, hq=h, A=A, b=b)
    return sol

###############################################################################
def mean_var_target_opt(rets, cov, var_target):
    '''maximize expected return subject to max target variance, no shorting
    this is a convex QCQP, which can be expressed as a SOCP and solved in cvxopt
    rets = expected returns vector
    cov = covariance matrix'''
    try:
        chol = np.transpose(np.linalg.cholesky(cov))
    except:
        raise Exception('Covariance matrix is not positive definite')
    
    #put into SOCP standard form
    n = len(rets)
    f = -rets
    F = np.ones((1,n))
    g = 1
    A = [chol] + [np.zeros((1,n)) for i in range(n)]
    b = [np.zeros((n,1))] + [[[0]] for i in range(n)]
    c = [np.zeros((n,1))]
    for i in range(n):
        ci = np.zeros((n,1))
        ci[i,0] = 1
        c.append(ci)
    d = [[[np.sqrt(var_target)]]] + [[[0]] for i in range(n)]
    
    sol = solve_SOCP(f,A,b,c,d,F,g)
    return sol, np.array(sol['x'])


###############################################################################
def mean_var_sharpe_opt(rets, cov, rf=0):
    '''Formulates max Sharpe ratio portfolio as quadratic program, solves in CVXOPT
    rets = expected returns vector, cov = covariance matrix, rf = riskfree rate'''
    assert np.max(rets) - rf > 0
    retsr = rets - np.ones_like(rets)*rf
    
    #set up inputs to CVXOPT solver
    n = len(retsr)
    P = matrix(cov,tc='d')
    q = matrix(np.zeros((n,1)),tc='d')
    G = matrix(-np.eye(n),tc='d')
    A = np.zeros((1,n))
    A[0,:] = np.transpose(retsr)
    A = matrix(A,tc='d')
    b = matrix(1,tc='d')
    
    sol = solvers.qp(P=P, q=q, G=G, h=q, A=A, b=b)
    if sol['primal infeasibility'] == None:
        raise Exception('Primal infeasibility in Sharpe opt')
    else:
        y = np.array(sol['x'])
        x = y/np.sum(y)
    return sol, x
    
    
    










