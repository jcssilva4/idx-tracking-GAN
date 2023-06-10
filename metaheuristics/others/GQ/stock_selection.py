import numpy as np
from random import sample
from metaheuristics.solveIT import *

def get_S(selected_solver, r, T, k, RCL_size):

    S = [] # list of selected stocks (trial solution)
    S_c = [stock for stock in range(r.shape[0]-1)] # list of stocks not included in the trial solution

    # stock selection - solution construction phase
    while len(S) < k:
        #print("S: " + str(S))
        #print("S_c: " + str(S_c))
        g_sorted = get_greedy_vals(S, S_c, r, T) # get sorted greedy vals
        RCL = g_sorted[0:RCL_size]
        s = sample(RCL,1)
        s = s[0]
        #print("s: " + str(s['stock_idx']))
        S.append(s['stock_idx'])
        S_c.remove(s['stock_idx'])

    S.sort()
    S_binary_encoded = []
    for i in range(r.shape[0]-1):
    	if(i in S):
    		S_binary_encoded.append(1)
    	else:
    		S_binary_encoded.append(0)

    f_val_temp = selected_solver.solving(S_binary_encoded,1)

    return S, S_c, f_val_temp

def get_greedy_vals(S, S_c, r, T):

    INDX_returns = r[0,:] # get index returns
    g = [] # greedy values is a list of dictionaries where val1: stock idx, val2: greedy val
    naive_ratio = 1/(1+len(S))

    te = []
    for t in range(T):
        te_temp = INDX_returns[t]
        te_temp -= naive_ratio*np.sum([r[s+1,t] for s in S])
        te.append(te_temp)

    for i in S_c:
        f = np.sum([abs(te[t] - (naive_ratio*r[i+1,t])) for t in range(T)])
        g.append({"stock_idx":i,"g":f})

    g_sorted = sorted(g, key = lambda i: i['g']) # sort by value (greedy value)

    return g_sorted
