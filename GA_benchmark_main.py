# utilities
import random
from datetime import datetime
import os
# importing data science and visualization libraries
import pandas as pd
import numpy as np
import seaborn as sns
# GA
from metaheuristics.GA.simpleGA import simpleGA 
from utils import *
from experiment_parameters.parameters import *


######################################## Main Script ################################################

# Set random seed for reproducibility
'''
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)
'''

# get experiment parameters
expParameters = get_parameters()
# set test_parameters
deltaT =  expParameters["deltaT"]
test_size = expParameters["test_size"]
w = expParameters["w"]
b = expParameters["b"]
f = w - b
# set portfolio problem
model_pars = dict([])
model_pars["K"] = expParameters["K"]
model_pars["lb"] = expParameters["lb"]
model_pars["ub"] = expParameters["ub"]
objs =  expParameters["objs"]
# GA parameters
nRuns = expParameters["nRuns"]
lookback_windows = expParameters["lookback_windows"]

# get the dataset
ibovDB = pd.read_excel("data/IBOV_DB_useThis_extended.xlsx") 
ibovDB = ibovDB.reindex(index=ibovDB.index[::-1]) # reverse the data set rows order
ibovDB = ibovDB.drop(ibovDB.columns[0], axis = 1)
numpy_data = ibovDB.to_numpy()
# get returns and the number of assets in the universe
priceDB = ibovDB.drop(ibovDB.columns[0], axis = 1)
returnDB = priceDB.pct_change()[1:]
nAssets = returnDB.shape[1]
# get asset symbols
symbols = ibovDB.columns[1:].to_list()
# get dates
dates = numpy_data[:,0]


# load simulations for each epoch
returnDB_ = returnDB.values
returnDB_test = returnDB_[returnDB_.shape[0]-test_size:,:]
dates_test = dates[returnDB_.shape[0]-test_size + 1:] # +1 to adjust to return data
print("test period: " + str(dates_test[0]) + " - " + str(dates_test[len(dates_test)-1]))
obj = "hist_TE"
for run in range(nRuns):
	print("Execution " + str(run+1))
	for b_size in lookback_windows:
		f_timeIdx = b
		while f_timeIdx < test_size:
			# get returns for this window (Mb)
			#hist_data = returnDB_test[f_timeIdx,1:]
			start_time_idx = returnDB_.shape[0] - test_size + f_timeIdx - b_size 
			end_time_idx = returnDB_.shape[0] - test_size + f_timeIdx
			hist_data = returnDB_[start_time_idx : end_time_idx, :]
			#print("b_size: " + str(b_size) + "/  hist_data.shape: " + str(hist_data.shape))

			#run GA
			# set problem obj
			model_pars["obj"] = obj
			#get GA solutions with obj = "mean"
			best_TE, best_sol = simpleGA(S = hist_data, model_pars = model_pars, nGenerations = 200)
			#write GA solutions for this epoch and this f_timeIdx
			filepath = "results/GA_test/deltaT_" + str(deltaT) + "/run_" + str(run+1) + "/" + obj + "/windowsize_" + str(b_size) + "/f_timeIdx_" + str(f_timeIdx) + "/"
			if not os.path.exists(filepath):
				os.makedirs(filepath) # create this path if it not exists
			filehandle = open(filepath + "best_objval.txt", 'w')
			filehandle.write(str(best_TE))
			filehandle.close()
			filehandle = open(filepath + "best_sol.txt", 'w')
			for var_port in best_sol:
				filehandle.write(str(var_port) + ", ")
			filehandle.close()
			

			f_timeIdx += deltaT