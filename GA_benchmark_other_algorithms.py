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
heuristics = ['GA_santanna2017','GA_ruiz-torrubiano2009']
models = ['ruiz-torrubiano2009_relaxed','wang2012_relaxed']
model_cmpct = dict([])
model_cmpct['ruiz-torrubiano2009_relaxed'] = 'RT09'
model_cmpct['wang2012_relaxed'] = 'W12'
for model in models:
	for this_algorithm in heuristics:
		for run in range(nRuns):
			experimentsDB = dict([])
			experimentsDB["f_timeIdx"] = []
			experimentsDB["b_size"] = []
			experimentsDB["obj"] = []
			experimentsDB["best_objval"] = []
			experimentsDB["best_sol"] = []
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
					#best_TE, best_sol = simpleGA(S = hist_data, model_pars = model_pars, nGenerations = 100)
					tr1, best_TE, tr2, tr3, best_sol = get_solution_approx(model, this_algorithm, hist_data[:,1:], hist_data[:,0], 10, hist_data.shape[0], 1, 0)
					#write GA solutions for this epoch and this f_timeIdx
					experimentsDB["f_timeIdx"].append(f_timeIdx)
					experimentsDB["b_size"].append(b_size)
					experimentsDB["obj"].append(obj)
					experimentsDB["best_objval"].append(best_TE)
					best_sol = str(best_sol.tolist())
					best_sol = best_sol.replace('[',"").replace("]","")
					experimentsDB["best_sol"].append(best_sol)

					f_timeIdx += deltaT

			# save the final file
			df_exp = pd.DataFrame.from_dict(experimentsDB)
			df_exp.to_csv("results/GA_test/deltaT_" + str(deltaT) + "/run_" + str(run+1) + "/" + model_cmpct[model] + "_" + this_algorithm + "_experimentsDB.csv")
