import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import os
import random
from experiment_parameters.parameters import *


GA_data_mean = dict([])
GA_data_max = dict([])

# get experiment parameters
expParameters = get_parameters()
# set GAN parameters
total_epochs =  expParameters["total_epochs"]
delta_epoch =  expParameters["delta_epoch"]
nModels =  expParameters["nModels"]
w = expParameters["w"]
b = expParameters["b"]
f = w - b
# set test_parameters
deltaT =  expParameters["deltaT"]
test_size = expParameters["test_size"]
# set portfolio problem
model_pars = dict([])
model_pars["K"] = expParameters["K"]
model_pars["lb"] = expParameters["lb"]
model_pars["ub"] = expParameters["ub"]
# set ga with gan parameters
objs =  expParameters["objs"]
n_sims = expParameters["n_sims"]
epoch_GA = [400, 800, 8000] # for mean_ms -> best_epoch = 2300, for mean -> best_epoch = 

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

# analysis window
returnDB_ = returnDB.values
returnDB_test = returnDB_[returnDB_.shape[0]-test_size:,:]
dates_test = dates[returnDB_.shape[0]-test_size + 1:] # +1 to adjust to return data

sns.set(rc={'figure.figsize':(9.7,6.27)})

metaheuristics = dict([])
metaheuristics['ms_mean'] = 'SDM-SBDGA-GAN' 
metaheuristics['mean'] = 'SDM-SAAGA-GAN' 

max_epoch = total_epochs
epoch_step_size = delta_epoch
main_folder = "results/GA_test/deltaT_" + str(deltaT) + "/"

data_GA = dict([])
data_GA["TE"] = []
data_GA["Obj"] = []
data_GA["Epoch"] = []
data_GA["Algorithm"] = []
for run in range(nModels):
	experimentsDB = pd.read_csv("results/GA_test/deltaT_" + str(deltaT) + "/run_" + str(run+1) + "/experimentsDB.csv")
	for obj in objs:
		epoch = delta_epoch
		while epoch <= max_epoch:
			f_timeIdx = b
			TE = 0
			next_rebal = f_timeIdx
			w = []
			while f_timeIdx < returnDB_test.shape[0]:				
				# check if we need to rebalance
				if(f_timeIdx == next_rebal): # if rebalance, then get weights... 
					query = (experimentsDB["f_timeIdx"] == f_timeIdx) & (experimentsDB["obj"] == obj) & (experimentsDB["epoch"] == epoch-1) & (experimentsDB["n_sims"] == n_sims)
					this_row = experimentsDB[query]
					w =  this_row["best_sol"].values[0]
					w =  w.split(",")
					w = [float(raw_w) for raw_w in w[:nAssets-1]]
					next_rebal = next_rebal + deltaT
					#print("weights: " + str(w))

				range_oos = min(deltaT, returnDB_test.shape[0] - f_timeIdx)
				for t in range(range_oos): #loop over each t of the simulation
					data_GA["TE"].append((np.dot(w,returnDB_test[f_timeIdx,1:]) - returnDB_test[f_timeIdx,0])**2)
					data_GA["Obj"].append(obj)
					data_GA["Epoch"].append(epoch)
					data_GA["Algorithm"].append(metaheuristics[obj])
					f_timeIdx += 1

			epoch += epoch_step_size

# consolidated for all periods
df = pd.DataFrame(data_GA)
sns.set_theme(style="ticks", font_scale=1.4)
sns_plot = sns.lineplot(data=df, x="Epoch", y="TE", hue="Algorithm")#, palette="inferno")
sns_plot.figure.autofmt_xdate()
sns_plot.figure.savefig(main_folder  + "sim" + str(n_sims) + "/compareObj_TE_epoch.png")
sns_plot.figure.clf()