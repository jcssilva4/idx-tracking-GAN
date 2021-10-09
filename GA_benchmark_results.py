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
# set test_parameters
deltaT =  expParameters["deltaT"]
test_size = expParameters["test_size"]
w = expParameters["w"]
b = expParameters["b"]
f = w - b
nModels = expParameters["nModels"]
total_epochs =  expParameters["total_epochs"]
# set portfolio problem
model_pars = dict([])
model_pars["K"] = expParameters["K"]
model_pars["lb"] = expParameters["lb"]
model_pars["ub"] = expParameters["ub"]
objs =  expParameters["objs"]
# GA parameters
nRuns = expParameters["nRuns"]
lookback_windows = expParameters["lookback_windows"]
# GA with gan parameters
n_sims = expParameters["n_sims"]
best_lookback_windows = [40, 60]

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

returnDB_ = returnDB.values
returnDB_test = returnDB_[returnDB_.shape[0]-test_size:,:]
dates_test = dates[returnDB_.shape[0]-test_size + 1:] # +1 to adjust to return data

sns.set(rc={'figure.figsize':(9.7,6.27)})
main_folder = "results/GA_test/deltaT_" + str(deltaT) + "/"
# plot out-of-sample tracking error x time (multiple epochs)
epoch = total_epochs
objs = ["mean", "max", "hist_TE"]
data_GA = dict([])
data_GA["Cumulative Return"] = []
data_GA["Date"] = []
data_GA["Model"] = []
for obj in objs:
	continue_b = True
	for b_size in best_lookback_windows:
		if continue_b:
			for run in range(nModels):
				f_timeIdx = b
				cumRet = 1
				next_rebal = f_timeIdx
				w = []
				filepath = main_folder + "run_" + str(run+1) + "/experimentsDB.csv"
				if not obj in ["mean", "max"]:
					filepath = main_folder + "run_" + str(run+1) + "/benchmark_experimentsDB.csv"

				experimentsDB = pd.read_csv(filepath) 
				f_timeIdx = b
				cumRet = 1
				next_rebal = f_timeIdx
				w = []
				while f_timeIdx < returnDB_test.shape[0]:
					# check if we need to rebalance
					if(f_timeIdx == next_rebal): # if rebalance, then get weights... 
						w = []
						if not obj in ["mean", "max"]:
							query = (experimentsDB["f_timeIdx"] == f_timeIdx) & (experimentsDB["b_size"] == b_size) 
						else:
							query = (experimentsDB["f_timeIdx"] == f_timeIdx) & (experimentsDB["obj"] == obj) & (experimentsDB["epoch"] == epoch-1) & (experimentsDB["n_sims"] == n_sims)
						next_rebal = next_rebal + deltaT
						# get weights
						this_row = experimentsDB[query]
						w =  this_row["best_sol"].values[0]
						w =  w.split(",")
						w = [float(raw_w) for raw_w in w[:nAssets-1]]
						#print("weights: " + str(w))

					range_oos = min(deltaT, returnDB_test.shape[0] - f_timeIdx)
					for t in range(range_oos): #loop over each t of the simulation
						cumRet = cumRet*(1+np.dot(w,returnDB_test[f_timeIdx,1:]))
						data_GA["Cumulative Return"].append(cumRet)
						data_GA["Date"].append(dates_test[f_timeIdx])
						if not obj == "hist_TE":
							model = "GAN - " + obj
						else:
							model = "GA - " + str(b_size)
						data_GA["Model"].append(model)
						f_timeIdx += 1

			if not obj == "hist_TE":
				continue_b = False

# get index data
cumRet = 1
for t in range(b,returnDB_test.shape[0]):
	cumRet = cumRet*(1+returnDB_test[t,0])
	data_GA["Cumulative Return"].append(cumRet)
	data_GA["Date"].append(dates_test[t])
	data_GA["Model"].append("Ibovespa")

# consolidated plot
df = pd.DataFrame(data_GA)
sns.set_theme(style="ticks", font_scale=1.4)
sns_plot = sns.lineplot(data=df, x="Date", y="Cumulative Return", hue="Model")#, palette="inferno")
sns_plot.figure.autofmt_xdate()
sns_plot.figure.savefig(main_folder + "sim" + str(n_sims) + "/"  + "benchmark_CumRet_Date.png")
sns_plot.figure.clf()

# plot TE x f_timeIdx for each combination of GA + simulated/real data model
objs = ["mean", "max"]
objs.extend([str(b_size) for b_size in best_lookback_windows])
data_GA = dict([])
data_GA["TE"] = []
data_GA["rebalance date"] = []
data_GA["Model"] = []
for run in range(nModels):
	for obj in objs:
		b_size = 0
		filepath = main_folder + "run_" + str(run+1) + "/experimentsDB.csv"
		if not obj in ["mean", "max"]:
			filepath = main_folder + "run_" + str(run+1) + "/benchmark_experimentsDB.csv"
			b_size = int(obj)

		experimentsDB = pd.read_csv(filepath) 
		f_timeIdx = b
		TE = 0
		next_rebal = f_timeIdx
		w = []
		while f_timeIdx < returnDB_test.shape[0]:
			current_f_ = f_timeIdx
			# check if we need to rebalance
			if(f_timeIdx == next_rebal): # if rebalance, then get weights... 
				w = []
				query = []
				if not obj in ["mean", "max"]:
					query = (experimentsDB["f_timeIdx"] == f_timeIdx) & (experimentsDB["b_size"] == b_size) 
				else:
					query = (experimentsDB["f_timeIdx"] == f_timeIdx) & (experimentsDB["obj"] == obj) & (experimentsDB["epoch"] == epoch-1) & (experimentsDB["n_sims"] == n_sims)

				next_rebal = next_rebal + deltaT
				# get weights
				this_row = experimentsDB[query]
				w =  this_row["best_sol"].values[0]
				w =  w.split(",")
				w = [float(raw_w) for raw_w in w[:nAssets-1]]

			range_oos = min(deltaT, returnDB_test.shape[0] - f_timeIdx)
			for t in range(range_oos): #loop over each t of the simulation
				data_GA["TE"].append((np.dot(w,returnDB_test[f_timeIdx,1:]) - returnDB_test[f_timeIdx,0])**2)
				data_GA["rebalance date"].append(dates_test[current_f_])
				if not obj in ["mean", "max"]:
					model = "GA - " + str(b_size)
				else:
					model = "GAN - " + obj
				data_GA["Model"].append(model)
				f_timeIdx += 1

# consolidated for all periods
df = pd.DataFrame(data_GA)
sns.set_theme(style="ticks", font_scale=1.4)
sns_plot = sns.lineplot(data=df, x="rebalance date", y="TE", hue = "Model")#, palette="inferno")
#sns_plot.figure.autofmt_xdate()
sns_plot.figure.savefig(main_folder + "sim" + str(n_sims) + "/" + "benchmark_TE_outOfSample.png")
sns_plot.figure.clf()



'''

# plot TE x f_timeIdx for each combination of GA + simulated/real data model
objs = ["mean", "max"]
objs.extend([str(b_size) for b_size in best_lookback_windows])
data_GA = dict([])
data_GA["TE"] = []
data_GA["time index"] = []
data_GA["Model"] = []
for run in range(nModels):
	for obj in objs:
		b_size = 0
		if not obj in ["mean", "max"]:
			b_size = int(obj)

		f_timeIdx = b
		TE = 0
		next_rebal = f_timeIdx
		w = []
		while f_timeIdx < returnDB_test.shape[0]:
			current_f_ = f_timeIdx
			# check if we need to rebalance
			if(f_timeIdx == next_rebal): # if rebalance, then get weights... 
				w = []
				if not obj in ["mean", "max"]:
					filepath = main_folder + "run_" + str(run+1) + "/hist_TE/windowsize_" + str(b_size) + "/f_timeIdx_" + str(f_timeIdx) + "/"
				else:
					filepath = main_folder + "run_" + str(run+1) + "/sim" + str(n_sims) + "/" + obj + "/epoch_" + str(epoch-1) + "/f_timeIdx_" + str(f_timeIdx) + "/"

				next_rebal = next_rebal + deltaT
				# get weights
				w =  [raw_w for raw_w in filehandle]
				w =  [raw_w.replace(" ", "") for raw_w in w[0].split(",")]
				w = [float(raw_w) for raw_w in w[:nAssets-1]]
				#print("weights: " + str(w))

			range_oos = min(deltaT, returnDB_test.shape[0] - f_timeIdx)
			for t in range(range_oos): #loop over each t of the simulation
				data_GA["TE"].append((np.dot(w,returnDB_test[f_timeIdx,1:]) - returnDB_test[f_timeIdx,0])**2)
				data_GA["time index"].append(f_timeIdx)
				if not obj in ["mean", "max"]:
					model = "GA - " + str(b_size)
				else:
					model = "GAN - " + obj
				data_GA["Model"].append(model)
				f_timeIdx += 1

# consolidated for all periods
df = pd.DataFrame(data_GA)
sns.set_theme(style="ticks", font_scale=1.4)
sns_plot = sns.lineplot(data=df, x="time index", y="TE", hue = "Model")#, palette="inferno")
#sns_plot.figure.autofmt_xdate()
sns_plot.figure.savefig(main_folder + "sim" + str(n_sims) + "/" + "benchmark_TE_outOfSample.png")
sns_plot.figure.clf()
'''

'''
# plot TE x lookback_windows to select the best window sizes
data_oos_period = dict([])
obj = "hist_TE"
for run in range(nModels):
	for b_size in lookback_windows:
		f_timeIdx = b
		TE = 0
		next_rebal = f_timeIdx
		w = []
		while f_timeIdx < returnDB_test.shape[0]:
			if not str(f_timeIdx) in data_oos_period.keys():
				data_GA = dict([])
				data_GA["TE"] = []
				data_GA["window size"] = []
				data_oos_period[str(f_timeIdx)] = data_GA
			# check if we need to rebalance
			if(f_timeIdx == next_rebal): # if rebalance, then get weights... 
				w = []
				filepath = main_folder + "run_" + str(run+1) + "/" + obj + "/windowsize_" + str(b_size) + "/f_timeIdx_" + str(f_timeIdx) + "/"
				next_rebal = next_rebal + deltaT
				# get weights
				w =  [raw_w for raw_w in filehandle]
				w =  [raw_w.replace(" ", "") for raw_w in w[0].split(",")]
				w = [float(raw_w) for raw_w in w[:nAssets-1]]
				#print("weights: " + str(w))

			data_GA = data_oos_period[str(f_timeIdx)]
			range_oos = min(deltaT, returnDB_test.shape[0] - f_timeIdx)
			for t in range(range_oos): #loop over each t of the simulation
				data_GA["TE"].append((np.dot(w,returnDB_test[f_timeIdx,1:]) - returnDB_test[f_timeIdx,0])**2)
				data_GA["window size"].append(b_size)
				f_timeIdx += 1

# consolidated for all periods
for f_timeIdx in data_oos_period.keys():
	data_GA = data_oos_period[f_timeIdx]
	df = pd.DataFrame(data_GA)
	sns.set_theme(style="ticks", font_scale=1.4)
	sns_plot = sns.lineplot(data=df, x="window size", y="TE")#, palette="inferno")
	sns_plot.figure.autofmt_xdate()
	sns_plot.figure.savefig(main_folder + "sim" + str(n_sims) + "/" + f_timeIdx + "benchmark_TE_bsize.png")
	sns_plot.figure.clf()
'''

# plot TE x lookback_windows to select the best window sizes
obj = "hist_TE"
data_GA = dict([])
data_GA["TE"] = []
data_GA["window size"] = []
for run in range(nModels):
	experimentsDB = pd.read_csv(main_folder + "run_" + str(run+1) + "/benchmark_experimentsDB.csv") 
	for b_size in lookback_windows:
		f_timeIdx = b
		TE = 0
		next_rebal = f_timeIdx
		w = []
		while f_timeIdx < returnDB_test.shape[0]:
			# check if we need to rebalance
			if(f_timeIdx == next_rebal): # if rebalance, then get weights... 
				w = []
				query = (experimentsDB["f_timeIdx"] == f_timeIdx) & (experimentsDB["b_size"] == b_size) 
				next_rebal = next_rebal + deltaT
				# get weights
				this_row = experimentsDB[query]
				w =  this_row["best_sol"].values[0]
				w =  w.split(",")
				w = [float(raw_w) for raw_w in w[:nAssets-1]]

			range_oos = min(deltaT, returnDB_test.shape[0] - f_timeIdx)
			for t in range(range_oos): #loop over each t of the simulation
				data_GA["TE"].append((np.dot(w,returnDB_test[f_timeIdx,1:]) - returnDB_test[f_timeIdx,0])**2)
				data_GA["window size"].append(b_size)
				f_timeIdx += 1

# consolidated for all periods
df = pd.DataFrame(data_GA)
sns.set_theme(style="ticks", font_scale=1.4)
sns_plot = sns.lineplot(data=df, x="window size", y="TE")#, palette="inferno")
sns_plot.figure.autofmt_xdate()
sns_plot.figure.savefig(main_folder + "sim" + str(n_sims) + "/benchmark_TE_bsize.png")
sns_plot.figure.clf()
