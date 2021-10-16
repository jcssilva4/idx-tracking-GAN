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
epoch_GA = [100, 1000, 2500] # for mean_ms -> best_epoch = 2300, for mean -> best_epoch = 

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

max_epoch = total_epochs
epoch_step_size = delta_epoch
main_folder = "results/GA_test/deltaT_" + str(deltaT) + "/"
# plot out-of-sample tracking error x time (multiple epochs)
for obj in objs:
	data_GA = dict([])
	data_GA["Cumulative Return"] = []
	data_GA["Date"] = []
	data_GA["Epoch"] = []
	for run in range(nModels):
		# open experiments file for this run
		experimentsDB = pd.read_csv("results/GA_test/deltaT_" + str(deltaT) + "/run_" + str(run+1) + "/experimentsDB.csv")
		for epoch in epoch_GA:
			f_timeIdx = b
			cumRet = 1
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

				range_oos = min(deltaT, returnDB_test.shape[0] - f_timeIdx)
				for t in range(range_oos): #loop over each t of the simulation
					cumRet = cumRet*(1+np.dot(w,returnDB_test[f_timeIdx,1:]))
					data_GA["Cumulative Return"].append(cumRet)
					data_GA["Date"].append(dates_test[f_timeIdx])
					data_GA["Epoch"].append(epoch)
					f_timeIdx += 1

		# get index data
		cumRet = 1
		for t in range(b,returnDB_test.shape[0]):
			cumRet = cumRet*(1+returnDB_test[t,0])
			data_GA["Cumulative Return"].append(cumRet)
			data_GA["Date"].append(dates_test[t])
			data_GA["Epoch"].append("Ibovespa")

		#df = pd.DataFrame(data_GA)
		#sns.set_theme(style="ticks", font_scale=1.4)
		#sns_plot = sns.lineplot(data=df, x="Date", y="Cumulative Return", hue="Epoch", palette="inferno")
		#sns_plot.figure.autofmt_xdate()
		#sns_plot.figure.savefig(main_folder  + "obj_" +  obj + "_CumRet_Date.png")
		#sns_plot.figure.clf()

	# consolidated plot
	df = pd.DataFrame(data_GA)
	sns.set_theme(style="ticks", font_scale=1.4)
	sns_plot = sns.lineplot(data=df, x="Date", y="Cumulative Return", hue="Epoch")#, palette="inferno")
	sns_plot.figure.autofmt_xdate()
	sns_plot.figure.savefig(main_folder  + "sim" + str(n_sims) + "/obj_" +  obj + "_CumRet_Date.png")
	sns_plot.figure.clf()

'''
# plot mean out-of-sample tracking error (with sd) x epoch = 1,...,Max Epoch
#  TE =  (portfolio_ret - s[0,t])**2
objs = ["mean", "max"]
data_GA = dict([])
data_GA["TE"] = []
data_GA["Obj"] = []
data_GA["Epoch"] = []
for run in range(nModels):
	experimentsDB = pd.read_csv("results/GA_test/deltaT_" + str(deltaT) + "/run_" + str(run+1) + "/experimentsDB.csv")
	for obj in objs:
		epoch = 20
		while epoch <= max_epoch:
			f_timeIdx = b
			TE = 0
			next_rebal = f_timeIdx
			w = []
			while f_timeIdx + deltaT <= returnDB_test.shape[0]:
				# check if we need to rebalance
				if(f_timeIdx == next_rebal): # if rebalance, then get weights... 
					next_rebal = next_rebal + deltaT
					# get weights
					query = (experimentsDB["f_timeIdx"] == f_timeIdx) & (experimentsDB["obj"] == obj) & (experimentsDB["epoch"] == epoch-1) & (experimentsDB["n_sims"] == n_sims)
					this_row = experimentsDB[query]
					w =  this_row["best_sol"].values[0]
					w =  w.split(",")
					w = [float(raw_w) for raw_w in w[:nAssets-1]]

				for t in range(f): #loop over each t of the simulation
					data_GA["TE"].append((np.dot(w,returnDB_test[f_timeIdx,1:]) - returnDB_test[t,0])**2)
					data_GA["Obj"].append(obj)
					data_GA["Epoch"].append(epoch)
					f_timeIdx += 1

			epoch += epoch_step_size

	#df = pd.DataFrame(data_GA)
	#sns.set_theme(style="ticks", font_scale=1.4)
	#sns_plot = sns.lineplot(data=df, x="Epoch", y="TE", hue="Obj", palette="inferno")
	#sns_plot.figure.autofmt_xdate()
	#sns_plot.figure.savefig(main_folder  + "compareObj_TE_epoch.png")
	#sns_plot.figure.clf()

# consolidated plot
df = pd.DataFrame(data_GA)
sns.set_theme(style="ticks", font_scale=1.4)
sns_plot = sns.lineplot(data=df, x="Epoch", y="TE", hue="Obj")#, palette="inferno")
sns_plot.figure.autofmt_xdate()
sns_plot.figure.savefig(main_folder  + "compareObj_TE_epoch.png")
sns_plot.figure.clf()
'''

'''
# plot mean out-of-sample tracking error (with sd) x epoch = 1,...,Max Epoch 
# for each out-of-sample period 
#  TE =  (portfolio_ret - s[0,t])**2
objs = ["mean", "max"]
data_oos_period = dict([])
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
				if not str(f_timeIdx) in data_oos_period.keys():
					data_GA = dict([])
					data_GA["TE"] = []
					data_GA["Obj"] = []
					data_GA["Epoch"] = []
					data_oos_period[str(f_timeIdx)] = data_GA
				# check if we need to rebalance
				if(f_timeIdx == next_rebal): # if rebalance, then get weights... 
					next_rebal = next_rebal + deltaT
					# get weights
					query = (experimentsDB["f_timeIdx"] == f_timeIdx) & (experimentsDB["obj"] == obj) & (experimentsDB["epoch"] == epoch-1) & (experimentsDB["n_sims"] == n_sims)
					this_row = experimentsDB[query]
					w =  this_row["best_sol"].values[0]
					w =  w.split(",")
					w = [float(raw_w) for raw_w in w[:nAssets-1]]

				data_GA = data_oos_period[str(f_timeIdx)]
				range_oos = min(deltaT, returnDB_test.shape[0] - f_timeIdx)
				for t in range(range_oos): #loop over each t of the simulation
					data_GA["TE"].append((np.dot(w,returnDB_test[f_timeIdx,1:]) - returnDB_test[f_timeIdx,0])**2)
					data_GA["Obj"].append(obj)
					data_GA["Epoch"].append(epoch)
					f_timeIdx += 1

			epoch += epoch_step_size

# consolidated for all periods
for f_timeIdx in data_oos_period.keys():
	data_GA = data_oos_period[f_timeIdx]
	df = pd.DataFrame(data_GA)
	sns.set_theme(style="ticks", font_scale=1.4)
	sns_plot = sns.lineplot(data=df, x="Epoch", y="TE", hue="Obj")#, palette="inferno")
	sns_plot.figure.autofmt_xdate()
	sns_plot.figure.savefig(main_folder  + "sim" + str(n_sims) + "/" + f_timeIdx + "compareObj_TE_epoch.png")
	sns_plot.figure.clf()
'''

# plot mean out-of-sample tracking error (with sd) x epoch = 1,...,Max Epoch 
# for each out-of-sample period 
#  TE =  (portfolio_ret - s[0,t])**2
data_GA = dict([])
data_GA["TE"] = []
data_GA["Obj"] = []
data_GA["Epoch"] = []
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
					f_timeIdx += 1

			epoch += epoch_step_size

# consolidated for all periods
df = pd.DataFrame(data_GA)
sns.set_theme(style="ticks", font_scale=1.4)
sns_plot = sns.lineplot(data=df, x="Epoch", y="TE", hue="Obj")#, palette="inferno")
sns_plot.figure.autofmt_xdate()
sns_plot.figure.savefig(main_folder  + "sim" + str(n_sims) + "/compareObj_TE_epoch.png")
sns_plot.figure.clf()


# plot, for max_epoch of each obj function type, out-of-sample tracking error x time

# plot out-of-sample TE x time for 3 epoch values
data_oos_period = dict([])
for run in range(nModels):
	experimentsDB = pd.read_csv("results/GA_test/deltaT_" + str(deltaT) + "/run_" + str(run+1) + "/experimentsDB.csv")
	for epoch in epoch_GA:
		for obj in objs:
			if not obj in data_oos_period.keys():
				data_GA = dict([])
				data_GA["TE"] = []
				data_GA["rebalance date"] = []
				data_GA["Epoch"] = []
			else:
				data_GA = data_oos_period[obj]

			f_timeIdx = b
			TE = 0
			next_rebal = f_timeIdx       
			w = []
			while f_timeIdx < returnDB_test.shape[0]:
				current_f = f_timeIdx
				# check if we need to rebalance
				if(f_timeIdx == next_rebal): # if rebalance, then get weights... 
					next_rebal = next_rebal + deltaT
					# get weights
					query = (experimentsDB["f_timeIdx"] == f_timeIdx) & (experimentsDB["obj"] == obj) & (experimentsDB["epoch"] == epoch-1) & (experimentsDB["n_sims"] == n_sims)
					this_row = experimentsDB[query]
					w =  this_row["best_sol"].values[0]
					w =  w.split(",")
					w = [float(raw_w) for raw_w in w[:nAssets-1]]

				
				range_oos = min(deltaT, returnDB_test.shape[0] - f_timeIdx)
				for t in range(range_oos): #loop over each t of the simulation
					data_GA["TE"].append((np.dot(w,returnDB_test[f_timeIdx,1:]) - returnDB_test[f_timeIdx,0])**2)
					data_GA["rebalance date"].append(dates_test[current_f])
					data_GA["Epoch"].append(epoch)
					f_timeIdx += 1
			data_oos_period[obj] = data_GA


# consolidated for all periods
for obj in data_oos_period.keys():
	df = pd.DataFrame(data_oos_period[obj])
	sns.set_theme(style="ticks", font_scale=1.4)
	sns_plot = sns.lineplot(data=df, x="rebalance date", y="TE", hue = "Epoch", palette="Dark2")
	#sns_plot.figure.autofmt_xdate()
	sns_plot.figure.savefig(main_folder  + "sim" + str(n_sims) + "/obj_" + obj + "_TE_outOfSample.png")
	sns_plot.figure.clf()


