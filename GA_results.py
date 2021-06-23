import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import os
import random

GA_data_mean = dict([])
GA_data_max = dict([])

# Set random seed for reproducibility
manualSeed = 999
random.seed(manualSeed)
np.random.seed(manualSeed)

# portfolio problem
model_pars = dict([])
model_pars["K"] = 10
model_pars["lb"] = 0
model_pars["ub"] = 1
objs = ["mean", "max"]

# get the dataset
ibovDB = pd.read_excel("data/IBOV_DB_useThis.xlsx") 
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
test_size = 120
w = 60
b = 40
f = w - b
returnDB_ = returnDB.values
returnDB_test = returnDB_[returnDB_.shape[0]-test_size:,:]
dates_test = dates[returnDB_.shape[0]-test_size + 1:] # +1 to adjust to return data

sns.set(rc={'figure.figsize':(9.7,6.27)})

main_folder = "results/GA_final/"
max_epoch = 250
epoch_step_size = 10

# plot out-of-sample tracking error x time (multiple epochs)
epoch_GA = [1, 10, 25, 40, 50]
objs = ["mean", "max"]
deltaT = f
for obj in objs:
	data_GA = dict([])
	data_GA["Cumulative Return"] = []
	data_GA["Date"] = []
	data_GA["Epoch"] = []
	for epoch in epoch_GA:
		f_timeIdx = b
		cumRet = 1
		next_rebal = f_timeIdx
		w = []
		while f_timeIdx + deltaT <= returnDB_test.shape[0]:
			# check if we need to rebalance
			if(f_timeIdx == next_rebal): # if rebalance, then get weights... 
				w = []
				filepath = main_folder + obj + "/epoch_" + str(epoch-1) + "/f_timeIdx_" + str(f_timeIdx) + "/"
				filehandle = open(filepath + "best_sol.txt", 'r')
				next_rebal = next_rebal + f
				# get weights
				w =  [raw_w for raw_w in filehandle]
				w =  [raw_w.replace(" ", "") for raw_w in w[0].split(",")]
				w = [float(raw_w) for raw_w in w[:nAssets-1]]
				#print("weights: " + str(w))

			for t in range(f): #loop over each t of the simulation
				cumRet = cumRet*(1+np.dot(w,returnDB_test[f_timeIdx,1:]))
				data_GA["Cumulative Return"].append(cumRet - 1)
				data_GA["Date"].append(dates_test[f_timeIdx])
				data_GA["Epoch"].append(epoch)
				f_timeIdx += 1

	# get index data
	cumRet = 1
	for t in range(b,returnDB_test.shape[0]):
		cumRet = cumRet*(1+returnDB_test[t,0])
		data_GA["Cumulative Return"].append(cumRet - 1)
		data_GA["Date"].append(dates_test[t])
		data_GA["Epoch"].append("Ibovespa")

	df = pd.DataFrame(data_GA)
	sns.set_theme(style="ticks", font_scale=1.4)
	sns_plot = sns.lineplot(data=df, x="Date", y="Cumulative Return", hue="Epoch", palette="inferno")
	sns_plot.figure.autofmt_xdate()
	sns_plot.figure.savefig(main_folder  + "obj_" +  obj + "_CumRet_Date.png")
	sns_plot.figure.clf()


# plot mean out-of-sample tracking error (with sd) x epoch = 1,...,Max Epoch
#  TE =  (portfolio_ret - s[0,t])**2
objs = ["mean", "max"]
deltaT = f
data_GA = dict([])
data_GA["TE"] = []
data_GA["Obj"] = []
data_GA["Epoch"] = []
for obj in objs:
	epoch = 10
	while epoch <= max_epoch:
		f_timeIdx = b
		TE = 0
		next_rebal = f_timeIdx
		w = []
		while f_timeIdx + deltaT <= returnDB_test.shape[0]:
			# check if we need to rebalance
			if(f_timeIdx == next_rebal): # if rebalance, then get weights... 
				w = []
				filepath = main_folder + obj + "/epoch_" + str(epoch-1) + "/f_timeIdx_" + str(f_timeIdx) + "/"
				filehandle = open(filepath + "best_sol.txt", 'r')
				next_rebal = next_rebal + f
				# get weights
				w =  [raw_w for raw_w in filehandle]
				w =  [raw_w.replace(" ", "") for raw_w in w[0].split(",")]
				w = [float(raw_w) for raw_w in w[:nAssets-1]]
				#print("weights: " + str(w))

			for t in range(f): #loop over each t of the simulation
				data_GA["TE"].append((np.dot(w,returnDB_test[f_timeIdx,1:]) - returnDB_test[t,0])**2)
				data_GA["Obj"].append(obj)
				data_GA["Epoch"].append(epoch)
				f_timeIdx += 1

		epoch += epoch_step_size

df = pd.DataFrame(data_GA)
sns.set_theme(style="ticks", font_scale=1.4)
sns_plot = sns.lineplot(data=df, x="Epoch", y="TE", hue="Obj", palette="inferno")
sns_plot.figure.autofmt_xdate()
sns_plot.figure.savefig(main_folder  + "compareObj_TE_epoch.png")
sns_plot.figure.clf()

# plot, for max_epoch of each obj function type, out-of-sample tracking error x time


