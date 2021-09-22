# utilities
import random
from datetime import datetime
import os
# importing data science and visualization libraries
import pandas as pd
import numpy as np
import seaborn as sns
# PyTorch libs
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.autograd as autograd
# GA
from metaheuristics.GA.simpleGA import simpleGA 
from utils import *

import matplotlib.pyplot as plt
import math as mth

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
n_sims = expParameters["n_sims"]

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

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0
USE_CUDA = False

[allMb, allMf] = get_dataSet(TSData = returnDB, b = b, f = f, step = 1)
torch_Mb = torch.from_numpy(np.array(allMb)).float()
torch_Mf = torch.from_numpy(np.array(allMf)).float()
M_raw = torch.cat((torch_Mb, torch_Mf), dim = 2)
M_train = M_raw[0:M_raw.shape[0] - test_size,:,:]
M_test = M_raw[M_raw.shape[0] - test_size:,:,:]  # dim1: number of examples, dim2: num assets, dim3: return time series size (w)
#print(M_test[0].shape)
#print("M_raw: " + str(M_raw.shape))
#print("M_train:" + str(M_train.shape))
#print(M_train)
#print("M_test:" + str(M_test.shape))
#print(M_test)

# load simulations for each epoch
filepath_model = "models/" + str(total_epochs) + "_epochs/"
simulations = dict([])
screenshots_epochs = [50, 1000, 2500]
screenshots_assets = [0, 10, 15, 26, 35]
screenshots_model = dict([])
screenshots_model["Epoch"] = []
screenshots_model["Model"] = []
screenshots_model["Loss"] = []
screenshots_model["Exec"] = []
returnDB_ = returnDB.values
returnDB_test = returnDB_[returnDB_.shape[0]-test_size:,:]
dates_test = dates[returnDB_.shape[0]-test_size + 1:] # +1 to adjust to return data
print("test period: " + str(dates_test[0]) + " - " + str(dates_test[len(dates_test)-1]))

# loop over all models
for run in range(nModels):
	epoch = delta_epoch - 1
	while epoch < total_epochs:
		# initialize a generator instance from saved checkpoints
		# load the model state
		filename = filepath_model + "run_" + str(run+1) + "/GAN_state_" + str(epoch+1) + ".pth"
		print("running simulations using " + filename)
		if not torch.cuda.is_available():
			checkpoint = torch.load(filename, map_location = "cpu")
		else:
			checkpoint = torch.load(filename)
		# create an instance of the model
		generator = Generator(nAssets,f)
		generator.load_state_dict(checkpoint['G_state_dict'])
		screenshots_model["Epoch"].append(epoch+1)
		screenshots_model["Model"].append("Generator")
		screenshots_model["Loss"].append(float(checkpoint['G_loss'].cpu().detach().numpy()))
		screenshots_model["Exec"].append(run+1)
		screenshots_model["Epoch"].append(epoch+1)
		screenshots_model["Model"].append("Discriminator")
		screenshots_model["Loss"].append(float(checkpoint['D_loss'].cpu().detach().numpy()))
		screenshots_model["Exec"].append(run+1)

		current_epoch = epoch
		epoch += delta_epoch
		'''
		# uncomment this if you want to save simulation plots into img files
		# generate simulations
		f_timeIdx = b
		while f_timeIdx + deltaT <= M_test.shape[0]:
			Mb = M_test[f_timeIdx:f_timeIdx+1,:,0:b]
			#Mf_real = M_test[f_timeIdx:f_timeIdx+1,:,b:f]
			sim_data = []
			# run simulations for this window (Mb)
			for sim in range(n_sims):
				Mf_fake = generator(Mb)
				Mf_fake = Mf_fake[0].cpu().detach().numpy()
				#print(Mf_fake.shape)
				sim_data.append(Mf_fake)

			simulations[str(f_timeIdx) + str(current_epoch)] = sim_data

			f_timeIdx += deltaT

	screenshots_dates_idxs = [b, len(dates_test)-deltaT] # save plots in the first rebalancing period and in the last rebalacing period
	fig, axes = plt.subplots(len(screenshots_dates_idxs), len(screenshots_assets), figsize=(7, 20), sharex=False)
	screenshots_data = dict([])
	screenshots_data["Date"] = []
	screenshots_data["Epoch"] = []
	screenshots_data["Return"] = []
	screenshots_data["Data type"] = []
	screenshots_data["Ticker"] = []
	screenshots_data["Initial time"] = []
	for screenshots_time in screenshots_dates_idxs: 
		returnDB_vals = returnDB.values
		#print(returnDB.shape)
		for epoch in screenshots_epochs:
			sims = simulations[str(screenshots_time) + str(epoch-1)]
			# sample a simulation from sims
			sim = random.choice(sims)
			for assetIdx in screenshots_assets:
				dateIdx = screenshots_time
				for val in sim[assetIdx,:]:
					screenshots_data["Date"].append(dates_test[dateIdx])
					screenshots_data["Epoch"].append(str(epoch))
					screenshots_data["Return"].append(val)
					screenshots_data["Data type"].append("simulated")
					screenshots_data["Ticker"].append(symbols[assetIdx])
					screenshots_data["Initial time"].append(screenshots_time)
					dateIdx += 1
					#print(dateIdx)

		#sns.set_theme(style='whitegrid')
		#sns.set(font_scale = 0.7)
		#sns.set(rc={'figure.figsize':(11.7,8.27)})
		for assetIdx in screenshots_assets:
			print("getting screenshots for:" + symbols[assetIdx])
			dateIdx = screenshots_time
			# get real data
			for val in range(deltaT):
				screenshots_data["Date"].append(dates_test[dateIdx])
				screenshots_data["Epoch"].append("real data")
				screenshots_data["Return"].append(returnDB_test[dateIdx, assetIdx])
				screenshots_data["Data type"].append("real")
				screenshots_data["Ticker"].append(symbols[assetIdx])
				screenshots_data["Initial time"].append(screenshots_time)
				dateIdx += 1
				#sns.set_theme(style="ticks", font_scale=1.4)
				#sns_plot = sns.lineplot(data=df, x="Date", y="Return", hue="Epoch", style = "data_type", palette="inferno").set_title(symbols[assetIdx] + " (" + str(dates_test[screenshots_time]) + " - " + str(dates_test[screenshots_time+deltaT-1]) +  ")")
				#sns_plot._legend.remove()
				#sns_plot.figure.autofmt_xdate()

	df = pd.DataFrame(screenshots_data)
	screen_shot_path = "results/screen_shots_simulations/run_" + str(run+1) + "/"
	if not os.path.exists(screen_shot_path):
		os.makedirs(screen_shot_path)
	linestyle = ["-" for epoch in screenshots_epochs]
	linestyle.append("--")
	kw = {'ls' : linestyle}
	g = sns.FacetGrid(df, row = 'Initial time', col='Ticker', hue='Epoch', sharey = False, sharex = False, hue_kws = kw)
	g1 = g.map(sns.lineplot, 'Date', 'Return').add_legend()
	for axes in g.axes.flat:
		_ = axes.set_xticklabels(axes.get_xticklabels(), rotation=45)
	g.tight_layout()
	g.savefig(screen_shot_path + "consolidated_model" + str(run+1) + ".png")
	plt.close()
	'''

# save model loss plot 
#fig, axes = plt.plot(figsize=(7, 20))
df = pd.DataFrame(screenshots_model)
sns.set_theme(style="ticks", font_scale=1.4)
sns_plot = sns.lineplot(data=df, x="Epoch", y="Loss", hue="Model", palette="inferno")
sns_plot.figure.savefig("results/screen_shots_simulations/modelLoss.png")
sns_plot.figure.clf()







