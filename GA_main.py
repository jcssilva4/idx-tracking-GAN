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
'''
# pytorch examples #
https://github.com/pytorch/examples/blob/master/dcgan/main.py
https://louisenaud.github.io/time_series_prediction.html
https://github.com/GitiHubi/deepAI/blob/master/GTC_2018_Lab.ipynb 
https://colab.research.google.com/github/GitiHubi/deepAD/blob/master/KDD_2019_Lab.ipynb--> [Schreyer et al., 2019]

Wasserstein GANs with Gradient Penalty: https://github.com/eriklindernoren/PyTorch-GAN
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
'''

# transformed TS data set
def get_dataSet(TSData, b, f, step):
	allMb = [] # Mb belongs to lookback data (in-sample)
	allMf = [] # Mf belongs to forward data (out-of-sample)
	t = 0 # current time index
	datasetIdx = 0
	allPrices = TSData.values
	while t <= TSData.shape[0] - (b + f): 
		Mb = allPrices[t : (t + b),:]
		#print("before: " + str(Mb))
		#print("after: " + str(np.transpose(Mb)))
		Mf = allPrices[t + b : (t + b + f),:]
		allMb.append(np.transpose(Mb))
		allMf.append(np.transpose(Mf))
		t += step
	
	return allMb, allMf

class Discriminator(nn.Module):
	def __init__(self, nAssets):
		'''
		:param nAssets: int, number of stocks
        :param T: int, number of look back points b associated with M_b
		'''
		# The super call delegates the function call to the parent class, which is nn.Module in your case.
		# This is needed to initialize the nn.Module properly. Have a look at the Python docs 175 for more information.
		super(Discriminator, self).__init__()

		self.crit_conv1 = nn.Conv1d(in_channels = nAssets, out_channels = 2*nAssets, kernel_size = 4, stride=2, padding = 1)
		self.leakyrelu1 = nn.LeakyReLU()
		self.crit_conv2 = nn.Conv1d(in_channels = 2*nAssets, out_channels = 4*nAssets, kernel_size = 4, stride=2, padding = 1)
		self.leakyrelu2 = nn.LeakyReLU()
		self.crit_conv3 = nn.Conv1d(in_channels = 4*nAssets, out_channels = 8*nAssets, kernel_size = 4, stride=2, padding = 1)
		self.leakyrelu3 = nn.LeakyReLU()
		self.crit_conv4 = nn.Conv1d(in_channels = 8*nAssets, out_channels = 16*nAssets, kernel_size = 4, stride=2, padding = 1)
		self.leakyrelu4 = nn.LeakyReLU()
		self.crit_conv5 = nn.Conv1d(in_channels = 16*nAssets, out_channels = 32*nAssets, kernel_size = 4, stride=2, padding = 1)
		self.leakyrelu5 = nn.LeakyReLU()
		self.crit_dense6 = nn.Linear(in_features =  32*nAssets, out_features = 1)
		self.leakyrelu6 = nn.LeakyReLU()

	def forward(self, x):
		#print("input: " + str(x.shape))
		# Layer 1
		out = self.leakyrelu1(self.crit_conv1(x))
		#print("ouput layer 1: " + str(out.shape))

		# Layer 2:
		out = self.leakyrelu2(self.crit_conv2(out))
		#print("ouput layer 2: " + str(out.shape))

		# Layer 3:
		out = self.leakyrelu3(self.crit_conv3(out))
		#print("ouput layer 3: " + str(out.shape))

		# Layer 4:
		out = self.leakyrelu4(self.crit_conv4(out))
		#print("ouput layer 4: " + str(out.shape))

		# Layer 5:
		out = self.leakyrelu5(self.crit_conv5(out))
		#print("ouput layer 5: " + str(out.shape))

		# Final layer - critic value
		out = torch.flatten(out, start_dim = 1, end_dim =2)
		out = self.leakyrelu6(self.crit_dense6(out))
		#print("ouput final layer: " + str(out.shape))

		return out

class Generator(nn.Module):
	def __init__(self, nAssets):
		'''
		:param nAssets: int, number of stocks
        :param T: int, number of look back points b associated with M_b
		'''
		# The super call delegates the function call to the parent class, which is nn.Module in your case.
		# This is needed to initialize the nn.Module properly. Have a look at the Python docs 175 for more information.
		super(Generator, self).__init__()

		'''
		Conditioning
		'''
		# first layer
		# input 
		self.cond_conv1 = nn.Conv1d(in_channels = nAssets, out_channels = 2*nAssets, kernel_size = 5, stride=2, padding = 1)
		self.relu1 = nn.ReLU()

		# second layer
		self.cond_conv2 = nn.Conv1d(in_channels = 2*nAssets, out_channels = 2*nAssets, kernel_size = 5, stride=2, padding = 1)
		self.relu2 = nn.ReLU()

		# third layer
		self.cond_conv3 = nn.Conv1d(in_channels = 2*nAssets, out_channels = 2*nAssets, kernel_size = 5, stride=2, padding = 1)
		self.relu3 = nn.ReLU()  

		# fourth layer
		self.cond_conv4 = nn.Conv1d(in_channels = 2*nAssets, out_channels = 2*nAssets, kernel_size = 5, stride=2, padding = 1)
		self.relu4 = nn.ReLU()

		# output layer fully connected / dense
		self.cond_dense5 = nn.Linear(in_features = 2*nAssets, out_features = nAssets)
		self.relu5 = nn.ReLU()

		'''
		Simulator
		'''
		# first layer - dense
		self.sim_dense1 = nn.Linear(in_features = 3*nAssets, out_features = f*nAssets)
		self.relu1 = nn.ReLU()

		# second layer
		self.sim_conv2 = nn.ConvTranspose1d(in_channels = 4*nAssets, out_channels = 2*nAssets, kernel_size = 4, stride=2, padding = 1)
		self.relu2 = nn.ReLU()

		# third layer
		self.sim_conv3 = nn.ConvTranspose1d(in_channels = 2*nAssets, out_channels = nAssets, kernel_size = 4, stride=2, padding = 1)
		self.relu3 = nn.ReLU()  

	def forward(self, x):
		'''
		conditioning forward pass
		'''
		# Layer 1
		out = self.relu1(self.cond_conv1(x))

		# Layer 2:
		out = self.relu2(self.cond_conv2(out))

		# Layer 3:
		out = self.relu3(self.cond_conv3(out))

		# Layer 4:
		out = self.relu4(self.cond_conv4(out))

		# Final layer
		out = torch.flatten(out, start_dim = 1, end_dim =2)
		cond_out = self.relu5(self.cond_dense5(out))

		'''
		simulator forward pass
		'''
		# sample a prior latent vector from a normal distribution
		lambd_prior = torch.from_numpy(np.array([np.random.normal(size = 2*nAssets) for train_example in range(cond_out.shape[0])])).float()
		#print("prior shape: " + str(lambd_prior.shape))
		# concat cond_out with the latent vector sampled from the prior
		sim_input = torch.cat((cond_out, lambd_prior), dim = 1)
		#print("concat shape: " + str(sim_input.shape))

		# First layer
		#print("first") 
		out = self.relu1(self.sim_dense1(sim_input))

		# Layer 2:
		# unflatten the out tensor
		out = out.view(out.shape[0], 4*nAssets, -1) #https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71
		#print("unflatten shape: " + str(out.shape))
		#print("second_sim")
		out = self.relu2(self.sim_conv2(out))
		#print(out.shape)

		# Layer 3:
		#print("third_sim")
		out = self.relu3(self.sim_conv3(out))
		#print(out.shape)

		return out

Tensor = torch.FloatTensor

######################################## Main Script ################################################

# Set random seed for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
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
w = 60
b = 40
f = w - b

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0
USE_CUDA = False

[allMb, allMf] = get_dataSet(TSData = returnDB, b = b, f = f, step = 1)
torch_Mb = torch.from_numpy(np.array(allMb)).float()
torch_Mf = torch.from_numpy(np.array(allMf)).float()
M_raw = torch.cat((torch_Mb, torch_Mf), dim = 2)
test_size = 120
M_train = M_raw[0:M_raw.shape[0] - test_size,:,:]
M_test = M_raw[M_raw.shape[0] - test_size:,:,:]  # dim1: number of examples, dim2: num assets, dim3: return time series size (w)
#print(M_test[0].shape)
#print("M_raw: " + str(M_raw.shape))
#print("M_train:" + str(M_train.shape))
#print(M_train)
#print("M_test:" + str(M_test.shape))
#print(M_test)

# load simulations for each epoch
filepath_model = "models/final/"
total_epochs = 250
delta_epoch = 10
n_sims = 30
simulations = dict([])
screenshots_epochs = [10, 50, 100, 200, 250]
screenshots_assets = [0, 10, 15, 26, 35]
screenshots_model = dict([])
screenshots_model["Epoch"] = []
screenshots_model["Model"] = []
screenshots_model["Loss"] = []
returnDB_ = returnDB.values
returnDB_test = returnDB_[returnDB_.shape[0]-test_size:,:]
dates_test = dates[returnDB_.shape[0]-test_size + 1:] # +1 to adjust to return data
print("test period: " + str(dates_test[0]) + " - " + str(dates_test[len(dates_test)-1]))
epoch = 9
while epoch < total_epochs:
	# initialize a generator instance from saved checkpoints
	# load the model state
	filename = filepath_model + "GAN_state_" + str(epoch+1) + ".pth"
	print("running simulations using " + filename)
	checkpoint = torch.load(filename)
	# create an instance of the model
	generator = Generator(nAssets)
	generator.load_state_dict(checkpoint['G_state_dict'])
	screenshots_model["Epoch"].append(epoch+1)
	screenshots_model["Model"].append("Generator")
	screenshots_model["Loss"].append(float(checkpoint['G_loss'].cpu().detach().numpy()))
	screenshots_model["Epoch"].append(epoch+1)
	screenshots_model["Model"].append("Discrimnator")
	screenshots_model["Loss"].append(float(checkpoint['D_loss'].cpu().detach().numpy()))

	# generate simulations
	deltaT = f
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

		simulations[str(f_timeIdx) + str(epoch)] = sim_data
		
		#run GA
		for obj in objs:
			# set problem obj
			model_pars["obj"] = obj # "mean" or "max"
			#get GA solutions with obj = "mean"
			best_TE, best_sol = simpleGA(S = sim_data, model_pars = model_pars, nGenerations = 100)
			#write GA solutions for this epoch and this f_timeIdx
			filepath = "results/GA_test/" + obj + "/epoch_" + str(epoch) + "/f_timeIdx_" + str(f_timeIdx) + "/"
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

	epoch += delta_epoch


# save plots into img files
screenshots_dates_idxs = [b, len(dates_test)-deltaT] # save plots in the first rebalancing period and in the last rebalacing period
for screenshots_time in screenshots_dates_idxs: 
	screenshots_data = dict([])
	returnDB_vals = returnDB.values
	print(returnDB.shape)
	for epoch in screenshots_epochs:
		sims = simulations[str(screenshots_time) + str(epoch-1)]
		# sample a simulation from sims
		sim = random.choice(sims)
		for assetIdx in screenshots_assets:
			this_asset_data = dict([])
			if not symbols[assetIdx] in screenshots_data.keys():
				this_asset_data["Date"] = []
				this_asset_data["Epoch"] = []
				this_asset_data["Return"] = []
			else:
				this_asset_data = screenshots_data[symbols[assetIdx]]
			dateIdx = screenshots_time
			for val in sim[assetIdx,:]:
				this_asset_data["Date"].append(dates_test[dateIdx])
				this_asset_data["Epoch"].append(str(epoch))
				this_asset_data["Return"].append(val)
				dateIdx += 1
				#print(dateIdx)
			screenshots_data[symbols[assetIdx]] = this_asset_data

	sns.set_theme(style='whitegrid')
	sns.set(font_scale = 0.7)
	sns.set(rc={'figure.figsize':(11.7,8.27)})
	for assetIdx in screenshots_assets:
		print("getting screenshots for:" + symbols[assetIdx])
		this_asset_data = screenshots_data[symbols[assetIdx]]
		dateIdx = screenshots_time
		# get real data
		for val in range(deltaT):
			this_asset_data["Date"].append(dates_test[dateIdx])
			this_asset_data["Epoch"].append("real data")
			this_asset_data["Return"].append(returnDB_test[dateIdx, assetIdx])
			dateIdx += 1
		df = pd.DataFrame(this_asset_data)
		sns.set_theme(style="ticks", font_scale=1.4)
		sns_plot = sns.lineplot(data=df, x="Date", y="Return", hue="Epoch", style = "Epoch", palette="inferno").set_title(symbols[assetIdx] + " (" + str(dates_test[screenshots_time]) + " - " + str(dates_test[screenshots_time+deltaT-1]) +  ")")
		sns_plot.figure.autofmt_xdate()
		sns_plot.figure.savefig("results/screen_shots_simulations/" + symbols[assetIdx] + "_" + str(screenshots_time) + ".png")
		sns_plot.figure.clf()

# save model loss plot		
df = pd.DataFrame(screenshots_model)
sns.set_theme(style="ticks", font_scale=1.4)
sns_plot = sns.lineplot(data=df, x="Epoch", y="Loss", hue="Model", palette="inferno")
sns_plot.figure.savefig("results/screen_shots_simulations/modelLoss.png")
sns_plot.figure.clf()






