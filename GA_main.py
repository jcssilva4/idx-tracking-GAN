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
# set GAN parameters
total_epochs =  expParameters["total_epochs"]
delta_epoch =  expParameters["delta_epoch"]
nModels =  expParameters["nModels"]
w = expParameters["w"]
b = expParameters["b"]
f = w - b
filepath_model = "models/" + str(total_epochs) + "_epochs/"
# set test_parameters
deltaT =  expParameters["deltaT"]
test_size = expParameters["test_size"]
# set ga with gan parameters
objs =  expParameters["objs"]
n_sims = expParameters["n_sims"]
# set portfolio problem
model_pars = dict([])
model_pars["K"] = expParameters["K"]
model_pars["lb"] = expParameters["lb"]
model_pars["ub"] = expParameters["ub"]

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

# perform simulations for each epoch
simulations = dict([])
returnDB_ = returnDB.values
returnDB_test = returnDB_[returnDB_.shape[0]-test_size:,:]
dates_test = dates[returnDB_.shape[0]-test_size + 1:] # +1 to adjust to return data
print("test period: " + str(dates_test[0]) + " - " + str(dates_test[len(dates_test)-1]))

#print(M_test.shape)
#print(M_test[60])
for run in range(nModels):
	experimentsDB = dict([])
	experimentsDB["epoch"] = []
	experimentsDB["f_timeIdx"] = []
	experimentsDB["n_sims"] = []
	experimentsDB["obj"] = []
	experimentsDB["best_objval"] = []
	experimentsDB["best_sol"] = []
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
		generator = Generator(nAssets, f)
		generator.load_state_dict(checkpoint['G_state_dict'])

		# generate simulations
		f_timeIdx = b
		#while f_timeIdx + deltaT <= M_test.shape[0]:
		while f_timeIdx < M_test.shape[0]:
			Mb = M_test[f_timeIdx-1:f_timeIdx,:,f:] # we do f: instead of 0:b, because there is no real_market part for the discriminator now...
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
				best_TE, best_sol = simpleGA(S = sim_data, model_pars = model_pars, nGenerations = 200)
				#write GA solutions for this epoch and this f_timeIdx
				experimentsDB["epoch"].append(epoch)
				experimentsDB["f_timeIdx"].append(f_timeIdx)
				experimentsDB["n_sims"].append(n_sims)
				experimentsDB["obj"].append(obj)
				experimentsDB["best_objval"].append(best_TE)
				best_sol = str(best_sol.tolist())
				best_sol = best_sol.replace('[',"").replace("]","")
				experimentsDB["best_sol"].append(best_sol)
				'''
				if not os.path.exists(filepath):
					os.makedirs(filepath) # create this path if it not exists
				filehandle = open(filepath + "best_objval.txt", 'w')
				filehandle.write(str(best_TE))
				filehandle.close()
				filehandle = open(filepath + "best_sol.txt", 'w')
				for var_port in best_sol:
					filehandle.write(str(var_port) + ", ")
				filehandle.close()
				'''
			

			f_timeIdx += deltaT

		epoch += delta_epoch
	# save the final file
	df_exp = pd.DataFrame.from_dict(experimentsDB)
	df_exp.to_csv("results/GA_test/deltaT_" + str(deltaT) + "/run_" + str(run+1) + "/experimentsDB.csv")
