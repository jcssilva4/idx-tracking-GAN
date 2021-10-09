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

model_eval_epochs = [2499]
gen_evals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#gen_evals = [10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 300]
all_sims = [5,10,20]
#all_sims = [5]

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
running_experiments = False
if running_experiments:
	for run in range(nModels):
		experimentsDB = dict([])
		experimentsDB["epoch"] = []
		experimentsDB["f_timeIdx"] = []
		experimentsDB["n_gens"] = []
		experimentsDB["obj"] = []
		experimentsDB["n_sims"] = []
		experimentsDB["best_objval"] = []
		experimentsDB["best_sol"] = []
		for epoch in model_eval_epochs:
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
			Mb = M_test[f_timeIdx-1:f_timeIdx,:,f:] # we do f: instead of 0:b, because we use f:w now (getting the last b values of each time series)
			#Mf_real = M_test[f_timeIdx:f_timeIdx+1,:,b:f]
			for n_sims in all_sims:
				print(n_sims)
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
					for n_gens in gen_evals:
						# set problem obj
						model_pars["obj"] = obj # "mean" or "max"
						#get GA solutions with obj = "mean"
						best_TE, best_sol = simpleGA(S = sim_data, model_pars = model_pars, nGenerations = n_gens)
						#write GA solutions for this epoch and this f_timeIdx
						experimentsDB["epoch"].append(epoch)
						experimentsDB["f_timeIdx"].append(f_timeIdx)
						experimentsDB["n_gens"].append(n_gens)
						experimentsDB["obj"].append(obj)
						experimentsDB["n_sims"].append(n_sims)
						experimentsDB["best_objval"].append(best_TE)
						best_sol = str(best_sol.tolist())
						best_sol = best_sol.replace('[',"").replace("]","")
						experimentsDB["best_sol"].append(best_sol)


		# save the final file
		df_exp = pd.DataFrame.from_dict(experimentsDB)
		df_exp.to_csv("results/GA_test/deltaT_" + str(deltaT) + "/run_" + str(run+1) + "/experimentsDB_modeleval.csv")

		experimentsDB = dict([])
		experimentsDB["f_timeIdx"] = []
		experimentsDB["b_size"] = []
		experimentsDB["n_gens"] = []
		experimentsDB["best_objval"] = []
		experimentsDB["best_sol"] = []
		print("Execution " + str(run+1))
		obj = "hist_TE"
		for n_gens in gen_evals:
			for b_size in [40, 60, 80, 100, 120]:
				f_timeIdx = b
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
				best_TE, best_sol = simpleGA(S = hist_data, model_pars = model_pars, nGenerations = n_gens)
				#write GA solutions for this epoch and this f_timeIdx
				experimentsDB["f_timeIdx"].append(f_timeIdx)
				experimentsDB["b_size"].append(b_size)
				experimentsDB["n_gens"].append(n_gens)
				experimentsDB["best_objval"].append(best_TE) 
				best_sol = str(best_sol.tolist())
				best_sol = best_sol.replace('[',"").replace("]","")
				experimentsDB["best_sol"].append(best_sol)

				f_timeIdx += deltaT

		# save the final file
		df_exp = pd.DataFrame.from_dict(experimentsDB)
		df_exp.to_csv("results/GA_test/deltaT_" + str(deltaT) + "/run_" + str(run+1) + "/benchmark_experimentsDB_modeleval.csv")
		
# plot some figures
else:
	print("plots")
	# plot mean out-of-sample tracking error (with sd) x epoch = 1,...,Max Epoch 
	# for each out-of-sample period 
	#  TE =  (portfolio_ret - s[0,t])**2
	for obj in objs:
		for epoch in model_eval_epochs:
			data_GA = dict([])
			data_GA["TE_insample"] = []
			data_GA["TE_outsample"] = []
			data_GA["Generations"] = []
			data_GA["n_sims"] = []
			for run in range(nModels):
				experimentsDB = pd.read_csv("results/GA_test/deltaT_" + str(deltaT) + "/run_" + str(run+1) + "/experimentsDB_modeleval.csv")
				f_timeIdx = b
				for n_sims in all_sims:
					for n_gens in gen_evals:
						query = (experimentsDB["f_timeIdx"] == f_timeIdx) & (experimentsDB["obj"] == obj) & (experimentsDB["epoch"] == epoch) & (experimentsDB["n_gens"] == n_gens) & (experimentsDB["n_sims"] == n_sims) 
						this_row = experimentsDB[query]
						insample_TE =  this_row["best_objval"].values[0]
						data_GA["TE_insample"].append(insample_TE)
						data_GA["Generations"].append(n_gens)
						data_GA["n_sims"].append(n_sims)
						# get weights
						this_row = experimentsDB[query]
						w =  this_row["best_sol"].values[0]
						w =  w.split(",")
						w = [float(raw_w) for raw_w in w[:nAssets-1]]

						range_oos = min(deltaT, returnDB_test.shape[0] - f_timeIdx)
						TE_out = []
						for t in range(range_oos): #loop over each t of the simulation
							TE_out.append((np.dot(w,returnDB_test[f_timeIdx+t,1:]) - returnDB_test[f_timeIdx+t,0])**2)
						data_GA["TE_outsample"].append(np.mean(TE_out))

			# consolidated for all periods
			df = pd.DataFrame(data_GA)
			sns.set_theme(style="ticks", font_scale=1.4)
			# in sample
			sns_plot = sns.lineplot(data=df, x="Generations", y="TE_insample", hue="n_sims")#, palette="inferno")
			sns_plot.figure.autofmt_xdate()
			sns_plot.figure.savefig("results/GA_test/deltaT_" + str(deltaT) + "/model_eval/" + obj + str(epoch+1) + "_TE_insample.png")
			sns_plot.figure.clf()
			# out of sample
			sns_plot = sns.lineplot(data=df, x="Generations", y="TE_outsample", hue="n_sims")#, palette="inferno")
			sns_plot.figure.autofmt_xdate()
			sns_plot.figure.savefig("results/GA_test/deltaT_" + str(deltaT) + "/model_eval/" + obj + str(epoch+1) + "_TE_outsample.png")
			sns_plot.figure.clf()
	
	data_GA = dict([])
	data_GA["TE_insample"] = []
	data_GA["TE_outsample"] = []
	data_GA["b_size"] = []
	data_GA["Generations"] = []
	obj = "hist_TE"
	for run in range(nModels):
		experimentsDB = pd.read_csv("results/GA_test/deltaT_" + str(deltaT) + "/run_" + str(run+1) + "/benchmark_experimentsDB_modeleval.csv")
		for n_gens in gen_evals:
			for b_size in [40]:
				f_timeIdx = b
				query = (experimentsDB["f_timeIdx"] == f_timeIdx) & (experimentsDB["b_size"] == b_size)  & (experimentsDB["n_gens"] == n_gens)
				this_row = experimentsDB[query]
				insample_TE =  this_row["best_objval"].values[0]
				data_GA["TE_insample"].append(insample_TE)
				data_GA["Generations"].append(n_gens)
				data_GA["b_size"].append(b_size)
				# get weights
				w =  this_row["best_sol"].values[0]
				w =  w.split(",")
				w = [float(raw_w) for raw_w in w[:nAssets-1]]

				range_oos = min(deltaT, returnDB_test.shape[0] - f_timeIdx)
				TE_out = []
				for t in range(range_oos): #loop over each t of the simulation
					TE_out.append((np.dot(w,returnDB_test[f_timeIdx+t,1:]) - returnDB_test[f_timeIdx+t,0])**2)
				data_GA["TE_outsample"].append(np.mean(TE_out))

	# consolidated for all periods
	df = pd.DataFrame(data_GA)
	sns.set_theme(style="ticks", font_scale=1.4)
	# in sample	
	sns_plot = sns.lineplot(data=df, x="Generations", y="TE_insample", hue="b_size")#, palette="inferno")
	sns_plot.figure.autofmt_xdate()
	sns_plot.figure.savefig("results/GA_test/deltaT_" + str(deltaT) + "/model_eval/" + obj + "_TE_insample.png")
	sns_plot.figure.clf()
	#out of sample
	sns_plot = sns.lineplot(data=df, x="Generations", y="TE_outsample", hue="b_size")#, palette="inferno")
	sns_plot.figure.autofmt_xdate()
	sns_plot.figure.savefig("results/GA_test/deltaT_" + str(deltaT) + "/model_eval/" + obj + "_TE_outsample.png")
	sns_plot.figure.clf()