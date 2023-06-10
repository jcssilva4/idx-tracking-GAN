import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import os
import random
from experiment_parameters.parameters import *
# PyTorch libs
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.autograd as autograd
from utils import *

metaheuristics = dict([])
metaheuristics['ms_mean'] = 'SDM-SBDGA-GAN' 
metaheuristics['mean'] = 'SDM-SAAGA-GAN' 

# get experiment parameters
expParameters = get_parameters()
# set test_parameters
deltaT =  expParameters["deltaT"]
total_epochs =  expParameters["total_epochs"]
delta_epoch =  expParameters["delta_epoch"]
max_epoch = total_epochs
epoch_step_size = delta_epoch

test_size = expParameters["test_size"]
w = expParameters["w"]
b = expParameters["b"]
f = w - b
nModels = expParameters["nModels"]
#top5_gans = [r for r in range(nModels)]
total_epochs = 8000 #expParameters["total_epochs"]
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


# first, get the RD-GA results
objs = [str(b_size) for b_size in best_lookback_windows]
# data for oos TE
data_GA = dict([])
data_GA["TE"] = []
data_GA["rebalance date"] = []
data_GA["Model"] = []
data_GA["Algorithm"] = []
data_GA["run"] = []
# data for cumRet
data_GAc = dict([])
data_GAc["Cumulative Return"] = []
data_GAc["Date"] = []
data_GAc["Model"] = []
data_GAc["Algorithm"] = []
for obj in objs:
	#obj_namee = obj
	#if not obj in ["mean", "ms_mean"]:
	#	obj_namee = 'hist_TE' + obj 
	#for run in top5_models[obj_namee]:
	for run in range(1,nRuns+1):
		b_size = 0

		filepath = main_folder + "run_" + str(run) + "/benchmark_experimentsDB.csv"
		b_size = int(obj)

		experimentsDB = pd.read_csv(filepath) 
		f_timeIdx = b
		TE = 0
		next_rebal = f_timeIdx
		w = []
		cumRet = 1
		while f_timeIdx < returnDB_test.shape[0]:
			current_f_ = f_timeIdx
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
				# data - oos TE
				data_GA["TE"].append((np.dot(w,returnDB_test[f_timeIdx,1:]) - returnDB_test[f_timeIdx,0])**2)
				data_GA["rebalance date"].append(dates_test[current_f_])
				model = "GA - " + str(b_size)
				data_GA["Model"].append(model)
				data_GA["Algorithm"].append("RDM-GA")
				data_GA["run"].append(run)
				# data - cumRet
				cumRet = cumRet*(1+np.dot(w,returnDB_test[f_timeIdx,1:]))
				data_GAc["Cumulative Return"].append(cumRet)
				data_GAc["Date"].append(dates_test[f_timeIdx])
				data_GAc["Model"].append(model)
				data_GAc["Algorithm"].append("RDM-GA")
				f_timeIdx += 1

df_raw = pd.DataFrame(data_GA)
df_rawc = pd.DataFrame(data_GAc)
df_rdga = df_raw[df_raw["Model"] == "GA - 40"]
df_rdgac = df_rawc[df_rawc["Model"] == "GA - 40"]
# get runs table
#df_raw = df[df['Model'] == "GA - 60"]
data_GA_grouped = df_raw.groupby(['Model']).mean()
#df_raw = df[df['Model'] == "GA - 60"]
data_GA_grouped2 = df_raw.groupby(['Model']).std()
data_GA_grouped["std(TE)"] = data_GA_grouped2["TE"]
#print(data_GA_grouped)

# prepare rdga boxlot data
boxplot_raw = df_rdga.copy()
boxplot_rdga_final = boxplot_raw.groupby(['run']).mean()
boxplot_rdga_final["Epoch"] = [1]*30
boxplot_rdga_final["Algorithm"] = ["RDM-GA"]*30
boxplot_rdga_final["Objective"] = ["real"]*30
boxplot_rdga_final["Model"] = [1]*30
#print(boxplot_rdga_final)

# get the mean OoS tracking error of the chosen GA 
mean_ga_hist = data_GA_grouped[data_GA_grouped.index == 'GA - 40'].TE.values[0]
print("RDM-GA performance: \n" + str(data_GA_grouped[data_GA_grouped.index == 'GA - 40']))

# for each out-of-sample period 
#  TE =  (portfolio_ret - s[0,t])**2
data_GA = dict([])
data_GA["TE"] = []
data_GA["Obj"] = []
data_GA["Epoch"] = []
data_GA["Model"] = []
objs =  expParameters["objs"]
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
					data_GA["Model"].append(run+1)
					f_timeIdx += 1

			epoch += epoch_step_size

df = pd.DataFrame(data_GA)
# plot mean out-of-sample TE for each epoch for each GAN 
df_avg_best = dict([])
df_avg_best["TE"] = []
df_avg_best["Epoch"] = []
df_avg_best["Algorithm"] = []
df_avg_best["Objective"] = []
df_avg_best["Model"] = []
for run in range(nModels):
	df_this_gan_raw = df[df["Model"] == run+1] # get data for a specific gan
	for obj in objs:
		df_this_gan = df_this_gan_raw[df_this_gan_raw["Obj"] == obj]
		df_gan_epoch = df_this_gan.groupby(['Epoch']).mean()
		#print("before sorting")
		#print(df_gan_epoch)
		#print("after sorting")
		df_gan_sorted = df_gan_epoch.sort_values('TE')
		#print(df_gan_sorted.head(1))
		#df_gan_final = df_gan_epoch[df_gan_epoch["TE"] <= mean_ga_hist]
		df_gan_final = df_gan_sorted.head(1)
		#print(df_gan_final)
		block_len = len(df_gan_final.index.values)
		df_avg_best["TE"].extend(df_gan_final["TE"].values)
		df_avg_best["Epoch"].extend(df_gan_final.index.values)
		df_avg_best["Model"].extend(df_gan_final["Model"].values)
		df_avg_best["Algorithm"].extend([metaheuristics[obj]]*block_len)
		df_avg_best["Objective"].extend([obj]*block_len)


#print(df_avg_best)

# plot hist, x = Epoch
# As the epoch grows, the probability of having better models is higher?
df = pd.DataFrame(df_avg_best)
df_after_plot = df.copy() # we need to plot this in the end of the code, because FacetGrid cause problems for some reason...
sns.set_theme(style="ticks", font_scale=1.4)
sns_plot = sns.histplot(data=df, x="Epoch", hue = "Algorithm")
sns_plot.figure.savefig(main_folder + "sim" + str(n_sims) + "/new_results2/epoch_hist" + str(deltaT) + ".png")
sns_plot.figure.clf()


# select the best GAN models
frames = []
best_obj = "ms_mean"
best_model = []
for obj in objs:
	df_obj = df[df['Algorithm'] == metaheuristics[obj]]
	df_obj_sorted = df_obj.sort_values('TE')
	print(df_obj_sorted)
	frames.append(df_obj_sorted)
	if obj == "ms_mean":
		best_model = df_obj_sorted.head(1)
df_selected_models = pd.concat(frames)
print("best model: ")
print(str(best_model))
frames_grouped = df_selected_models.groupby(['Algorithm']).mean()
frames_grouped2= df_selected_models.groupby(['Algorithm']).std()
frames_grouped['std(TE)'] = frames_grouped2['TE']
print(frames_grouped)

#print("df_boxplot_data")
df_boxplot_data = pd.concat([df_selected_models,boxplot_rdga_final])
#print(df_boxplot_data)
#boxplot for oosTE
sns_plot = sns.boxplot(x="Algorithm", y="TE", data=df_boxplot_data)#, palette="Set3")
sns_plot.figure.savefig(main_folder + "sim" + str(n_sims) + "/new_results2/benchmark_oosTE_boxplot" + str(deltaT) + ".png")
sns_plot.figure.clf()




# compare the RD-GA-40, and the SDM-GAS mean oos TE
data_GA = dict([])
data_GA["TE"] = []
data_GA["rebalance date"] = []
data_GA["Model"] = []
data_GA["Algorithm"] = []
data_GA["run"] = []
data_GA["Epoch"] = []
# data for cumRet
data_GAc = dict([])
data_GAc["Cumulative Return"] = []
data_GAc["Date"] = []
data_GAc["Model"] = []
data_GAc["Algorithm"] = []
for index, row in df_selected_models.iterrows():
	this_epoch = row["Epoch"]
	this_model = row["Model"]
	this_obj = row["Objective"]
	# read the associated DB
	experimentsDB = pd.read_csv("results/GA_test/deltaT_" + str(deltaT) + "/run_" + str(this_model) + "/experimentsDB.csv")
	f_timeIdx = b
	next_rebal = f_timeIdx
	w = []
	cumRet = 1
	while f_timeIdx < returnDB_test.shape[0]:
		current_f_ = f_timeIdx
		# check if we need to rebalance
		if(f_timeIdx == next_rebal): # if rebalance, then get weights... 
			w = []
			query = (experimentsDB["f_timeIdx"] == f_timeIdx) & (experimentsDB["obj"] == this_obj) & (experimentsDB["epoch"] == this_epoch-1) & (experimentsDB["n_sims"] == n_sims)
			next_rebal = next_rebal + deltaT
			# get weights
			this_row = experimentsDB[query]
			w =  this_row["best_sol"].values[0]
			w =  w.split(",")
			w = [float(raw_w) for raw_w in w[:nAssets-1]]

		range_oos = min(deltaT, returnDB_test.shape[0] - f_timeIdx)
		for t in range(range_oos): #loop over each t of the simulation
			# data oos TE
			data_GA["TE"].append((np.dot(w,returnDB_test[f_timeIdx,1:]) - returnDB_test[f_timeIdx,0])**2)
			data_GA["rebalance date"].append(dates_test[current_f_])
			data_GA["Model"].append("GAN - " + this_obj)
			data_GA["Algorithm"].append(metaheuristics[this_obj])
			data_GA["run"].append(this_model)
			data_GA["Epoch"].append(this_epoch)
			# data - cumRet
			cumRet = cumRet*(1+np.dot(w,returnDB_test[f_timeIdx,1:]))
			data_GAc["Cumulative Return"].append(cumRet)
			data_GAc["Date"].append(dates_test[f_timeIdx])
			data_GAc["Model"].append("GAN - " + this_obj)
			data_GAc["Algorithm"].append(metaheuristics[this_obj])
			f_timeIdx += 1

# consolidated for all periods
df_sdga = pd.DataFrame(data_GA)
df = pd.concat([df_sdga, df_rdga])
'''
sns.set_theme(style="ticks", font_scale=1.4)
sns_plot = sns.lineplot(data=df, x="rebalance date", y="TE", hue = "Model")#, palette="inferno")
#sns_plot.figure.autofmt_xdate()
sns_plot.figure.savefig(main_folder + "sim" + str(n_sims) + "/new_results/benchmark_TE_outOfSample" + str(deltaT) + ".png")
sns_plot.figure.clf()
'''
df_1 = df.copy()
# get the best and worst rebalance dates
best_model_query = (df_sdga["run"] == best_model["Model"].values[0])  & (df_sdga["Epoch"] == best_model["Epoch"].values[0]) & (df_sdga["Model"] == "GAN - ms_mean")
df_sdga_best = df_sdga[best_model_query]
df_sdga_best_rebal = df_sdga_best.groupby(['rebalance date']).mean()
best_rebl_date = df_sdga_best_rebal.sort_values('TE').head(1).index.values[0]
worst_rebl_date = df_sdga_best_rebal.sort_values('TE', ascending = False).head(1).index.values[0]
print("best_rebl_date: " + str(best_rebl_date))
print("worst_rebl_date: " + str(worst_rebl_date))

# compare the RD-GA-40, and the SD-GA objs cumulative return
# concat the RD-GA data
data_GAc["Cumulative Return"].extend(df_rdgac["Cumulative Return"].values)
data_GAc["Date"].extend(df_rdgac["Date"].values)
data_GAc["Model"].extend(df_rdgac["Model"].values)
data_GAc["Algorithm"].extend(df_rdgac["Algorithm"].values)
# concat index data
cumRet = 1
for t in range(b,returnDB_test.shape[0]):
	cumRet = cumRet*(1+returnDB_test[t,0])
	data_GAc["Cumulative Return"].append(cumRet)
	data_GAc["Date"].append(dates_test[t])
	data_GAc["Model"].append("Ibovespa")
	data_GAc["Algorithm"].append("Ibovespa")

# consolidated plot
#df_sdgac = pd.DataFrame(data_GAc)
#df = pd.concat([df_sdgac, df_rdgac])
df = pd.DataFrame(data_GAc)
'''
sns.set_theme(style="ticks", font_scale=1.4)
sns_plot = sns.lineplot(data=df, x="Date", y="Cumulative Return", hue="Model")#, palette="inferno")
sns_plot.figure.autofmt_xdate()
sns_plot.figure.savefig(main_folder + "sim" + str(n_sims) + "/new_results/benchmark_CumRet_Date" + str(deltaT) + ".png")
sns_plot.figure.clf()
'''
df_2 = df.copy()


fig, axs = plt.subplots(ncols = 2)
sns.lineplot(x="rebalance date", y="TE", hue="Algorithm", data = df_1, ax = axs[0])
sns.lineplot(x = 'Date', y = 'Cumulative Return', hue="Algorithm", data = df_2, ax = axs[1])
fig.set_size_inches(27, 9)
fig.savefig(main_folder + "sim" + str(n_sims) + "/new_results2/benchmark_oosTE_CumRet_Date" + str(deltaT) + ".png")



'''
# Table: row: Models, columns: Rebalancing dates
df_1g = df_1.groupby(['rebalance date','Model']).mean()
#print(df_1g.columns)
df_1g = df_1g.drop(['run'], axis = 1)
print(df_1g)
table = pd.pivot_table(df_1g, values='TE', columns = ['rebalance date'], index=['Model'])
table["Model"] = table.index
table = table[["Model"] + [col for col in table if col not in ["Model"]]]
print(table)

print(table.to_latex(index=False))  
with open(main_folder + "sim" + str(n_sims) + "/new_results/table" + str(deltaT) + '.txt', 'w', encoding='utf-8') as f:
    f.write(table.to_latex(index=False))
'''

# Mean simulation trajectory -----------

[allMb, allMf] = get_dataSet(TSData = returnDB, b = b, f = f, step = 1)
torch_Mb = torch.from_numpy(np.array(allMb)).float()
torch_Mf = torch.from_numpy(np.array(allMf)).float()
M_raw = torch.cat((torch_Mb, torch_Mf), dim = 2)
M_train = M_raw[0:M_raw.shape[0] - test_size,:,:]
M_test = M_raw[M_raw.shape[0] - test_size:,:,:]  # dim1: number of examples, dim2: num assets, dim3: return time series size (w)

filepath_model = "models/" + str(total_epochs) + "_epochs/"
simulations = dict([])
#screenshots_assets = [0, 10, 15, 26, 35]

this_model = best_model["Model"].values[0]
this_epoch = best_model["Epoch"].values[0]
# get top 4 assets for the best models on each selected period
experimentsDB = pd.read_csv("results/GA_test/deltaT_" + str(deltaT) + "/run_" + str(this_model) + "/experimentsDB.csv")

# initialize a generator instance from saved checkpoints
# load the model state
filename = filepath_model + "run_" + str(this_model) + "/GAN_state_" + str(this_epoch) + ".pth"
print("running simulations using " + filename)
if not torch.cuda.is_available():
	checkpoint = torch.load(filename, map_location = "cpu")
else:
	checkpoint = torch.load(filename)
# create an instance of the model
generator = Generator(nAssets,f)
generator.load_state_dict(checkpoint['G_state_dict'])
current_epoch = this_epoch
# uncomment this if you want to save simulation plots into img files
# generate simulations
f_timeIdx = b
screenshots_dates_idxs = dict([]) # save plots for these rebalancing periods and in the last rebalacing period
w_screenshots = dict([])
while f_timeIdx + deltaT <= M_test.shape[0]:
	if dates_test[f_timeIdx] in [best_rebl_date, worst_rebl_date]:
		w = []
		query = (experimentsDB["f_timeIdx"] == f_timeIdx) & (experimentsDB["obj"] == this_obj) & (experimentsDB["epoch"] == this_epoch-1) & (experimentsDB["n_sims"] == n_sims)
		next_rebal = next_rebal + deltaT
		# get weights
		this_row = experimentsDB[query]
		w =  this_row["best_sol"].values[0]
		w =  w.split(",")
		w = [float(raw_w) for raw_w in w[:nAssets-1]]
		# run simulations for this window (Mb)
		print(dates_test[f_timeIdx])
		Mb = M_test[f_timeIdx:f_timeIdx+1,:,0:b]
		#Mf_real = M_test[f_timeIdx:f_timeIdx+1,:,b:f]
		sim_data = []
		for sim in range(n_sims):
			Mf_fake = generator(Mb)
			Mf_fake = Mf_fake[0].cpu().detach().numpy()
			#print(Mf_fake.shape)
			sim_data.append(Mf_fake)
		simulations[str(f_timeIdx) + str(current_epoch)] = sim_data
		# get top 4 assets
		w_top_idx = np.flip(np.argsort(w, kind=None, order=None))[:4]
		print("w: " + str(w))
		print("w_top_idx: " + str(w_top_idx))
		if dates_test[f_timeIdx] == best_rebl_date:
			screenshots_dates_idxs['best'] = f_timeIdx
			w_screenshots['best'] = [0]
			w_screenshots['best'].extend([this_asset + 1 for this_asset in w_top_idx])
		else:
			screenshots_dates_idxs['worst'] = f_timeIdx
			w_screenshots['worst'] = [0]
			w_screenshots['worst'].extend([this_asset + 1 for this_asset in w_top_idx])

	f_timeIdx += deltaT

print(w_screenshots)

fig, axes = plt.subplots(len(screenshots_dates_idxs.keys()), len(w_screenshots['best']), figsize=(50, 23), sharex=False)
screenshots_data = dict([])
screenshots_data["Date"] = []
screenshots_data["Epoch"] = []
screenshots_data["Return"] = []
screenshots_data["Data type"] = []
screenshots_data["Ticker"] = []
screenshots_data["Initial time"] = []
for rebl_prd_type in ['best', 'worst']: 
	returnDB_vals = returnDB.values
	#print(returnDB.shape)
	sims = simulations[str(screenshots_dates_idxs[rebl_prd_type]) + str(current_epoch)]
	#sims = random.sample(sims, 5)
	# loop over all simulations in sims
	scenario_cnt = 1
	for sim in sims:
		for assetIdx in w_screenshots[rebl_prd_type]:
			dateIdx = screenshots_dates_idxs[rebl_prd_type]
			for val in sim[assetIdx,:]:
				screenshots_data["Date"].append(dates_test[dateIdx])
				screenshots_data["Epoch"].append(str(current_epoch))
				screenshots_data["Return"].append(val)
				screenshots_data["Data type"].append("simulated")
				#screenshots_data["Data type"].append("scenario " + str(scenario_cnt))
				screenshots_data["Ticker"].append(symbols[assetIdx])
				screenshots_data["Initial time"].append(screenshots_dates_idxs[rebl_prd_type])
				dateIdx += 1
				#print(dateIdx)
		scenario_cnt += 1

	#sns.set_theme(style='whitegrid')
	#sns.set(font_scale = 0.7)
	#sns.set(rc={'figure.figsize':(11.7,8.27)})
	for assetIdx in w_screenshots[rebl_prd_type]:
		print("getting screenshots for:" + symbols[assetIdx])
		dateIdx =screenshots_dates_idxs[rebl_prd_type]
		# get real data
		for val in range(deltaT):
			screenshots_data["Date"].append(dates_test[dateIdx])
			screenshots_data["Epoch"].append("real data")
			screenshots_data["Return"].append(returnDB_test[dateIdx, assetIdx])
			screenshots_data["Data type"].append("real")
			screenshots_data["Ticker"].append(symbols[assetIdx])
			screenshots_data["Initial time"].append(screenshots_dates_idxs[rebl_prd_type])
			dateIdx += 1
			#sns.set_theme(style="ticks", font_scale=1.4)
			#sns_plot = sns.lineplot(data=df, x="Date", y="Return", hue="Epoch", style = "data_type", palette="inferno").set_title(symbols[assetIdx] + " (" + str(dates_test[screenshots_time]) + " - " + str(dates_test[screenshots_time+deltaT-1]) +  ")")
			#sns_plot._legend.remove()
			#sns_plot.figure.autofmt_xdate()

df = pd.DataFrame(screenshots_data)
linestyle = ["-"]
#linestyle = ["-" for sim in sims]
linestyle.append("--")
kw = {'ls' : linestyle}
sns.set_theme(style="ticks", font_scale=0.9)

for rebl_prd_type in ['best','worst']:
	dfx = df[df['Initial time'] == screenshots_dates_idxs[rebl_prd_type]]
	g = sns.FacetGrid(dfx, col='Ticker', hue='Data type', sharey = False, sharex = False, hue_kws = kw)
	g1 = g.map(sns.lineplot, 'Date', 'Return').add_legend()
	for axes in g.axes.flat:
		_ = axes.set_xticklabels(axes.get_xticklabels(), rotation=45)
	g.tight_layout()
	g.savefig(main_folder + "sim" + str(n_sims) + "/new_results2/sims_" + rebl_prd_type + str(deltaT) + ".png")
	plt.close()

''''
axi = 0
for rebl_prd_type in ['best', 'worst']:
	axj = 0
	dfx_raw = df[df['Initial time'] == screenshots_dates_idxs[rebl_prd_type]]
	for ticker in w_screenshots[rebl_prd_type]:
		print("axi: " + str(axi) + ", axj: " + str(axj))
		dfx = dfx_raw[dfx_raw['Ticker'] == symbols[ticker]] 
		sns.lineplot(data = dfx, x = 'Date', y='Return', hue='Data type', style = "Data type", ax = axes[axi, axj], legend = False)
		axj += 1
	axi += 1
#fig.set_size_inches(27, 9)

fig.savefig(main_folder + "sim" + str(n_sims) + "/new_results/sims_best_worst" + str(deltaT) + ".png")
'''
'''
sns_plot = sns.relplot(data=df, x="Date", y="Return", hue="Data type", col="Ticker", style = "Data type", kind = "line", col_wrap=5)
sns_plot.savefig(main_folder + "sim" + str(n_sims) + "/new_results/sims_best_worst" + str(deltaT) + ".png")
'''


############### Final plot.... ###################
# Are models that produce better mean oos tracking errors concentrated in higher epochs??
sns.set(rc={'figure.figsize':(25,16)})
sns.set_theme(style="ticks", font_scale=1.0)
g = sns.FacetGrid(data = df_after_plot, col="Algorithm")
g.map(sns.histplot,"Epoch","TE")
g.savefig(main_folder + "sim" + str(n_sims) + "/new_results2/epoch_TE_hist" + str(deltaT) + ".png")

