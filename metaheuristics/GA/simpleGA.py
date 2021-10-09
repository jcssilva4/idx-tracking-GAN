import numpy as np
import itertools as it
from metaheuristics.GA.tools import *
from metaheuristics.GA.operators import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from random import sample

def simpleGA(S, model_pars, nGenerations = 500):

	best_TE = []
	bestsol = []

	# check if the objective is associated with multi-scenario optimization
	if model_pars["obj"] in ["ms_mean"]:
		top = 10
		X_ms = np.array([]) # containig a population of optimized solutions from all scenarios
		model_pars_temp = model_pars.copy()
		model_pars_temp["obj"] = "hist_TE"
		model_pars_temp["ms_init_process"] = True 
		for s in S:
			best_TE, trsh2, optim_pop = simpleGA(S = s.T, model_pars = model_pars_temp, nGenerations = 100)
			#print(optim_pop)
			if X_ms.shape[0] == 0:
				X_ms = optim_pop
			else:
				X_ms = np.concatenate((X_ms, optim_pop), axis=0)
			#print("optim_pop: " + str(optim_pop.shape))
			#print("x_ms: " + str(X_ms.shape))

	# define your optimization problem
	nAssets = S[0].shape[0] - 1 # -1 to exclude the index 
	lb = model_pars["lb"] 
	ub = model_pars["ub"]
	k = model_pars["K"]
	obj = model_pars["obj"] 

	# set hiperparameters
	nIndividuals = 40;
	p_c = 1.0 # crossover probability
	p_m = 1/nAssets # mutation probability

	# get initial population
	if obj in ["ms_mean"]: # multi-scenario setup
		R = X_ms
		# get ref point
		max_s, min_s = get_scenarios_extremes((get_fitness_individuals(R, nAssets, S, obj)))
		ref_z = get_reference_point_ms(max_s, min_s)
		#nIndividuals = R.shape[0]
	else: # single-scenario (based on historical data) setup
		pop =  get_initial_pop(nIndividuals, nAssets, k)
		R = pop	

	for gen in range(nGenerations):

		#if obj in ["ms_mean"]:
		#	print("generation: " + str(gen + 1))

		##  new population P_(t+1) ##
		if obj in ["ms_mean"]:
			# get fitness of R elements
			fit = get_fitness_individuals(R, nAssets, S, obj)
			# non-dominated fronts
			F = ms_non_dominated_sort(fit)
			# initialize the new population P_(t+1) 
			P, lastFrontIdx, frank = set_new_pop(F, nIndividuals)  # try to get better performance by not calculating any front when the new population is complete
			# get cr_metric
			cr_metric = get_CR_metric(fit, max_s, min_s)
			# set the final new population
			P = set_new_pop(F, nIndividuals, cr_metric, lastFrontIdx, P)
			# get best sol and best_TE until now
			best_sol, best_TE, best_inddx = ms_get_best_sol(P, R, S, nAssets, cr_metric)
			#print("\n----")
			#print("best_TE: " + str(best_TE))
			#print("best_TE: " + str(get_samples_TE(best_sol[:nAssets],S)))
			#print("best_idx: " + str(best_inddx))
			#print("----\n")
		else:
			# get fitness of R elements
			fit = get_fitness_individuals(R, nAssets, S, obj)
			# sort by best individuals
			F = np.argsort(fit) # sort by ascending order
			# get final new population P_(t+1) 
			P = list(F[:nIndividuals]) # P[0] is the best individual (contains the minimum TE among TE_samples from S)
			# get best_sol and best_TE until now
			best_sol = R[P[0],:] 
			# best TE
			best_TE = fit[P[0]]

		## new offspring Q_(t+1) ##

		# initialize R_(t+1)
		R_new = np.zeros((nIndividuals, 2*nAssets))
		for ind in range(nIndividuals):
			R_new[ind,:] = R[P[ind],:]

		# binary tournament selection
		if obj in ["ms_mean"]:
			mating_pool = ms_bin_tournament_selection(P, frank, cr_metric, fit)
		else:
			mating_pool = bin_tournament_selection(P, fit)
		# crossover
		Q = uniform_crossover_chang(mating_pool, R, nAssets, k, nIndividuals, p_c)
		#Q = uniform_crossover(mating_pool, R, nAssets, k, nIndividuals, p_c)
		# mutation
		Q = santanna_mutation(Q, nAssets, nIndividuals, 1, p_m)
		# R_(t+1) = union(P_(t+1),Q_(t+1))
		R = np.concatenate((R_new, Q))

	'''
	if obj in ["ms_mean"]:
		print("----")
		this_tes = get_samples_TE(best_sol[:nAssets],S) 
		fit = np.array(fit)
		min_s = np.array([np.min(fit[:,s]) for s in range(fit.shape[1])])
		print("best_TE: " + str(np.mean(this_tes)))
		print("----")
		#print("max_s_after: " + str(np.array([np.max(fit[:,s]) for s in range(fit.shape[1])])) + ", max_s_before: " + str(max_s))
		#print("min_s_after: " + str(np.array([np.min(fit[:,s]) for s in range(fit.shape[1])])) + ", min_s_before: " + str(min_s))
	'''

	if "ms_init_process" in model_pars.keys(): # check if this is running a multi-scenario init 
		return best_TE, best_sol, R_new
	else:
		return best_TE, best_sol






