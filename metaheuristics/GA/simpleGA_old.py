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
	pop =  get_initial_pop(nIndividuals, nAssets, k)
	R = pop

	if obj in ["per_sim", "per_sim_pnlty", "per_period","per_period_pnlty"]:
		s_idx = 0
		t_idx = 0
		S_all = S
		R_all_s = []
		best_sol_all_s = []
		all_sim_idxs = [ss for ss in range(len(S_all))]

	for gen in range(nGenerations):
		if obj in ["per_sim", "per_sim_pnlty"]:
			s_idx = sample(all_sim_idxs,1)
			S_ = S_all[s_idx[0]]
			S = S_.T

			'''
			S_ = S_all[s_idx]
			S = S_.T
			s_idx += 1
			if s_idx == len(S_all): # if the last market simulation was reached
				s_idx = 0 # go back to the first market iteration
			'''

		if obj in ["per_period","per_period_pnlty"]:
			S_ = []
			for s in S_all:
				S_.append(s[:,t_idx])
			S = np.array(S_)
			t_idx += 1
			if t_idx == S_all[0].shape[1]: # if the end of the simulated out of sample period was reached
				t_idx = 0 # go back to the first simulated out of sample period

		# if a penalty must be calculated
		if obj in ["per_sim_pnlty", "per_period_pnlty"]:
			if gen > 0: # update init_w using the best solution of the last generation
				init_w = best_sol[:nAssets]

		#print("generation: " + str(gen + 1))

		##  new population P_(t+1) ##

		# get fitness of R elements
		fit = get_fitness_individuals(R, nAssets, S, obj)
		# if a penalty is necessary....
		if obj in ["per_sim_pnlty","per_period_pnlty"] and gen > 0:
			fit = get_fit_penalty_individuals(R, nAssets, init_w, fit)

		# sort by best individuals
		F = np.argsort(fit) # sort by ascending order
		# get final new population P_(t+1) 
		P = list(F[:nIndividuals]) # P[0] is the best individual (contains the minimum TE among TE_samples from S)
		# get best_sol and best_TE until now
		best_sol = R[P[0],:] 
		if obj in ["per_sim", "per_sim_pnlty", "per_period","per_period_pnlty"]:
			best_TE = np.mean(get_samples_TE(best_sol[:nAssets], S_all)) #fit[P[0]]
		else:
			best_TE = fit[P[0]]

		## new offspring Q_(t+1) ##

		# initialize R_(t+1)
		R_new = np.zeros((nIndividuals, 2*nAssets))
		for ind in range(nIndividuals):
			R_new[ind,:] = R[P[ind],:]

		# binary tournament selection
		mating_pool = bin_tournament_selection(P, fit)
		# crossover
		Q = uniform_crossover_chang(mating_pool, R, nAssets, k, nIndividuals, p_c)
		#Q = uniform_crossover(mating_pool, R, nAssets, k, nIndividuals, p_c)
		# mutation
		Q = santanna_mutation(Q, nAssets, nIndividuals, 1, p_m)
		# R_(t+1) = union(P_(t+1),Q_(t+1))
		R = np.concatenate((R_new, Q))

		'''
		if obj in ["per_sim", "per_sim_pnlty", "per_period","per_period_pnlty"]:
			R_all_s.append(R)
			best_sol_all_s.append(best_sol)
			# check if we need to update the robust solution
			if s_idx == len(S_all) - 1 or gen + 1 == nGenerations:
				best_sol_robust, best_TE_robust, R_robust = get_best_sol_robust(R_all_s, best_sol_all_s, S_all, nAssets)
				best_TE = best_TE_robust 
				best_sol = best_sol_robust
				R = R_robust
				R_all_s = []
				best_sol_all_s = []
		'''
		'''
		# update the robust solution
		if obj in ["per_sim", "per_sim_pnlty", "per_period","per_period_pnlty"]:
			R_all_s.append(R_new)
			best_sol_all_s.append(best_sol)
			# compare the current geeneration's best solution with the last generation's best solution
			if gen > 1:
				best_sol_robust, best_TE_robust, R_robust = get_best_sol_robust(R_all_s, best_sol_all_s, S_all, nAssets)
				best_TE = best_TE_robust 
				best_sol = best_sol_robust
				R = R_robust
				R_all_s = [R_robust]
				best_sol_all_s = [best_sol_robust]
		'''

	return best_TE, best_sol






