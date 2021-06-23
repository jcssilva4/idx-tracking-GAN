import numpy as np
import itertools as it
from metaheuristics.GA.tools import *
from metaheuristics.GA.operators import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def simpleGA(S, model_pars, nGenerations = 1000):

	best_TE = []
	bestsol = []

	# define your optimisation problem
	nAssets = S[0].shape[0] - 1 # -1 to exclude the index 
	lb = model_pars["lb"] 
	ub = model_pars["ub"]
	k = model_pars["K"]
	obj = model_pars["obj"]
	

	# set real coded NSGA-II parameters
	nIndividuals = 10;
	p_c = 0.9 # crossover probability
	p_m = 1/nAssets # mutation probability


	# get initial population
	pop =  get_initial_pop(nIndividuals, nAssets, k)
	R = pop

	for gen in range(nGenerations):


		#print("generation: " + str(gen + 1))

		##  new population P_(t+1) ##

		# get fitness of R elements
		fit = get_fitness_individuals(R, nAssets, S, obj)
		# sort by best individuals
		F = np.argsort(fit) # sort by ascending order
		# get final new population P_(t+1) 
		P = list(F[:nAssets]) # P[0] is the best individual (contains the minimum TE among TE_samples from S)
		# get best_sol and best_TE until now
		best_sol = R[P[0],:] 
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
		print(" -- -- constraint satisfaction -- --")
		for ind in range(R.shape[0]):
			print("K = " + str(sum(R[ind,nAssets:2*nAssets])) + " and sum(w_i) = " + str(sum(R[ind,0:nAssets])) )
		'''

	return best_TE, best_sol






