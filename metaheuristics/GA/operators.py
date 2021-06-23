import random
import numpy as np
from metaheuristics.GA.tools import *

def bin_tournament_selection(P, fit):

	mating_pool = []

	# select samples so that each solution participates in two tournaments
	sample_part1 = random.sample(P, len(P))
	sample_part2 = random.sample(P, len(P))

	# run the tournaments
	for j in range(len(sample_part1)):
		winner = sample_part1[j] # assume that elements in part1 are winners
		#print(str(sample_part1[j]) + " frank: " + str(frank[sample_part1[j]]))
		#print(str(sample_part2[j]) +" frank: " + str(frank[sample_part2[j]]))
		if fit[sample_part2[j]] < fit[sample_part1[j]]: # test best frontier
			winner = sample_part2[j]
		#print("winner is: " + str(winner))
		mating_pool.append(winner)

	return mating_pool

def uniform_crossover(mating_pool, R, nAssets, k, nIndividuals, p_c): # Deb's book uniform crossover
	
	R_temp = R.copy()
	Q = [] #offspring
	i = 0
	while len(Q) != nIndividuals:
		parent1 = mating_pool[i]
		parent2 = mating_pool[i+1]
		for x in range(nAssets, 2*nAssets):
			if random.uniform(0,1) < p_c: # select which bit to exchange
				temp_val_p1 = R_temp[parent1, x]
				temp_val_p2 = R_temp[parent2, x]
				R_temp[parent1, x] = temp_val_p2
				R_temp[parent2, x] = temp_val_p1

		#print("ind " + str(parent1) + " before: " + str(R[parent1, nAssets:2*nAssets]))
		Q.append(repair_vars(R[parent1,:], k, nAssets, True))
		#print("ind " + str(parent1) + " after: " + str(Q[i][nAssets:2*nAssets]))
		Q.append(repair_vars(R[parent2,:], k, nAssets, True))
		i += 2 # jump to another pair of parents

	return Q

def uniform_crossover_chang(mating_pool, R, nAssets, k, nIndividuals, p_c): # chang2000 uniform crossover

	Q = [] #offspring
	i = 0
	while i < nIndividuals:
		parent1 = mating_pool[i]
		parent2 = mating_pool[i+1]
		if random.uniform(0,1) < p_c: # perform crossover?
			#print("p1: " + str(R[parent1, nAssets:2*nAssets]))
			#print("p2: " + str(R[parent2, nAssets:2*nAssets]))
			for x in range(nAssets, 2*nAssets):
				#if random.uniform(0,1) < p_c: # select which bit to exchange
				temp_Q = R[parent1,:].copy() # copy everything from parent1 to the single offspring
				temp_val_p1 = R[parent1, x]
				temp_val_p2 = R[parent2, x]
				if(temp_val_p2 != temp_val_p1): #this asset do not belong to both parents
					if random.uniform(0,1) < 0.5: # 50% chance of being 0 or 1
						temp_Q[x] = temp_val_p2
			#print("ind " + str(parent1) + " before: " + str(R[parent1, nAssets:2*nAssets]))
			#print("q: " + str(temp_Q[nAssets:2*nAssets]))
			Q.append(repair_vars(temp_Q[:], k, nAssets, True))
		else: # simply copy parents
			Q.append(R[parent1,:])
			Q.append(R[parent2,:])
		i += 2 # jump to another pair of parents

	return Q

def santanna_mutation(Q, nAssets, nIndividuals, num_mut, p_m):
	ind = 0
	Q_final = np.zeros((len(Q), 2*nAssets))
	for ind in range(len(Q)): # loop over all the children
		#'''
		if (random.uniform(0,1) < p_m): 
			#print("before mutation: " + str(Q[ind][nAssets:2*nAssets]))
			# mutate this ind
			selected_vec = [] # vector of selected stock idxx (ones)
			unselected_vec = [] # vector of unselected stock idxx (zeroes)
			count = 0
			for idx in range(nAssets,2*nAssets):
				if Q[ind][idx] == 0:
					unselected_vec.append(idx)
				else:
					selected_vec.append(idx)

			sample_selected = random.sample(selected_vec, num_mut) # change these to zero
			sample_unselected = random.sample(unselected_vec, num_mut) # change these to one
			for s in range(num_mut):
				Q[ind][sample_selected[s]] = 0
				Q[ind][sample_unselected[s]] = 1

			#print("after mutation: " + str(Q[ind][nAssets:2*nAssets]))
			psize = np.sum(Q[ind][nAssets:2*nAssets])
			if(psize!=10):
				print("K = " + str(psize))
		#'''
		## gaussian mutation for the weights ##
		# correct zero weights and mutate weights
		temp_var = Q[ind][:]
		#print("current bin: " + str(temp_var[nAssets:2*nAssets]))
		B =  temp_var[nAssets:2*nAssets]
		w_idx = 0
		for b in B:
			if b == 1: # check if associated w is zero
				if temp_var[w_idx] == 0: # 
					temp_var[w_idx] = random.uniform(0,1)
				temp_var[w_idx] =  min(max(np.random.normal(temp_var[w_idx], 0.15),0.001),1)
				# check if w_i is positive after mutation

			else:
				temp_var[w_idx] = 0
			w_idx += 1
		# budget constraint viability
		repaired = repair_weights(temp_var, nAssets)
		#print("weights_after: " + str(repaired[0:nAssets]))
		Q_final[ind,:] = repaired[:]

	return Q_final

	