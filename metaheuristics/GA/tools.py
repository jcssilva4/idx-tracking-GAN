import random
import numpy as np

def get_initial_pop(nIndividuals, nAssets, k):

	initialPop = np.zeros((nIndividuals, 2*nAssets))

	for individual in range(nIndividuals):

		temp_var = []
		for var in range(nAssets):
			temp_var.append(random.uniform(0,1)) # sample from gaussian dist with mean = 0.5 and std = 0.15

		repaired = repair_vars(temp_var, k, nAssets)
		final_sol = repair_weights(repaired, nAssets)
		initialPop[individual, :] = final_sol[:]

	return initialPop

def repair_vars(temp_var, k, nAssets, crossover = False, bug_ver = False):

	if crossover:
		#  when crossover occurs, some solutions may have length != k (santanna2017)
		current_psize = np.sum(temp_var[nAssets:2*nAssets])
		diff = current_psize - k # current violation magnitude
		#print("diff: " + str(diff))
		if diff > 0: # remove assets that contain small weights
			temp_var = repair_vars(temp_var[:nAssets], k, nAssets, bug_ver = True)

		elif diff < 0: # add assets
			idx_add = random.sample([i for i in range(nAssets, 2*nAssets)],nAssets) 
			count = 0
			while diff != 0:
				if(temp_var[idx_add[count]] == 0): # include this
					temp_var[idx_add[count]] = 1
					diff += 1
				count += 1

		return temp_var

	repaired = []
	# get the K top values 
	sorted_vec = sorted( [[x,i] for (i,x) in enumerate(temp_var)], reverse=True )[:k] 
	kth_largest_weight = sorted_vec[k-1]
	#print(kth_largest_weight)

	# only the k highest weights will be maintained
	w_var = [] # continuous vars
	z_var = [] # binary vars
	k_constr = 0
	for var in range(nAssets):
		value = 0
		value_bin = 0

		if temp_var[var] >= kth_largest_weight[0] and k_constr < k:
			value = temp_var[var]
			value_bin = 1
			k_constr += 1
		w_var.append(value)
		z_var.append(value_bin)

	repaired = w_var
	repaired.extend(z_var)

	return repaired

def repair_weights(temp_var, nAssets):

	repaired = []
	weights = temp_var[0:nAssets]
	sum_weights = np.sum(weights)
	weights_normalized = np.dot((1/sum_weights),weights)
	repaired = weights_normalized.tolist()
	repaired.extend(temp_var[nAssets:])

	return repaired

	
def get_fitness_individuals(R, nAssets, S, obj):
	fit = []
	for individual in R:
		samples_TE = get_samples_TE(individual[:nAssets],S)
		if obj == "mean":
			fit.append(np.mean(samples_TE))
		if obj == "max":
			fit.append(np.max(samples_TE))
	return fit

def get_samples_TE(weights, S):
	samples_TE = [] # TE samples from S
	for s in S: # loop over all the simulation set S, where s = M_f
		TE = 0
		for t in range(s.shape[1]): # loop over t = {1,2,...,T}
			portfolio_ret = np.dot(weights, s[1:,t])
			TE += (portfolio_ret - s[0,t])**2 # s[0,t] is R_t, for simulation s
		samples_TE.append(TE/s.shape[1])
	return samples_TE


def set_new_pop(F, nIndividuals, cdist = None, lastFrontIdx = None, P = []):

	if lastFrontIdx == None:
		#print("F: " + str(F))
		new_P = []
		i = 0
		while(len(new_P) + len(F[i]) < nIndividuals):
			new_P.extend(F[i])
			i += 1

		return new_P, i

	else:
		#print("P: " + str(P))
		#print("cdist: " + str(cdist))
		new_P = P
		remaining_elements = nIndividuals - len(P)
		cdist_unranked = []
		front = F[lastFrontIdx]
		for p in front:
			cdist_unranked.append(cdist[p])

		#print("cdist_unranked: " + str(cdist_unranked))

		# set the final new_P
		sorted_idx = [e[0] for e in sorted(enumerate(cdist_unranked), key=lambda x:x[1])]
		#print("sorted_idx: " + str(sorted_idx))
		for i in range(remaining_elements):
			new_P.append(front[sorted_idx[len(sorted_idx) - (i+1)]])

		#print("final_P: " + str(new_P))

		return new_P


def get_normalization_coefs(F, fit, lastFrontIdx, nObjectives):

	temp_pop_idxs = []
	fit_obj = dict([])
	f_max = [] # contains the population-maximum for each obj
	f_min = [] # contains the population-minimum for each obj

	for i in range(lastFrontIdx + 1):
		temp_pop_idxs.extend(F[i])

	# initialize fit_obj
	for obj in range(nObjectives):
		fit_obj[obj] = []
	
	# get fit vectors for each obj func of solutions contained in F[0:lastFronIdx]
	for p in temp_pop_idxs:
		for obj in range(nObjectives):
			fit_obj[obj].append(fit[p][obj])

	# get f_max, f_min
	for obj in range(nObjectives):
		f_max.append(max(fit_obj[obj]))
		f_min.append(min(fit_obj[obj]))

	return [1,1], [0,0]

def front_dist_rank(f_max, f_min, F, fit, lastFrontIdx, nObjectives):

	inf_const = 1e16 # represents a big number
	cdist = dict([]) # crowding distance
	frank = dict([]) # front rank
	for i in range(lastFrontIdx + 1): # loop over all fronts
		if(len(F[i]) < 3): # if F_i only have two solutions
			for p in F[i]:
				cdist[p] = inf_const 
				frank[p] = i + 1
		else: # F_i has intermediate solutions
			front = F[i]
			fit_obj = dict([])
			# initialize fit_obj
			for obj in range(nObjectives):
				fit_obj[obj] = []
			
			# index ranking for each obj 
			for p in front:
				frank[p] = i + 1
				cdist[p] = 0 # initialize cdist for the elements of this front
				for obj in range(nObjectives):
					fit_obj[obj].append(fit[p][obj])

			for obj in range(nObjectives):
				#sort front elements according to obj function m
				sorted_idx = [e[0] for e in sorted(enumerate(fit_obj[obj]), key=lambda x:x[1])]
				count = 0
				# get cdist for each element of this front
				for p in sorted_idx:
					if count == 0 or count == (len(front) - 1): # check if this is a border solution
						cdist[front[p]] = inf_const
					else:
						dist = (fit[front[sorted_idx[count+1]]][obj] - fit[front[sorted_idx[count-1]]][obj])/(f_max[obj] - f_min[obj])
						# get overall crowd dist
						if(dist > cdist[front[p]]):	
							cdist[front[p]] = dist
					count += 1

	return cdist, frank
	

	