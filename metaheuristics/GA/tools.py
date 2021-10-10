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
		if not obj in ["hist_TE"]:
			# if the objective is to minimize the mean or max TE in GAN simulations
			samples_TE = get_samples_TE(individual[:nAssets],S)
			if obj == "mean":
				fit.append(np.mean(samples_TE))
			if obj == "max":
				fit.append(np.max(samples_TE))
			if obj in ["ms_mean"]:
				fit.append(samples_TE)

		else:
			# minimize historical TE
			fit.append(get_TE(individual[:nAssets],S))

	return fit

def get_fit_penalty_individuals(R, nAssets, w_init, fit):
	final_fit = []
	count_ind = 0
	for individual in R:
		# calculate penalty for this individual
		penalty = (10**-3)*np.mean(np.absolute(w_init - individual[:nAssets])) # if this do not work, do not sum, do fit*(1+penalty)
		# penalize the fitness of this individual
		#print("fit: " + str(fit[count_ind]) + ", penalty: " + str(penalty) + ", final_fit: " + str(fit[count_ind] + penalty))
		final_fit.append(fit[count_ind] + penalty)
		count_ind += 1
	return final_fit

def get_best_sol_robust(R_all_s, best_sol_all_s, S_all, nAssets):
	best_sol_robust = []
	R_robust = []
	best_TE_robust = 1000
	counter_r = 0
	for best_w in best_sol_all_s:
		samples_TE = get_samples_TE(best_w[:nAssets], S_all)
		candidate_fit = np.mean(samples_TE)
		if candidate_fit < best_TE_robust:
			R_robust = R_all_s[counter_r]
			best_sol_robust = best_w
			best_TE_robust = candidate_fit
		counter_r += 1
	return best_sol_robust, best_TE_robust, R_robust 

def get_samples_TE(weights, S):
	samples_TE = [] # TE samples from S
	for s in S: # loop over all the simulation set S, where s = M_f
		portfolio_ret = np.matmul(weights, s[1:,:])
		TE = (portfolio_ret - s[0,:])**2 # s[0,t] is R_t, for simulation s
		samples_TE.append(np.sum(TE)/s.shape[1])
	return samples_TE

def get_TE(weights, s):
	portfolio_ret = np.matmul(s[:,1:], weights)
	TE = (portfolio_ret - s[:,0])**2 # s[0,t] is R_t, for simulation s
	return np.sum(TE)/s.shape[1]

def ms_non_dominated_sort(fit):
	F = [] # non-dominated fronts
	evaluating = [e for e in range(len(fit))]
	epsilon = 0#5*(10**(-6))
	while(len(evaluating)>0): # while there are elements being evaluated

		f_temp = [] # initialize a front
		for i in evaluating:
			# check if i is e-dominated by any element j in evaluating
			nonDominated = True # assume i is a non-dominated element
			j = 0
			eval_i = np.array(fit[i]) # the original nondominated sort only uses this line
			while nonDominated and j < len(evaluating):
				eval_j = np.array(fit[evaluating[j]]) # the original nondominated sort only uses this line
				count_dom = 0
				for s in range(len(eval_i)): # loop over all scenarios
					if(eval_j[s] < eval_i[s] - epsilon) : # if i is flipped-e-dominated in this scenario
						count_dom += 1
				if(count_dom == len(eval_i)): # check if i is flipped-e-dominated by at least one j
					nonDominated = False	
				j += 1
			if(nonDominated): # if i is not dominated
				f_temp.append(i) # add i to the current non-dominated front
		# remove elements that were included in the current front
		eval_temp = [] #remaining elements (dominated by elements of f_temp)
		for i in evaluating:
			if not i in f_temp:
				eval_temp.append(i)
		evaluating = eval_temp

		# update the non-dominated fronts vec
		F.append(f_temp)

		'''
		# save some computation by stopping the algorithm earlier
		new_P = [] # simulates a new pop
		for f in F: # loop over all current fronts
			new_P.extend(f)
		if(len(new_P) >= nIndividuals): 
			evaluating = 0 # the number of fronts are sufficient to compute the new population
		'''
	return F

def set_new_pop(F, nIndividuals, cdist = None, lastFrontIdx = None, P = []):

	if lastFrontIdx == None:
		#print("F: " + str(F))
		new_P = []
		lastidx_i = 0
		while(len(new_P) + len(F[lastidx_i]) < nIndividuals):
			new_P.extend(F[lastidx_i])
			lastidx_i += 1

		# get frank for each individual in the new population
		frank = dict([])
		for i in range(lastidx_i + 1): # loop over all fronts
			front = F[i]
			# index ranking for each obj 
			for p in front:
				frank[p] = i + 1

		return new_P, lastidx_i, frank

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
		sorted_idx = np.argsort(cdist_unranked)
		#print("remaining: " + str(remaining_elements))
		#print("sorted_idx: " + str(sorted_idx))
		for i in range(remaining_elements):
			new_P.append(front[sorted_idx[i]])

		#print("final_P: " + str(new_P))

		return new_P


def get_scenarios_extremes(fit):
	fit = np.array(fit)
	max_s = [] # maximum of each scenario
	min_s = [] # minimum of each scenario
	for s in range(fit.shape[1]):
		max_s.append(np.max(fit[:,s]))
		min_s.append(np.min(fit[:,s]))
	return max_s, min_s

def get_reference_point_ms(max_s, min_s):
	ref_z = 0.5*(np.array(max_s) + np.array(min_s)) # the reference point is the mean of the max and min in each scenario
	#ref_z = min_s
	return ref_z

def get_CR_metric_old(fit, ref_z, max_s, min_s):
	fit = np.array(fit)
	CR_metric = [-1 for ind in range(fit.shape[0])]
	dist = np.zeros((fit.shape[0], fit.shape[1]))
	rank_s = []
	for s in range(fit.shape[1]): # loop over all scenarios
		# normalize the fitness in each scenario
		norm_fit = ((fit[:,s] - min_s[s])/(max_s[s]-min_s[s]))
		ref_z_norm = (ref_z[s] - min_s[s])/(max_s[s]-min_s[s]) # normalize ref point
		# get the distance relative to the ref_point
		dist[:,s] = (norm_fit[:] - ref_z_norm)**2
	# sort individuals in each scenario
	rank_ = np.argsort(dist, axis = 0)
	CR_metric_final = dict([]) # convert to a dict
	for ind in range(fit.shape[0]):
		# compute the CR_metric of this individual
		if CR_metric[ind] == -1: # if 'ind' is not an extreme point in a scenario
			ranks = []
			for s in range(rank_.shape[1]):
				this_rank = np.where(rank_[:,s] == ind)
				ranks.append(this_rank[0][0])
			ranks = np.array(ranks)
			#print("ind" + str(ind) + " before: " + str(ranks) + "after: " + str(ranks[np.argsort(ranks)]))
			CR_metric[ind] = np.mean(ranks) + 2*np.std(ranks)
		CR_metric_final[ind] = CR_metric[ind] 

	return CR_metric_final

def get_CR_metric_new(fit, max_s, min_s):
	fit = np.array(fit)
	CR_metric_final = dict([]) # convert to a dict
	#for ind in range(fit.shape[0]):


def get_CR_metric(fit, max_sold, min_sold):
	fit = np.array(fit)
	max_s = np.array([np.max(fit[:,s]) for s in range(fit.shape[1])])
	min_s = np.array([np.min(fit[:,s]) for s in range(fit.shape[1])])
	#print("max: " + str(max_s))
	#print("min: " + str(min_s))
	dist_ref_max = np.zeros((fit.shape[0], fit.shape[1])) # dist(x_s, max_s)
	dist_ref_min = np.zeros((fit.shape[0], fit.shape[1])) # dist(x_s, min_s)
	dist_max = np.zeros((fit.shape[0], fit.shape[1])) # dist(x_s, max_s)
	dist_min = np.zeros((fit.shape[0], fit.shape[1])) # dist(x_s, min_s)
	CR_metric_final = dict([]) # convert to a dict
	for s in range(fit.shape[1]): # loop over all scenarios
		#dist[:,s] = (norm_fit[:] - ref_z[s])**2
		dist_ref_max[:,s] = (fit[:,s] - max_s[s])**2
		dist_ref_min[:,s] = (fit[:,s] - min_s[s])**2
		dist_max[:,s] = (fit[:,s] - max_s[s])
		dist_min[:,s] = (fit[:,s] - min_s[s])
	# sort individuals in each scenario
	for ind in range(fit.shape[0]):
		# compute the CR_metric of this individual
		X_min = []
		X_max = []
		sorted_fit = fit[ind,:]
		sorted_fit = sorted_fit[np.argsort(sorted_fit)]
		#print("ind" + str(ind) + ", fit: " + str(fit[ind,:]))
		#print("dist_max: " + str(dist_max[ind,:]))
		#print("dist_min: " + str(dist_min[ind,:]))	
		mean_dmin = np.mean(dist_min[ind,:])
		std_dmin = np.std(dist_min[ind,:])
		mean_dmax = np.mean(dist_max[ind,:])
		cluster = [sorted_fit[0]]
		cluster_cont = True
		#while s + 1 < (fit.shape[1]):
		for s in range(fit.shape[1]):	
			if np.absolute(dist_min[ind,s]) <  np.absolute(dist_max[ind,s]):
				X_min.append(fit[ind,s])
			else:
				X_max.append(fit[ind,s])
			'''
			# add elements to the cluster
			if cluster_cont and sorted_fit[s+1]/cluster[s] <= 1.2:
				cluster.append(sorted_fit[s+1])
			# stop adding elements to the cluster
			else:
				if len(X_min) == 0: # initialize X_min
					X_min = cluster
					cluster_cont = False
				X_max.append(sorted_fit[s+1])
			s += 1

		#print("sorted_fit: " + str(sorted_fit))
		#print("X_min: " + str(X_min) + ", X_max: " + str(X_max))
		if len(X_min) > 0:
			mean_xmin = np.mean(X_min)
		else:
			mean_xmin = np.mean(X_max)
		CR_metric_final[ind] = len(X_max) + mean_xmin

		'''
		'''
		# first case
		if mean_dmin < 0 and mean_dmax < 0:
			#print("first case, mean_dmin: " + str(mean_dmin) + ", mean_dmax: " + str(mean_dmax))
			CR_metric_final[ind] = len(X_max) + mean_dmin/(1+mean_dmax)
		# second case
		elif mean_dmin < 0 and mean_dmax > 0:
			#print("second case, mean_dmin: " + str(mean_dmin) + ", mean_dmax: " + str(mean_dmax))
			CR_metric_final[ind] = len(X_max) + mean_dmin/(1+mean_dmax)
		# third case
		elif mean_dmin > 0 and mean_dmax > 0:
			#print("third case, mean_dmin: " + str(mean_dmin) + ", mean_dmax: " + str(mean_dmax))
			CR_metric_final[ind] = len(X_max) + mean_dmin/(1-mean_dmax)
		# fourth case
		elif mean_dmin > 0 and mean_dmax < 0:
			#print("fourth case, mean_dmin: " + str(mean_dmin) + ", mean_dmax: " + str(mean_dmax))
			CR_metric_final[ind] = len(X_max) + mean_dmin/(1-mean_dmax)
		'''
		CR_metric_final[ind] = len(X_max) + np.mean(fit[ind,:])


		#CR_metric_final[ind] = np.mean(dist_min[ind,:])/np.mean(dist_max[ind,:])
		#CR_metric_final[ind] = -np.max(dist_max[ind,:])/(0.00000001 + np.min(dist_min[ind,:]))
		#print("CR_metric: " + str(CR_metric_final[ind]))

	return CR_metric_final


def ms_get_best_sol(P, R, S, nAssets, cr_metric):
	best_sol = []
	best_cr = 100000
	best_ind = -1
	for ind in P: # loop over the individuals contained in the best front
		#print("F[0], ind" + str(this_ind) + ", non-sorted" + str(TE_samples[:quantile]) + ", sorted: " + str(TE_samples[np.argsort(TE_samples[:quantile])]) + ", mean: " + str(np.mean(TE_samples[:quantile])))
		if cr_metric[ind] < best_cr	:
			best_sol =  R[ind,:]
			best_cr = cr_metric[ind]
			best_TE	=  np.mean(np.array(get_samples_TE(R[ind,:nAssets], S)))
			best_ind = ind
	#print("best_TE_samples: " + s
	return best_sol, best_TE, ind


def ms_get_best_sol_old(F, R, S, nAssets):
	best_sol = []
	best_TE = 100000
	for this_ind in F[0]: # loop over the individuals contained in the best front
		TE_samples = np.array(get_samples_TE(R[this_ind,:nAssets], S))
		#print("F[0], ind" + str(this_ind) + ", non-sorted" + str(TE_samples[:quantile]) + ", sorted: " + str(TE_samples[np.argsort(TE_samples[:quantile])]) + ", mean: " + str(np.mean(TE_samples[:quantile])))
		TE_test = np.mean(TE_samples)
		if TE_test < best_TE:
			best_sol =  R[this_ind,:]
			best_TE = TE_test
	#print("best_TE_samples: " + str(best_TE))
	return best_sol, best_TE

def ms_bin_tournament_selection(P, frank, cdist, fit):

	mating_pool = []

	# select samples so that each solution participates in two tournaments
	sample_part1 = random.sample(P, len(P))
	sample_part2 = random.sample(P, len(P))

	# run the tournaments
	for j in range(len(sample_part1)):
		#print("-----------------------------------------------")
		#print(str(sample_part1[j]) + " fit: " + str(fit[sample_part1[j]]))
		#print(str(sample_part2[j]) + " fit: " + str(fit[sample_part2[j]]))
		winner = sample_part1[j] # assume that elements in part1 are winners
		#print(str(sample_part1[j]) + " frank: " + str(frank[sample_part1[j]]))
		#print(str(sample_part2[j]) +" frank: " + str(frank[sample_part2[j]]))
		if frank[sample_part2[j]] < frank[sample_part1[j]]: # test best frontier
			winner = sample_part2[j]
		elif frank[sample_part2[j]] == frank[sample_part1[j]]:
			#print("equal ranks")
			#print(str(sample_part1[j]) + " cdist: " + str(cdist[sample_part1[j]]))
			#print(str(sample_part2[j]) + " cdist: " + str(cdist[sample_part2[j]]))
			if cdist[sample_part2[j]] < cdist[sample_part1[j]]: # test best rank, remember that the lower the rank (CR_metric) of the solution, the better it is
				winner = sample_part2[j]
		#print("winner is: " + str(winner))
		mating_pool.append(winner)

	return mating_pool

def ms_bin_tournament_selection_new(P, frank, cdist):

	mating_pool = []

	# select samples so that each solution participates in two tournaments
	sample_part1 = random.sample(P, len(P))
	sample_part2 = random.sample(P, len(P))

	# run the tournaments
	for j in range(len(sample_part1)):
		winner = sample_part1[j] # assume that elements in part1 are winners
		#print(str(sample_part1[j]) + " frank: " + str(frank[sample_part1[j]]))
		#print(str(sample_part2[j]) +" frank: " + str(frank[sample_part2[j]]))
		if frank[sample_part2[j]] < frank[sample_part1[j]]: # test best frontier
			winner = sample_part2[j]
		elif frank[sample_part2[j]] == frank[sample_part1[j]]:
			#print("equal ranks")
			rank_difference = cdist[sample_part1[j]] - cdist[sample_part2[j]]  
			#print("r1: " + str(cdist[sample_part1[j]]) + "r2: " + str(cdist[sample_part2[j]]) + "rank diff: " + str(rank_difference))
			rank_difference = np.where(rank_difference > 0)
			#print(len(rank_difference[0]))
			if len(rank_difference[0]) > 0.5*len(cdist[sample_part1[j]]): # test who wins in more scenarios
				winner = sample_part2[j]
				#print("sample_2 won!")
		#print("winner is: " + str(winner))
		mating_pool.append(winner)

	return mating_pool

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
	

	