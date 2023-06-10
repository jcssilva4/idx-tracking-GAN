import metaheuristics.GA_torrubiano.population as populationn
import random as rand
import time

def GA_torrubiano(model, r, k, T, num_executions, target_sol):

	time_solutions = dict([]) # dictionary containing CPUtime obtained in each run of the approx algorithm
	time_to_target_sol = dict([]) # dictionary containing time to obtain the target solution in each run of the approx algorithm
	f_solutions = dict([]) # dictionary containing f_val obtained in each run of the approx algorithm
	it_solutions = dict([]) # dictionary containing niters obtained in each run of the approx algorithm
	w_solutions = dict([]) # dictionary containing  w vector obtained in each run of the approx algorithm


	# GA parameters
	mutation_rate, popsize, w = 85, 100, 1  # inserted torrubiano parameters
	timelimit = 300 # stopping criteria 1: 5 min
	max_iter = 100  # stopping criteria 2: max number of iterations the algorithm can perform to improve best_fit
	num_mutation = 1 # 1 for single mutation 2 for double mutation

	pop = populationn.Population(model, popsize, k, num_mutation, r, T)
	id = pop.get_id()

	for run in range(num_executions): 

		print("execution = " + str(run+1) + " - GA_torrubiano")

		pop.gen_pop()
		population=pop.population.copy()    ##
		fitness=pop.fit_all(population,id)
		t0=time.time()
		t1=time.time()
		nit, bestfi, current_bestfi, iter_improv = 0, 100, 100, 0
		at_least_cplex_sol = False
		time_to_target = 0

		#while abs(t0-t1) < timelimit and nit < max_iter:
		while nit < max_iter:

			current_bestfi = bestfi

			parent1,parent2=pop.get_parents(population)
			child1=[]
			child1=pop.rar(parent1,parent2,w)
			binchild1=pop.bin(child1)

			m1=rand.randint(1,100)
			if m1<=mutation_rate:
				binchild1=pop.mutation(binchild1)

			child1=pop.tencode(binchild1)
			fitchild1=pop.fit(binchild1,id)
			population.append(child1)
			fitness.append(fitchild1)
			w1=maxx(fitness)

			population.pop(w1)
			fitness.pop(w1)
			t1=time.time()

			bestfi,bi=bestfit(fitness,2)

			if(bestfi <= target_sol and not at_least_cplex_sol):
				at_least_cplex_sol = True
				time_to_target = t1 - t0

			nit+=1 # go to the next iteration

		# get portfolio composition given the UNIVERSE of assets
		parent_idxs = population[bi]
		parents_weights = pop.fit(pop.bin(population[bi]),id,1)
		final_portfolio = []
		for asset_idx in range(r.shape[0]-1):
			if(asset_idx+1 in parent_idxs):
				final_portfolio.append(parents_weights[parent_idxs.index(asset_idx+1)])
			else:
				final_portfolio.append(0)

		time_solutions[str(run)] = t1 - t0
		if not at_least_cplex_sol:
			time_to_target = t1 - t0
		time_to_target_sol[str(run)] = time_to_target
		
		f_solutions[str(run)] =  bestfi
		w_solutions[str(run)] =  final_portfolio
		it_solutions[str(run)] = nit


	return time_solutions, f_solutions, it_solutions, time_to_target_sol, w_solutions




def maxx(list):                      #returns the index of the worst fitness parent
    i1=None
    w1=-100000
    #print("len(list): %d"%len(list))
    #print("Max given list: %s"%list)
    for j in range(len(list)):
        if list[j]>w1:
            w1,i1=list[j],j
    return i1

def worstfit(list,control=0):    
    aux,ind=0,None
    for i in range(len(list)):
        if list[i]>aux:
            ind=i
            aux=list[i]
    if control==0:
        return aux
    if control==1:
        return ind

def bestfit(list,control=0):   
    fit,ind=float('inf'),None
    for i in range(len(list)):
        if list[i]<fit:
            fit=list[i]
            ind=i
    if control==0:
        return fit
    if control==1:
        return ind
    if control==2:
        return fit,ind

def checkl(element,list):            #check if this element is already on a list
    for i in list:
        if element==i:
            return 1
    if len(element)==0:
        return 2
    return 0

def check(element1,element2):
    element1,element2=round(element1,12),round(element2,12)
    if abs(element1-element2)<=objerror:
    #if element1==element2:
        return 1
    else:
        return 0