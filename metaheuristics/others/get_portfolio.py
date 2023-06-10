from metaheuristics.GA_torrubiano.GA_torrubiano import GA_torrubiano 
from metaheuristics.GA_santanna.GA_santanna import GA_santanna 
from metaheuristics.GQ.GQ import GQ
from metaheuristics.GQ_MiLS.GQ_MiLS import GQ_MiLS
import numpy as np

def get_solution_approx(model, selected_metaheuristic, assets_return, index_returns, k, T, num_executions, target_sol):

	'''
	the following dictionaries contain information about the results obtained in each run of the metaheuristic:

	time_solutions = dict([]) # dictionary containing CPUtime obtained in each run of the approx algorithm
	time_to_target_sol = dict([]) # dictionary containing time to obtain the target solution in each run of the approx algorithm
	f_solutions = dict([]) # dictionary containing f_val obtained in each run of the approx algorithm
	it_solutions = dict([]) # dictionary containing niters obtained in each run of the approx algorithm
	w_solutions = dict([]) # dictionary containing  w vector obtained in each run of the approx algorithm

	- where: dict.key = "execution index"
	'''

	allReturns = [index_returns]
	allReturns.extend(assets_return)
	r = np.array(allReturns)

	if selected_metaheuristic == "GA_ruiz-torrubiano2009":

		print("select assets with GA_ruiz-torrubiano2009")
		time_solutions, f_solutions, it_solutions, time_to_target_sol,  w_solutions = GA_torrubiano(model, r, k, T, num_executions, target_sol)

	elif selected_metaheuristic == "GA_santanna2017":

		print("select assets with GA_santanna2017")
		time_solutions, f_solutions, it_solutions, time_to_target_sol, w_solutions = GA_santanna(model, r, k, T, num_executions, target_sol)

	elif selected_metaheuristic == "GQ":

		print("select assets with GQ")
		time_solutions, f_solutions, it_solutions, time_to_target_sol, w_solutions = GQ(model, r, k, T, num_executions, target_sol)

	elif selected_metaheuristic == "GQ-MiLS":

		print("select assets with GQ-MiLS")
		time_solutions, f_solutions, it_solutions, time_to_target_sol, w_solutions = GQ_MiLS(model, r, k, T, num_executions, target_sol)

	return time_solutions[0], f_solutions[0], it_solutions[0], time_to_target_sol[0], w_solutions[0]