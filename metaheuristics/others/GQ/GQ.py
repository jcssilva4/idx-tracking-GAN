from metaheuristics.GQ.stock_selection import *
from metaheuristics.GQ_MiLS.phiD_MiLS import *
from metaheuristics.GQ.MaLS import *
import time
import random

def GQ(model, r, k, T, num_executions, target_sol):

    time_solutions = dict([]) # dictionary containing CPUtime obtained in each run of the approx algorithm
    time_to_target_sol = dict([]) # dictionary containing time to obtain the target solution in each run of the approx algorithm
    f_solutions = dict([]) # dictionary containing f_val obtained in each run of the approx algorithm
    it_solutions = dict([]) # dictionary containing niters obtained in each run of the approx algorithm
    w_solutions = dict([]) # dictionary containing  w vector obtained in each run of the approx algorithm


    if model == "ruiz-torrubiano2009_relaxed":
        import metaheuristics.GA_torrubiano.solver_torrubiano as solver

    elif model == "wang2012_relaxed":
        import metaheuristics.GA_torrubiano.solver_wang as solver

    selected_solver = solver.Solve(T, r)

    # GQ parameters
    timelimit = 300 # stopping criteria_1: 5 min
    max_iter = 50  # stopping criteria_2: max number of iterations
    RCL_size = round(0.45*(r.shape[0]-1)) # round to the closest integer
    pMiLS = 0.4
    UB_MiLS = round(0.15*RCL_size) # round to the closest integet

    for run in range(num_executions): 

        print("execution = " + str(run+1) + " - GQ")

        t0=time.time()
        t1=time.time()
        nit, bestfi, current_bestfi, iter_improv = 0, 1000, 1000, 0
        at_least_cplex_sol = False
        time_to_target = 0
        S_final = []
        S_c_final = []

        while abs(t0-t1)<timelimit and nit < max_iter:

            current_bestfi = bestfi

            # Solution construction
            S, S_c, f_val_temp = get_S(selected_solver, r, T, k, RCL_size)

            if random.uniform(0,1) <= pMiLS:
                S_mils, fval_mils, Scmils = MiLS(UB_MiLS, S, S_c, f_val_temp, selected_solver, t0, t1, timelimit, r, 1)
                if fval_mils < f_val_temp:
                    f_val_temp = fval_mils
                    S = [element for element in S_mils]
                    S_c = [element for element in Scmils]

            # finished one GRASP iteration
            t1 = time.time()
            nit+=1

            if f_val_temp < current_bestfi:
                bestfi = f_val_temp
                S_final = S
                S_c_final = S_c
                # at least a solution better than cplex?
                at_least_cplex_sol, time_to_target = check_target(bestfi, at_least_cplex_sol, time_to_target, t1, t0, target_sol)

        # MaLS
        S_final, fval_mals, t1_mals = MaLS(S_final, S_c_final, bestfi, selected_solver, t0, t1, r, timelimit)

        # at least a solution better than cplex?
        at_least_cplex_sol, time_to_target = check_target(fval_mals, at_least_cplex_sol, time_to_target, t1_mals, t0, target_sol)

        # get portfolio composition given the UNIVERSE of assets
        final_portfolio = get_final_portfolio(S, selected_solver, r)

        # get running time and target to solution
        time_solutions[str(run) + "mils"] = t1 - t0
        time_solutions[str(run) + "mals"] = t1_mals - t0
        if not at_least_cplex_sol:
            time_to_target = t1_mals - t0
        time_to_target_sol[str(run)] = time_to_target
		
        # get solution and iterations
        f_solutions[str(run) + "mils"] =  bestfi
        f_solutions[str(run) + "mals"] =  fval_mals
        w_solutions[str(run)] =  final_portfolio
        it_solutions[str(run)] = nit

    return time_solutions, f_solutions, it_solutions, time_to_target_sol,  w_solutions

def get_final_portfolio (S, selected_solver, r):

    S_binary_encoded = []
    for i in range(r.shape[0]-1):
        if(i in S):
            S_binary_encoded.append(1)
        else:
            S_binary_encoded.append(0)

    w_vec = selected_solver.solving(S_binary_encoded)
    #print("w_vec:" + str(w_vec))
    #print("S:" + str(S))
    final_portfolio = []

    for i in range(r.shape[0]-1):
        if(i in S):
            final_portfolio.append(w_vec[S.index(i)])
        else:
            final_portfolio.append(0)

    return final_portfolio

def check_target(fval, at_least_cplex_sol, time_to_target, t1, t0, target_sol):

    if(fval <= target_sol and not at_least_cplex_sol):
        at_least_cplex_sol = True
        time_to_target = t1 - t0

    return at_least_cplex_sol, time_to_target


def print_gap(fval_heur):

    fval_cplex = 0.0028069754546620022 # wang2012_k10_cplex_opt_sol_idtrack1
    #fval_cplex = 0.0021865721119429964 # wang2012_k10_cplex_opt_sol_idtrack2
    gap = 100*((fval_heur - fval_cplex)/abs(fval_cplex))
    print("GAP: " + str(gap) + "/ fval: " + str(fval_heur))


