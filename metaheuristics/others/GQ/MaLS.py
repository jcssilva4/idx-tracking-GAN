from random import sample
import numpy as np
import random
import time

def MaLS(Sglobal, Scg, fvalGlobal, selected_solver, t0, t1_orig, r, timelimit):

    # Sg is the best combination of stocks (best global solution)
    # fvalG is the min of obj function value from quadraticProgramming(Sg)
    Sg = [element for element in Sglobal]
    Sc = [element for element in Scg]
    fvalg = fvalGlobal
    gain = 1
    iterMaLS = 0
    randomFit = -1 #do not use random fit
    t1 = t1_orig

    #fprintf('\nstarting Major Local Search\n');
    while gain and t1-t0 < timelimit:

        iterMaLS += 1
        index_i = []
        index_j = []
        index_fval = []
        for i in range(len(Sg)):

            for j in range(len(Sc)):

                St = [element for element in Sg] # we need to do this again when this loop restarts, since St is sorted
                St[i] = Sc[j]
                St.sort()
                S_binary_encoded = []
                for ii in range(r.shape[0]-1):
                    if(ii in St):
                        S_binary_encoded.append(1)
                    else:
                        S_binary_encoded.append(0)

                fvalt = selected_solver.solving(S_binary_encoded,1)
                if fvalt < fvalg:
                    index_i.append(i)
                    index_j.append(j)
                    index_fval.append(fvalt)

        if len(index_i) == 0:
            gain = 0

        else:

            if random.uniform(0,1) <= randomFit:

                randIdx = sample([e for e in range(len(index_i))],1)
                a = Sc[index_j[randIdx]]
                b = Sg[index_i[randIdx]]
                Sg[index_i[randIdx]] = a
                Sc[index_j[randIdx]] = b
                fvalg = index_fval[randIdx]

            else:

                x = np.argmin(index_fval)
                fvalg = index_fval[x]
                a = Sc[index_j[x]]
                b = Sg[index_i[x]]
                Sg[index_i[x]] = a
                Sc[index_j[x]] = b

        t1 = time.time()

    return Sg, fvalg, t1


def print_gap(fval_heur):

    fval_cplex = 0.0028069754546620022 # wang2012_k10_cplex_opt_sol
    gap = 100*((fval_heur - fval_cplex)/abs(fval_cplex))
    print("GAP: " + str(gap))
