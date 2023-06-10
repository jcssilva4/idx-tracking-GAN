from random import sample
import time

def MiLS(UB_MiLS, S, S_c, fval, selected_solver, t0, t1, time_limit, r, swapQuant):

	SMiLS = [element for element in S]
	Scmils = [element for element in S_c]
	fvalMiLS = fval 
	improved = 0
	imils = 1
	t1_mils = t1

	while imils <= UB_MiLS and t1_mils-t0 < time_limit:
		
		St = [element for element in S] # temp S
		s1 = sample([i for i in range(len(S))], swapQuant) # draw swapQuant samples from S
		s2 = sample([i for i in range(len(S_c))], swapQuant) # draw swapQuant samples from S_c

        # exchange s1 by s2
		for smpl in range(swapQuant):
			St[s1[smpl]] = S_c[s2[smpl]] 

		# optimize
		St.sort()
		S_binary_encoded = []
		for ii in range(r.shape[0]-1):
			if(ii in St):
				S_binary_encoded.append(1)
			else:
				S_binary_encoded.append(0)
        
		fvalt = selected_solver.solving(S_binary_encoded,1) 

		# check solution quality
		if(fvalt < fvalMiLS):
			improved = 1  
			fvalMiLS = fvalt
			SMiLS = [element for element in St]
			Scmils = [element for element in S_c]
			for smpl in range(swapQuant):
				Scmils[s2[smpl]] = S[s1[smpl]]

		imils += 1
		t1_mils = time.time()

	return SMiLS, fvalMiLS, Scmils