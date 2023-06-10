
from models.torrubiano_model import *
from models.wang_model import *

def solveIT(selected_model, opt, orig_r, T, S):

	sub_r = get_subset_r(orig_r, S)

	# choose a model and solve
	if selected_model == "ruiz-torrubiano2009_relaxed":
		print("UNDER CONSTRUCTION")

	elif selected_model == "wang2012_relaxed":
		[fval, S_weights] = solve_wang_relaxed(sub_r, T)

	#choose objects of interest
	if opt == "fval":
		return fval

	elif opt == "w":
		return S_weights

	elif opt == "all":
		return fval, S_weights


def get_subset_r(r, S):

	sub_r_temp = [r[0,:]]
	for s in S:
		sub_r_temp.append(r[1+s,:])

	sub_r = np.array(sub_r_temp)
	print(sub_r.shape)
	return 	sub_r




