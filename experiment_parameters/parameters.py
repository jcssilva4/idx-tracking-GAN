def get_parameters():
	print("reading parameters file...")
	parameters = dict([])
	filehandle = [raw for raw in open("experiment_parameters/parameters.txt", 'r')]
	#print(filehandle)

	for index, elem in enumerate(filehandle):
		# get GAN's parameters
		if "total_epochs" in elem:
			parameters["total_epochs"] = int(filehandle[index+1])
			#print("total_epochs: " + str(parameters["total_epochs"]))
		if "delta_epoch" in elem:
			parameters["delta_epoch"] = int(filehandle[index+1])
			#print("delta_epoch: " + str(parameters["delta_epoch"]))
		if "nModels" in elem:
			parameters["nModels"] = int(filehandle[index+1])
			#print("nModels: " + str(parameters["nModels"]))
		if "analysis period" in elem:
			parameters["w"] = int(filehandle[index+1])
			#print("w: " + str(parameters["w"]))
		if "condition period" in elem:
			parameters["b"] = int(filehandle[index+1])
			#print("b: " + str(parameters["b"]))

		# get test parameters
		if "test_size" in elem:
			parameters["test_size"] = int(filehandle[index+1])
			#print("test_size: " + str(parameters["test_size"]))
		if "deltaT" in elem:
			parameters["deltaT"] = int(filehandle[index+1])
			#print("deltaT: " + str(parameters["deltaT"]))

		# get parameters for the GA with GAN-generated data
		if "n_sims (number of simulations" in elem:
			parameters["n_sims"] = int(filehandle[index+1])
			#print("n_sims: " + str(parameters["n_sims"]))
		if "objectives used" in elem:
			item_=  filehandle[index+1]
			item_ = item_.replace("[","")
			item_ = item_.replace("]","")
			item_ = item_.replace("\n","")
			item_ = item_.split(",")
			parameters["objs"] = item_
			#print("objs: " + str(parameters["objs"]))

		# get parameters for the GA with historical data 
		if "lookback_windows" in elem:
			item_=  filehandle[index+1]
			item_ = item_.replace("[","")
			item_ = item_.replace("]","")
			item_ = item_.replace("\n","")
			item_ = [int(num) for num in item_.split(",")]
			parameters["lookback_windows"] = item_
			#print("lookback_windows: " + str(parameters["lookback_windows"]))
		if "nRuns" in elem:
			parameters["nRuns"] = int(filehandle[index+1])
			#print("nRuns: " + str(parameters["nRuns"]))

		# get index tracking model parameters
		if "cardinality" in elem:
			parameters["K"] = int(filehandle[index+1])
			#print("K: " + str(parameters["K"]))
		if "lower bound" in elem:
			parameters["lb"] = float(filehandle[index+1])
			#print("lb: " + str(parameters["lb"]))
		if "upper bound" in elem:
			parameters["ub"] = float(filehandle[index+1])
			#print("ub: " + str(parameters["ub"]))

	return parameters