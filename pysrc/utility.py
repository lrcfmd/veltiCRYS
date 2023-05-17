import os, glob
import statistics
import pickle

captime = 50000

def iterno(x):
	return(int(x.split('_')[-1].split('.')[0]))

def utility(folder, methods, lads):
	"""Function to calculate method utility for 
	structure relaxation.
	
	Parameters
	----------
	folder : str
		Path to structure folders
	methods : list (string)
		List of methods to calculate utilities for
	lads : list (double)
		Lambda values for preference declaration

	Returns
	-------
	dict[str, double]

	

	"""
	rates = {m: 0 for m in methods} # dict with method as key
	iters = [] # keep a list with iteration number and method per structure
	LEN = 0

	# Find successful experiments per method
	for struct in os.listdir(folder): # per structure folder
				
		# lists of iteration files per structure
		pkl_list = glob.glob(folder+struct+'/*.{}'.format('pkl'))
		cif_list = glob.glob(folder+struct+'/*.{}'.format('cif'))

		# per iteration
		pkl_list = sorted(pkl_list, key=iterno)
		cif_list = sorted(cif_list, key=iterno)

		# find eperiment's method
		method = None
		# if successful, keep total iterations
		iteration = None
			
		# get initial info
		with open(pkl_list[0], 'rb') as file:
			output = pickle.load(file)

		if len(pkl_list) > 1:
			# get info from first iteration
			with open(pkl_list[1], 'rb') as file:
				output = pickle.load(file)
				method = output['Method']
		else:
			print("No method found.")

		# get final info
		with open(pkl_list[-1], 'rb') as file:
			output = pickle.load(file)
			iteration = int(pkl_list[-1].split('_')[-1].split('.')[0])

			# check if the run was successful
			# if yes, count in method successes and keep iteration number
			try:
				if output['Optimised']:
					rates[method] += 1
					iteration = int(pkl_list[-1].split('_')[-1].split('.')[0])
					iters.append((method, iteration))
			except:
				print("Final iteration does not have outcome in data.") 


		# update length of structure collection
		LEN += 1

	# calculate utility per method
	utilities = {m: [] for m in methods}

	for lad in lads: # Find all experiment scores and get mean

		scores = {m: [] for m in methods}
		# Use successful experiments
		for pair in iters: # per structure folder

			method = pair[0]
			value = pair[1]
			success = rates[method]/LEN # calculate success rate

			# Formal Preferences
			score = lad*success + (1 - lad)*(1 - value/captime)
			scores[method].append(score)

		for method in scores.keys():
			u = statistics.mean(scores[method])
			print("Method:",method,"Lambda:",lad,"Utility:",u)
			utilities[method].append(u)

	return utilities


