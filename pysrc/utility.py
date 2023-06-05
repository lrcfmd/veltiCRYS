import os, glob
import statistics
import pickle
import argparse
import matplotlib.pyplot as plt

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
	if not methods:
		raise ValueError('No method defined.')

	rates = {} # dict with method as key
	iters = [] # keep a list with iteration number and method per structure
	LEN = 0
	step_methods = []

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
				method = output['Method']+'_'+folder.split('_')[-1].split('/')[0]
				if method not in rates.keys():
					rates[method] = 0
				if method not in step_methods:
					step_methods.append(method)
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
	utilities = {m: [] for m in step_methods}

	for lad in lads: # Find all experiment scores and get mean

		scores = {m: [] for m in step_methods}
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


def plot_utilities(lads, utilities, path):
	fig, ax = plt.subplots(figsize=(10,10))
	cg_colors = [[205/255,133/255,63/255],
		[255/255,99/255,71/255],
		[238/255,92/255,66/255], 
		[205/255,79/255,57/255], 
		[139/255,54/255,38/255]]
	gd_colors = [[152/255,251/255,152/255],
		[154/255,255/255,154/255],	 
		[144/255,238/255,144/255],	 
		[124/255,205/255,124/255],	 
		[84/255,139/255,84/255]]
	linestyles = ['-', ':', '--', '-.', '-']

	idg, idc = 0, 0
	for method in utilities.keys():
		if "GD" in method:
			ax.plot(lads, list(utilities[method]), 
				color=gd_colors[idg%5], label=method,
				linestyle=linestyles[idg%5], linewidth=5)
			idg += 1
		else:
			ax.plot(lads, list(utilities[m]), 
				color=cg_colors[idc%5], label=method, 
				linestyle=linestyles[idc%5], linewidth=5)
			idc += 1

	ax.grid(zorder=0, linestyle = '--')
	ax.legend(prop={'size': 23})
	ax.set_title("Utility Function", fontsize=33)

	ax.set_ylabel("Score", fontsize=23)
	ax.set_xlabel("Lambda", fontsize=23)
	ax.tick_params(axis='both', which='major', labelsize=23)
	plt.tight_layout()
	plt.show()

	fig.savefig(path+"utility.pdf")

	return utilities

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description='Define input')
	parser.add_argument(
		'-f', metavar='--folders', type=str,
		nargs='+', default=[],
		help='Folder with structures.')
	parser.add_argument(
		'-m', metavar='--method', type=str,
		nargs='+', default=[],
		help='Method to evaluate utility on')
	parser.add_argument(
		'-o', metavar='--output', type=str,
		default='./',
		help='Folder to place plot')
	
	args = parser.parse_args()
	utilities = {}
	lads = [lad/1000 for lad in range(0, 1000)]

	for folder in args.f:
		utilities.update(utility(folder, args.m, lads))

	plot_utilities(lads, utilities, args.o)


