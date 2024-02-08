import os, glob
import statistics
import pickle, re
import argparse
import matplotlib.pyplot as plt

captime = 70000

def iterno(x):
	return(int(x.split('_')[-1].split('.')[0]))

def get_records(folder, mainmethod, records={}):
	"""Function to get all iteration number
	records and outcome from experiments 
	across folders.
	
	Parameters
	----------
	folder : str
		Path to structure folders
	records : dict[str, list[tuple[int, bool]]]
		Dictionary with the information from the experiments

	Returns
	-------
	dict[str, list[tuple[int, bool]]]

	

	"""

	# Find successful experiments per method
	for struct in os.listdir(folder): # per structure folder

		# lists of iteration files per structure
		pkl_list = glob.glob(folder+struct+'/*.{}'.format('pkl'))
		cif_list = glob.glob(folder+struct+'/*.{}'.format('cif'))

		# per iteration
		pkl_list = sorted(pkl_list, key=iterno)
		cif_list = sorted(cif_list, key=iterno)

		# find experiment's method
		method = None
		# if successful, keep total iterations
		iteration = None
			
		# get initial info
		with open(pkl_list[0], 'rb') as file:
			output = pickle.load(file)

			if len(pkl_list) > 0:
				# get info from first iteration
				with open(pkl_list[0], 'rb') as file:
					output = pickle.load(file)

					# get whole method name from folder name
					method = mainmethod+'_'+folder.split('_'+mainmethod+'_')[1].split('/')[0]
					if method not in records.keys():
						records[method] = []

					# get info from final iteration
					with open(pkl_list[-1], 'rb') as file:
						output = pickle.load(file)
						iteration = int(pkl_list[-1].split('_')[-1].split('.')[0])
						try:
							records[method].append((iteration, output['Optimised']))
						except:
							print("Final iteration does not have outcome in data.") 
			else:
				print("No experiments found.")

	return records


def utility(records, lads):
	"""Function to calculate method utility for 
	structure relaxation.
	
	Parameters
	----------
	records : dict[str, list[tuple[int, bool]]]
		Dictionary with the information from the experiments

	Returns
	-------
	dict[str, dict[double, double]]

	

	"""
	methods_len = {method: len(records[method]) for method in records.keys()}
	print('Number of records per method:', methods_len) 

	# calculate utility per lambda per method
	utilities = {m: {lad: 0 for lad in lads} for m in records.keys()}
	for method, structs in records.items():

		# calculate success rate of method
		success = 0
		for struct in structs:
			if struct[1]:
				success += 1/methods_len[method]

		# get one lambda value mean for all structures
		for lad in lads:
			scores = []
			for struct in structs:
				# Formal Preferences 
				score = lad*success # c_0
				if struct[1]:
					score += (1-lad)*(1-struct[0]/captime) # c_1*utility
					scores.append(score)
			# get mean for all experiments of this method
			if len(scores)>0:
				u = statistics.mean(scores)
			else:
				u = 0
			utilities[method][lad] = u

	return utilities


def plot_utilities(utilities, path, mainmethod):
	fig, ax = plt.subplots(figsize=(10,10))
	cg_colors = [[205/255,133/255,63/255],
		[255/255,99/255,71/255],
		[238/255,92/255,66/255], 
		[205/255,79/255,57/255], 
		[139/255,54/255,38/255],
		[139/255,60/255,50/255]]
	gd_colors = [[152/255,251/255,152/255],
		[154/255,255/255,154/255],	 
		[144/255,238/255,144/255],	 
		[124/255,205/255,124/255],	 
		[84/255,139/255,84/255],
		[84/255,200/255,90/255]]
	linestyles = ['--', ':', '-', '-.']

	idg, idc = 0, 0
	methods = utilities.keys()
	for method in methods:
		X, y = utilities[method].keys(), utilities[method].values()
		if "GD" in method:
			ax.plot(X, y, 
				color=gd_colors[idg%6], label=method,
				linestyle=linestyles[idg%4], linewidth=5)
			idg += 1
		else:
			ax.plot(X, y,
				color=cg_colors[idc%6], label=method, 
				linestyle=linestyles[idc%4], linewidth=5)
			idc += 1

	ax.grid(zorder=0, linestyle = '--')
	ax.set_title("Scoring Function for "+mainmethod, fontsize=33)

	# get current handles and labels
	handles, labels = plt.gca().get_legend_handles_labels()

	# remove extra name
	labels = [label.lstrip(mainmethod+'_') for label in labels]

	# place in order
	num_labels = []
	for label in labels:
		try:
			num_labels.append("{:.1f}".format(float(label)))
		except:
			num_labels.append(label)
	indices = sorted(range(len(num_labels)), key=lambda k: num_labels[k])

	# call plt.legend() with the new values
	plt.legend([handles[i] for i in indices],[labels[i] for i in indices], prop={'size': 23})

	ax.set_ylabel("Score", fontsize=23)
	ax.set_xlabel("Lambda", fontsize=23)
	ax.tick_params(axis='both', which='major', labelsize=23)
	ax.set_ylim(-0.05, 1.05)
	plt.tight_layout()
	plt.show()

	fig.savefig(path+"utility"+"".join(mainmethod)+".pdf")

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
		help='Method to evaluate utility on')
	parser.add_argument(
		'-o', metavar='--output', type=str,
		default='./',
		help='Folder to place plot')
	
	args = parser.parse_args()
	utilities = {}
	lads = [lad/1000 for lad in range(0, 1000)]
	lads.append(1) 

	records = {}
	for folder in args.f:
		records = get_records(folder, args.m, records)
	utilities = utility(records, lads)

	plot_utilities(utilities, args.o, args.m)
	print(utilities)


