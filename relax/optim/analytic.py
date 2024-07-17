import os
import numpy as np
import pickle
import time

from ase.io import read as aread
from ase.cell import Cell
from ase.geometry import wrap_positions
from ase.io import write

from relax.analytic_potentials.potential import *
from relax.analytic_potentials.coulomb.coulomb import Coulomb
from relax.analytic_potentials.buckingham.buckingham import Buckingham
from relax.finite_differences import finite_diff_grad

import shutil
COLUMNS = shutil.get_terminal_size().columns
def prettyprint(dict_):
	np.set_printoptions(suppress=True)
	words = ""
	for key, value in dict_.items():
		if key=="Total energy":
			words += key+" "+str(value)+" "
		else:
			print("\n", key)
			print(value)
	print(words.center(COLUMNS,"-"))
 

def init(charge_dict, atoms, outdir):
	"""  INITIALISATION  """
	vects 				= np.asarray(atoms.get_cell())
	scaled_pos 			= np.asarray(atoms.get_scaled_positions())
	chemical_symbols	= np.array(atoms.get_chemical_symbols())
	N 					= len(atoms.positions)
	initial_energy 		= 0

	# Define Coulomb potential object
	libfile = "libraries/madelung.lib"
	Cpot = Coulomb(
		chemical_symbols=chemical_symbols,
		N=N,
		charge_dict=charge_dict,
		filename=libfile)
	Cpot.set_cutoff_parameters(
		vects=vects, 
		N=N)
	initial_energy += Cpot.energy(atoms)
	
	# Define Buckingham potential object
	libfile = "libraries/buck.lib"
	Bpot = Buckingham(
		filename=libfile, 
		chemical_symbols=chemical_symbols, 
		)
	Bpot.set_cutoff_parameters(
		vects=vects, 
		N=N)
	initial_energy += Bpot.energy(atoms)

	# Print Ewald-related parameters
	Cpot.print_parameters()
	Bpot.print_parameters()

	potentials = {}	
	potentials['Coulomb'] = Cpot
	potentials['Buckingham'] = Bpot

	if not os.path.isdir(outdir):
		os.mkdir(outdir) 

	prettyprint({
		'Chemical Symbols':chemical_symbols, 
		'Positions':atoms.positions, \
		'Cell':atoms.get_cell(), 
		'Electrostatic energy':Cpot.get_all_ewald_energies(), 
		'Interatomic energy':Bpot.get_all_ewald_energies(), \
		'Total energy':initial_energy})
 
	return potentials, vects, scaled_pos, initial_energy
 

def repeat(atoms, outdir, outfile, charge_dict, line_search_fn,
	optimizer, usr_flag=False, out=1, **kwargs):
	"""The function that performs the optimisation. It calls repetitively 
	iter_step for each updating step.

	Parameters
	----------
	init_energy : double
		The energy of the initial configuration.
	atoms : Python ASE's Atoms instance.
		Object with the parameters to optimise.
	potentials : dict[str, Potential]
		Dictionary containing the names and Potential 
		instances of the energy functions to be used 
		as objective functions.
	outdir : str
		Name of the folder to place the output files.
	outfile : str
		Name of the output files.
	step_func : function
		The function to be used for line minimisation.
	direction_func : function
		The function to be used for the calculation of
		the direction vector (optimiser).
	usr_flag : bool
		Flag that is used to stop after each iteration
		and wait for user input, if true.
	max_step : float
		The upper bound of the step size.
	out : int
		Frequency of produced output files -- after 
		how many iterations the ouput should be written
		to a file.

	Returns
	-------
	(int, dict[str, _])
	
	"""
	start_time = time.time()

	strains = np.ones((3,3))
	potentials, vects, _, initial_energy = init(charge_dict, atoms, outdir)
	pos = atoms.get_positions()
	N = len(pos)

	final_iteration = None
	history = []
	outfile = 'analytic_' + outfile

	if not os.path.isdir(outdir+"imgs"):
		os.mkdir(outdir+"imgs")
	if not os.path.isdir(outdir+"structs"):
		os.mkdir(outdir+"structs")
	if not os.path.isdir(outdir+"imgs/"+outfile+"/"):
		os.mkdir(outdir+"imgs/"+outfile)
	if not os.path.isdir(outdir+"structs/"+outfile+"/"):
		os.mkdir(outdir+"structs/"+outfile)
	
	# Gradient for current point on PES
	grad = np.zeros((N+3,3))
	for name in potentials:
		if hasattr(potentials[name], 'gradient'):
			grad += np.array(potentials[name].gradient(
			pos_array=pos, vects_array=vects, N_=N))
	# Print numerical derivatives
	if 'debug' in kwargs:
		if kwargs['debug']:
			finite_diff_grad(
				atoms, grad, N, strains, 0.00001, potentials)
	# Gradient norm
	gnorm = get_gnorm(grad, N)
	optimizer.lnscheduler.gnorm = gnorm

	# Normalise gradient
	if gnorm>0:
		grad_norm = grad/gnorm
	else:
		grad_norm = 0
  
	# Keep info of this iteration
	iteration = {
	'Time': time.time()-start_time,
	'Gradient':grad, 'Positions':atoms.positions.copy(), 
	'Strains':strains, 'Cell':np.array(atoms.get_cell()), 'Iter':optimizer.iterno, 
	'Step': 0, 'Gnorm':gnorm, 'Energy':initial_energy}
 
	# Iterations
	while(True):
		final_iteration = iteration
  
		# Check for termination
		prettyprint(iteration)
		if optimizer.completion_check(gnorm):
			print("Writing result to file",
			outfile+"_"+str(optimizer.iterno),"...")
			write(outdir+"imgs/"+outfile+"/"+outfile+"_"+\
				str(optimizer.iterno)+".png", atoms)
			write(outdir+"structs/"+outfile+"/"+outfile+"_"+\
				str(optimizer.iterno)+".cif", atoms)
			dict_file = open(
				outdir+"structs/"+outfile+"/"+outfile+"_"+\
				str(optimizer.iterno)+".pkl", "wb")
			pickle.dump(
				{**iteration, 'Optimised': True}, 
				dict_file)
			dict_file.close()				
			break
		elif (optimizer.iterno%out)==0:
			print("Writing result to file",
			outfile+"_"+str(optimizer.iterno),"...")
			write(outdir+"imgs/"+outfile+"/"+outfile+"_"+\
				str(optimizer.iterno)+".png", atoms)
			write(outdir+"structs/"+outfile+"/"+outfile+"_"+\
				str(optimizer.iterno)+".cif", atoms)
			dict_file = open(
				outdir+"structs/"+outfile+"/"+outfile+"_"+\
				str(optimizer.iterno)+".pkl", "wb")
			pickle.dump(
				{**iteration, 'Optimised': False}, 
				dict_file)
			dict_file.close()
		if (('iterno' in kwargs) & (kwargs['iterno'] <= optimizer.iterno)):
			break
 
		if usr_flag:
			usr = input()
			if 'n' in usr:
				return iteration
  
		''' 1 --- Apply an optimization step --- 1 '''
		params = np.concatenate((pos, np.ones((3,3))), axis=0)
		params = optimizer.step(grad_norm, gnorm, params, line_search_fn,
						  atoms=atoms, potentials=potentials)

		# Make a method history
		history.append(type(optimizer).__name__)

		''' 2 --- Update parameters --- 2 '''
		# Make sure ions stay in unit cell
		pos_temp = wrap_positions(params[:N], vects)
		# Update strains
		strains = (params[N:]-1)+np.identity(3)
	
		# Apply strains to all unit cell vectors
		pos = pos_temp @ strains.T
		vects = vects @ strains.T
  
		# Calculate new point on energy surface
		atoms.positions = pos
		atoms.set_cell(vects)
  
		# Assign parameters calculated with altered volume
		for name in potentials:
			if hasattr(potentials[name], 'set_cutoff_parameters'):
				potentials[name].set_cutoff_parameters(vects, N)
    
		''' 3 --- Re-calculate energy --- 3 '''
		# Calculate energy on current PES point
		energy =0
		for name in potentials:
			if hasattr(potentials[name], 'energy'):
				energy += potentials[name].energy(atoms)
    
		''' 4 --- Re-calculate derivatives --- 4 '''
		# Gradient of this point on PES
		grad = np.zeros((N+3,3))
		for name in potentials:
			grad += np.array(potentials[name].gradient(atoms))
		# Gradient norm
		gnorm = get_gnorm(grad,N)
		# Normalise gradient
		if gnorm>0:
			grad_norm = grad/gnorm
		else:
			grad_norm = 0

		# Print numerical derivatives
		if 'debug' in kwargs:
			if kwargs['debug']:
				finite_diff_grad(
					atoms, grad, N, params[N:], 0.00001, potentials)
			
		iteration = {
		'Time': time.time()-start_time,
		'Gradient':grad, 'Positions':atoms.positions.copy(), 
		'Strains':strains, 'Cell':np.array(atoms.get_cell()), 
		'Iter':optimizer.iterno, 'Method': history[-1], 
		'Step':optimizer.lnscheduler.curr_step, 'Gnorm':gnorm, 'Energy':energy}

	return final_iteration

