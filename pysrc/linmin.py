import argparse, sys
import numpy as np
from math import log, exp

from cysrc.potential import *
from cysrc.buckingham import *
from cysrc.coulomb import *

from cysrc.cutoff import check_lattice
from cysrc.operations import get_all_distances, get_min_dist

from ase import Atoms
from ase.io import read as aread
from ase.visualize import view as aview
from ase.cell import Cell
from ase.geometry import wrap_positions
from ase.geometry import cell_to_cellpar, cellpar_to_cell

import shutil
from multiprocessing import Pool
from itertools import repeat
COLUMNS = shutil.get_terminal_size().columns


def calculate_temp_energy(potentials, init_energy, pos, vects, 
	strains, step_temp, direction, N, update):
	"""Wrapper function that calculates a new point on the PES
	and the energy value of this point. 

	Parameters
	----------
	init_energy : double
		The energy of the initial configuration.
	potentials : dict[str, Potential]
		Dictionary containing the names and Potential 
		instances of the energy functions to be used 
		as objective functions.
	strains : 3x3 array (double)
		The lattice strains of the current configuration.
	pos : Nx3 array (double)
		The Cartesian coordinates of the ions inside the unit cell.
	vects : 3x3 array (double)
		The lattice vectors in Cartesian coordinates.
	N : int
		Number of atoms in unit cell.
	step_temp : float
		The step size used for the current update.
	direction : Nx3 array (double)
		The direction vector as calculated with an optimiser 
		(same as the normalised gradient for Steepest Descent).
	update: list (str)
		List with the names of parameters to be optimised. Can
		be 'ions' and/or 'lattice'.

	Returns
	-------
	dict[str, _]
	
	"""
	# step = [.0, .0]
	pos_temp = pos.copy()
	vects_temp = vects.copy()
	strains_temp = strains.copy()

	# if not isinstance(step_temp, list):
	# 	step = [step_temp, step_temp]
	# elif len(step_temp)==1:
	# 	step = [step_temp[0], step_temp[0]]
	# else:
	# 	step = step_temp 

	words = "UP "+str(update)+"\n"
	print(words.center(COLUMNS," "))

	if 'ions' in update:
		# Update ion positions
		pos_temp = position_update(
			pos, step_temp, direction, vects_temp)

	if 'lattice' in update:
		# Update lattice
		vects_temp, strains_temp, pos_temp = lattice_update(
			potentials, strains, step_temp, direction, pos_temp, vects_temp)
	
	# Calculate temporary energy
	temp_energy, temp_val = 0, 0
	for name in potentials:
		if hasattr(potentials[name], 'calc'):
			temp_val = potentials[name].calc(
				pos_array=pos_temp, vects_array=vects_temp, N_=N)
			
			# If Buckingham energy is exactly 0
			# catastrophe check might have failed
			if (temp_val==0) & (name=='Buckingham'):
				return {}
			
			temp_energy += temp_val

	print("Initial energy: ",init_energy,
		" New energy: ",temp_energy, "Step: ",step_temp)

	return {'Energy': temp_energy, 'Positions': pos_temp, 'Cell': vects_temp,
	'Strains': strains_temp, 'Step': step_temp, 'Catastrophe': 0}


def position_update(pos, step_temp, direction, vects):
	"""Wrapper function that updates the ion positions.

	Parameters
	----------
	pos : Nx3 array (double)
		The Cartesian coordinates of the ions inside the unit cell.
	vects : 3x3 array (double)
		The lattice vectors in Cartesian coordinates.
	N : int
		Number of atoms in unit cell.
	step_temp : float
		The step size used for the current update.
	direction : Nx3 array (double)
		The direction vector as calculated with an optimiser 
		(same as the normalised gradient for Steepest Descent).

	Returns
	-------
	Nx3 array (double)
	
	"""
	N = len(pos)
	pos_temp = wrap_positions(
		pos + step_temp*direction[:N], vects)
	return pos_temp


def lattice_update(potentials, strains, step_temp, direction, pos_temp, vects):
	"""Wrapper function that updates the lattice vectors using
	their strains as the optimised parameters. Once the vectors have 
	been updated, the potential parameters related to cutoff are 
	accordingly changed.

	Parameters
	----------
	potentials : dict[str, Potential]
		Dictionary containing the names and Potential 
		instances of the energy functions to be used 
		as objective functions.
	strains : 3x3 array (double)
		The lattice strains of the current configuration.
	pos_temp : Nx3 array (double)
		The Cartesian coordinates of the ions inside the unit cell.
	vects : 3x3 array (double)
		The lattice vectors in Cartesian coordinates.
	N : int
		Number of atoms in unit cell.
	step_temp : float
		The step size used for the current update.
	direction : Nx3 array (double)
		The direction vector as calculated with an optimiser 
		(same as the normalised gradient for Steepest Descent).

	Returns
	-------
	3x3 array (double), 3x3 array (double), Nx3 array (double) 
	
	"""
	N = len(pos_temp)

	# Using strains and stress
	strains_temp = strains + step_temp*direction[N:]
	delta_strains_temp = (strains_temp-1)+np.identity(3)
	vects_temp = vects @ delta_strains_temp.T
	pos_temp = pos_temp @ delta_strains_temp.T

	# Assign parameters calculated with altered volume
	for name in potentials:
		if hasattr(potentials[name], 'set_cutoff_parameters'):
			potentials[name].set_cutoff_parameters(vects_temp,N)

	return vects_temp, strains_temp, pos_temp



def steady_step(atoms, strains, grad, gnorm, direction, potentials,
	init_energy, update, max_step=0.001, **kwargs):
	"""Wrapper function to perform one updating step 
	using constant step size.

	Parameters
	----------
	atoms : Python ASE's Atoms instance (optional).
		Object from Atoms class, can be used instead of 
		specifying the arrays below.  
	strains : 3x3 array (double)
		The lattice strains of the previous configuration.
	grad : (N+3)x3 array (double)
		Array containing the gradient w.r.t. ion positions
		and strains.
	direction : Nx3 array (double)
		The direction vector as calculated with an optimiser 
		(same as the normalised gradient for Steepest Descent).
	init_energy : double
		The energy of the initial configuration.
	potentials : dict[str, Potential]
		Dictionary containing the names and Potential 
		instances of the energy functions to be used 
		as objective functions.
	c1 : float
		Parameter of Wolfe conditions.
	max_step : double
		Upper bound of step size.
	update: list (str)
		List with the names of parameters to be optimised. Can
		be 'ions' and/or 'lattice'.
	
	Returns
	-------
	(dict[str, _], int)
	
	"""
	vects = np.array(atoms.get_cell())
	N = len(atoms.positions)
	pos = atoms.positions.copy()
	evals = 0
	
	res_dict = {
		'Positions':pos, 
		'Cell':vects, 
		'Step': None,
		'Energy':init_energy,
		'Strains':strains.copy()
	}
	
	words = "Using constant step: "+str(max_step)
	print(words.center(COLUMNS," "))

	temp_dict = calculate_temp_energy(
		potentials, init_energy, pos, vects, 
		strains, max_step, direction, N, update)
	evals += 1

	# Fill in results 
	if temp_dict:
		res_dict = temp_dict
	elif temp_dict=={}:
		energies[s] = None
		res_dict['Catastrophe'] = 1

	for name in potentials:
		if hasattr(potentials[name], 'print_parameters'):
			potentials[name].print_parameters()
	return res_dict, evals



def scheduled_bisection(atoms, strains, grad, gnorm, direction, potentials,
	init_energy, update, max_step=1, min_step=1e-5, **kwargs):
	"""Line minimisation with simple bisection algorithm.

	Parameters
	----------
	atoms : Python ASE's Atoms instance (optional).
		Object from Atoms class, can be used instead of 
		specifying the arrays below.  
	strains : 3x3 array (double)
		The lattice strains of the previous configuration.
	grad : (N+3)x3 array (double)
		Array containing the gradient w.r.t. ion positions
		and strains.
	direction : Nx3 array (double)
		The direction vector as calculated with an optimiser 
		(same as the normalised gradient for Steepest Descent).
	init_energy : double
		The energy of the initial configuration.
	potentials : dict[str, Potential]
		Dictionary containing the names and Potential 
		instances of the energy functions to be used 
		as objective functions.
	max_step : double
		Upper bound of step size.
	update: list (str)
		List with the names of parameters to be optimised. Can
		be 'ions' and/or 'lattice'.
	
	Returns
	-------
	(dict[str, _], int)
	
	"""

	vects = np.array(atoms.get_cell())
	N = len(atoms.positions)
	pos = atoms.positions.copy()
	evals = 0
	
	res_dict = {
		'Positions':pos, 
		'Cell':vects, 
		'Step': None,
		'Energy':init_energy,
		'Strains':strains.copy()
	}
	
	step_temp = max_step
	words = "Using step: "+str(step_temp)

	if (kwargs['iteration']>0) & (kwargs['iteration']%kwargs['schedule']==0):
		words = "Using median of max: "+str(max_step)+" and min: "+str(min_step)
		step_temp = (max_step+min_step)/2

	print(words.center(COLUMNS," "))

	temp_dict = calculate_temp_energy(
		potentials, init_energy, pos, vects, 
		strains, step_temp, direction, N, update)
	evals += 1

	# Fill in results 
	if temp_dict:
		res_dict = temp_dict
	elif temp_dict=={}:
		energies[s] = None
		res_dict['Catastrophe'] = 1

	for name in potentials:
		if hasattr(potentials[name], 'print_parameters'):
			potentials[name].print_parameters()
	return res_dict, evals


def scheduled_exp(atoms, strains, grad, gnorm, direction, potentials,
	init_energy, update, max_step=1, min_step=1e-5, **kwargs):
	"""Line minimisation with simple bisection algorithm.

	Parameters
	----------
	atoms : Python ASE's Atoms instance (optional).
		Object from Atoms class, can be used instead of 
		specifying the arrays below.  
	strains : 3x3 array (double)
		The lattice strains of the previous configuration.
	grad : (N+3)x3 array (double)
		Array containing the gradient w.r.t. ion positions
		and strains.
	direction : Nx3 array (double)
		The direction vector as calculated with an optimiser 
		(same as the normalised gradient for Steepest Descent).
	init_energy : double
		The energy of the initial configuration.
	potentials : dict[str, Potential]
		Dictionary containing the names and Potential 
		instances of the energy functions to be used 
		as objective functions.
	max_step : double
		Upper bound of step size.
	update: list (str)
		List with the names of parameters to be optimised. Can
		be 'ions' and/or 'lattice'.
	
	Returns
	-------
	(dict[str, _], int)
	
	"""

	vects = np.array(atoms.get_cell())
	N = len(atoms.positions)
	pos = atoms.positions.copy()
	evals = 0
	
	res_dict = {
		'Positions':pos, 
		'Cell':vects, 
		'Step': None,
		'Energy':init_energy,
		'Strains':strains.copy()
	}
	
	step_temp = max_step
	if (kwargs['iteration']>0):
		if step_temp > min_step:
			step_temp = max_step*0.999
			
	words = "Using step: "+str(step_temp)
	print(words.center(COLUMNS," "))

	temp_dict = calculate_temp_energy(
		potentials, init_energy, pos, vects, 
		strains, step_temp, direction, N, update)
	evals += 1

	# Fill in results 
	if temp_dict:
		res_dict = temp_dict
	elif temp_dict=={}:
		energies[s] = None
		res_dict['Catastrophe'] = 1

	for name in potentials:
		if hasattr(potentials[name], 'print_parameters'):
			potentials[name].print_parameters()
	return res_dict, evals

import math
def gnorm_scheduled_bisection(atoms, strains, grad, gnorm, direction, potentials,
	init_energy, update, max_step=1, min_step=1e-5, **kwargs):
	"""Line minimisation with simple bisection algorithm.

	Parameters
	----------
	atoms : Python ASE's Atoms instance (optional).
		Object from Atoms class, can be used instead of 
		specifying the arrays below.  
	strains : 3x3 array (double)
		The lattice strains of the previous configuration.
	grad : (N+3)x3 array (double)
		Array containing the gradient w.r.t. ion positions
		and strains.
	direction : Nx3 array (double)
		The direction vector as calculated with an optimiser 
		(same as the normalised gradient for Steepest Descent).
	init_energy : double
		The energy of the initial configuration.
	potentials : dict[str, Potential]
		Dictionary containing the names and Potential 
		instances of the energy functions to be used 
		as objective functions.
	max_step : double
		Upper bound of step size.
	update: list (str)
		List with the names of parameters to be optimised. Can
		be 'ions' and/or 'lattice'.
	
	Returns
	-------
	(dict[str, _], int)
	
	"""

	vects = np.array(atoms.get_cell())
	N = len(atoms.positions)
	pos = atoms.positions.copy()
	evals = 0
	
	res_dict = {
		'Positions':pos, 
		'Cell':vects, 
		'Step': None,
		'Energy':init_energy,
		'Strains':strains.copy()
	}
	
	step_temp = max_step
	words = "Using step: "+str(step_temp)

	if kwargs['gmag']>0:
		words = "Using median of max: "+str(max_step)+" and min: "+str(min_step)
		step_temp = (max_step+min_step)/2

	print(words.center(COLUMNS," "))

	temp_dict = calculate_temp_energy(
		potentials, init_energy, pos, vects, 
		strains, step_temp, direction, N, update)
	evals += 1

	# Fill in results 
	if temp_dict:
		res_dict = temp_dict
	elif temp_dict=={}:
		energies[s] = None
		res_dict['Catastrophe'] = 1

	for name in potentials:
		if hasattr(potentials[name], 'print_parameters'):
			potentials[name].print_parameters()
	return res_dict, evals

