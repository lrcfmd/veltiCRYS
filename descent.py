import argparse, os
import numpy as np
import pandas as pd
import pickle

from ase.io import read as aread
from ase.cell import Cell
from ase.geometry import wrap_positions, get_distances
from ase.visualize import view
from ase.io import write
from ase.geometry import cell_to_cellpar, cellpar_to_cell

from cysrc.potential import *
from cysrc.buckingham import *
from cysrc.coulomb import *

from pysrc.utils import prettyprint
from pysrc.finite_differences import finite_diff_grad
from pysrc.direction import *
from pysrc.linmin import *


class Descent:
	"""Class with procedures that perform optimisation steps.

	Attributes
    ----------
    iterno : int
		Upper bound of iterations' number.
	ftol : float
		Energy difference tolerance between iterations.
	gtol : float
		Gradient norm tolerance.
	tol : float
		Tolerance of step*||direction||/||parameter_array||
	iters : int
		Iteration number counter.
	methods : list[str]
		List of used functions for direction vector calculation.
	gmax : float
		Tolerance of absolute value of maximum gradient component.
	emin : float
		Stores the last energy value found.

	"""
	def __init__(self, ftol=0.00001, gtol=0.001, 
		tol=0.001, iterno=None, gmax=0.01):
		self.iterno = iterno
		self.ftol = ftol
		self.gtol = gtol
		self.tol = tol
		self.iters = 0
		self.methods = []
		self.cattol = 5
		self.gmax = gmax
		self.emin = 0
		self.step_tol = 1e-5
		self.gmag = 0

	def completion_check(self, last_iteration, iteration, N):
		"""Function to check if termination conditions have 
		been met.

		Parameters
		----------
		last_iteration : dict[str, _]
			Dictionary that holds all information related to 
			the previous iteration.
		iteration : dict[str, _]
			Dictionary that holds all information related to 
			the lastly performed iteration.
		N : int
			Number of atoms in unit cell.

		Returns
		-------
		bool

		"""
		de = abs(last_iteration['Energy']-iteration['Energy'])

		if iteration['Gnorm']<self.gtol:
			print("Iterations: {} Final Gnorm: {} Tolerance: {}".format(
				self.iters, iteration['Gnorm'], self.gtol))
			return True

		if de <= self.ftol:
			print("Iterations: {} Energy: {} Tolerance: {}".format(
					self.iters, iteration['Energy'], self.ftol),end=" ")
			
			if np.amax(np.absolute(iteration['Gradient'])) <= self.gmax:
				
				x = np.append(iteration['Positions'],iteration['Cell'],axis=0)
				x_norm = (np.sum(x**2))**(1/2)
				direction_norm = np.linalg.norm(last_iteration['Direction'])

				if iteration['Step']:
					dx_norm = abs(iteration['Step']*direction_norm)/x_norm
					if dx_norm <= self.tol:
						print("Final Difference Magnitude: {} Tolerance: {}".format(
							dx_norm, self.gtol))
						return True
			else:
				print()
		return False

	def iter_step(self, atoms, potentials, last_iter={}, 
		step_func=steady_step, direction_func=GD, max_step=1, 
		update=['ions', 'lattice'], **kwargs):
		"""Updating iteration step.

		Parameters
		----------
		atoms : Python ASE's Atoms instance.
			Object with the parameters to optimise.
		potentials : dict[str, Potential]
			Dictionary containing the names and Potential 
			instances of the energy functions to be used 
			as objective functions.
		last_iter : dict[str, _]
			Dictionary that holds all information related to 
			the previous iteration.
		step_func : function
			The function to be used for line minimisation.
		direction_func : function
			The function to be used for the calculation of
			the direction vector (optimiser).
		max_step : float
			The upper bound of the step size.

		Returns
		-------
		dict[str, _]
		
		"""

		# Get number of ions
		N = len(atoms.positions)
		
		# Normalise already obtained gradient
		if last_iter['Gnorm']>0:
			grad_norm = last_iter['Gradient']/last_iter['Gnorm']
		else:
			grad_norm = 0

		# Calculate new energy
		res_dict, evals = step_func(
			atoms=atoms, 
			strains=last_iter['Strains'], 
			grad=grad_norm, 
			gnorm=last_iter['Gnorm'], 
			direction=last_iter['Direction'], 
			potentials=potentials, 
			init_energy=last_iter['Energy'],
			iteration=self.iters, 
			max_step=max_step,
			step_tol=self.step_tol,
			update=update, 
			schedule=100,
			gmag=self.gmag)

		# Calculate new point on energy surface
		atoms.set_cell(Cell.new(res_dict['Cell']))
		atoms.positions = res_dict['Positions']
		self.iters += 1

		# Change method if stuck
		if 'reset' in kwargs:
			C = 3*N+9
			if kwargs['reset'] & (
				((not res_dict['Step']) & (res_dict['Energy']==self.emin)) \
				or (self.iters % C == 0)
				):
				direction_func = GD

		# Reset strains every _ iterations
		if (self.iters % (3*N+9) == 0):
			res_dict['Strains'] = np.ones((3,3))

		# Assign new vectors
		pos = np.array(atoms.positions)
		vects = np.array(atoms.get_cell())

		# Gradient of this point on PES
		grad = np.zeros((N+3,3))
		for name in potentials:
			grad += np.array(potentials[name].calc_drv(
			pos_array=pos, vects_array=vects, N_=N))

		# Print numerical derivatives
		if 'debug' in kwargs:
			if kwargs['debug']:
				finite_diff_grad(
					atoms, grad, N, res_dict['Strains'], 0.0001, potentials)
		
		# Gradient norm
		gnorm = get_gnorm(grad,N)
		# Gradient norm difference in magnitude with last gnorm
		self.gmag = -math.floor(math.log(gnorm, math.e))+math.floor(math.log(last_iter['Gnorm'], math.e))

		# Normalise gradient
		if gnorm>0:
			grad_norm = grad/gnorm
		else:
			grad_norm = 0

		# Named arguments for direction function
		args = {
			'Residual': -last_iter['Gradient']/last_iter['Gnorm'],
			# 'Centered': True,
			**last_iter
		}

		# New direction vector -- the returned dict includes
		# all information that need to be passed to the
		# next direction calculation
		dir_dict = direction_func(grad_norm, **args)
		
		iteration = {
		'Gradient':grad, **dir_dict, 'Positions':atoms.positions.copy(), 
		'Strains':res_dict['Strains'], 'Cell':np.array(atoms.get_cell()), 
		'Iter':self.iters, 'Method': self.methods[-1], 
		'Step':res_dict['Step'], 'Gnorm':gnorm, 'Energy':res_dict['Energy'], 
		'Evaluations':evals, 'Catastrophe': res_dict['Catastrophe']}

		# Limit for Buckingham catastrophe
		if 'Buckingham' in potentials:
			iteration['Limit'] = potentials['Buckingham'].get_limit()

		# Add name of used method to list
		self.methods += [direction_func.__name__]

		return iteration

	def repeat(self, init_energy, atoms, potentials, outdir, outfile,
		step_func=steady_step, direction_func=GD, 
		strains=np.ones((3,3)), usr_flag=False, max_step=1, out=1, **kwargs):
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
		strains : 3x3 array (double)
			The lattice strains of the initial configuration.
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

		pos = np.array(atoms.positions)
		vects = np.array(atoms.get_cell())
		N = len(atoms.positions)
		
		count_non = 0
		total_evals = 1
		final_iteration = None
		update = kwargs['params'] if 'params' in kwargs else ['ions', 'lattice']

		if not os.path.isdir(outdir+"imgs"):
			os.mkdir(outdir+"imgs")
		if not os.path.isdir(outdir+"structs"):
			os.mkdir(outdir+"structs")
		if not os.path.isdir(outdir+"imgs/"+outfile+"/"):
			os.mkdir(outdir+"imgs/"+outfile)
		if not os.path.isdir(outdir+"structs/"+outfile+"/"):
			os.mkdir(outdir+"structs/"+outfile)

		self.emin = init_energy
		
		# Gradient for this point on PES
		grad = np.zeros((N+3,3))
		for name in potentials:
			if hasattr(potentials[name], 'calc_drv'):
				grad += np.array(potentials[name].calc_drv(
				pos_array=pos, vects_array=vects, N_=N))

		# Print numerical derivatives
		if 'debug' in kwargs:
			if kwargs['debug']:
				finite_diff_grad(
					atoms, grad, N, strains, 0.0001, potentials)

		# Gradient norm
		gnorm = get_gnorm(grad,N)

		# Normalise gradient
		if gnorm>0:
			grad_norm = grad/gnorm
		else:
			grad_norm = 0
		
		# Direction -- the returned dict includes
		# all information that need to be passed to the
		# next direction calculation
		if direction_func.__name__=='GD' or direction_func.__name__=='CG':
			dir_dict = GD(grad_norm)
			# Add name of used method to list
			self.methods += ['GD']
		else:
			dir_dict = direction_func(grad_norm, Positions=pos)
			# Add name of used method to list
			self.methods += [direction_func.__name__]	

		# Keep info of this iteration
		iteration = {
		'Gradient':grad, **dir_dict, 'Positions':atoms.positions.copy(), 
		'Strains':strains, 'Cell':np.array(atoms.get_cell()), 'Iter':self.iters, 
		'Step': 0, 'Gnorm':gnorm, 'Energy':init_energy, 'Evaluations':total_evals,
		'Catastrophe': 0}

		final_iteration = iteration

		# OUTPUT
		prettyprint(iteration)
		print("Writing result to file",outfile+"_"+\
			str(self.iters),"...")
		write(outdir+"imgs/"+outfile+"/"+outfile+"_"+\
			str(self.iters)+".png", atoms)
		write(outdir+"structs/"+outfile+"/"+outfile+"_"+\
			str(self.iters)+".cif", atoms)
		dict_file = open(
			outdir+"structs/"+outfile+"/"+outfile+"_"+\
			str(self.iters)+".pkl", "wb")
		pickle.dump(
			iteration, 
			dict_file)
		dict_file.close()

		if usr_flag:
			usr = input()
			if 'n' in usr:
				return total_evals, iteration

		# Iterations #
		i = 1
		while(True):
			last_iteration = iteration
			iteration = self.iter_step(
				atoms=atoms, 
				potentials=potentials, 
				last_iter=last_iteration, 
				step_func=step_func, 
				direction_func=direction_func, 
				max_step=last_iteration['Step'] if last_iteration['Step']>0 else max_step,
				update=update,
				**kwargs)

			final_iteration = iteration

			# Keep the newly found energy value
			self.emin = iteration['Energy']
			total_evals += iteration['Evaluations']

			prettyprint(iteration)

			# Check for termination
			if self.completion_check(last_iteration, iteration, N):
				print("Writing result to file",
				outfile+"_"+str(self.iters),"...")
				write(outdir+"imgs/"+outfile+"/"+outfile+"_"+\
					str(self.iters)+".png", atoms)
				write(outdir+"structs/"+outfile+"/"+outfile+"_"+\
					str(self.iters)+".cif", atoms)
				dict_file = open(
					outdir+"structs/"+outfile+"/"+outfile+"_"+\
					str(self.iters)+".pkl", "wb")
				pickle.dump(
					{**iteration, 'Optimised': True}, 
					dict_file)
				dict_file.close()				
				break
			elif (i%out)==0:
				print("Writing result to file",
				outfile+"_"+str(self.iters),"...")
				write(outdir+"imgs/"+outfile+"/"+outfile+"_"+\
					str(self.iters)+".png", atoms)
				write(outdir+"structs/"+outfile+"/"+outfile+"_"+\
					str(self.iters)+".cif", atoms)
				dict_file = open(
					outdir+"structs/"+outfile+"/"+outfile+"_"+\
					str(self.iters)+".pkl", "wb")
				pickle.dump(
					{**iteration, 'Optimised': False}, 
					dict_file)
				dict_file.close()

			# Count consecutive failed line mins
			if ( last_iteration['Step']==None ) & ( iteration['Step']==None ):
				count_non += 1
				if count_non>4:
					print("Line minimisation failed.")
					break
			else:
				count_non = 0

			# Check if max iteration number is reached
			if i == self.iterno:
				break

			if usr_flag:
				if not 'more' in usr:
					usr = input()
				if 'n' in usr:
					return total_evals, iteration

			i += 1

		return total_evals, final_iteration


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description='Define input')
	parser.add_argument(
		'-i', metavar='--input', type=str,
		help='.cif file to read')
	args = parser.parse_args()
	atoms = aread(args.i)
