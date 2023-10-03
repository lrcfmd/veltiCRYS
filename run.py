import sys, os
import shutil
import argparse
import fileinput
import numpy as np

from cmath import pi
from cmath import exp
import cmath
import math

from ase import *
from ase.visualize import view
from ase.io import read as aread
from ase.io import write as awrite
from ase.visualize.plot import plot_atoms

from relax.potentials.operations import get_min_dist
from relax.potentials.buckingham.buckingham import *
from relax.potentials.coulomb.coulomb import *
from relax.direction import *

import timeit

charge_dict = {
	'O' : -2.,
	'Sr':  2.,
	'Ti':  4.,
	'Cl': -1.,
	'Na':  1.,
	'S' : -2.,
	'Zn':  2.
}

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description='Define input')
	parser.add_argument(
		'-i', metavar='--input', type=str,
		help='.cif file to read')
	parser.add_argument(
		'-r', '--relax', action='store_true',
		help='Perform structural relaxation')
	parser.add_argument(
		'-u', '--user', action='store_true',
		help='Wait for user input after every iteration')
	parser.add_argument(
		'-out', metavar='--out_frequency', type=int,
		help='Print files every n iterations')
	parser.add_argument(
        '-o', metavar='--output', type=str,
        help='Output directory')
	parser.add_argument(
		'-su', metavar='--max_step', type=float,
		help='Use upper bound step size')
	parser.add_argument(
		'-sl', metavar='--min_step', type=float,
		help='Use lower bound step size')
	parser.add_argument(
        '-m', metavar='--relaxation_method', type=str,
        help='Choose updating method')
	parser.add_argument(
        '-res', '--reset', action='store_true',
        help='Force direction reset every 3N+9 iterations.')
	parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Print numerical derivatives.')
	
	args = parser.parse_args()

	############################## INPUT ##############################
	structure = None
	filename = args.i
	if filename:
		atoms = aread(filename)
		name = filename.rstrip('.cif').split('/')[-1]
		if name.isnumeric():
			name = "structure_"+name
		structure = name
	else:
		atoms = Atoms("O2",

					cell=[[4, 0.00, 0.00],	
						[0.00, 4, 0.00],
						[0.00, 0.00, 4]],
					positions=[[1, 1, 1],
								 [3.5, 3.5, 3.5]],
					pbc=True)
		structure = "O"
		print("Using custom Atoms object as input.")
	
	
	########################## INITIALISATION #########################
	N 					= len(atoms.positions)
	vects 				= np.array(atoms.get_cell())
	volume 				= abs(np.linalg.det(vects))
	accuracy			= 0.000000000000000000001
	chemical_symbols	= np.array(atoms.get_chemical_symbols())

	# avoid truncating too many terms
	assert((-np.log(accuracy)/N**(1/6)) >= 1)


	########################## DEFINITIONS ############################	
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
	Cpot.energy(atoms)
	coulomb_energies = Cpot.get_all_ewald_energies()
	
	# Define Buckingham potential object
	libfile = "libraries/buck.lib"
	Bpot = Buckingham(
		filename=libfile, 
		chemical_symbols=chemical_symbols, 
		)
	Bpot.set_cutoff_parameters(
		vects=vects, 
		N=N)
	Bpot.energy(atoms)
	buckingham_energies = Bpot.get_all_ewald_energies()

	# Print Ewald-related parameters
	Cpot.print_parameters()
	Bpot.print_parameters()


	########################### RELAXATION #############################
	from relax.descent import *
	import time

	potentials = {}
	initial_energy = 0

	desc = Descent(iterno=50000)
	
	initial_energy += coulomb_energies['All']
	potentials['Coulomb'] = Cpot

	initial_energy += buckingham_energies['All']
	potentials['Buckingham'] = Bpot

	outdir = args.o if args.o else "output/"
	if not os.path.isdir(outdir):
		os.mkdir(outdir) 
	
	direction = GD
	if args.m:
		if "CG" in args.m:
			direction = CG
	
	prettyprint({
		'Chemical Symbols':chemical_symbols, 
		'Positions':atoms.positions, \
		'Cell':atoms.get_cell(), 
		'Electrostatic energy':coulomb_energies, 
		'Interatomic energy':buckingham_energies, \
		'Total energy':initial_energy})

	iteration = {'Energy': initial_energy}
	
	from relax.linmin import *
	from utility import utility

	if args.relax:

		desc.iterno = 70000
		evals, iteration = desc.repeat(
			init_energy=iteration['Energy'],
			atoms=atoms, 
			potentials=potentials, 
			outdir=outdir,
			outfile=structure,
			direction_func=direction,
			step_func=steady_step,
			usr_flag=args.user,
			max_step=args.su if args.su else 0.01,
			min_step=args.sl if args.sl else 0.00001,
			out=args.out if args.out else 1,
			reset=args.reset if args.reset else False,
			debug=args.debug if args.debug else False,
			params=['ions','lattice']
			)
