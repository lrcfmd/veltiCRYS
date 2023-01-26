import sys, os
import shutil
import argparse
import fileinput
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cmath import pi
from cmath import exp
import cmath
import math

from ase import *
from ase.visualize import view
from ase.io import read as aread
from ase.io import write as awrite
from ase.visualize.plot import plot_atoms

from cysrc.operations import get_min_dist
from cysrc.buckingham import *
from cysrc.coulomb import *

from pysrc.utils import prettyprint
import pysrc.utils as utils
from pysrc.direction import *

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

atom_types = {
	'O':  1,
	'Sr': 2,
	'Ti': 3,
	# 'Na': 4,
	# 'Cl': 5,
	# 'S' : 6,
	# 'Zn': 7
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
		'-s', metavar='--max_step', type=float,
		# nargs='*',
		help='Use upper bound step size')
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

	nrg_list = []
	# for d in np.arange(-0.01, 0.01, 0.001):
	(folder, structure, atoms) = utils.get_input(filename=args.i)

	params = utils.initialise(atoms)

	# avoid truncating too many terms
	assert((-np.log(params['accuracy'])/params['N']**(1/6)) >= 1)

	libfile = utils.DATAPATH+"libraries/madelung.lib"
	Cpot = Coulomb(
		chemical_symbols=params['chemical_symbols'],
		N=params['N'],
		charge_dict=charge_dict,
		filename=libfile)
	Cpot.set_cutoff_parameters(
		vects=params['vects'], 
		N=params['N'])
	Cpot.calc(atoms)
	coulomb_energies = Cpot.get_energies()
	
	dist_limit = 0
	libfile = utils.DATAPATH+"libraries/buck.lib"
	libfile2 = utils.DATAPATH+"libraries/radii.lib"
	Bpot = Buckingham(
		filename=libfile, 
		chemical_symbols=params['chemical_symbols'], 
		radius_lib=libfile2,
		limit=dist_limit
		)
	Bpot.set_cutoff_parameters(
		vects=params['vects'], 
		N=params['N'])
	Bpot.calc(atoms)
	buckingham_energies = Bpot.get_energies()

	Cpot.print_parameters()
	Bpot.print_parameters()

	dict = { **coulomb_energies,
			'Elect_LAMMPS': 0, 'E_madelung': None, 'Interatomic': 0,
			'Inter_LAMMPS': 0, 'Total_GULP': 0}
	# utils.print_template(dict)


	########################### RELAXATION #############################
	from descent import *
	import time

	potentials = {}
	initial_energy = 0

	start_time = time.time()
	desc = Descent(iterno=50000)
	
	initial_energy += coulomb_energies['All']
	potentials['Coulomb'] = Cpot

	initial_energy += buckingham_energies['All']
	potentials['Buckingham'] = Bpot

	# nrg_list += [initial_energy]
	# plt.plot(np.arange(-0.01, 0.01, 0.001), nrg_list)
	# plt.show()


	outdir = args.o if args.o else "output/"

	if not os.path.isdir(outdir):
		os.mkdir(outdir) 
	
	direction = GD
	if args.m:
		if "CG" in args.m:
			direction = CG
		if "RMS" in args.m:
			direction = RMSprop
		if "Adam" in args.m:
			direction = Adam
	
	prettyprint({'Chemical Symbols':atoms.get_chemical_symbols(), 'Positions':atoms.positions, \
		'Cell':atoms.get_cell(), 'Electrostatic energy':coulomb_energies, 'Interatomic energy':Bpot.get_energies(), \
		'Total energy':initial_energy})

	iteration = {'Energy': initial_energy}
	
	import math
	from pysrc.linmin import *

	if args.relax:

		desc.iterno = 50000
		evals, iteration = desc.repeat(
			init_energy=iteration['Energy'],
			atoms=atoms, 
			potentials=potentials, 
			outdir=outdir,
			outfile=folder+structure,
			direction_func=direction,
			step_func=steady_step,
			usr_flag=args.user,
			max_step=args.s if args.s else 0.01,
			out=args.out if args.out else 1,
			reset=args.reset if args.reset else False,
			debug=args.debug if args.debug else False,
			params=['ions','lattice']
			)

	print("Time: "+str(time.time() - start_time)+"\n")
