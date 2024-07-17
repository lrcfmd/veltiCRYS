import os, torch
from pathlib import Path
import argparse
import numpy as np

from ase import *
from ase.io import read as aread
from ase.io import write as awrite
from ase.visualize import view

from relax.optim.gradient_descent import GD
from relax.optim.conjugate_gradient import *
from relax.optim.lbfgs import *
from relax.optim.cubic_minimization.main import *
from relax.optim.linmin import *

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
        'choose_mode', metavar='--mode', type=str,
        help='Choose between packages: \'analytic\' or \'auto\'.')
	parser.add_argument(
		'-i', metavar='--input', type=str,
		help='.cif file to read')
	parser.add_argument(
		'-r', '--relax', type=int,
		help='Perform structural relaxation for given iterations')
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
        '-d', '--debug', action='store_true',
        help='Print numerical derivatives.')
	parser.add_argument(
		'-ln', metavar='--line_search',
		nargs='*', 
		help='Type name of line search method and optional parameter value. One of: \n\
      			-gnorm_scheduled_bisection <order>\n\
				-scheduled_bisection <schedule>\n\
				-scheduled_exp <exponent>\n\
				-steady_step')
	args = parser.parse_args()


	"""  INPUT  """
	structure = None
	filename = args.i

	if filename:
		atoms = aread(filename)
		structure = Path(filename).stem
		
		# Give name to structure
		if structure.isnumeric():
			structure = "structure_"+structure+"_"+str(atoms.symbols)
	else:
		from examples import *
		atoms, structure = get_example3()
		print("Using custom Atoms object as input.")
	
	
	"""  INITIALISATION  """
	N 					= len(atoms.positions)
	accuracy			= 0.000000000000000000001
	outdir = args.o if args.o else "output/"
 
	# Avoid truncating too many terms
	assert((-np.log(accuracy)/N**(1/6)) >= 1)

	# Choosing package
	if 'an' in args.choose_mode:
		from relax.optim.analytic import repeat
	else:
		from relax.optim.autodiff import repeat


	"""  RELAXATION  """   
	lnsearch = LnSearch(
		max_step=args.su if args.su else 1e-3,
		min_step=args.sl if args.sl else 1e-5,
		schedule=100,
		exponent=0.999,
		order=10,
		gnorm=0
	)
	if args.ln:
		line_search_fn = args.ln[0]
	else:
		line_search_fn = 'steady_step'
	if args.ln:
		if len(args.ln)==2:
			if args.ln[0] == 'gnorm_scheduled_bisection':
				lnsearch.order = float(args.ln[1])
			elif args.ln[0] == 'scheduled_bisection':
				lnsearch.schedule = int(args.ln[1])
			elif args.ln[0] == 'scheduled_exp':
				lnsearch.exp = float(args.ln[1])

	iterno = 0
	if args.relax is not None:
		iterno = args.relax

	# Special case for BFGS, can change in the future
	if args.m=='BFGS':
		import sys
		from relax.optim.bfgs import BFGS
		optimizer = BFGS(charge_dict=charge_dict,
				   atoms=atoms,
				   max_iter=iterno,
				   outfile=outdir+structure+'/'+structure)
		optimizer.run()
		sys.exit()
  
	optimizer = GD(lnsearch)
	if args.m is not None:
		optimizer = globals()[args.m](lnsearch)	
		
	iteration = repeat(
		atoms=atoms, 
		outdir=outdir,
		outfile=structure,
		charge_dict=charge_dict,
		optimizer=optimizer, 
		line_search_fn=line_search_fn,
		usr_flag=args.user, 
		out=args.out if args.out else 1, 
		debug=args.debug if args.debug else False,
		iterno=iterno
	)
	view(atoms)
