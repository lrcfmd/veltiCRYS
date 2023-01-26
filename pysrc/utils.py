import shutil
import math, scipy.optimize
import numpy as np
from cmath import pi

from ase import *
from ase.calculators.gulp import GULP
from ase.calculators.lammpslib import LAMMPSlib
from ase.io import read as aread
from ase.io import write as awrite
from ase.visualize import view

DATAPATH = "./"


def print_template(dict):
	columns = shutil.get_terminal_size().columns
	print("\n")

	# Coulomb
	lines = columns/2 - len("COULOMB")
	for c in range(lines):
		print("=", end="")
	print("COULOMB", end="")
	for c in range(lines):
		print("=", end="")
	print()

	lines = columns/2 - len("CUSTOM IMPLEMENTATION")
	for c in range(lines):
		print("-", end="")
	print("CUSTOM IMPLEMENTATION", end="")
	for c in range(lines):
		print("-", end="")
	print()

	print("Real:\t\t"+str(dict['Real']))
	print("Self:\t\t"+str(dict['Self']))
	print("Recip:\t\t"+str(dict['Reciprocal']))
	print("Electrostatic:\t"+str(dict['Electrostatic']))

	lines = columns/2 - len("LAMMPS")
	for c in range(lines):
		print("-", end="")
	print("LAMMPS", end="")
	for c in range(lines):
		print("-", end="")
	print()

	print("Electrostatic:\t"+str(dict['Elect_LAMMPS']))

	lines = columns/2 - len("MADELUNG")
	for c in range(lines):
		print("-", end="")
	print("MADELUNG", end="")
	for c in range(lines):
		print("-", end="")
	print()

	if dict['E_madelung'] == None:
		print("No Madelung constant for this structure.")
	else:
		print("Electrostatic:\t"+str(dict['E_madelung']))

	for c in range(columns):
		print("-", end="")
	print()
	print("\n")
	
	# Buckingham
	lines = columns/2 - len("BUCKINGHAM")
	for c in range(lines):
		print("=", end="")
	print("BUCKINGHAM", end="")
	for c in range(lines):
		print("=", end="")
	print()

	lines = columns/2 - len("CUSTOM IMPLEMENTATION")
	for c in range(lines):
		print("-", end="")
	print("CUSTOM IMPLEMENTATION", end="")
	for c in range(lines):
		print("-", end="")
	print()

	print("Interatomic:\t"+str(dict['Interatomic']))

	lines = columns/2 - len("LAMMPS")
	for c in range(lines):
		print("-", end="")
	print("LAMMPS", end="")
	for c in range(lines):
		print("-", end="")
	print()

	print("Interatomic:\t"+str(dict['Inter_LAMMPS']))
	
	for c in range(columns):
		print("-", end="")
	print()
	print("\n")

	# Total
	lines = columns/2 - len("TOTAL")
	for c in range(lines):
		print("=", end="")
	print("TOTAL", end="")
	for c in range(lines):
		print("=", end="")
	print()

	lines = columns/2 - len("CUSTOM IMPLEMENTATION")
	for c in range(lines):
		print("-", end="")
	print("CUSTOM IMPLEMENTATION", end="")
	for c in range(lines):
		print("-", end="")
	print()

	print("Total lattice:\t"+str(dict['Electrostatic'] + dict['Interatomic']))
	
	lines = columns/2 - len("GULP")
	for c in range(lines):
		print("-", end="")
	print("GULP", end="")
	for c in range(lines):
		print("-", end="")
	print()
	
	print("Total lattice:\t"+str(dict['Total_GULP']))
	
	for c in range(columns):
		print("-", end="")
	print()

import shutil
COLUMNS = shutil.get_terminal_size().columns
def prettyprint(dict_):
	import pprint
	np.set_printoptions(suppress=True)
	words = ""
	for key, value in dict_.items():
		if key=="Total energy":
			words += key+" "+str(value)+" "
		else:
			print("\n", key)
			print(value)
	print(words.center(COLUMNS,"-"))


def get_input(filename=None, atom_coord=None, displacement=0):
	"""Choose input from file or manual.

	"""
	folder = None
	structure = None
	if filename:
		atoms = aread(filename)
		name = filename.rstrip('.cif').split('/')[-1]
		if name.isnumeric():
			name = "structure_"+name
		structure = name
		try:
			idata = filename.rstrip('.cif').split('/').index("Data")
			folder = filename.rstrip('.cif').split('/')[idata+1]+"_"
			if atom_coord:
				atoms.positions[atom_coord[0],atom_coord[1]] += displacement
				structure += "_dis"+str(displacement)
			if folder==structure:
				folder = ""
		except:
			folder = ""		
		print("Using file as input.")
	else:
		atoms = Atoms("O",

					cell=[[2.4, 0.00, 0.00],	
						[0.00, 2.4, 0.00],
						[0.00, 0.00, 2.4]],
					positions=[[0, 0, 0]],
								 # [3.5, 0, 0],
								 # [0, 3.5, 0],
								 # [0, 0, 3.5],
								 # [3.5, 3.5, 0],
								 # [0, 3.5, 3.5],
								 # [3.5, 0, 3.5],
								 # [3.5, 3.5, 3.5]],
					pbc=True)
		structure = "O"
		folder = ""
		print("Using custom Atoms object as input.")

	# atoms.cell *= 10

	return (folder,structure,atoms)


def lammps_energy(atoms):
	LAMMPSlib.default_parameters['lammps_header'] = ['units metal', 'atom_style charge', \
														'atom_modify map array sort 0 0']
	charge_cmds = [
					"set type 1 charge 2.0",  # Sr
					"set type 2 charge -2.0", # O
					"set type 3 charge 4.0",  # Ti
					# "set type 4 charge 1.0",  # Na
					# "set type 5 charge -1.0", # Cl
					# "set type 6 charge -2.0", # S
					# "set type 7 charge  2.0"  # Zn
					]
	param_cmds  = [
					"pair_style coul/long "+str(real_cut),
					"pair_coeff * *",
					"kspace_style ewald "+str(accuracy)]
	cmds = charge_cmds + param_cmds
	lammps = LAMMPSlib(lmpcmds=cmds, atom_types=atom_types, log_file=LOG)
	atoms.set_initial_charges(Cpot.charges)
	atoms.set_calculator(lammps)
	elect_LAMMPS = atoms.get_potential_energy()

	cmds = [
			"pair_style buck 10",
			"pair_coeff 1 2 1952.39 0.33685 19.22", # Sr-O
			"pair_coeff 1 3 4590.7279 0.261 0.0",   # Ti-O
			"pair_coeff 1 1 1388.77 0.36262 175",   # O-O
			"pair_coeff 2 2 0.0 1.0 0.0",
			"pair_coeff 3 3 0.0 1.0 0.0",
			"pair_coeff 2 3 0.0 1.0 0.0"]
	lammps = LAMMPSlib(lmpcmds=cmds, atom_types=atom_types, log_file=LOG)
	atoms.set_calculator(lammps)
	inter_LAMMPS = atoms.get_potential_energy()

	return (elect_LAMMPS,inter_LAMMPS)


def gulp_energy(atoms):
	from ase.calculators.gulp import GULP
	g_keywords = 'conp full nosymm unfix gradient'
	calc = GULP(keywords=g_keywords, \
					library='buck.lib')
	atoms.set_calculator(calc)
	total_GULP = atoms.get_potential_energy()
	return total_GULP


def initialise(atoms):
	LOG 				= "src/test.log"
	N 					= len(atoms.positions)
	vects 				= np.array(atoms.get_cell())
	volume 				= abs(np.linalg.det(vects))
	# accuracy 			= 0.00001  # Demanded accuracy of terms 
	accuracy			= 0.000000000000000000001
	chemical_symbols	= np.array(atoms.get_chemical_symbols())

	return {"LOG": LOG, "N": N, "vects": vects, "volume": volume,				
	"accuracy": accuracy, "chemical_symbols": chemical_symbols}
	

def atoms_to_html(atoms):
    'Return the html representation the atoms object as string'

    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile('r+', suffix='.html') as ntf:
        atoms.write(ntf.name, format='html')
        ntf.seek(0)
        html = ntf.read()
    return html