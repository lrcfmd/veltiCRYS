from scipy.special import erfc
import numpy as np

from cmath import pi
from cmath import exp
import cmath
import math

from ase import *
from ase.visualize import view
from ase.geometry import Cell
from ase.io import read as aread

class Potential:
	def set_parameters(self):
		pass

	def get_shifts(self, cut_off, vects):
		"""Get all possible lattice positions:   
		 (2cut_off+1)^3 - {case of (cut_off,cut_off,cut_off)}
		 combinations in R^3  
		
		"""
		shifts = np.zeros(((2*cut_off+1)**3 - 1, 3))
		tmp = np.array([cut_off, cut_off, cut_off])

		i = 0
		for shift in np.ndindex(2*cut_off+1, 2*cut_off+1, 2*cut_off+1):
			if shift != (cut_off, cut_off, cut_off):
				shifts[i, ] = shift
				shifts[i, ] = shifts[i, ] - tmp
				i = i+1
		shifts = shifts@vects
		return shifts


class Coulomb(Potential):
	"""Calculations for the Coulomb electrostatic energy contribution.
	The Ewald summation method used for long range. Each sum is calculated
	on a NxN matrix, where N the number of atoms in the unit cell. First 
	the upper triangular matrix is evaluated and the rest is merely repeated,
	thanks to the symmetry of the interactions' effect.

	"""
	def set_parameters(self, alpha, real_cut_off, \
			recip_cut_off, chemical_symbols, charge_dict,
			filename=None):
		"""Set following parameters:

		real_cut_off 	: Images of ions are included up to the 
		larger integer closest this number x real lattice vectors
		per dimension
		recip_cut_off	: Same as real space but number x reciprocal
		lattice vectors
		alpha			: Constant in erfc that controls balance 
		between reciprocal and real space term contribution
		made_const		: Madelung constant if it is to be used
		charges 		: List of ions' charges in respective positions
		chemical_symbols: Ions' chemical symbols in resp. positions

		
		also:
		filename		: Library file for Madelung way potential

		"""
		self.real_cut_off = math.ceil(real_cut_off)
		self.recip_cut_off = math.ceil(recip_cut_off)
		self.alpha = alpha
		self.made_const = 0
		self.charges = [charge_dict[x] for x in chemical_symbols]
		self.chemical_symbols = chemical_symbols

		if filename:
			try:
				with open(filename, "r") as fin:
					for line in fin:
						line = line.split()
						found = True
						for symbol in chemical_symbols:
							if symbol not in line:
									# chemical formula does not match
								found = False
								break
						if found == True:  # correct formula
							self.made_const = float(line[-1])
			except IOError:
				print("No Madelung library file found.")

	def get_reciprocal_vects(self, volume, vects):
		"""Calculate reciprocal vectors
		
		"""
		rvects = np.zeros((3, 3))
		for i in np.arange(3):
			rvects[i, ] = 2*pi*np.cross(vects[(1+i) % 3, ],
										vects[(2+i) % 3]) / volume
		return rvects

	def get_charges_mult(self, index1, index2):
		"""Find respective charges and
		 return their product
		
		"""
		return (self.charges[index1]*self.charges[index2])

	def calc_self(self, N):
		"""Calculate self interaction term
		
		"""
		esum = np.zeros((N, N))
		for i in range(0, N):
			esum[i, i] -= (self.get_charges_mult(i, i) *
						   (self.alpha / math.sqrt(pi)))
		return esum

	def calc_real(self, pos, vects, N):
		"""Calculate short range
		
		"""
		esum = np.zeros((N, N))
		shifts = self.get_shifts(self.real_cut_off, vects)
		for ioni in range(0, N):
			for ionj in range(ioni, N):
				if ioni != ionj:  # skip in case it's the same ion in original unit cell
					dist = np.linalg.norm(pos[ioni, ] - pos[ionj, ])
					esum[ioni, ionj] += (self.get_charges_mult(ioni, ionj) *
										 math.erfc(self.alpha*dist)/(2*dist))
				# take care of the rest lattice (+ Ln)
				for shift in shifts:
					dist = np.linalg.norm(
						pos[ioni, ] + shift - pos[ionj, ])
					esum[ioni, ionj] += (self.get_charges_mult(ioni, ionj) *
										 math.erfc(self.alpha*dist)/(2*dist))
		return esum

	def calc_recip(self, pos, vects, N):
		"""Calculate long range
		
		"""
		volume = abs(np.linalg.det(vects))
		rvects = self.get_reciprocal_vects(volume, vects)

		esum = np.zeros((N, N))
		shifts = self.get_shifts(self.recip_cut_off, rvects)
		for ioni in range(0, N):
			for ionj in range(ioni, N):

				rij = pos[ioni, ] - pos[ionj, ]
				for k in shifts:
					po = -np.dot(k, k)/(4*self.alpha**2)
					numerator = 4 * (pi**2) * (math.exp(po)) * \
						math.cos(np.dot(k, rij))

					# print(np.dot(k,k))
					# print(np.dot(k, rij))

					denominator = np.dot(k, k) * 2 * pi * volume
					esum[ioni, ionj] += ((self.get_charges_mult(ioni, ionj)) *
										 (numerator/denominator))
		return esum

	def calc_complete(self, N, esum):
		"""Complete lower triangular matrix of monopole to monopole
		
		"""
		esum *= 14.399645351950543  # electrostatic constant
		for ioni in range(0, N):
			for ionj in range(0, ioni):
				esum[ioni, ionj] = esum[ionj, ioni]
		return esum

	def calc_madelung(self, pos, N):
		if not self.made_const:
			return None
		esum = 0
		for ioni in range(N):
			for ionj in range(N):
				if ioni != ionj:  # skip in case it's the same atom
												  # in original unit cell
					dist = np.linalg.norm(pos[ioni, ] - pos[ionj, ])
					esum += (self.get_charges_mult(ioni, ionj)
							 * self.made_const / dist)
		esum *= 14.399645351950543 / 2  # Coulomb constant
		return esum

	def calc(self, atoms, **kwargs):
		"""This function needs either the whole Atoms object or
		named arguments for positions (ion positions), vects (unit cell vectors)
		and N (number of atoms in unit cell)

		"""
		if atoms:
			positions = atoms.positions
			vects = atoms.get_cell()
			N = len(positions)
		else:
			positions = kwargs['positions']
			vects = kwargs['vects']
			N = kwargs['N']		
		energies = {}
		Er_array = self.calc_real(positions, vects, N)
		Erc_array = self.calc_recip(positions, vects, N)
		Es_array = self.calc_self(N)
		Eupper = Er_array + Es_array + Erc_array

		energies['Real'] = sum(sum(self.calc_complete(N, Er_array)))
		energies['Self'] = sum(sum(self.calc_complete(N, Es_array)))
		energies['Reciprocal'] = sum(sum(self.calc_complete(N, Erc_array)))
		energies['Electrostatic'] = sum(sum(self.calc_complete(N, Eupper)))
		return energies


class Buckingham(Potential):
	"""Calculations for the Buckingham energy contribution. It
	corresponds to the interatomic forces exercised among entities.
	
	"""
	def set_parameters(self, filename, chemical_symbols):
		"""Set atom_i-atom_j parameters as read from library file:
		 - par: [A(eV), rho(Angstrom), C(eVAngstrom^6)]
		 - lo : min radius (Angstrom)
		 - hi : max radius (Angstrom)

		"""
		self.chemical_symbols = chemical_symbols
		self.buck = {}

		try:
			with open(filename, "r") as fin:
				for line in fin:
					line = line.split()
					if (len(line) < 4):
						continue
					pair = (min(line[0], line[2]), max(line[0], line[2]))
					self.buck[pair] = {}
					self.buck[pair]['par'] = list(map(float, line[4:7]))
					self.buck[pair]['lo'] = float(line[7])
					self.buck[pair]['hi'] = float(line[-1])
		except IOError:
			print("No library file found for Buckingham constants.")

	def get_cutoff(self, vects, hi):
		"""Find how many cells away to check
		 using the minimum cell vector value
		
		"""
		cutoff = math.ceil(hi/min(vects[vects != 0]))
		return cutoff

	def calc(self, atoms):
		"""Interatomic potential
		
		"""
		pos = atoms.get_positions()
		N = len(atoms.get_positions())
		vects = atoms.get_cell()
		return self.calc_real(pos, vects, N)

	def calc_real(self, pos, vects, N):
		esum = 0
		for ioni in range(N):
			for ionj in range(N):
				# Find the pair we are examining
				pair = (min(self.chemical_symbols[ioni], self.chemical_symbols[ionj]),
						max(self.chemical_symbols[ioni], self.chemical_symbols[ionj]))
				if (pair in self.buck):
					# Pair of ions is listed in parameters file
					A = self.buck[pair]['par'][0]
					rho = self.buck[pair]['par'][1]
					C = self.buck[pair]['par'][2]

					dist = np.linalg.norm(pos[ioni] - pos[ionj])
					# Check if distance of ions allows interaction
					if (dist < self.buck[pair]['hi']) & (ioni != ionj):
						esum += A*math.exp(-1.0*dist/rho) - C/dist**6

					# Check interactions with neighbouring cells
					cutoff = self.get_cutoff(vects, self.buck[pair]['hi'])
					shifts = self.get_shifts(cutoff, vects)
					for shift in shifts:
						dist = np.linalg.norm(
							pos[ioni] + shift - pos[ionj])
						# Check if distance of ions allows interaction
						if (dist < self.buck[pair]['hi']):
							esum += A*math.exp(-1.0*dist/rho) - C/dist**6
		return esum/2


def buckingham_coulomb(positions, Coulomb, Buckingham, N, vects):
	"""A wrapper for the result of the two potentials.

	"""
	Eupper = Coulomb.calc_real(positions, vects, N) + \
			 Coulomb.calc_recip(positions, vects, N) + \
			 Coulomb.calc_self(N)
	electrostatic = sum(sum(Coulomb.calc_complete(N, Eupper)))
	interatomic = Buckingham.calc_real(positions, vects, N)
	return electrostatic+interatomic


if __name__ == '__main__':
	"""The following parameters were chosen according to GULP
	documentation. This is just a test with the final calculated
	energy printed.

	"""
	atoms = aread("../../../Data/random/RandomStart_Sr3Ti3O9/1.cif")
	vects = np.array(atoms.get_cell())
	volume = abs(np.linalg.det(vects))
	N = len(atoms.get_positions())
	accuracy = 0.00001  # Demanded accuracy of terms 
	alpha = N**(1/6) * pi**(1/2) / volume**(1/3)
	real_cut = (-np.log(accuracy))**(1/2)/alpha
	recip_cut = 2*alpha*(-np.log(accuracy))**(1/2)


	Cpot = Coulomb()
	Cpot.set_parameters(alpha=alpha, real_cut_off=real_cut,
						recip_cut_off=recip_cut, 
						chemical_symbols=atoms.get_chemical_symbols(),
						charge_dict={'O' : -2., 'Sr':  2., 'Ti':  4.})

	Bpot = Buckingham()
	Bpot.set_parameters("../../../Data/Libraries/buck.lib", atoms.get_chemical_symbols())
	energy = buckingham_coulomb(atoms.positions, Cpot, Bpot, N, vects)
	print(energy)