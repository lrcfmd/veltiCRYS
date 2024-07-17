import numpy as np
cimport numpy as cnp

from cython.view cimport array as cvarray
from libc.stdlib cimport malloc, free
from cython.parallel import prange,parallel

from libc cimport bool
from libc.stdio cimport *                                                                
from libc.math cimport *
from libc.float cimport *
from libc.limits cimport *
from libc.stdio cimport printf
import cython, shutil

from relax.analytic_potentials.cutoff cimport inflated_cell_truncation as get_shifts
from relax.analytic_potentials.operations cimport det as det3_3

from relax.analytic_potentials.coulomb.energy cimport *
from relax.analytic_potentials.coulomb.gradient cimport *

'''																						'''
'''									  COULOMB 											'''
'''																						'''


cdef class Coulomb(EwaldPotential):
	"""Calculations for the Coulomb electrostatic energy contribution.
	Ewald summation method used for long range. Each sum per ion is stored
	in a NxN matrix, where N the number of atoms in the unit cell. First 
	the upper triangular matrix is evaluated and the rest is merely copied,
	thanks to the symmetry of the interactions' effect. 
	
	Attributes
    ----------
	real_cut_off : double
		Images of ions in real space (short range) inluded in the energy sum. 
	recip_cut_off : double
		Images of ions in reciprocal space (long range) inluded in the energy sum. 
	alpha : double
		Constant in erfc that controls real-reciprocal spaces' balance 
	made_const : double
		Madelung constant, if it is to be used
	charges : Nx1 array (int)
		List of ions' charges in respective positions
	chemical_symbols : Nx1 array (str)
		Ions' chemical symbols in respective positions 
	grad : (N+3)x3 array (double)
		The gradient of the Coulomb energy w.r.t. ion positions and lattice strain.		
	stresses : 3x3 array
		The gradient of the Coulomb energy w.r.t. lattice strain.
	eself : double
		Self term energy.
	ereal : double
		Real space energy.
	erecip : double
		Reciprocal space energy.

	"""
	def __cinit__(self, chemical_symbols, N, charge_dict, filename=None):
		"""Initialise Coulomb object attributes and allocate 
		arrays used for derivatives.

		Parameters
		----------
		chemical_symbols : Nx1 array (str)
			Ions' chemical symbols in respective positions.
		N : int
			Number of atoms in unit cell.
		charge_dict : dict[str, int]
			Dictionary mapping element chemical symbol to charge value.
		alpha : double
			Constant in erfc that controls real-reciprocal spaces' balance 
		filename : str
			Name of file with Madelung constant.
		

		Returns
		-------
		Coulomb (Potential) instance
		
		"""
		self.alpha = 0
		self.made_const = 0
		self.real_cut_off = 0
		self.recip_cut_off = 0

		self.chemical_symbols = None
		self.charges = None
		self.grad = None
		self.hessian = None

		self.param_flag = False

		self.grad = cvarray(shape=(N+3,3), \
					itemsize=sizeof(double), format="d")
		self.stresses = cvarray(shape=(3,3), \
					itemsize=sizeof(double), format="d")
		self.hessian = cvarray(shape=(3*N,3*N), \
					itemsize=sizeof(double), format="d")
		self.charges = cvarray(shape=(N,), \
					itemsize=sizeof(int), format="i")
		self.chemical_symbols = chemical_symbols.copy()

		cdef Py_ssize_t count = 0
		for elem in chemical_symbols: 
			# for every ion get its ch.symbol "elem"
			self.charges[count] = <int>charge_dict[elem]
			count += 1
		
		if filename:
			try:
				with open(filename,"r") as fin:
					for line in fin:
						line = line.split()
						found = True
						for symbol in chemical_symbols:
							if symbol not in line: # chemical formula does not match
								found = False
								break
							if found == True:  # correct formula
								self.made_const = float(len(line)-2)
			except IOError:
				printf("No Madelung library file found.")


	@cython.boundscheck(False)
	@cython.wraparound(False)
	cpdef set_cutoff_parameters(self, double[:,:] vects=None, int N=0, 
		double accuracy=1e-21, double real=0, double reciprocal=0):
		"""Function to set the real and reciprocal cutoff values. This
		also sets variables holding energy values to zero. 

		Parameters
		----------
		vects : 3x3 array (double)
			The lattice vectors in Cartesian coordinates.
		N : int
			Number of atoms in unit cell.
		accuracy : double
			Desired accuracy of energy summands (used for auto cutoff calculation).
		real : double
			Value to set real cutoff to (optional).
		reciprocal : double
			Value to set reciprocal cutoff to (optional).
		

		"""
		cdef double volume

		if vects!=None:
			volume = abs(det3_3(vects))

			self.alpha = self.get_alpha(N, volume)
			self.real_cut_off = sqrt(-np.log(accuracy))/self.alpha
			self.recip_cut_off = 2*self.alpha*sqrt(-np.log(accuracy))
		else:
			self.real_cut_off = real
			self.recip_cut_off = reciprocal


		self.made_const = 0		
		self.eself = 0
		self.ereal = 0
		self.erecip = 0
		
		self.param_flag = True

	
	cpdef print_parameters(self):
		"""Print alpha, real cutoff and reciprocal cutoff."""
		printf("COULOMB   \t ALPHA: %f\tREAL CUT: %f\tRECIP CUT: %f\n",
			self.alpha, self.real_cut_off,self.recip_cut_off)

	
	cpdef double energy(self, atoms=None, \
		double[:,:] pos_array=None, double[:,:] vects_array=None, int N_=0) except? 0:
		"""Wrapper function to calculate all the electrostatic energy 
		with Coulomb energy potential.

		Parameters
		----------
		atoms : Python ASE's Atoms instance (optional).
			Object from Atoms class, can be used instead of 
			specifying the arrays below.  
		pos_array : Nx3 array (double)
			The Cartesian coordinates of the ions inside the unit cell.
		vects_array : 3x3 array (double)
			The lattice vectors in Cartesian coordinates.
		N_ : int
			Number of atoms in unit cell.
		
		Returns
		-------
		double
		
		"""
		cdef double[:,:] positions 
		cdef double[:,:] vects, rvects
		cdef Py_ssize_t N

		# if atoms object is given then use it
		if atoms:
			positions = atoms.positions
			vects = np.array(atoms.get_cell())
			N = len(positions)
		else:
			positions = pos_array
			vects = vects_array
			N = N_      

		if not self.param_flag:
			raise ValueError("Coulomb potential parameters are not set.")

		# get reciprocal vectors
		rvects = self.get_reciprocal_vects(vects)

		# calculate all parts from ewald sum 		
		self.ereal = ewald_real(positions, vects, self.charges, 
			self.alpha, self.real_cut_off, N)
		self.erecip = ewald_recip(positions, vects, rvects, self.charges,
			self.alpha, self.recip_cut_off, N)
		self.eself = ewald_self(self.charges, self.alpha, N)

		return self.ereal+self.erecip+self.eself

	
	cpdef get_all_ewald_energies(self):
		"""Getter function for calculated energies.

		Returns
		-------
		Dictionary with \"Real\", \"Reciprocal\", \"Self\" energy and their sum.
		
		"""
		energies = {}
		energies['Real'] = self.ereal
		energies['Self'] = self.eself
		energies['Reciprocal'] = self.erecip
		energies['All'] = self.ereal+self.erecip+self.eself
		return energies

	
	cpdef double[:,:] gradient(self, atoms=None, double[:,:] pos_array=None, 
		double[:,:] vects_array=None, int N_=0):
		"""Wrapper function to initialise gradient vector and
		call the functions that calculate real and reciprocal derivatives 
		under the Ewald summation expansion. It returns the gradient
		corresponding to the electrostatical forces. 
	
		atoms : Python ASE's Atoms instance (optional).
			Object from Atoms class, can be used instead of 
			specifying the arrays below.  
		pos_array : Nx3 array (double)
			The Cartesian coordinates of the ions inside the unit cell.
		vects_array : 3x3 array (double)
			The lattice vectors in Cartesian coordinates.
		N_ : int
			Number of atoms in unit cell.

		Returns
		-------
		Nx3 array (double)

		"""
		cdef double[:,:] positions
		cdef double[:,:] vects
		cdef Py_ssize_t ioni, dim, N
		cdef double volume, alpha_drv

		if atoms:
			positions = atoms.positions
			vects = np.array(atoms.get_cell())
			N = len(positions)
		else:
			positions = pos_array
			vects = vects_array
			N = N_
		volume = abs(det3_3(vects))

		if not self.param_flag:
			raise ValueError("Coulomb potential parameters are not set.")
		
		for ioni in range(N+3):
			for dim in range(3):
				self.grad[ioni][dim] = 0

		for l in range(3):
			for m in range(3):
				self.stresses[l][m] = 0
		
		# get the derivative of alpha parameter
		alpha_drv = self.calc_alpha_drv(N, volume)
		# get reciprocal vectors
		rvects = self.get_reciprocal_vects(vects)

		self_drv(self.stresses, self.charges, alpha_drv, volume, N)			
		real_drv(self.grad, self.stresses, 
			positions, vects, self.charges, 
			self.real_cut_off, self.alpha, alpha_drv, volume, N)
		recip_drv(self.grad, self.stresses, 
			positions, vects, rvects, self.charges, 
			self.recip_cut_off, self.alpha, alpha_drv, volume, N)
		fill_stresses(self.grad, self.stresses, volume)

		return self.grad


	cpdef double[:,:] get_stresses(self):
		"""Function to get the strain partial derivatives
		matrix not divided by volume. It should be called 
		after the gradient has been calculated.
		
		Returns
		-------
		3x3 array (double)
		
		"""
		return self.stresses