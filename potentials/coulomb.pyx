import numpy as np
cimport numpy as cnp
import warnings
import heapq
import csv
import sys

from cython.view cimport array as cvarray
from libc.stdlib cimport malloc, free
from cython.parallel import prange,parallel

from libc cimport bool
from libc.stdio cimport *                                                                
from libc.math cimport *
from libc.float cimport *
from libc.limits cimport *
from libc.stdio cimport printf
from libc.stdlib cimport malloc,free,realloc
import cython, shutil

from cysrc.cutoff cimport inflated_cell_truncation as get_shifts

from cysrc.operations cimport det as det3_3
from cysrc.operations cimport get_real_distance 
from cysrc.operations cimport get_recip_distance
from cysrc.operations cimport get_distance_vector
from cysrc.operations cimport norm_m,norm
from cysrc.operations import get_all_distances, get_min_dist

from cpython.array cimport array, clone

'''																						'''
'''									  COULOMB 											'''
'''																						'''



cdef class Coulomb(Potential):
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
	def __cinit__(self, chemical_symbols, N, charge_dict, alpha=0, filename=None):
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

		if alpha > 0:
			self.alpha_flag = False
			self.alpha = alpha
		elif alpha == 0:
			self.alpha_flag = True
		else:
			raise ValueError("Alpha parameter is negative.")

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

			if self.alpha_flag:
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
				
	
	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef double calc_self(self, int N) except? 0:                             
		"""Calculate self interaction term.

		Parameters
		----------
		N : int
			Number of atoms in unit cell.

		Returns
		-------
		double

		"""
		cdef Py_ssize_t i
		cdef double eself = 0
		cdef double k_e = 14.399645351950543 # Coulomb constant

		for i in range(N):
			eself -= (self.charges[i]*self.charges[i] *
						   (self.alpha / sqrt(pi)))
		self.eself = eself*k_e  # electrostatic constant
		return eself

	
	@cython.boundscheck(False)
	@cython.wraparound(False)   
	cdef double calc_real(self, double[:,:] pos, double[:,:] vects, int N) except? 0:
		"""Calculate short range energy.

		Parameters
		----------
		pos : Nx3 array (double)
			The Cartesian coordinates of the ions inside the unit cell.
		vects : 3x3 array (double)
			The lattice vectors in Cartesian coordinates.
		N : int
			Number of atoms in unit cell.
		
		Returns
		-------
		double
		
		"""
		if pos.shape[1]!=3 or vects.shape[1]!=3:
			raise IndexError("Points are not 3-dimensional.")

		cdef double cutoff, dist, ereal = 0
		cdef double** esum
		cdef Py_ssize_t ioni, ionj, shift, shifts_no, i
		cdef double[:,:] shifts
		cdef double[:] nil_array

		shifts = get_shifts(vects, self.real_cut_off)
		if shifts == None:
			shifts_no = 0
		else:
			shifts_no = len(shifts)
		
		nil_array = array("d",[0,0,0])

		self.ereal = 0
		cutoff = self.real_cut_off

		# create array with sums for each N*N position
		esum = <double **> malloc(sizeof(double *) * N)
		for ioni in range(N):
			esum[ioni] = <double *> malloc(sizeof(double) * N)
			for ionj in range(N):
				esum[ioni][ionj] = 0

		with nogil, parallel():
			for ioni in prange(N, schedule='static'):

				# Allocate thread-local memory for distance vector
				rij = <double *> malloc(sizeof(double) * 3)

				for ionj in range(ioni, N):
					if ioni != ionj:  # skip in case it's the same ion in original unit cell
						dist = get_real_distance(pos[ioni], pos[ionj], nil_array)
						# Avoid division by zero
						if dist == 0:
							dist = DBL_EPSILON
						esum[ioni][ionj] += (self.charges[ioni]*self.charges[ionj] *
											erfc(self.alpha*dist)/dist)

					# Take care of the rest lattice (+ Ln)
					# Start with whole unit cell images
					for shift in range(shifts_no):
						dist = get_real_distance(pos[ioni], pos[ionj], shifts[shift])
						# Avoid division by zero
						if dist == 0:
							dist = DBL_EPSILON
						esum[ioni][ionj] += (self.charges[ioni]*self.charges[ionj] *
													erfc(self.alpha*dist)/dist)

		
		# Fill lower triangular matrix with symmetric values
		for ioni in range(N):
			for ionj in range(0, ioni):
				esum[ioni][ionj] = esum[ionj][ioni]
			for ionj in range(N):
				ereal += esum[ioni][ionj]

		# Deallocation
		for ioni in prange(N, nogil=True, schedule='static'):
			free(esum[ioni])
		free(esum)

		ereal = ereal*14.399645351950543/2  # electrostatic constant
		self.ereal = ereal
		return ereal

	
	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef double calc_recip(self, double[:,:] pos, double[:,:] vects, int N) except? 0:
		"""Calculate long range energy in reciprocal space.
		
		Parameters
		----------
		pos : Nx3 array (double)
			The Cartesian coordinates of the ions inside the unit cell.
		vects : 3x3 array (double)
			The lattice vectors in Cartesian coordinates.
		N : int
			Number of atoms in unit cell.
		
		Returns
		-------
		double

		"""
		if pos.shape[1]!=3 or vects.shape[1]!=3:
			raise IndexError("Points are not 3-dimensional.")
		
		cdef double* rij
		cdef double** esum
		cdef Py_ssize_t ioni, ionj, shift, shifts_no, i
		cdef double cutoff, volume = abs(det3_3(vects))
		cdef double[:] nil_array
		cdef double[:,:] rvects = self.get_reciprocal_vects(vects, volume)
		cdef double[:,:] shifts
		cdef double k_2, krij, frac, term, alpha = self.alpha
		cdef double k_e = 14.399645351950543 # Coulomb constant
		cdef double erecip = 0
		
		shifts = get_shifts(rvects, self.recip_cut_off)
		if shifts == None:
			shifts_no = 0
		else:
			shifts_no = len(shifts)

		self.erecip = 0
		cutoff = self.recip_cut_off
		nil_array = array("d",[0,0,0])

		# create array with sums for each N*N position
		esum = <double **> malloc(sizeof(double *) * N)
		for ioni in range(N):
			esum[ioni] = <double *> malloc(sizeof(double) * N)
			for ionj in range(N):
				esum[ioni][ionj] = 0

		with nogil, parallel():
			for ioni in prange(N, schedule='static'):

				# Allocate thread-local memory for distance vector
				rij = <double *> malloc(sizeof(double) * 3)

				for ionj in range(ioni, N): 

					# Get distance vector
					rij = get_distance_vector(rij, pos[ioni], pos[ionj], nil_array)  

					for shift in range(shifts_no):
						# shift on 2nd power
						k_2 = shifts[shift, 0]*shifts[shift, 0]+ \
								shifts[shift, 1]*shifts[shift, 1]+ \
								shifts[shift, 2]*shifts[shift, 2]
						krij = get_recip_distance(rij, shifts[shift])
						term = exp(-k_2/(4*pow(alpha,2)))
						# actual calculation
						frac = 2*pi*term*cos(krij) / (k_2*volume)
						esum[ioni][ionj] += self.charges[ioni]*self.charges[ionj]*frac

				# Deallocate distance vector
				free(rij)
				rij = NULL

		# Fill lower triangular matrix with symmetric values
		for ioni in prange(N, nogil=True, schedule='static'):
			for ionj in range(0, ioni):
				esum[ioni][ionj] = esum[ionj][ioni]
			for ionj in range(N):
				erecip += esum[ioni][ionj]
		
		# Deallocation
		for ioni in prange(N, nogil=True, schedule='static'):
			free(esum[ioni])
		free(esum)

		erecip = erecip*k_e  # electrostatic constant
		self.erecip = erecip
		return erecip

	
	@cython.boundscheck(False)
	cpdef calc_madelung(self, double[:,:] pos, int N):
		"""Calculate electrostatic energy using Madelung constant.

		Parameters
		----------
		pos : Nx3 array (double)
			The Cartesian coordinates of the ions inside the unit cell.
		vects : 3x3 array (double)
			The lattice vectors in Cartesian coordinates.
		N : int
			Number of atoms in unit cell.
		
		Returns
		-------
		double
		
		"""
		if not self.made_const:
			return None

		cdef double dist, esum = 0
		cdef Py_ssize_t ioni, ionj
		cdef double k_e = 14.399645351950543 # Coulomb constant

		for ioni in prange(N, nogil=True, schedule='static'):
			for ionj in range(N):
				if ioni != ionj:  # skip in case it's the same atom
												  # in original unit cell
					dist = (pos[ioni, 0]-pos[ionj, 0])*(pos[ioni, 0]-pos[ionj, 0])+ \
							(pos[ioni, 1]-pos[ionj, 1])*(pos[ioni, 1]-pos[ionj, 1])+ \
							(pos[ioni, 2]-pos[ionj, 2])*(pos[ioni, 2]-pos[ionj, 2])
					esum += (self.charges[ioni]*self.charges[ionj]* \
								self.made_const / dist)
		esum *= k_e / 2  # Coulomb constant
		return esum

	
	cpdef double calc(self, atoms=None, \
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
		cdef double[:,:] vects
		cdef Py_ssize_t N

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

		# printf("\nCalculating Coulomb term with Ewald sum:\n")
		self.calc_real(positions,vects,N)
		# printf("Real sum...................%f\n",self.ereal)#\x1B[32m ok\033[0m\n")
		self.calc_recip(positions,vects,N)
		# printf("Reciprocal sum.............%f\n",self.erecip)#\x1B[32m ok\033[0m\n")
		self.calc_self(N)
		# printf("Self term..................%f\n",self.eself)#\x1B[32m ok\033[0m\n")

		return self.ereal+self.erecip+self.eself

	
	cpdef get_energies(self):
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

		
	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef double[:,:] calc_self_drv(self, double volume, int N):
		"""Derivative of the self term with respect to strains.
		This must not be used when alpha parameter is a constant.

		Parameters
		----------
		volume : double
			The volume of the unit cell.
		N : int
			Number of atoms in unit cell.

		Returns
		-------
		Nx3 array (double)

		"""
		cdef Py_ssize_t i, l, m
		cdef double k_e = 14.399645351950543

		for i in range(N):
			for l in range(3):
					self.stresses[l][l] -= self.charges[i]*self.charges[i] * \
						self.calc_alpha_drv(N,volume)/sqrt(pi) * k_e

		return self.stresses

	
	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef double[:,:] calc_real_drv(self, double[:,:] pos, double[:,:] vects, int N, double volume):
		"""Calculate short range electrostatic forces in form of the
		energy function's gradient (forces = -gradient). 
		This function calculates the real space derivatives.

		Parameters
		----------
		pos : Nx3 array (double)
			The Cartesian coordinates of the ions inside the unit cell.
		vects : 3x3 array (double)
			The lattice vectors in Cartesian coordinates.
		N : int
			Number of atoms in unit cell.
		volume : double
			The volume of the unit cell.

		Returns
		-------
		Nx3 array (double)

		"""
		if pos.shape[1]!=3 or vects.shape[1]!=3:
			raise IndexError("Points are not 3-dimensional.")

		cdef double cutoff, dist, a2pi
		cdef double drv, term, k_e, alpha = self.alpha
		cdef double* rij
		cdef Py_ssize_t ioni, ionj, dim
		cdef Py_ssize_t shift, shifts_no, i, l, m
		cdef double[:,:] shifts
		cdef double[:] nil_array, part_drv

		shifts = get_shifts(vects, self.real_cut_off)
		cutoff = self.real_cut_off
		if shifts == None:
			shifts_no = 0
		else:
			shifts_no = len(shifts)

		nil_array = array("d",[0,0,0])
		part_drv = array("d",[0,0,0])

		a2pi = 2*alpha/sqrt(pi)
		k_e = 14.399645351950543    

		with nogil:

			# Allocate thread-local memory for distance vector
			rij = <double *> malloc(sizeof(double) * 3)

			for ioni in range(N):
				for ionj in range(N):
					if ioni != ionj:  # skip in case it's the same atom or it is constant

						rij = get_distance_vector(rij, pos[ioni], pos[ionj], nil_array)
						dist = get_real_distance(pos[ioni], pos[ionj], nil_array)
						term = exp(-alpha*alpha*dist*dist)
						drv = - self.charges[ioni]*self.charges[ionj] * \
							(
								a2pi*term / (dist*dist) + \
								erfc(alpha*dist) / (dist*dist*dist)
							) # partial derivative without position vector

						part_drv[0] = drv*rij[0]*k_e/2 # Coulomb constant
						part_drv[1] = drv*rij[1]*k_e/2 
						part_drv[2] = drv*rij[2]*k_e/2 

						# partial deriv with respect to ioni
						self.grad[ioni][0] += part_drv[0]
						self.grad[ioni][1] += part_drv[1]
						self.grad[ioni][2] += part_drv[2]

						# partial deriv with respect to ionj
						self.grad[ionj][0] -= part_drv[0]
						self.grad[ionj][1] -= part_drv[1]
						self.grad[ionj][2] -= part_drv[2]

						# deriv with respect to strain epsilon_lm (Ln not present in summand)
						for l in range(3):
							for m in range(l,3):
								self.stresses[l][m] += part_drv[l]*(pos[ioni][m]-pos[ionj][m])
								
								# Add the following if alpha is not constant
								if self.alpha_flag & (l==m):
									drv = self.charges[ioni]*self.charges[ionj]*k_e * \
										-self.calc_alpha_drv(N,volume)*term/sqrt(pi)
									self.stresses[l][m] += drv

					# take care of the rest lattice (+ Ln)
					for shift in range(shifts_no):
						rij = get_distance_vector(rij, pos[ioni], pos[ionj], shifts[shift])
						dist = get_real_distance(pos[ioni], pos[ionj], shifts[shift])
						term = exp(-alpha*alpha*dist*dist)

						# Avoid division by zero
						if dist == 0:
							dist = DBL_EPSILON

						# Partial derivative
						drv = - self.charges[ioni]*self.charges[ionj] * \
							(
								a2pi*term / (dist*dist) + \
								erfc(alpha*dist) / (dist*dist*dist)
							)

						part_drv[0] = drv*rij[0]*k_e/2 # Coulomb constant
						part_drv[1] = drv*rij[1]*k_e/2
						part_drv[2] = drv*rij[2]*k_e/2

						# partial deriv with respect to ioni
						self.grad[ioni][0] += part_drv[0]
						self.grad[ioni][1] += part_drv[1]
						self.grad[ioni][2] += part_drv[2]

						# partial deriv with respect to ionj
						self.grad[ionj][0] -= part_drv[0] 
						self.grad[ionj][1] -= part_drv[1] 
						self.grad[ionj][2] -= part_drv[2] 

						# deriv with respect to strain epsilon_lm (Ln present in summand)
						for l in range(3):
							for m in range(l,3):
								self.stresses[l][m] += part_drv[l]*( pos[ioni][m]+shifts[shift][m]-pos[ionj][m] )
								
								if self.alpha_flag & (l==m):
									drv = self.charges[ioni]*self.charges[ionj]*k_e * \
										-self.calc_alpha_drv(N,volume)*term/sqrt(pi)
									self.stresses[l][m] += drv
			# Deallocate distance vector
			free(rij)
			rij = NULL

		return self.grad

	
	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef double[:,:] calc_recip_drv(self, double[:,:] pos, double[:,:] vects, int N, double volume):
		"""Calculate long range electrostatic forces in form of the
		energy function's gradient (forces = -gradient). 
		This function calculates the reciprocal space derivatives.

		Parameters
		----------
		pos : Nx3 array (double)
			The Cartesian coordinates of the ions inside the unit cell.
		vects : 3x3 array (double)
			The lattice vectors in Cartesian coordinates.
		N : int
			Number of atoms in unit cell.
		volume : double
			The volume of the unit cell.


		Returns
		-------
		Nx3 array (double)

		"""
		cdef Py_ssize_t ioni, ionj, dim, shift, i, l, m
		cdef double cutoff, drv, k_2, stemp
		cdef double krij, term, alpha = self.alpha
		cdef double strain_temp
		cdef double* rij
		cdef double[:,:] recip_vects = self.get_reciprocal_vects(vects, volume)
		cdef double[:,:] shifts
		cdef double[:] nil_array
		cdef Py_ssize_t shifts_no
		cdef double k_e = 14.399645351950543 # Coulomb constant

		shifts = get_shifts(recip_vects, self.recip_cut_off)
		nil_array = array("d",[0,0,0])

		# Check interactions with neighbouring cells
		if shifts == None:
			shifts_no = 0
		else:
			shifts_no = len(shifts)
		cutoff = self.recip_cut_off

		with nogil:

			# Allocate thread-local memory for distance vector
			rij = <double *> malloc(sizeof(double) * 3)

			for ioni in range(N):
				for ionj in range(N):
					
					# Get distance vector
					rij = get_distance_vector(rij, pos[ioni], pos[ionj], nil_array)
					
					for shift in range(shifts_no):
						# shift on 2nd power
						k_2 = shifts[shift, 0]*shifts[shift, 0]+ \
								shifts[shift, 1]*shifts[shift, 1]+ \
								shifts[shift, 2]*shifts[shift, 2]

						krij = get_recip_distance(rij, shifts[shift])
						term = exp(-k_2/(4*pow(alpha,2)))
						drv = - self.charges[ioni]*self.charges[ionj] * \
										   2*pi*k_e*term*sin(krij)/(k_2*volume)
				
						# partial deriv with respect to ioni
						self.grad[ioni][0] += drv*shifts[shift,0]
						self.grad[ioni][1] += drv*shifts[shift,1]
						self.grad[ioni][2] += drv*shifts[shift,2]

						# partial deriv with respect to ionj
						self.grad[ionj][0] -= drv*shifts[shift,0]
						self.grad[ionj][1] -= drv*shifts[shift,1]
						self.grad[ionj][2] -= drv*shifts[shift,2]

						drv = 2*pi*k_e/volume * \
							self.charges[ioni]*self.charges[ionj] * \
							term/k_2 * cos(krij)

						for l in range(3):
							for m in range(l,3):					
								# deriv with respect to strain epsilon_lm 
								self.stresses[l][m] += \
									drv*(
										( 1/(2*alpha*alpha)+2/k_2 ) * shifts[shift][m] * \
										shifts[shift][l]
										)
								# Add the following if alpha is not constant
								if self.alpha_flag & (l==m):
									self.stresses[l][m] -= \
										drv*(1 - \
											k_2/(2*pow(alpha,3)) * self.calc_alpha_drv(N,volume))
								elif (l==m):
									self.stresses[l][m] -= drv

			# Deallocate distance vector
			free(rij)
			rij = NULL

		return self.grad

	
	cdef double[:,:] fill_stresses(self, double volume=1):
		"""Function to fill in the lower part of the stress tensor.
		The stress tensor is symmetrical, so the upper triangular 
		array is copied to the respective symmetrical positions. 

		Parameters
		----------
		volume : double
			The volume of the unit cell.
		
		Returns
		-------
		3x3 array (double)
		
		"""
		cdef Py_ssize_t i, j
		for i in range(3):
			for j in range(i+1,3):
				self.stresses[j][i] = self.stresses[i][j]

		for i in range(-1,-4,-1):
			for j in range(-1,i-1,-1):
				self.grad[i][j] = self.stresses[i][j]/volume
				self.grad[j][i] = self.stresses[i][j]/volume

		return self.stresses

	
	cpdef double[:,:] calc_drv(self, atoms=None, double[:,:] pos_array=None, 
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
		cdef double volume 

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
		
		# printf("Calculating Coulomb derivatives")
		
		if self.alpha_flag:
			self.calc_self_drv(volume,N)	
			# printf("...self")
		
		self.calc_real_drv(positions,vects,N,volume)
		# printf("...real")
		
		self.calc_recip_drv(positions,vects,N,volume)
		# printf("...recip")
		
		self.fill_stresses(volume)
		# printf("...stress")

		# printf("...\x1B[32m ok\033[0m\n")
		return self.grad