import numpy as np
cimport numpy as cnp
import warnings
import heapq
import csv
import sys

from cython.view cimport array as cvarray
from libc.stdlib cimport malloc, free
from cython.parallel import prange,parallel

from cysrc.potential cimport Potential

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
'''									BUCKINGHAM											'''
'''																						'''


buck = {}
cdef class Buckingham(Potential):
	"""Calculations for the Buckingham energy contribution. It
	corresponds to the interatomic forces exercised among entities. The dispersion
	term is expanded using the Ewald summation.

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
	limit : double
		Percentage of sum(radius1 + radius2) used as lower limit pairwise distances.
	radii : Nx1 array (double)
		Array that stores each ion radius according to their position in chemical_symbols array.
	
	"""
	def __cinit__(self,  filename=None, chemical_symbols=None, alpha=0, 
		radius_lib=None, radii=None, limit=0.4):
		"""Initialise Buckingham object attributes and allocate 
		arrays used for derivatives. Also, set atom_i-atom_j 
		parameters as found in library file:
		
		 - par: [A(eV), rho(Angstrom), C(eVAngstrom^6)]
		 - lo : min radius (Angstrom)
		 - hi : max radius (Angstrom)

		Parameters
		----------
		chemical_symbols : Nx1 array (str)
			Ions' chemical symbols in respective positions.
		charge_dict : dict[str, int]
			Dictionary mapping element chemical symbol to charge value.
		alpha : double
			Constant in erfc that controls real-reciprocal spaces' balance 
		filename : str
			Name of file with Buckingham constants.
		limit : double
			Percentage of sum(radius1 + radius2) used as lower limit pairwise distances.
		radii : Nx1 array (double)
			Array that stores each ion radius according to their position in chemical_symbols array.
		radius_lib: str
			Library file containing ion radius per element.

		Returns
		-------
		Buckingham (Potential) instance

		"""
		cdef Py_ssize_t ioni, dim, l, m

		self.grad = None
		self.radii = None
		self.close_pairs = 0
		self.limit = limit
		self.real_cut_off = 0
		self.recip_cut_off = 0

		self.param_flag = False

		if alpha > 0:
			self.alpha_flag = False
			self.alpha = alpha
		elif alpha == 0:
			self.alpha_flag = True
		else:
			raise ValueError("Alpha parameter is negative.")

		self.chemical_symbols = chemical_symbols.copy()
		self.grad = cvarray(shape=(len(chemical_symbols)+3,3), \
								itemsize=sizeof(double), format="d")

		for ioni in range(len(chemical_symbols)+3):
			for dim in range(3):
				self.grad[ioni][dim] = 0
		
		self.stresses = cvarray(shape=(3,3), \
								itemsize=sizeof(double), format="d")
		for l in range(3):
			for m in range(3):
				self.stresses[l][m] = 0

		if filename:
			try:
				with open(filename, "r") as fin:
					for line in fin:
						line = line.split()
						if (len(line) < 4):
							continue
						pair = (min(line[0], line[2]), max(line[0], line[2]))
						buck[pair] = {}
						buck[pair]['par'] = list(map(float, line[4:7]))
						buck[pair]['lo'] = float(line[7])
						buck[pair]['hi'] = float(line[-1])
			except IOError:
				printf("No library file found for Buckingham constants.")

		radius_dict = {}

		if radius_lib:
			self.radii = cvarray(shape=(len(chemical_symbols),), \
					itemsize=sizeof(double), format="d")
			try:
				with open(radius_lib, "r") as fin:
					for line in fin:
						line = line.split()
						radius_dict[line[0]] = float(line[1])
			except IOError:
				printf("No library file found for radius values.")
			for s in range(len(chemical_symbols)):
				self.radii[s] = radius_dict[chemical_symbols[s]]
		else:
			self.radii = np.copy(radii)

	
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

		self.eself = 0
		self.ereal = 0
		self.erecip = 0
		
		self.param_flag = True

	
	cpdef print_parameters(self):
		"""Print alpha, real cutoff and reciprocal cutoff."""
		printf("BUCKINGHAM\t ALPHA: %f\tREAL CUT: %f\tRECIP CUT: %f\n",
			self.alpha, self.real_cut_off,self.recip_cut_off)

	
	cpdef double get_limit(self) except? -1:
		"""Print limit value (see class attributes)."""
		return self.limit
	
	
	cpdef double get_max_radius(self) except? -1:
		return max(self.radii)	
	
	
	cdef double calc_self(self, double[:,:] vects, int N) except? 0:                             
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
		cdef double C, eself = 0
		cdef double volume = abs(det3_3(vects))

		for ioni in range(N):
			for ionj in range(N):
				pair = (min(self.chemical_symbols[ioni], self.chemical_symbols[ionj]),
						max(self.chemical_symbols[ioni], self.chemical_symbols[ionj]))
				if (pair in buck):
					# Pair of ions is listed in parameters file
					C = buck[pair]['par'][2]
					eself -= C/(3*volume) * pow(pi,1.5)*self.alpha**3
			pair = (self.chemical_symbols[ioni], self.chemical_symbols[ioni])
			if (pair in buck):
				# Pair of ions is listed in parameters file
				C = buck[pair]['par'][2] 
				eself += C*self.alpha**6/6 

		self.eself = eself/2
		return eself


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
		cdef Py_ssize_t ioni, ionj, shift, shifts_no, i
		cdef double cutoff, volume = abs(det3_3(vects))
		cdef double[:] nil_array, 
		cdef double[:,:] rvects = self.get_reciprocal_vects(vects, volume)
		cdef double[:,:] shifts
		cdef double k_2, krij, frac, term, alpha = self.alpha
		cdef double A, rho, C, k_3, erecip = 0
		
		shifts = get_shifts(rvects, self.recip_cut_off)
		if shifts == None:
			shifts_no = 0
		else:
			shifts_no = len(shifts)

		self.erecip = 0
		cutoff = self.recip_cut_off
		nil_array = array("d",[0,0,0])

		for ioni in range(N):

			# Allocate thread-local memory for distance vector
			rij = <double *> malloc(sizeof(double) * 3)

			for ionj in range(N): 

				pair = (min(self.chemical_symbols[ioni], self.chemical_symbols[ionj]),
						max(self.chemical_symbols[ioni], self.chemical_symbols[ionj]))
				if (pair in buck):
					# Pair of ions is listed in parameters file
					A = buck[pair]['par'][0]
					rho = buck[pair]['par'][1]
					C = buck[pair]['par'][2]

					# Get distance vector
					rij = get_distance_vector(rij, pos[ioni], pos[ionj], nil_array)  

					for shift in range(shifts_no):
						# shift on 2nd power
						k_2 = shifts[shift, 0]*shifts[shift, 0]+ \
								shifts[shift, 1]*shifts[shift, 1]+ \
								shifts[shift, 2]*shifts[shift, 2]
						k_3 = k_2*sqrt(k_2)
						krij = get_recip_distance(rij, shifts[shift])
						term = exp(-k_2/(4*pow(alpha,2)))
						# actual calculation
						erecip -= C*pow(pi,1.5)/(12*volume)*cos(krij)*k_3* \
							(
								sqrt(pi)*erfc(sqrt(k_2)/(2*alpha)) + \
								(4*pow(alpha,3)/k_3 - 2*alpha/sqrt(k_2))*term
								
							)

			# Deallocate distance vector
			free(rij)
			rij = NULL

		self.erecip = erecip/2
		return erecip

	
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
		cdef double A, rho, C, dist, ereal = 0
		cdef Py_ssize_t ioni, ionj, shifts_no, i
		cdef double cutoff, alpha=self.alpha
		cdef double[:] nil_array
		cdef double[:,:] shifts

		nil_array = array("d",[0,0,0])
		cutoff = self.real_cut_off
		self.ereal = 0
		self.close_pairs = 0

		# Check interactions with neighbouring cells
		shifts = get_shifts(vects, cutoff)
		if shifts == None:
			shifts_no = 0
		else:
			shifts_no = len(shifts)

		for ioni in range(N):
			for ionj in range(N):
				# Find the pair we are examining
				pair = (min(self.chemical_symbols[ioni], self.chemical_symbols[ionj]),
						max(self.chemical_symbols[ioni], self.chemical_symbols[ionj]))
				if (pair in buck):
					# Pair of ions is listed in parameters file
					A = buck[pair]['par'][0]
					rho = buck[pair]['par'][1]
					C = buck[pair]['par'][2]

					if ioni != ionj:
						dist = get_real_distance(pos[ioni], pos[ionj], nil_array)
	
						# Avoid division by zero
						if dist == 0:
							dist = DBL_EPSILON
						ereal += A*exp(-1.0*dist/rho)
						ereal -= C/pow(dist,6) * \
							(1+pow(alpha*dist,2)+pow(alpha*dist,4)/2) * \
							exp(-pow(alpha*dist,2))

						# Check for small distances
						if dist < self.limit*(self.radii[ioni]+self.radii[ionj]):
							# print("Pair",pair,"is too close dist=",dist,"Aexp=", A*exp(-1.0*dist/rho),"-C/*6=", - C/pow(dist,6))
							self.close_pairs += 1

					for shift in range(shifts_no):
						dist = get_real_distance(pos[ioni], pos[ionj], shifts[shift])

						# Avoid division by zero
						if dist == 0:
							dist = DBL_EPSILON
						ereal += A*exp(-1.0*dist/rho)
						ereal -= C/pow(dist,6) * \
							(1+pow(alpha*dist,2)+pow(alpha*dist,4)/2) * \
							exp(-pow(alpha*dist,2))

						# Check for small distances
						if dist < self.limit*(self.radii[ioni]+self.radii[ionj]):
							# print("Pair",pair,"is too close dist=",dist,"Aexp=", A*exp(-1.0*dist/rho),"-C/*6=", - C/pow(dist,6))
							self.close_pairs += 1
		self.ereal = ereal/2
		return ereal
	
	
	cpdef double calc(self, atoms=None, \
		double[:,:] pos_array=None, double[:,:] vects_array=None, int N_=0) except? 0:
		"""Wrapper function to calculate all the interatomic energy 
		with Buckingham energy potential.

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
			raise ValueError("Buckingham potential parameters are not set.")

		# printf("\nCalculating Buckingham term with Ewald sum:\n")
		self.calc_real(positions,vects,N)
		if (self.catastrophe_check()>0):
			return 0
		# printf("Real sum...................%f\n",self.ereal)#\x1B[32m ok\033[0m\n")
		self.calc_recip(positions,vects,N)
		# printf("Reciprocal sum.............%f\n",self.erecip)#\x1B[32m ok\033[0m\n")
		self.calc_self(vects,N)
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

	
	cdef int catastrophe_check(self) except? -1:
		"""Checks if Buckingham catastrophe has occurred on multiple levels.
		First the energy and gradient norm change is compared to find out if 
		energy is dropping but gradient norm is increasing. If this happens
		repeatedly then the function reports it.
		
		"""
		# cdef double gnorm_change, energy_change

		# gnorm_change = gnorm-old_gnorm
		# energy_change = energy-old_energy

		if self.close_pairs>0:
			printf("Catastrophe check..........:\t\x1B[31m Close pairs:\033[0m %d\n",self.close_pairs)
			# printf("\x1B[33mWarning:\033[0m Abnormally close pairs detected.\n")
			return 1

		return 0


	cdef double[:,:] calc_real_drv(self, double[:,:] pos, double[:,:] vects, int N, double volume):
		"""Calculate short range interatomic forces in form of the
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

		cdef double cutoff, dist, a2pi, alpha_drv
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
		alpha_drv = self.calc_alpha_drv(N,volume)

		# Allocate thread-local memory for distance vector
		rij = <double *> malloc(sizeof(double) * 3)

		for ioni in range(N):
			for ionj in range(N):
				# Find the pair we are examining
				pair = (min(self.chemical_symbols[ioni], self.chemical_symbols[ionj]),
						max(self.chemical_symbols[ioni], self.chemical_symbols[ionj]))
				if (pair in buck):
					# Pair of ions is listed in parameters file
					A = buck[pair]['par'][0]
					rho = buck[pair]['par'][1]
					C = buck[pair]['par'][2]

					if ioni != ionj:  # skip in case it's the same atom or it is constant

						rij = get_distance_vector(rij, pos[ioni], pos[ionj], nil_array)
						dist = get_real_distance(pos[ioni], pos[ionj], nil_array)
						term = exp(-alpha*alpha*dist*dist)
						
						# Buckingham repulsion
						drv = -A/(rho*dist)*exp(-dist/rho)

						# Dispersion
						drv += C*term/pow(dist,6) * \
						(
							6/pow(dist,2) + \
							6*pow(alpha,2) + 
							3*pow(alpha,4)*pow(dist,2) + \
							pow(alpha,6)*pow(dist,4)
						)
						# partial derivative without position vector

						part_drv[0] = drv*rij[0]/2
						part_drv[1] = drv*rij[1]/2
						part_drv[2] = drv*rij[2]/2

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
								self.stresses[l][m] += part_drv[l]*rij[m]
								
								# Add the following if alpha is not constant
								if self.alpha_flag & (l==m):
									drv =  C*term*alpha_drv*pow(alpha,5)
									self.stresses[l][m] += drv/2

					# take care of the rest lattice (+ Ln)
					for shift in range(shifts_no):
						rij = get_distance_vector(rij, pos[ioni], pos[ionj], shifts[shift])
						dist = get_real_distance(pos[ioni], pos[ionj], shifts[shift])
						term = exp(-alpha*alpha*dist*dist)

						# Avoid division by zero
						if dist == 0:
							dist = DBL_EPSILON

						# Buckingham repulsion
						drv = -A/(rho*dist)*exp(-dist/rho)

						# Dispersion
						drv += C*term/pow(dist,6) * \
						(
							6/pow(dist,2) + \
							6*pow(alpha,2) + 
							3*pow(alpha,4)*pow(dist,2) + \
							pow(alpha,6)*pow(dist,4)
						)

						part_drv[0] = drv*rij[0]/2
						part_drv[1] = drv*rij[1]/2
						part_drv[2] = drv*rij[2]/2

						# partial deriv with respect to ioni
						self.grad[ioni][0] += part_drv[0]
						self.grad[ioni][1] += part_drv[1]
						self.grad[ioni][2] += part_drv[2]

						# partial deriv with respect to ionj
						self.grad[ionj][0] -= part_drv[0] 
						self.grad[ionj][1] -= part_drv[1] 
						self.grad[ionj][2] -= part_drv[2] 

						# # deriv with respect to strain epsilon_lm (Ln present in summand)
						for l in range(3):
							for m in range(l,3):
								self.stresses[l][m] += part_drv[l]*rij[m]
								
								# Add the following if alpha is not constant
								if self.alpha_flag & (l==m):
									drv =  C*term*alpha_drv*pow(alpha,5)
									self.stresses[l][m] += drv/2

		# Deallocate distance vector
		free(rij)
		rij = NULL

		return self.grad


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
		if not self.param_flag:
			raise ValueError("Buckingham potential parameters are not set.")

		cdef double cutoff, drv, dist
		cdef double term_exp, term_c, term_erfc, term_a
		cdef double A, rho, C, alpha_drv, alpha = self.alpha
		cdef double[:,:] shifts
		cdef Py_ssize_t shifts_no, i
		cdef double* rij
		cdef double[:] nil_array, part_drv
		cdef double[:,:] recip_vects = self.get_reciprocal_vects(vects, volume)

		nil_array = array("d",[0,0,0])
		part_drv = array("d",[0,0,0])
		shifts = get_shifts(recip_vects, self.recip_cut_off)

		# Check interactions with neighbouring cells
		if shifts == None:
			shifts_no = 0
		else:
			shifts_no = len(shifts)
		cutoff = self.recip_cut_off

		# allocate memory for distance vector
		rij = <double *> malloc(sizeof(double) * 3)

		for ioni in range(N):
			for ionj in range(N):
				# Find the pair we are examining
				pair = (min(self.chemical_symbols[ioni], self.chemical_symbols[ionj]),
						max(self.chemical_symbols[ioni], self.chemical_symbols[ionj]))
				if (pair in buck):
					# Pair of ions is listed in parameters file
					A = buck[pair]['par'][0]
					rho = buck[pair]['par'][1]
					C = buck[pair]['par'][2]

					rij[0] = 0
					rij[1] = 0
					rij[2] = 0

					# Get distance vector
					rij = get_distance_vector(rij, pos[ioni], pos[ionj], nil_array)
					
					for shift in range(shifts_no):
						# shift on 2nd power
						k_2 = shifts[shift, 0]*shifts[shift, 0]+ \
								shifts[shift, 1]*shifts[shift, 1]+ \
								shifts[shift, 2]*shifts[shift, 2]
						k_3 = k_2*sqrt(k_2)

						krij = get_recip_distance(rij, shifts[shift])

						# Calculate some parts
						term_exp = exp(-k_2/(4*pow(alpha,2)))
						term_c = C*pow(pi,1.5)/(12*volume)
						term_erfc = sqrt(pi)*erfc(sqrt(k_2)/(2*alpha))
						term_a = (4*pow(alpha,3)-2*alpha*k_2)

						# Calculate partial derivative real value
						drv = term_c*sin(krij)/2 * \
							(k_3*term_erfc+term_a*term_exp)
				
						# partial deriv with respect to ioni
						self.grad[ioni][0] += drv*shifts[shift,0]
						self.grad[ioni][1] += drv*shifts[shift,1]
						self.grad[ioni][2] += drv*shifts[shift,2]

						# partial deriv with respect to ionj
						self.grad[ionj][0] -= drv*shifts[shift,0]
						self.grad[ionj][1] -= drv*shifts[shift,1]
						self.grad[ionj][2] -= drv*shifts[shift,2]

						drv = term_c*cos(krij)/2 * \
							(
								3*term_erfc*sqrt(k_2) - 6*alpha*term_exp
							)
						alpha_drv = self.calc_alpha_drv(N,volume)

						for l in range(3):
							for m in range(l,3):					
								# deriv with respect to strain epsilon_lm 
								self.stresses[l][m] += drv * \
									shifts[shift][m]*shifts[shift][l]
									
								# Add the following if alpha is not constant
								if self.alpha_flag & (l==m):
									self.stresses[l][m] -= term_c*cos(krij)/2 * \
										(
											-term_erfc*k_3 - \
											(term_a-12*pow(alpha,2)*alpha_drv)*term_exp											
										)

		# Deallocate distance vector
		free(rij)
		rij = NULL

		return self.grad


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
		cdef Py_ssize_t i, j, l, m
		cdef double alpha = self.alpha

		for i in range(N):
			for j in range(N):
				# Find the pair we are examining
				pair = (min(self.chemical_symbols[i], self.chemical_symbols[j]),
						max(self.chemical_symbols[i], self.chemical_symbols[j]))
				if (pair in buck):
					# Pair of ions is listed in parameters file
					C = buck[pair]['par'][2]
					for l in range(3):
						self.stresses[l][l] -= C/(3*volume)*pow(pi,1.5) * alpha*alpha * \
							(3*self.calc_alpha_drv(N,volume)-alpha)/2
			# Find the pair we are examining
			pair = (self.chemical_symbols[i], self.chemical_symbols[i])
			if (pair in buck):
				# Pair of ions is listed in parameters file
				C = buck[pair]['par'][2]
				for l in range(3):
					self.stresses[l][l] += C*pow(alpha,5)*self.calc_alpha_drv(N,volume)/2

		return self.stresses

	
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
		corresponding to the interatomic forces. 
	
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
		
		# printf("Calculating Buckingham derivatives")
		
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


	cpdef double calc_simple(self, atoms=None, \
		double[:,:] pos_array=None, double[:,:] vects_array=None, int N_=0) except? 0:
		"""Calculate interatomic energy with simple form of dispersion term.

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
		cdef double[:,:] vects
		cdef double energy
		cdef int N

		if atoms:
			pos = atoms.positions
			vects = np.array(atoms.get_cell())
			N = len(atoms.positions)
		else:
			pos = pos_array
			vects = vects_array
			N = N_  

		cdef double A, rho, C, dist, esum = 0
		cdef Py_ssize_t ioni, ionj, shifts_no, i
		cdef double cutoff
		cdef double[:] nil_array
		cdef double[:,:] shifts

		if not self.param_flag:
			raise ValueError("Buckingham potential parameters are not set.")

		nil_array = array("d",[0,0,0])
		cutoff = self.real_cut_off
		self.close_pairs = 0

		# Check interactions with neighbouring cells
		shifts = get_shifts(vects, cutoff)
		if shifts == None:
			shifts_no = 0
		else:
			shifts_no = len(shifts)

		for ioni in range(N):
			for ionj in range(N):
				# Find the pair we are examining
				pair = (min(self.chemical_symbols[ioni], self.chemical_symbols[ionj]),
						max(self.chemical_symbols[ioni], self.chemical_symbols[ionj]))
				if (pair in buck):
					# Pair of ions is listed in parameters file
					A = buck[pair]['par'][0]
					rho = buck[pair]['par'][1]
					C = buck[pair]['par'][2]

					dist = get_real_distance(pos[ioni], pos[ionj], nil_array)

					# Check if distance of ions allows interaction
					if (dist <= buck[pair]['hi']) & (ioni != ionj):
						# Avoid division by zero
						if dist == 0:
							dist = DBL_EPSILON
						esum += A*exp(-1.0*dist/rho) - C/pow(dist,6)

						# Check for small distances
						if dist < self.limit*(self.radii[ioni]+self.radii[ionj]):
							print("Pair",pair,"is too close dist=",dist,"Aexp=", A*exp(-1.0*dist/rho),"-C/*6=", - C/pow(dist,6))
							self.close_pairs += 1

					for shift in range(shifts_no):
						dist = get_real_distance(pos[ioni], pos[ionj], shifts[shift])
						# Check if distance of ions allows interaction
						if dist <= buck[pair]['hi']:
							# Avoid division by zero
							if dist == 0:
								dist = DBL_EPSILON
							esum += A*exp(-1.0*dist/rho) - C/dist**6

							# Check for small distances
							if dist < self.limit*(self.radii[ioni]+self.radii[ionj]):
								print("Pair",pair,"is too close dist=",dist,"Aexp=", A*exp(-1.0*dist/rho),"-C/*6=", - C/pow(dist,6))
								self.close_pairs += 1
		return esum/2
