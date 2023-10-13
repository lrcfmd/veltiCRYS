import numpy as np
cimport numpy as cnp
import warnings
import heapq
import csv
import sys

from cython.view cimport array as cvarray
from libc.stdlib cimport calloc, free
from cython.parallel import prange,parallel

from relax.potentials.potential cimport EwaldPotential

from libc cimport bool
from libc.stdio cimport *                                                                
from libc.math cimport *
from libc.float cimport *
from libc.limits cimport *
from libc.stdio cimport printf
from libc.stdlib cimport calloc,free,realloc
import cython, shutil

from relax.potentials.cutoff cimport inflated_cell_truncation as get_shifts

from relax.potentials.operations cimport det as det3_3
from relax.potentials.operations cimport get_real_distance 
from relax.potentials.operations cimport get_recip_distance
from relax.potentials.operations cimport get_distance_vector
from relax.potentials.operations cimport norm_m,norm
from relax.potentials.operations import get_all_distances, get_min_dist

from cpython.array cimport array, clone


'''																						'''
'''									BUCKINGHAM											'''
'''																						'''

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double ewald_self(buck, double[:,:] vects, cnp.ndarray chemical_symbols, 
	double alpha, int N) except? 0:                             
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
			pair = (min(chemical_symbols[ioni], chemical_symbols[ionj]),
					max(chemical_symbols[ioni], chemical_symbols[ionj]))
			if (pair in buck):
				# Pair of ions is listed in parameters file
				C = buck[pair]['par'][2]
				eself -= C/(3*volume) * pow(pi,1.5)*alpha**3
		pair = (chemical_symbols[ioni], chemical_symbols[ioni])
		if (pair in buck):
			# Pair of ions is listed in parameters file
			C = buck[pair]['par'][2] 
			eself += C*alpha**6/6 

	return eself/2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double ewald_recip(buck, double[:,:] pos, double[:,:] vects, double[:,:] rvects, 
	cnp.ndarray chemical_symbols, double alpha, double recip_cut_off, int N) except? 0:
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
	cdef double volume = abs(det3_3(vects))
	cdef double[:] nil_array, 
	cdef double[:,:] shifts
	cdef double k_2, krij, frac, term
	cdef double A, rho, C, k_3, erecip = 0
	
	shifts = get_shifts(rvects, recip_cut_off)
	if shifts == None:
		shifts_no = 0
	else:
		shifts_no = len(shifts)

	# set an array as zero shift
	nil_array = array("d",[0,0,0])

	for ioni in range(N):

		# Allocate thread-local memory for distance vector
		rij = <double *> calloc(3, sizeof(double))

		for ionj in range(N): 

			pair = (min(chemical_symbols[ioni], chemical_symbols[ionj]),
					max(chemical_symbols[ioni], chemical_symbols[ionj]))
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

	return erecip/2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double ewald_real(buck, double[:,:] pos, double[:,:] vects, 
	cnp.ndarray chemical_symbols, double alpha, double real_cut_off, int N) except? 0:
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
	cdef double[:] nil_array
	cdef double[:,:] shifts

	# set an array as zero shift
	nil_array = array("d",[0,0,0])

	# Check interactions with neighbouring cells
	shifts = get_shifts(vects, real_cut_off)
	if shifts == None:
		shifts_no = 0
	else:
		shifts_no = len(shifts)

	for ioni in range(N):
		for ionj in range(N):
			# Find the pair we are examining
			pair = (min(chemical_symbols[ioni], chemical_symbols[ionj]),
					max(chemical_symbols[ioni], chemical_symbols[ionj]))
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

				for shift in range(shifts_no):
					dist = get_real_distance(pos[ioni], pos[ionj], shifts[shift])

					# Avoid division by zero
					if dist == 0:
						dist = DBL_EPSILON
					ereal += A*exp(-1.0*dist/rho)
					ereal -= C/pow(dist,6) * \
						(1+pow(alpha*dist,2)+pow(alpha*dist,4)/2) * \
						exp(-pow(alpha*dist,2))
	return ereal/2