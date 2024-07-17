import numpy as np
cimport numpy as cnp
import warnings
import heapq
import csv
import sys

from cython.view cimport array as cvarray
from cython.parallel import prange,parallel

from libc cimport bool
from libc.stdio cimport *                                                                
from libc.math cimport *
from libc.float cimport *
from libc.limits cimport *
from libc.stdio cimport printf
from libc.stdlib cimport calloc,free
import cython, shutil

from relax.analytic_potentials.cutoff cimport inflated_cell_truncation as get_shifts

from relax.analytic_potentials.operations cimport det as det3_3
from relax.analytic_potentials.operations cimport get_real_distance 
from relax.analytic_potentials.operations cimport get_recip_distance
from relax.analytic_potentials.operations cimport get_distance_vector
from relax.analytic_potentials.operations cimport norm_m,norm
from relax.analytic_potentials.operations cimport get_all_distances, get_min_dist

from cpython.array cimport array, clone


'''																						'''
'''									  COULOMB 											'''
'''																						'''


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double ewald_self(int[:] charges, double alpha, int N) except? 0:                             
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
		eself -= (charges[i]*charges[i] *
					   (alpha / sqrt(pi)))
	return eself*k_e  # electrostatic constant


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)  
cdef double ewald_real(double[:,:] pos, double[:,:] vects, 
	int[:] charges, double alpha, double real_cut_off, int N) except? 0:
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

	cdef double  dist, ereal = 0	# initialise energy value
	cdef double** esum
	cdef Py_ssize_t ioni, ionj, shift, shifts_no, i
	cdef double[:,:] shifts
	cdef double[:] nil_array

	shifts = get_shifts(vects, real_cut_off)
	if shifts == None:
		shifts_no = 0
	else:
		shifts_no = len(shifts)
	
	# set an array as zero shift
	nil_array = array("d",[0,0,0])

	# create array with sums for each N*N position
	esum = <double **> calloc(N, sizeof(double *))
	for ioni in range(N):
		esum[ioni] = <double *> calloc(N, sizeof(double))

	with nogil, parallel():
		for ioni in prange(N, schedule='static'):

			# Allocate thread-local memory for distance vector
			rij = <double *> calloc(3, sizeof(double))

			for ionj in range(ioni, N):
				if ioni != ionj:  # skip in case it's the same ion in original unit cell
					dist = get_real_distance(pos[ioni], pos[ionj], nil_array)
					# Avoid division by zero
					if dist == 0:
						dist = DBL_EPSILON
					esum[ioni][ionj] += (charges[ioni]*charges[ionj] *
										erfc(alpha*dist)/dist)

				# Take care of the rest lattice (+ Ln)
				# Start with whole unit cell images
				for shift in range(shifts_no):
					dist = get_real_distance(pos[ioni], pos[ionj], shifts[shift])
					# Avoid division by zero
					if dist == 0:
						dist = DBL_EPSILON
					esum[ioni][ionj] += (charges[ioni]*charges[ionj] *
												erfc(alpha*dist)/dist)

			# Deallocate distance vector
			free(rij)
	
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
	return ereal


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double ewald_recip(double[:,:] pos, double[:,:] vects, double[:,:] rvects,
	int[:] charges, double alpha, double recip_cut_off, int N) except? 0:
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
	cdef double volume = abs(det3_3(vects))
	cdef double[:] nil_array
	cdef double[:,:] shifts
	cdef double k_2, krij, frac, term
	cdef double k_e = 14.399645351950543 # Coulomb constant
	cdef double erecip = 0
	
	shifts = get_shifts(rvects, recip_cut_off)
	if shifts == None:
		shifts_no = 0
	else:
		shifts_no = len(shifts)

	# set an array as zero shift
	nil_array = array("d",[0,0,0])

	# create array with sums for each N*N position
	esum = <double **> calloc(N, sizeof(double *))
	for ioni in range(N):
		esum[ioni] = <double *> calloc(N, sizeof(double))
		for ionj in range(N):
			esum[ioni][ionj] = 0

	with nogil, parallel():
		for ioni in prange(N, schedule='static'):

			# Allocate thread-local memory for distance vector
			rij = <double *> calloc(3, sizeof(double))


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
					esum[ioni][ionj] += charges[ioni]*charges[ionj]*frac

			# Deallocate distance vector
			free(rij)

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
	return erecip


@cython.boundscheck(False)
cpdef madelung(double made_const, double[:,:] pos, int[:] charges, int N):
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
	if not made_const:
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
				esum += (charges[ioni]*charges[ionj]* \
							made_const / dist)
	esum *= k_e / 2  # Coulomb constant
	return esum