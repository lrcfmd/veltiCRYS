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

from relax.potentials.cutoff cimport inflated_cell_truncation as get_shifts

from relax.potentials.operations cimport det as det3_3
from relax.potentials.operations cimport get_real_distance 
from relax.potentials.operations cimport get_recip_distance
from relax.potentials.operations cimport get_distance_vector
from relax.potentials.operations cimport norm_m,norm
from relax.potentials.operations import get_all_distances, get_min_dist

from cpython.array cimport array, clone


'''																						'''
'''									  COULOMB 											'''
'''																						'''


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double[:,:] self_drv(double[:,:] stresses, int[:] charges, 
	double alpha_drv, double volume, int N):
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
				stresses[l][l] -= charges[i]*charges[i] * \
					alpha_drv/sqrt(pi) * k_e
	return stresses


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double erfc_drv(double alpha, int charges, double dist) nogil:
	"""The derivative of the real function that depends only 
	on the pairwise distance between two ions.
	
	"""
	cdef double drv, a2pi, term
	a2pi = 2*alpha/sqrt(pi)
	term = exp(-alpha*alpha*dist*dist)

	drv = - charges * \
		(
			a2pi*term / dist + \
			erfc(alpha*dist) / pow(dist, 2)
		)
	return drv


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double[:,:] real_drv(double[:,:] grad, double[:,:] stresses, 
	double[:,:] pos, double[:,:] vects,  int[:] charges,
	double real_cut_off, double alpha, double alpha_drv, double volume, int N):
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

	cdef double dist
	cdef double drv, term, k_e
	cdef double* rij
	cdef Py_ssize_t ioni, ionj, dim
	cdef Py_ssize_t shift, shifts_no, i, l, m
	cdef double[:,:] shifts
	cdef double[:] nil_array, part_drv
	cdef int charges_val = 0

	shifts = get_shifts(vects, real_cut_off)
	if shifts == None:
		shifts_no = 0
	else:
		shifts_no = len(shifts)

	nil_array = array("d",[0,0,0])
	part_drv = array("d",[0,0,0])
	k_e = 14.399645351950543   

	with nogil:

		# Allocate thread-local memory for distance vector
		rij = <double *> malloc(sizeof(double) * 3)

		for ioni in range(N):
			for ionj in range(N):
				if ioni != ionj:  # skip in case it's the same atom or it is constant

					rij = get_distance_vector(rij, pos[ioni], pos[ionj], nil_array)
					dist = get_real_distance(pos[ioni], pos[ionj], nil_array)
					charges_val = charges[ioni]*charges[ionj]

					# Partial derivative
					drv = erfc_drv(alpha, charges_val, dist)

					part_drv[0] = drv*rij[0]/dist*k_e/2 # Coulomb constant
					part_drv[1] = drv*rij[1]/dist*k_e/2 
					part_drv[2] = drv*rij[2]/dist*k_e/2 

					# partial deriv with respect to ioni
					grad[ioni][0] += part_drv[0]
					grad[ioni][1] += part_drv[1]
					grad[ioni][2] += part_drv[2]

					# partial deriv with respect to ionj
					grad[ionj][0] -= part_drv[0]
					grad[ionj][1] -= part_drv[1]
					grad[ionj][2] -= part_drv[2]

					# deriv with respect to strain epsilon_lm (Ln not present in summand)
					for l in range(3):
						for m in range(l,3):
							stresses[l][m] += part_drv[l]*(pos[ioni][m]-pos[ionj][m])
							
							# Add the following if alpha is not constant
							if (l==m):
								term = exp(-alpha*alpha*dist*dist)
								drv = charges_val*k_e * \
									-alpha_drv*term/sqrt(pi)
								stresses[l][m] += drv

				# take care of the rest lattice (+ Ln)
				for shift in range(shifts_no):
					rij = get_distance_vector(rij, pos[ioni], pos[ionj], shifts[shift])
					dist = get_real_distance(pos[ioni], pos[ionj], shifts[shift])
					charges_val = charges[ioni]*charges[ionj]

					# Avoid division by zero
					if dist == 0:
						dist = DBL_EPSILON

					# Partial derivative
					drv = erfc_drv(alpha, charges_val, dist)

					part_drv[0] = drv*rij[0]/dist*k_e/2 # Coulomb constant
					part_drv[1] = drv*rij[1]/dist*k_e/2
					part_drv[2] = drv*rij[2]/dist*k_e/2

					# partial deriv with respect to ioni
					grad[ioni][0] += part_drv[0]
					grad[ioni][1] += part_drv[1]
					grad[ioni][2] += part_drv[2]

					# partial deriv with respect to ionj
					grad[ionj][0] -= part_drv[0] 
					grad[ionj][1] -= part_drv[1] 
					grad[ionj][2] -= part_drv[2] 

					# deriv with respect to strain epsilon_lm (Ln present in summand)
					for l in range(3):
						for m in range(l,3):
							stresses[l][m] += part_drv[l]*( pos[ioni][m]+shifts[shift][m]-pos[ionj][m] )
							
							if (l==m):
								term = exp(-alpha*alpha*dist*dist)
								drv = charges_val*k_e* \
									-alpha_drv*term/sqrt(pi)
								stresses[l][m] += drv
		# Deallocate distance vector
		free(rij)

	return grad


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double[:,:] recip_drv(double[:,:] grad, double[:,:] stresses, 
	double[:,:] pos, double[:,:] vects,  double[:,:] rvects, int[:] charges,
	double recip_cut_off, double alpha, double alpha_drv, double volume, int N):
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
	cdef double drv, k_2, stemp
	cdef double krij, term
	cdef double strain_temp
	cdef double* rij
	cdef double[:,:] shifts
	cdef double[:] nil_array
	cdef Py_ssize_t shifts_no
	cdef double k_e = 14.399645351950543 # Coulomb constant

	shifts = get_shifts(rvects, recip_cut_off)
	nil_array = array("d",[0,0,0])

	# Check interactions with neighbouring cells
	if shifts == None:
		shifts_no = 0
	else:
		shifts_no = len(shifts)

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
					drv = - charges[ioni]*charges[ionj] * \
									   2*pi*k_e*term*sin(krij)/(k_2*volume)
			
					# partial deriv with respect to ioni
					grad[ioni][0] += drv*shifts[shift,0]
					grad[ioni][1] += drv*shifts[shift,1]
					grad[ioni][2] += drv*shifts[shift,2]

					# partial deriv with respect to ionj
					grad[ionj][0] -= drv*shifts[shift,0]
					grad[ionj][1] -= drv*shifts[shift,1]
					grad[ionj][2] -= drv*shifts[shift,2]

					drv = 2*pi*k_e/volume * \
						charges[ioni]*charges[ionj] * \
						term/k_2 * cos(krij)

					for l in range(3):
						for m in range(l,3):					
							# deriv with respect to strain epsilon_lm 
							stresses[l][m] += \
								drv*(
									( 1/(2*alpha*alpha)+2/k_2 ) * shifts[shift][m] * \
									shifts[shift][l]
									)
							# Add the following if alpha is not constant
							if (l==m):
								stresses[l][m] -= \
									drv*(1 - \
										k_2/(2*pow(alpha,3)) * alpha_drv)

		# Deallocate distance vector
		free(rij)

	return grad


cdef double[:,:] fill_stresses(double[:,:] grad, double[:,:] stresses, double volume=1):
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
			stresses[j][i] = stresses[i][j]

	for i in range(-1,-4,-1):
		for j in range(-1,i-1,-1):
			grad[i][j] = stresses[i][j]/volume
			grad[j][i] = stresses[i][j]/volume

	return stresses