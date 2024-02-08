import numpy as np
cimport numpy as cnp
import warnings
import heapq
import csv
import sys

from cython.view cimport array as cvarray
from libc.stdlib cimport calloc, free
from cython.parallel import prange,parallel

from libc cimport bool
from libc.stdio cimport *                                                                
from libc.math cimport *
from libc.float cimport *
from libc.limits cimport *
from libc.stdio cimport printf
import cython, shutil

from relax.potentials.cutoff cimport inflated_cell_truncation as get_shifts

from relax.potentials.operations cimport det as det3_3
from relax.potentials.operations cimport get_real_distance 
from relax.potentials.operations cimport get_recip_distance
from relax.potentials.operations cimport get_distance_vector
from relax.potentials.operations cimport norm_m,norm
from relax.potentials.operations cimport get_all_distances, get_min_dist

from cpython.array cimport array, clone


'''																						'''
'''									BUCKINGHAM											'''
'''																						'''
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double cterm_drv(double A, double C, double rho, double dist, double alpha, double term):
	"""The derivative of the real function that depends only 
	on the pairwise distance between two ions.
	
	"""
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
	return drv

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double[:,:] real_drv(buck, double[:,:] grad, double[:,:] stresses, 
	double[:,:] pos, double[:,:] vects,  cnp.ndarray chemical_symbols,
	double real_cut_off, double alpha, double alpha_drv, double volume, int N):
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

	cdef double cutoff, dist, a2pi
	cdef double drv, term, k_e
	cdef double* rij
	cdef Py_ssize_t ioni, ionj, dim
	cdef Py_ssize_t shift, shifts_no, i, l, m
	cdef double[:,:] shifts
	cdef double[:] nil_array, part_drv
	cdef double A, rho, C

	shifts = get_shifts(vects, real_cut_off)
	if shifts == None:
		shifts_no = 0
	else:
		shifts_no = len(shifts)

	nil_array = array("d",[0,0,0])
	part_drv = array("d",[0,0,0])  

	# Allocate thread-local memory for distance vector
	rij = <double *> calloc(3, sizeof(double))

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

				if ioni != ionj:  # skip in case it's the same atom or it is constant

					rij = get_distance_vector(rij, pos[ioni], pos[ionj], nil_array)
					dist = get_real_distance(pos[ioni], pos[ionj], nil_array)
					term = exp(-alpha*alpha*dist*dist)
					
					# Partial derivative real value
					drv = cterm_drv(A, C, rho, dist, alpha, term)

					part_drv[0] = drv*rij[0]/2
					part_drv[1] = drv*rij[1]/2
					part_drv[2] = drv*rij[2]/2

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
							stresses[l][m] += part_drv[l]*rij[m]
							
							# Add the following if alpha is not constant
							if (l==m):
								drv =  C*term*alpha_drv*pow(alpha,5)
								stresses[l][m] += drv/2

				# take care of the rest lattice (+ Ln)
				for shift in range(shifts_no):
					rij = get_distance_vector(rij, pos[ioni], pos[ionj], shifts[shift])
					dist = get_real_distance(pos[ioni], pos[ionj], shifts[shift])
					term = exp(-alpha*alpha*dist*dist)

					# Avoid division by zero
					if dist == 0:
						dist = DBL_EPSILON

					# Partial derivative real value
					drv = cterm_drv(A, C, rho, dist, alpha, term)

					part_drv[0] = drv*rij[0]/2
					part_drv[1] = drv*rij[1]/2
					part_drv[2] = drv*rij[2]/2

					# partial deriv with respect to ioni
					grad[ioni][0] += part_drv[0]
					grad[ioni][1] += part_drv[1]
					grad[ioni][2] += part_drv[2]

					# partial deriv with respect to ionj
					grad[ionj][0] -= part_drv[0] 
					grad[ionj][1] -= part_drv[1] 
					grad[ionj][2] -= part_drv[2] 

					# # deriv with respect to strain epsilon_lm (Ln present in summand)
					for l in range(3):
						for m in range(l,3):
							stresses[l][m] += part_drv[l]*rij[m]
							
							# Add the following if alpha is not constant
							if (l==m):
								drv =  C*term*alpha_drv*pow(alpha,5)
								stresses[l][m] += drv/2

	# Deallocate distance vector
	free(rij)

	return grad


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double[:,:] recip_drv(buck, double[:,:] grad, double[:,:] stresses, 
double[:,:] pos, double[:,:] vects,  double[:,:] rvects, cnp.ndarray chemical_symbols,
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

	cdef double cutoff, drv, dist
	cdef double term_exp, term_c, term_erfc, term_a
	cdef double[:,:] shifts
	cdef Py_ssize_t shifts_no, i
	cdef double* rij
	cdef double[:] nil_array, part_drv
	cdef double A, rho, C, k_2, k_3

	nil_array = array("d",[0,0,0])
	part_drv = array("d",[0,0,0])
	shifts = get_shifts(rvects, recip_cut_off)

	# Check interactions with neighbouring cells
	if shifts == None:
		shifts_no = 0
	else:
		shifts_no = len(shifts)

	# allocate memory for distance vector
	rij = <double *> calloc(3, sizeof(double))

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
					grad[ioni][0] += drv*shifts[shift,0]
					grad[ioni][1] += drv*shifts[shift,1]
					grad[ioni][2] += drv*shifts[shift,2]

					# partial deriv with respect to ionj
					grad[ionj][0] -= drv*shifts[shift,0]
					grad[ionj][1] -= drv*shifts[shift,1]
					grad[ionj][2] -= drv*shifts[shift,2]

					drv = term_c*cos(krij)/2 * \
						(
							3*term_erfc*sqrt(k_2) - 6*alpha*term_exp
						)

					for l in range(3):
						for m in range(l,3):					
							# deriv with respect to strain epsilon_lm 
							stresses[l][m] += drv * \
								shifts[shift][m]*shifts[shift][l]
								
							# Add the following if alpha is not constant
							if (l==m):
								stresses[l][m] -= term_c*cos(krij)/2 * \
									(
										-term_erfc*k_3 - \
										(term_a-12*pow(alpha,2)*alpha_drv)*term_exp											
									)

	# Deallocate distance vector
	free(rij)

	return grad


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:,:] self_drv(buck, double[:,:] stresses, cnp.ndarray chemical_symbols,
double alpha, double alpha_drv, double volume, int N):
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
	cdef double A, rho, C

	for i in range(N):
		for j in range(N):
			# Find the pair we are examining
			pair = (min(chemical_symbols[i], chemical_symbols[j]),
					max(chemical_symbols[i], chemical_symbols[j]))
			if (pair in buck):
				# Pair of ions is listed in parameters file
				C = buck[pair]['par'][2]
				for l in range(3):
					stresses[l][l] -= C/(3*volume)*pow(pi,1.5) * alpha*alpha * \
						(3*alpha_drv-alpha)/2
		# Find the pair we are examining
		pair = (chemical_symbols[i], chemical_symbols[i])
		if (pair in buck):
			# Pair of ions is listed in parameters file
			C = buck[pair]['par'][2]
			for l in range(3):
				stresses[l][l] += C*pow(alpha,5)*alpha_drv/2

	return stresses


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


