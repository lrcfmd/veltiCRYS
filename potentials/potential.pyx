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

cpdef double get_volume(double[:,:] vects):
	return abs(det3_3(vects))


cdef class Potential:
	"""Generic class for defining potentials."""

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef double[:,:] get_reciprocal_vects(self, double[:,:] vects, double volume):
		"""Calculate reciprocal vectors.
		
		Parameters
		----------
		vects : 3x3 array (double)
			The lattice vectors in Cartesian coordinates.
		volume : double
			The volume of the unit cell.
			
		Returns
		-------
		3x3 array (double)
		"""

		cdef size_t i,a,b
		cdef double[:,:] rvects_view

		rvects = cvarray(shape=(3,3),itemsize=sizeof(double),format="d")
		for i in range(3):
			a = (1+i) % 3
			b = (2+i) % 3
			rvects[i,:] = 2*pi*np.cross(vects[a,:],vects[b,:]) / volume

		rvects_view = rvects
		return rvects_view


	@cython.boundscheck(False)
	@cython.wraparound(False)
	cpdef double get_alpha(self, double N, double volume) except? 0:
		"""
		Function to calculate the balance between terms in real 
		and reciprocal space.

		Parameters
		----------
		N : int
			Number of atoms in unit cell.
		volume : float
			Volume of unit cell.

		Returns
		-------
		double

		"""

		cdef double alpha
		alpha = pow(N,1/6) * sqrt(pi) / pow(volume,1/3)
		return alpha


	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef double calc_alpha_drv(self, double N, double volume) nogil except? 0:
		""" Gives the derivative w.r.t. volume multiplied by the
		derivative of volume w.r.t. strain """
		cdef double alpha
		alpha = - pow(N,1/6) * sqrt(pi) / (3*pow(volume,1/3))
		return alpha

	
	cpdef int[:] get_cubic_cutoff(self, double[:,:] vects, double init_cutoff):
		"""
		Function to calculate the cutoff for cubic unit cells.

		Parameters
		----------
		N : int
			Number of atoms in unit cell.
		volume : float
			Volume of unit cell.

		Returns
		-------
		1D array (int)
		
		"""
		cdef size_t i
		cdef int[:] cutoff_view
		cdef double[3] normal
		cdef double normal_norm, volume

		cutoff_view = cvarray(shape=(3,),itemsize=sizeof(double),format="d")

		for i in range(3):
			cutoff_view[i] = 0
			normal = np.cross(
				vects[(i+1)%3,],vects[(i+2)%3,])
			normal_norm = sqrt(normal[0]*normal[0] + \
			 normal[1]*normal[1] + normal[2]*normal[2])
			volume = abs(det3_3(vects))

			cutoff_view[i] = int(ceil(init_cutoff/(volume/normal_norm)))
		return cutoff_view


cpdef double get_gnorm(double[:,:] grad, int N): 
	"""
	Function to calculate the cutoff for cubic unit cells.

	Parameters
	----------
	grad : Nx3 array (double)
		Volume of unit cell.
	N : int
		Number of atoms in unit cell.
	

	Returns
	-------
	double
	
	"""
	cdef double gnorm, temp
	cdef int i, j, count

	gnorm = 0
	for i in range(N):
		for j in range(3):
			gnorm += pow(grad[i][j],2)

	for i in range(3):
		for j in range(i,3):
			gnorm += pow(grad[N+i][j],2)
			
	return sqrt(gnorm)/(3*N+6)
