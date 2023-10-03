import numpy as np
cimport numpy as cnp
import warnings
import heapq
import csv
import sys

from cysrc.potential cimport Potential

cdef class Buckingham(Potential):
	"""Calculations for the Buckingham energy contribution. It
	corresponds to the interatomic forces exercised among entities.
	
	"""
	cdef cnp.ndarray chemical_symbols
	cdef double eself, ereal, erecip
	cdef bint param_flag, alpha_flag
	cdef double e, alpha
	cdef double[:,:] grad, stresses
	cdef double[:] radii
	cdef double real_cut_off
	cdef double recip_cut_off, limit
	cdef int close_pairs

	cpdef set_cutoff_parameters(self, double[:,:] vects=*, int N=*, double accuracy=*, double real =*, double reciprocal=*)
	cpdef print_parameters(self)
	cpdef double get_max_radius(self) except? -1
	cpdef double get_limit(self) except? -1
	cdef int catastrophe_check(self) except? -1

	cdef double calc_self(self, double[:,:] vects, int N) except? 0
	cdef double calc_recip(self, double[:,:] pos, double[:,:] vects, int N) except? 0
	cdef double calc_real(self, double[:,:] pos, double[:,:] vects, int N) except? 0	
	cpdef double calc(self, atoms=*, double[:,:] pos_array=*, double[:,:] vects_array=*, int N_=*) except? 0
	cpdef get_energies(self)
	
	cdef double[:,:] calc_recip_drv(self, double[:,:] pos, double[:,:] vects, int N, double volume)
	cdef double[:,:] calc_self_drv(self, double volume, int N)
	cdef double[:,:] calc_real_drv(self, double[:,:] pos, double[:,:] vects, int N, double volume)
	cpdef double[:,:] calc_drv(self, atoms=*, double[:,:] pos_array=*, double[:,:] vects_array=*, int N_=*)
	cdef double[:,:] fill_stresses(self, double volume=*)

	cpdef double calc_simple(self, atoms=*, \
		double[:,:] pos_array=*, double[:,:] vects_array=*, int N_=*) except? 0
	