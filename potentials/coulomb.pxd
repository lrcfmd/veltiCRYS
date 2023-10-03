import numpy as np
cimport numpy as cnp

from cysrc.potential cimport Potential

cdef class Coulomb(Potential):
	"""Calculations for the Coulomb energy contribution. It
	corresponds to the electrostatic forces exercised among entities.
	
	"""
	cdef double alpha, made_const
	cdef double eself, ereal, erecip
	cdef double[:,:] grad, stresses, hessian
	cdef cnp.ndarray chemical_symbols
	cdef double real_cut_off
	cdef double recip_cut_off
	cdef int[:] charges
	cdef bint param_flag, alpha_flag

	cpdef set_cutoff_parameters(self, double[:,:] vects=*, int N=*, double accuracy=*, double real =*, double reciprocal=*)
	cpdef print_parameters(self)
	
	cdef double calc_self(self, int N) except? 0
	cdef double calc_real(self, double[:,:] pos, double[:,:] vects, int N) except? 0
	cdef double calc_recip(self, double[:,:] pos, double[:,:] vects, int N) except? 0
	cpdef calc_madelung(self, double[:,:] pos, int N)
	cpdef double calc(self, atoms=*, double[:,:] pos_array=*, double[:,:] vects_array=*, int N_=*) except? 0
	cpdef get_energies(self)

	cdef double[:,:] calc_self_drv(self, double volume, int N)
	cdef double[:,:] calc_real_drv(self, double[:,:] pos, double[:,:] vects, int N, double volume)	
	cdef double[:,:] calc_recip_drv(self, double[:,:] pos, double[:,:] vects, int N, double volume)
	cpdef double[:,:] calc_drv(self, atoms=*, double[:,:] pos_array=*, double[:,:] vects_array=*, int N_=*)
	cdef double[:,:] fill_stresses(self, double volume=*)

