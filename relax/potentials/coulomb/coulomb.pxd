import numpy as np
cimport numpy as cnp

from relax.potentials.potential cimport EwaldPotential


'''																						'''
'''									  COULOMB 											'''
'''																						'''


cdef class Coulomb(EwaldPotential):
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
	cdef bint param_flag

	cpdef set_cutoff_parameters(self, double[:,:] vects=*, int N=*, double accuracy=*, double real =*, double reciprocal=*)
	cpdef print_parameters(self)

	cpdef double energy(self, atoms=*, double[:,:] pos_array=*, double[:,:] vects_array=*, int N_=*) except? 0
	cpdef get_all_ewald_energies(self)
	cpdef double[:,:] gradient(self, atoms=*, double[:,:] pos_array=*, double[:,:] vects_array=*, int N_=*)
	cpdef double[:,:] get_stresses(self)
