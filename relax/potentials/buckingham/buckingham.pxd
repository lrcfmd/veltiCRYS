cimport numpy as cnp
from relax.potentials.potential cimport EwaldPotential


'''																						'''
'''									BUCKINGHAM											'''
'''																						'''


cdef class Buckingham(EwaldPotential):
	"""Calculations for the Buckingham energy contribution. It
	corresponds to the interatomic forces exercised among entities.
	
	"""
	cdef cnp.ndarray chemical_symbols
	cdef double eself, ereal, erecip
	cdef bint param_flag
	cdef double alpha
	cdef double[:,:] grad, stresses
	cdef double real_cut_off
	cdef double recip_cut_off

	cpdef set_cutoff_parameters(self, double[:,:] vects=*, 
		int N=*, double accuracy=*, double real =*, double reciprocal=*)
	cpdef print_parameters(self)

	cpdef double energy(self, atoms=*, 
		double[:,:] pos_array=*, double[:,:] vects_array=*, int N_=*) except? 0
	cpdef get_all_ewald_energies(self)

	cpdef double[:,:] gradient(self, atoms=*, double[:,:] pos_array=*, 
		double[:,:] vects_array=*, int N_=*)