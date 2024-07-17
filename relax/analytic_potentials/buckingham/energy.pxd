cimport numpy as cnp


'''																						'''
'''									BUCKINGHAM											'''
'''																						'''


# Function that returns the self term of the dispersion energy value when expanded with the Ewald sum
cdef double ewald_self(buck, double[:,:] vects, cnp.ndarray chemical_symbols, 
	double alpha, int N) except? 0

# Function that returns the reciprocal term of the dispersion energy value when expanded with the Ewald sum
cdef double ewald_recip(buck, double[:,:] pos, double[:,:] vects, double[:,:] rvects, 
	cnp.ndarray chemical_symbols, double alpha, double recip_cut_off, int N) except? 0

# Function that returns the real term of the dispersion energy value when expanded with the Ewald sum
cdef double ewald_real(buck, double[:,:] pos, double[:,:] vects, 
	cnp.ndarray chemical_symbols, double alpha, double real_cut_off, int N) except? 0