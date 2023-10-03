cimport numpy as cnp


'''																						'''
'''									BUCKINGHAM											'''
'''																						'''


# Function that returns the derivative of the Coulomb energy self term when expanded with the Ewald sum
cdef double[:,:] self_drv(buck, double[:,:] stresses, cnp.ndarray chemical_symbols, 
	double alpha, double alpha_drv, double volume, int N)

# Function that returns the derivative of the Coulomb energy real term when expanded with the Ewald sum
cdef double[:,:] real_drv(buck, double[:,:] grad, double[:,:] stresses, 
	double[:,:] pos, double[:,:] vects,  cnp.ndarray chemical_symbols,
	double real_cut_off, double alpha, double alpha_drv, double volume, int N)
# # Function that returns the portion of the real Ewald part 
cdef double cterm_drv(double A, double C, double rho, double dist, double alpha, double term)

# Function that returns the derivative of the Coulomb energy reciprocal term when expanded with the Ewald sum
cdef double[:,:] recip_drv(buck, double[:,:] grad, double[:,:] stresses, 
	double[:,:] pos, double[:,:] vects,  double[:,:] rvects, cnp.ndarray chemical_symbols,
	double recip_cut_off, double alpha, double alpha_drv, double volume, int N)

# Function to fill the lower triangular part of the stress tensor with symmetrical values.
cdef double[:,:] fill_stresses(double[:,:] grad, double[:,:] stresses, double volume=*)