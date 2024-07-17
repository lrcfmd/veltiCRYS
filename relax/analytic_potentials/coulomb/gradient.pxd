'''																						'''
'''									  COULOMB 											'''
'''																						'''

# Function that returns the derivative of the Coulomb energy self term when expanded with the Ewald sum
cdef double[:,:] self_drv(double[:,:] stresses, int[:] charges, 
	double alpha_drv, double volume, int N)

# Function that returns the derivative of the Coulomb energy real term when expanded with the Ewald sum
cdef double[:,:] real_drv(double[:,:] grad, double[:,:] stresses, 
	double[:,:] pos, double[:,:] vects,  int[:] charges,
	double real_cut_off, double alpha, double alpha_drv, double volume, int N)
# Function that returns the portion of the real Ewald part with the erfc derivative
cdef double erfc_drv(double alpha, int charges, double dist) nogil

# Function that returns the derivative of the Coulomb energy reciprocal term when expanded with the Ewald sum
cdef double[:,:] recip_drv(double[:,:] grad, double[:,:] stresses, 
	double[:,:] pos, double[:,:] vects,  double[:,:] rvects, int[:] charges,
	double recip_cut_off, double alpha, double alpha_drv, double volume, int N)

# Function to fill the lower triangular part of the stress tensor with symmetrical values.
cdef double[:,:] fill_stresses(double[:,:] grad, double[:,:] stresses, double volume=*)