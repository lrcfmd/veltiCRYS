import numpy as np
cimport numpy as cnp


cdef class Potential:
	cdef double[:,:] get_reciprocal_vects(self, double[:,:] vects, double volume)
	cpdef int[:] get_cubic_cutoff(self, double[:,:] vects, double init_cutoff)
	cpdef double get_alpha(self, double N, double volume) except? 0
	cdef double calc_alpha_drv(self, double N, double volume) nogil except? 0


cpdef double get_gnorm(double[:,:] grad, int N)
