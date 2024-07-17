'''																						'''
'''									  COULOMB 											'''
'''																						'''

# Function that returns the self term of the Coulomb energy value when expanded with the Ewald sum
cdef double ewald_self(int[:] charges, double alpha, int N) except? 0

# Function that returnsthe real term of the Coulomb energy value when expanded with the Ewald sum
cdef double ewald_real(double[:,:] pos, double[:,:] vects, 
	int[:] charges, double alpha, double real_cut_off, int N) except? 0

# Function that returns the reciprocal term of the Coulomb energy value when expanded with the Ewald sum
cdef double ewald_recip(double[:,:] pos, double[:,:] vects, double[:,:] rvects,
	int[:] charges, double alpha, double recip_cut_off, int N) except? 0

# Function that returns the Coulomb energy value calculated using the Madelung constant
cpdef madelung(double made_const, double[:,:] pos, int[:] charges, int N)