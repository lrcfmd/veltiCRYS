cdef double dot_vv(double[:] a, double[:] b) except? 0
cdef double[:] dot_vm(double[:] A, double[:, :] B) except *
cdef double[:,:] dot_mm(double[:,:] A, double[:, :] B)
cdef double norm(double[:] a)
cdef double norm_m(double[:,:] a)
cdef double det(double[:,:] arr) except? -1
cdef double get_real_distance(double[:] ioni, double[:] ionj, double[:] shift) nogil
cdef double get_recip_distance(double* rij, double[:] shift) nogil
cdef double* get_distance_vector(double* rij, double[:] ioni, double[:] ionj, double[:] shift) nogil
cpdef double[:,:] get_all_distances(double[:,:] pos, int N)
cpdef double[:] get_min_dist(double[:,:] pos, int N)