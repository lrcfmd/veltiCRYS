cdef double[:,:] get_normals(double[:,:] vects)
cpdef double[:,:] inflated_cell_truncation(double[:,:] vects, double cutoff)

cpdef double[:,:] translation_wrapper(double[:,:] vects, double cutoff)
cpdef double[:,:] get_shifts(double[:,:] vects, double[:,:] borders)
cpdef int check_lattice(double[:,:] vects, double old_volume, double volume, double min_length=*, double max_length_per=*)