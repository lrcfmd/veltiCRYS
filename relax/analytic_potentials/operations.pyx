import cython
from libc.math cimport sqrt
from cython.view cimport array as cvarray
from libc.stdio cimport printf
from cpython.array cimport array


@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cdef double dot_vv(double[:] a, double[:] b) except? 0:
    """ Dot product of (vector a) x (vector B)

    """ 
    cdef Py_ssize_t M = a.shape[0]
    cdef Py_ssize_t N = b.shape[0]

    if not M==N:
        raise ValueError("Error in vectors' dimensions.")
    
    cdef double result = 0

    for k in range(M):
        result += a[k]*b[k]

    return result


@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cdef double[:] dot_vm(double[:] a, double[:,:] B) except *:
    """ Dot product of (vector a) x (matrix B) where matrix B
    has 2 dimensions

    """ 
    cdef Py_ssize_t M = a.shape[0]
    cdef Py_ssize_t Nb = B.shape[0]
    cdef Py_ssize_t K = B.shape[1]

    if not M==Nb:
        raise ValueError("Error in vectors' dimensions.")
    
    cdef double[:] result
    cdef double t

    result = cvarray(shape=(K,), itemsize=sizeof(double), format="d")

    for k in range(K):
        t = 0
        for n in range(M):
            t += a[n] * B[n, k]
        result[k] = t

    return result


@cython.boundscheck(False) 
@cython.wraparound(False) 
cdef double[:,:] dot_mm(double[:,:] A, double[:, :] B):
    """ Dot product of (matrix A) x (matrix B) of 2 dimensions each.

    """ 
    cdef Py_ssize_t M = A.shape[0]
    cdef Py_ssize_t Na = A.shape[1]
    cdef Py_ssize_t Nb = B.shape[0]
    cdef Py_ssize_t K = B.shape[1]

    if not Na==Nb:
        raise ValueError("Error in matrices' dimensions.")

    cdef double[:,:] result
    cdef double t

    result = cvarray(shape=(M,K), itemsize=sizeof(double), format="d")

    for m in range(M):
        for k in range(K):
            result[m,k] = 0
            for n in range(Na):
                result[m,k] += A[m,n] * B[n,k]

    return result


@cython.boundscheck(False) 
@cython.wraparound(False) 
cdef double norm(double[:] a):

    cdef Py_ssize_t i
    cdef double result = 0
    
    for i in range(a.shape[0]):
        result += a[i]*a[i]
        
        # printf("%f ",a[i])              ######### DEBUG
    # printf("\n%f\n",result)

    if result==0:
        printf("\x1B[31mWarning:\033[0m Vector norm is zero.\n")

    return sqrt(result)


@cython.boundscheck(False) 
@cython.wraparound(False) 
cdef double norm_m(double[:,:] a):

    cdef Py_ssize_t i
    cdef double result = 0
    
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            result += a[i][j]*a[i][j]

    return sqrt(result)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double det(double[:,:] arr) except? -1:
    """Returns the determinant of an 3x3 matrix.

    """
    if (not arr.shape[0]==3) or (not arr.shape[1]==3):
        raise ValueError("Error in matrix's dimensions.")

    cdef double det
    det = arr[0,0]*(arr[1,1]*arr[2,2]-arr[1,2]*arr[2,1])- \
            arr[0,1]*(arr[1,0]*arr[2,2]-arr[1,2]*arr[2,0])+ \
            arr[0,2]*(arr[1,0]*arr[2,1]-arr[1,1]*arr[2,0])

    return det


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double get_real_distance(double[:] ioni, double[:] ionj, double[:] shift) nogil:
    cdef double dist
    dist = (ioni[0]+shift[0]-ionj[0])*(ioni[0]+shift[0]-ionj[0])+ \
            (ioni[1]+shift[1]-ionj[1])*(ioni[1]+shift[1]-ionj[1])+ \
            (ioni[2]+shift[2]-ionj[2])*(ioni[2]+shift[2]-ionj[2])
    dist = sqrt(dist)
    return dist


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double get_recip_distance(double* rij, double[:] shift) nogil:
    cdef double krij
    # dot product to find image
    krij = shift[0]*rij[0] + shift[1]*rij[1] + shift[2]*rij[2]
    return krij


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double* get_distance_vector(double* rij, double[:] ioni, double[:] ionj, double[:] shift) nogil:
    rij[0] = ioni[0]+shift[0]-ionj[0] # distance vector
    rij[1] = ioni[1]+shift[1]-ionj[1]
    rij[2] = ioni[2]+shift[2]-ionj[2]
    return rij


cpdef double[:,:] get_all_distances(double[:,:] pos, int N):
    cdef Py_ssize_t ioni, ionj
    cdef double[:,:] dists
    cdef double[:] nil_array

    nil_array = array("d",[0,0,0])
    dists = cvarray(shape=(N,N), itemsize=sizeof(double), format="d")
    
    for ioni in range(N):
        for ionj in range(ioni, N):
            dists[ioni][ionj] = get_real_distance(pos[ioni], pos[ionj], nil_array)
            dists[ionj][ioni] = dists[ioni][ionj]
    return dists


cpdef double[:] get_min_dist(double[:,:] pos, int N):
    cdef Py_ssize_t i, j
    cdef Py_ssize_t ioni, ionj
    cdef double dist
    cdef double[:] nil_array = array("d",[0,0,0])
    cdef double[:] mins = array("d",[0,0,0])
    
    for ioni in range(N):
        for ionj in range(ioni, N):
            dist = get_real_distance(pos[ioni], pos[ionj], nil_array)
            if (ioni==1) & (ionj==2):
                mins[0] = dist
                mins[1] = ioni
                mins[2] = ionj
            elif (dist>0) & (dist<mins[0]):
                mins[0] = dist
                mins[1] = ioni
                mins[2] = ionj

    return mins