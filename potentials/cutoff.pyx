cimport potentials.operations as ops
import numpy as np

from ase import *

from libc.math cimport sqrt, ceil
from libc.stdio cimport printf
from libc.stdlib cimport malloc,free

from itertools import combinations 
from cython.view cimport array as cvarray
from cpython.array cimport array, clone


cdef double[:,:] get_normals(double[:,:] vects):
	"""Returns the normal vectors for each
	of the 3 of unit cell faces defined by the given 
	lattice vectors in the following order:
	ab-plane, bc-plane, ac-plane. The product is
	calculated using the determinant formula.
	
	"""

	cdef size_t i, v1, v2
	cdef double[:,:] normals_view
	cdef double[:,:] normals

	normals = cvarray(shape=(3,3), itemsize=sizeof(double), format="d")

	for i in range(3):
		v1 = i%3
		v2 = (i+1)%3
		normals[i][0] = vects[v1][1]*vects[v2][2] - vects[v2][1]*vects[v1][2]
		normals[i][1] = vects[v1][2]*vects[v2][0] - vects[v1][0]*vects[v2][2]
		normals[i][2] = vects[v1][0]*vects[v2][1] - vects[v1][1]*vects[v2][0]

	return normals


cpdef double[:,:] inflated_cell_truncation(double[:,:] vects, double cutoff):

	cdef size_t i, count, shifts_no
	cdef double nnorm, volume
	cdef double[:] centre, translate_view
	cdef double[:,:] normals, shifts
	cdef double[3] translate

	volume = abs(ops.det(vects))
	centre = ops.dot_vm(array("d", [0.5, 0.5, 0.5]), vects)
	normals = get_normals(vects)

	# Find translation distance 
	for i in range(3):
		nnorm = ops.norm(normals[i])
		height = volume / nnorm
		translate[(i+2)%3] = ceil(round((cutoff-height/2) / height))

		if translate[(i+2)%3]<0:
			printf("\x1B[33mWarning:\033[0m The translate vector component %ld  \
				is negative: %f \n",(i+2)%3,translate[(i+2)%3])
			return None

	shifts_no = (2*int(translate[0])+1) * \
				(2*int(translate[1])+1) * \
				(2*int(translate[2])+1)-1

	if shifts_no == 0:
		return None

	translate[0] = ceil(translate[0])
	translate[1] = ceil(translate[1])
	translate[2] = ceil(translate[2])

	shifts_no = (2*int(translate[0])+1) * \
				(2*int(translate[1])+1) * \
				(2*int(translate[2])+1)-1

	try:
		shifts = cvarray(shape=(shifts_no,3), itemsize=sizeof(double), format="d")
	except:
		return None

	count = 0
	for shift in np.ndindex(
		2*int(translate[0])+1, 2*int(translate[1])+1, 2*int(translate[2])+1 ):
		
		if shift!=(translate[0],translate[1],translate[2]):
			shifts[count][0] = shift[0] - translate[0]
			shifts[count][1] = shift[1] - translate[1]
			shifts[count][2] = shift[2] - translate[2]
			count += 1

	return ops.dot_mm(shifts,vects)


cpdef double[:,:] translation_wrapper(double[:,:] vects, double cutoff):

	cdef size_t i, count, shifts_no
	cdef double nnorm, volume
	cdef double[:] translate, centre
	cdef double[:,:] normals, borders

	volume = abs(ops.det(vects))
	centre = ops.dot_vm(np.asarray([0.5, 0.5, 0.5]).T, vects)
	borders = cvarray(shape=(2,3), itemsize=sizeof(double), format="d")

	normals = get_normals(vects)
	translate = get_translation_vector(
		vects, volume, cutoff, centre, normals)

	# Find how many times parallelepiped height fits
	for i in range(3):
		nnorm = ops.norm(normals[(i+1)%3])
		height = volume / nnorm
		# Store integer max border
		borders[0][i] = ceil(round(translate[(i+1)%3] / height, 10))
		# Store percentage of remainder of length inside enlarged unit cell
		borders[1][i] = (translate[(i+1)%3] % height) / height

	return borders


cpdef double[:,:] get_shifts(double[:,:] vects, double[:,:] borders):
	cdef Py_ssize_t i, count, shifts_no

	shifts_no = (2*int(borders[0][0])+1) * \
				(2*int(borders[0][1])+1) * \
				(2*int(borders[0][2])+1)-1

	if shifts_no == 0:
		return None
	else:
		shifts = cvarray(shape=(shifts_no,3), itemsize=sizeof(double), format="d")

	count = 0
	for shift in np.ndindex(
		2*int(borders[0][0])+1, 2*int(borders[0][1])+1, 2*int(borders[0][2])+1 ):
		
		if shift!=(borders[0][0],borders[0][1],borders[0][2]):
			shifts[count][0] = shift[0] - borders[0][0]
			shifts[count][1] = shift[1] - borders[0][1]
			shifts[count][2] = shift[2] - borders[0][2]
			count += 1

	# return ops.dot_mm(shifts,vects)
	return shifts
	
cpdef int check_lattice(double[:,:] vects, double old_volume, double volume,
double min_length=1, double max_length_per=1000):
	cdef double[:] norms
	max_length = 0
	norms = cvarray(shape=(3,), itemsize=sizeof(double), format="d")
	for i in range(3):
		norms[i] = ops.norm(vects[i])
		if norms[i]>max_length:
			max_length = norms[i]

	if norms[0]<min_length or \
	norms[1]<min_length or norms[2]<min_length:
		printf("\x1B[31mWarning:\033[0mLattice vectors under limit.\n")
		return 1	

	if volume > 2*old_volume:
		return 1

	# if max_length>max_length_per*norms[0] or \
	# max_length>max_length_per*norms[1] or max_length>max_length_per*norms[2]:
	# 	printf("\x1B[31mWarning:\033[0mLattice vector over limit.\n")
	# 	return 1
	return 0