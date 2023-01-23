from ase import *
from ase.visualize import view
from ase.io import read as aread
from ase.io import write as awrite
from ase.calculators.gulp import GULP
from ase.calculators.lammpslib import LAMMPSlib
from ase.visualize.plot import plot_atoms
import math


from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def check_vertices(vects, vect_combs, shift_cell, cutoff, centre_0):
	vects = np.asarray(vects)
	for comb in vect_combs:
		vertice = shift_cell + comb[0]*vects[0] + comb[1]*vects[1] + comb[2]*vects[2]
		# print(vertice, np.linalg.norm(vertice-centre_0),cutoff)
		if np.linalg.norm(vertice-centre_0)>cutoff:
			return 0
	return 1


def get_shifts(vects, cutoff):
	"""In this procedure we suppose that the centre of
	mass of the unit cell (0,0,0) coincides with the 
	centre of a sphere with a radius whose length is 
	equal to cutoff.

	We produce triplets of integers and retrieve the
	centre of mass for each unit cell image characterised 
	by the said numbers. If the centre of mass lies in the
	sphere, then the shift is kept in list shifts. If it
	lies outside the sphere, then we check if there is any 
	vertice of this image inside the sphere. If there is,
	then the image is kept as a border shift, otherwise we
	keep the last accepted shift as a border 

	Each loop increases a number in the triplet until a 
	centre of mass is found to have the corresponding 
	coordinate with greater absolute value than the cutoff.
	"""

	# Combinations of 0,1 in triplets of (k,l,m)
	# for k*vect[0] + l*vect[1] + m*vect[2]
	vect_combs = [x for x in np.ndindex(2,2,2)]

	# Calculate and keep centre of mass of image (0,0,0)
	centre = np.dot(np.asarray([0.5, 0.5, 0.5]).T, vects)
	centre_0 = centre

	i,j,k = 0,0,0
	shifts = [] # images of whole unit cells inside sphere
	border = [] # images of unit cells on the border of sphere
	accept_flag = False

	# Variables to use for switching opposite directions
	j_up, i_right, k_front = True, True, True

	# Check sphere capacity of unit cells per xy-plane
	k = 0
	k_front = True
	while(True):

		# Check capacity of sphere on y-axis
		j = 0
		j_up = True
		while(True):

			# Check capacity of sphere on x-axis
			i = 0
			i_right = True
			while(True):

				# Get shift and new image centre of mass
				shift = np.asarray([i,j,k])
				shift_cell = shift @ vects
				centre = centre_0 + shift_cell

				# Check if the new centre of mass is outside sphere
				if (np.linalg.norm(shift_cell)>cutoff):
					# Check if there are vertices of this image inside sphere
					if check_vertices(vects, vect_combs, shift_cell, cutoff, centre_0):
						border.append(shift_cell)
					# Check if list is empty
					elif (shifts!=[]) & (accept_flag):
						# If all vertices outside of sphere, keep
						# last accepted image as border
						border.append(shifts.pop())
					accept_flag = False
				elif not np.all((shift == 0)):
					# Keep whole new image
					shifts.append(shift_cell)
					accept_flag = True

				dist = centre[0]-centre_0[0]
				# If centre of mass has x-coord <= cutoff
				# keep checking (0-->positive)

				# If centre of mass has x-coord > cutoff and
				# was going to positive x then check 
				# opposite direction (0-->negative)
				
				# If centre of mass has x-coord < -cutoff
				# it means we checked both directions and
				# need to go up y-axis
				if dist > cutoff:
					i_right = False
					i = -1
					continue
				elif dist < -cutoff:
					break
				else:
					if i_right:
						i += 1
					else:
						i -= 1

			dist = centre[1]-centre_0[1]
			# If centre of mass has y-coord <= cutoff
			# keep checking (0-->positive)

			# If centre of mass has y-coord > cutoff and
			# was going to positive y then check 
			# opposite direction (0-->negative)
			
			# If centre of mass has y-coord < -cutoff
			# it means we checked 4 directions and need
			# to go to new depth
			if dist > cutoff:
				j_up = False
				j = -1
				continue
			elif dist < -cutoff:
				break
			else:
				if j_up:
					j += 1
				else:
					j -= 1

		dist = centre[2]-centre_0[2]
		# If centre of mass has z-coord <= cutoff
		# keep checking (0-->positive)

		# If centre of mass has z-coord > cutoff and
		# was going to positive x then check 
		# opposite direction (0-->negative)
		
		# If centre of mass has z-coord < -cutoff
		# it means we checked all directions
		if dist > cutoff:
			k_front = False
			k = -1
			continue
		elif dist < -cutoff:
			break
		else:
			if k_front:
				k += 1
			else:
				k -= 1

	return np.array(shifts+border)


def check_border(ri, rj, shift, cutoff):
	if np.linalg.norm(ri+shift-rj) > cutoff:
		return 0
	return 1


def get_shifts_old(vects, cut_off):
	"""Returns an array of all possible lattice positions:   
	 (2cut_off+1)^3 - {case of (cut_off,cut_off,cut_off)}
	 combinations in R^3  
	
	"""
	dim = (2*cut_off+1)**3-1
	shifts = np.zeros((dim, 3))

	i = 0
	for shift in np.ndindex(2*cut_off+1, 2*cut_off+1, 2*cut_off+1):
		if shift != (cut_off,cut_off,cut_off):
			shifts[i] = shift - np.array([cut_off,cut_off,cut_off])
			i += 1
	shifts = np.dot(shifts,vects)
	return shifts

if __name__=="__main__":
	atoms = Atoms("SrTiO3",

					  cell=[[4.00, 0.00, 0.00],
							[0.00, 4.00, 0.00],
							[0.00, 0.00, 4.00]],

					  positions=[[0, 0, 0],
								 [2, 2, 2],
								 [0, 2, 2],
								 [2, 0, 2],
								 # [1.5, .5, 2], # put Os too close
								 [0, 2, 0]],
					  pbc=True)

	cutoff = 8
	vects = atoms.cell
	shifts_all = get_shifts(vects, cutoff)
	shifts_all_old = get_shifts_old(vects, cutoff)

	print(np.sort(shifts_all))
	print(shifts_all_old)
