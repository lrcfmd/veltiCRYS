import os,sys
import numpy as np
from ase.geometry import wrap_positions, get_distances

from relax.analytic_potentials.coulomb.coulomb import *
from relax.analytic_potentials.buckingham.buckingham import *
from ase.cell import Cell

def finite_diff_grad(atoms, analytical_grad, ions, 
	strains, displacement, potentials):
	"""Defining local slope using finite differences 
	(like the derivative limit). 

	A displacement -h and +h is added to the coordinate of 
	each ion per iter. The potential is evaluated using the 
	new positions f(x-h), f(x+h) and then the numerical 
	derivative ( f(x+h)-f(x-h) )/2h is calculated.

	"""

	grad = np.zeros((len(atoms.positions)+3,3))
	h = np.zeros(3)
	vects = np.array(atoms.get_cell())			
	N = len(atoms.positions)

	for name in potentials:
		potentials[name].set_cutoff_parameters(vects,N)

	for ioni in range(ions):
		for coord in range(3):
			h[:3] = 0
			h[coord] = displacement
			
			# f(x-h)
			# add perturbation to one ion coordinate
			atoms_cp = atoms.copy()
			init_pos = atoms_cp.positions[ioni][coord]
			positions_cp = atoms_cp.positions.copy()
			positions_cp[ioni] -= h
			new_pos = positions_cp[ioni][coord]

			# pos = wrap_positions(positions_cp,vects)
			menergy = 0 

			# calculate (f(x+h)-f(x))/h
			for name in potentials:
				menergy += potentials[name].energy(
					pos_array = positions_cp,
					vects_array = vects,
					N_ = N)

			# f(x+h)
			# add perturbation to one ion coordinate
			atoms_cp = atoms.copy()
			init_pos = atoms_cp.positions[ioni][coord]
			positions_cp = atoms_cp.positions.copy()
			positions_cp[ioni] += h
			new_pos = positions_cp[ioni][coord]

			# pos = wrap_positions(positions_cp,vects)
			penergy = 0 

			# calculate (f(x+h)-f(x))/h
			for name in potentials:
				penergy += potentials[name].energy(
					pos_array = positions_cp,
					vects_array = vects,
					N_ = N)

			# calculate (f(x+h)-f(x))/h
			grad[ioni][coord] = ( penergy-menergy )/(2*h[coord])

	for i in range(3):
		for j in range(i,3):

			strains_cp = strains.copy()
			strains_cp[i][j] -= displacement
			# strains_cp[j][i] = strains_cp[i][j]
			atoms_cp = atoms.copy()

			delta_strains = (strains_cp-1)+np.identity(3)
			vects = atoms_cp.get_cell() @ delta_strains.T
			pos = atoms_cp.positions @ delta_strains.T

			for name in potentials:
				potentials[name].set_cutoff_parameters(vects=vects,N=N)

			# calculate (f(x+h)-f(x))/h
			penergy = 0
			for name in potentials:
				penergy += potentials[name].energy(
					pos_array = pos,
					vects_array = vects,
					N_ = N)

			strains_cp = strains.copy()
			strains_cp[i][j] += displacement
			# strains_cp[j][i] = strains_cp[i][j]
			atoms_cp = atoms.copy()

			delta_strains = (strains_cp-1)+np.identity(3)
			vects = atoms_cp.get_cell() @ delta_strains.T
			pos = atoms_cp.positions @ delta_strains.T

			for name in potentials:
				potentials[name].set_cutoff_parameters(vects=vects,N=N)

			# calculate (f(x+h)-f(x))/h
			menergy = 0
			for name in potentials:
				menergy += potentials[name].energy(
					pos_array = pos,
					vects_array = vects,
					N_ = N)

			grad[N+i][j] = ( penergy-menergy )/(2*displacement*np.linalg.det(vects))
			grad[N+j][i] = grad[N+i][j]

			for name in potentials:
				potentials[name].set_cutoff_parameters(
					vects=np.array(atoms_cp.get_cell()),N=N)

	print("analytical-numerical:\n",np.absolute(analytical_grad[:N, :]-grad[:N, :]))
	print(np.absolute(analytical_grad[N:, :])-np.absolute(grad[N:, :]))
	return grad


def finite_diff_hess(atoms, analytical_hess, strains, 
	displacement, potentials):
	"""Defining local curvature using finite differences 
	(like the derivative limit). 

	A displacement -h and +h is added to each parameter.
	The partial derivatives are evaluated using the new 
	parameter and then we calculate the numerical second
	derivative ( g(x+h)-g(x-h) )/2h.

	"""

	hessian = np.zeros((len(atoms.positions)+3,len(atoms.positions)+3,3,3))
	h = np.zeros(3)
	vects = np.array(atoms.get_cell())
	volume = np.linalg.det(vects)		
	N = len(atoms.positions)

	for name in potentials:
		potentials[name].set_cutoff_parameters(vects,N)

	print('DEBUG')
	for ionj,b in np.ndindex((N,3)):
		pgrad = np.zeros((len(atoms.positions)+3,3))
		mgrad = np.zeros((len(atoms.positions)+3,3))
		pstress = np.zeros((3,3))
		mstress = np.zeros((3,3))

		# add perturbation to one parameter
		atoms_cp = atoms.copy()
		positions_cp = atoms_cp.positions
		positions_cp[ionj][b] -= displacement

		for name in potentials:
			mgrad += potentials[name].gradient(
				pos_array=positions_cp, 
				vects_array=vects, N_=N)

		# add perturbation to one parameter
		atoms_cp = atoms.copy()
		positions_cp = atoms_cp.positions
		positions_cp[ionj][b] += displacement

		for name in potentials:
			pgrad +=  potentials[name].gradient(
				pos_array=positions_cp, 
				vects_array=vects, N_=N)

		# POS-POS
		for ioni,a in np.ndindex((N,3)): ### DO NOT TOUCH
			hessian[ionj][ioni][b][a] += \
				(pgrad[ioni][a]-mgrad[ioni][a])/(2*displacement)
		# POS-STRAINS
		for l,m in np.ndindex((3,3)): ### DO NOT TOUCH
			hessian[ionj][N+l][b][m] += \
				(pgrad[N+l][m]-mgrad[N+l][m])/(2*displacement)

	for l,m in np.ndindex((3,3)):
		pstress = np.zeros((3,3))
		mstress = np.zeros((3,3))
		pgrad = np.zeros((len(atoms.positions)+3,3))
		mgrad = np.zeros((len(atoms.positions)+3,3))

		# add perturbation to one parameter
		strains_cp = strains.copy()
		strains_cp[l][m] -= displacement
		atoms_cp = atoms.copy()

		delta_strains = (strains_cp-1)+np.identity(3)
		vects = atoms_cp.get_cell() @ delta_strains.T
		positions_cp = atoms_cp.positions @ delta_strains.T

		for name in potentials:
			potentials[name].set_cutoff_parameters(vects=vects,N=N)

		for name in potentials:
			mgrad += potentials[name].gradient(
				pos_array=positions_cp, 
				vects_array=vects, N_=N)
			mstress = potentials[name].get_stresses().copy()

		# add perturbation to one parameter
		strains_cp = strains.copy()
		strains_cp[l][m] += displacement
		atoms_cp = atoms.copy()

		delta_strains = (strains_cp-1)+np.identity(3)
		vects = atoms_cp.get_cell() @ delta_strains.T
		positions_cp = atoms_cp.positions @ delta_strains.T

		for name in potentials:
			potentials[name].set_cutoff_parameters(vects=vects,N=N)

		for name in potentials:
			pgrad += potentials[name].gradient(
				pos_array=positions_cp, 
				vects_array=vects, N_=N)
			pstress = potentials[name].get_stresses().copy()

		# STRAINS
		for n,ksi in np.ndindex((3,3)):
			hessian[N+l][N+n][m][ksi] = \
				(pstress[n][ksi]-mstress[n][ksi])/(2*displacement*np.linalg.det(vects))
		# STRAINS-POS
		for ioni,a in np.ndindex((N,3)): #### DO NOT TOUCH
			hessian[N+l][ioni][m][a] += \
				(pgrad[ioni][a]-mgrad[ioni][a])/(2*displacement*np.linalg.det(vects))

		for name in potentials:
			potentials[name].set_cutoff_parameters(
				vects=np.array(atoms_cp.get_cell()),N=N)

	usr = input("Select and type from \"pos\", \"pos-strain\", \"strain\", \"all\": ")
	print("analytical-numerical: {} atoms\n".format(N))
	if (usr=="all") or (usr=="pos"):
		print("POSITIONAL ONLY")
		count = 0
		for row in np.absolute(analytical_hess[:N, :N, :, :]-hessian[:N, :N, :, :]):
			print(count, row)
			count += 1
	if (usr=="all") or ("pos-strain" in usr):
		print("POSITIONAL-STRAINS")
		count = 0
		for row in np.absolute(analytical_hess[:N, -3:, :, :]-hessian[:N, N:, :, :]):
			print(count, row)
			count += 1
		print("STRAINS-POSITIONAL")
		count = 0
		for row in np.absolute(analytical_hess[N:, :N, :, :]-hessian[N:, :N, :, :]):
			print(count, row)
			count += 1
	if (usr=="all") or (usr=="strain"):
		print("STRAINS ONLY")
		count = 0
		for row in np.absolute(analytical_hess[N:, N:, :, :]-hessian[N:, N:, :, :]):
			print(count, row)
			count += 1
	return hessian


def flatten(positions):
	vector = []
	for sublist in positions:
		for element in sublist:
			vector.append(element)
	return vector
