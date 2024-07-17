import numpy as np
import torch, math
from torch import Tensor
from math import pi
from typing import Dict
from torch.cuda import device

class EwaldPotential:
	"""Generic class for defining Ewald sum potentials."""
	def __init__(self, device: device=None) -> None:
		if device is not None:
			self.device = device
		else:
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.energy = torch.tensor(0.)
		self.grad = {}

	def get_reciprocal_vects(self, vects: Tensor) -> Tensor:
		volume = torch.det(vects)
		rvect1 = 2*pi*torch.div(torch.cross(vects[1,:],vects[2,:]), volume)
		rvect2 = 2*pi*torch.div(torch.cross(vects[2,:],vects[0,:]), volume)
		rvect3 = 2*pi*torch.div(torch.cross(vects[0,:],vects[1,:]), volume)
		rvects = torch.cat((rvect1, rvect2, rvect3))
		return torch.reshape(rvects, shape=(3,3))

	def get_alpha(self, N: float, volume: Tensor) -> Tensor:
		accuracy = N**(1/6) * math.sqrt(pi)
		alpha = torch.div(accuracy, torch.pow(volume, 1/3))
		if (volume<0):
			raise ValueError('Volume has a negative value.')
		return alpha

	def get_gradient(self, energy: Tensor, scaled_pos: Tensor, N: int,
              vects: Tensor, strains: Tensor, volume: Tensor) -> Dict[Tensor, Tensor]:
		if not volume:
			volume = torch.det(vects)

		scaled_grad = torch.autograd.grad(
			energy, 
			(scaled_pos, strains), 
			torch.ones_like(energy), 
			create_graph=True, 
			retain_graph=True,
			materialize_grads=True)
  
		self.grad = {
      		'positions': torch.matmul(scaled_grad[0], torch.inverse(vects)),
            'strains': torch.div(scaled_grad[1], volume.item())
        }
		return self.grad

	def get_hessian(self, grad: Dict[Tensor, Tensor], scaled_pos: Tensor, 
             vects: Tensor, strains: Tensor, volume: Tensor) -> Tensor:
		if not volume:
			volume = torch.det(vects)

		N = len(scaled_pos)
		hessian = torch.tensor(np.zeros((3*N+6, 3*N+6)), device=self.device)
		pos_grad = grad['positions']
		for ioni, beta in np.ndindex((len(scaled_pos), 3)):
			partial_pos_i_hessian_scaled  = torch.autograd.grad(
				pos_grad[ioni][beta], 
				(scaled_pos, strains),
				grad_outputs=(torch.ones_like(pos_grad[ioni][beta])),
				retain_graph=True,
				materialize_grads=True)
			partial_pos_i_hessian = (
				torch.matmul(partial_pos_i_hessian_scaled[0], torch.inverse(vects)),
				torch.div(partial_pos_i_hessian_scaled[1], volume.item())
			)
			for ionj, gamma in np.ndindex(N, 3):
				hessian[3*ioni+beta][3*ionj+gamma] = partial_pos_i_hessian[0][ionj][gamma]
			for straini in range(6):
				hessian[3*ioni+beta][3*N+straini] = partial_pos_i_hessian[1][straini]

		strains_grad = grad['strains']
		for straini in range(6):
			partial_strain_i_hessian_scaled  = torch.autograd.grad(
				strains_grad[straini], 
				(scaled_pos, strains),
				grad_outputs=(torch.ones_like(strains_grad[straini], device=self.device)),
				retain_graph=True,
				materialize_grads=True)
			partial_strain_i_hessian = (
				torch.matmul(partial_strain_i_hessian_scaled[0], torch.inverse(vects)),
				partial_strain_i_hessian_scaled[1]
			)
			for ionj, gamma in np.ndindex(N, 3):
				hessian[3*N+straini][3*ionj+gamma] = \
					partial_strain_i_hessian[0][ionj][gamma]
			for strainb in range(6):
				hessian[3*N+straini][3*N+strainb] = \
					partial_strain_i_hessian[1][strainb]
		
		self.hessian = hessian
		return self.hessian

	def get_pairwise_dists(self, pos: Tensor, other: Tensor=None, mask: Tensor=None) -> Tensor:
		if other is None:
			other = pos.clone()

		rij = pos[:, None] - other[None, :]
		dists = torch.sum(torch.pow(rij, 2), 2)
		mask_ = dists!=0  # dispose self-self interactions in central cell
		if mask is not None:
			mask_ = mask.logical_and(mask_)
		return torch.sqrt(dists[mask_]), mask_
		
	def get_gnorm(grad: Dict[Tensor, Tensor]):
		size, gnorm = 0, 0
		for param in grad:
			gnorm += torch.sum(grad[param]**2)
			size += torch.numel(grad[param])
		return math.sqrt(gnorm)/size

	def get_Hnorm(hessian: Dict[Tensor, Tensor]):
		hnorm = torch.sum(hessian**2)
		size = torch.numel(hessian)
		return math.sqrt(hnorm)/size