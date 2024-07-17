"""                    COULOMB                    """																			
import torch
from torch import Tensor
import numpy as np
import math
import numpy.typing as npt
from typing import Dict, Tuple
from torch.cuda import device

from .ewaldpotential import *
from .cutoff import *

class Coulomb(EwaldPotential):

	def __init__(self, chemical_symbols: npt.ArrayLike, charge_dict: Dict, 
			  get_shifts: callable, device: device):
		super().__init__(device=device)
  
		self.alpha = None
		self.real_cut_off = 0
		self.recip_cut_off = 0

		self.chemical_symbols = chemical_symbols.copy()
		self.charges = np.zeros((len(chemical_symbols),))
		self.get_shifts = get_shifts

		count = 0
		for elem in chemical_symbols: 
			# Save charge per ion position according to given charges
			self.charges[count] = charge_dict[elem]
			count += 1

		self.real = None
		self.recip = None
		self.self_energy = None

	
	def set_cutoff_parameters(self, vects: Tensor=None, N: int=0, 
		accuracy: float=1e-21, real: float=0, reciprocal: float=0,
		alpha: float= 0): 

		if vects is not None:
			volume = torch.det(vects)
			self.alpha = self.get_alpha(N, volume)
			self.real_cut_off = torch.div(math.sqrt(-np.log(accuracy)), self.alpha)
			self.recip_cut_off = torch.mul(self.alpha, 2*math.sqrt(-np.log(accuracy)))
		else:
			if (real == 0) and (reciprocal == 0) or (alpha == 0):
				raise ValueError('Cutoffs and alpha need to be defined.')
			self.real_cut_off = torch.tensor(real, device=self.device)
			self.recip_cut_off = torch.tensor(reciprocal, device=self.device)
			self.alpha = torch.tensor(alpha, device=self.device)


	def ewald_real_energy(self, pos: Tensor, vects: Tensor) -> Tensor:
		shifts = self.get_shifts(vects, self.real_cut_off.item())
		if shifts == None:
			shifts_no = 0
		else:
			shifts_no = len(shifts)		
		N = len(pos)

		esum = torch.tensor(0., device=self.device)
		charges = torch.ones((N,N), requires_grad=False, device=self.device)
		for ioni in range(N):
			for ionj in range(N):
				charges[ioni][ionj] = self.charges[ioni]*self.charges[ionj]
		
		# Keep off-diagonal elements
		central_dists, mask = self.get_pairwise_dists(pos)
		term = torch.erfc(torch.mul(self.alpha, central_dists))
		offcharges = charges[mask]
		esum = torch.mul(offcharges , torch.div(term, central_dists)).sum()

		# Cell neighboring images
		for shift in range(shifts_no):
			pos_shift = torch.add(pos.clone(), torch.tile(shifts[shift], (N,1)))
			img_dists, mask = self.get_pairwise_dists(pos, pos_shift)
			term = torch.erfc(torch.mul(self.alpha, img_dists))
			divisor = torch.div(term, img_dists)
			offcharges = charges[mask]
			esum = torch.add(esum, torch.mul(offcharges, divisor).sum())
		
		esum = torch.mul(esum, 14.399645351950543/2)  # electrostatic constant
		self.real = esum

		return esum

	
	def ewald_recip_energy(self, pos: Tensor, vects: Tensor, volume: Tensor) -> Tensor:
		if not volume:
			volume = torch.det(vects)

		rvects = self.get_reciprocal_vects(vects)
		shifts = self.get_shifts(rvects, self.recip_cut_off.item())
		if shifts == None:
			shifts_no = 0
		else:
			shifts_no = len(shifts)
		N = len(pos)

		charges = torch.ones((N,N), device=self.device)
		for ioni in range(N):
			for ionj in range(N):
				charges[ioni][ionj] = self.charges[ioni]*self.charges[ionj]

		rij = pos[:, None] - pos[None, :]
		rij_all = rij.reshape(-1, 3)
		
		esum = torch.tensor(0., device=self.device)
		for shift in range(shifts_no):
			k_2 = torch.dot(shifts[shift], shifts[shift])	
			km = torch.tile(shifts[shift], (rij_all.shape[0],1))
			krij_all = torch.einsum('ij,ij->i', km, rij_all)
			power = -torch.div(k_2, torch.mul(4, torch.pow(self.alpha, 2)))
			cos_term = torch.mul(torch.mul(2*pi, torch.exp(power)), torch.cos(krij_all))
			# actual calculation
			frac = torch.mul(charges.reshape(1, -1), torch.div(cos_term, torch.mul(k_2, volume)))
			esum = torch.add(esum,  frac.sum())

		esum = torch.mul(esum, 14.399645351950543)  # electrostatic constant
		self.recip = esum

		return esum


	def ewald_self_energy(self, pos: Tensor) -> Tensor:
		N = len(pos)
		charges = torch.tensor(self.charges.copy(), device=self.device)
		alphapi = torch.div(self.alpha, math.sqrt(pi))
		esum = sum(-torch.mul(torch.square(charges), alphapi))

		esum = torch.mul(esum, 14.399645351950543)  # electrostatic constant
		self.self_energy = esum
		return esum


	def all_energy(self, pos: Tensor, vects: Tensor, volume: Tensor) -> Tensor:
		real_energy = self.ewald_real_energy(pos, vects)	
		recip_energy = self.ewald_recip_energy(pos, vects, volume)
		self_energy = self.ewald_self_energy(pos)
	
		energy = torch.add(real_energy, recip_energy)
		self.energy = torch.add(energy, self_energy)
		return self.energy



