import math, torch
import numpy as np
from ase.geometry import wrap_positions
from relax.optim.optimizer import Optimizer


class LBFGS(Optimizer):
    
	def __init__(self, lnsearch, max_iter=20, max_eval=100,
			  ftol=1e-5, gtol=1e-3, tol=1e-5, history_size=50):
		super().__init__(lnsearch, ftol, gtol, tol)
	
		self.reg_value = None
		self.requires_hessian = False
		self.optargs = { 
			'lr': 1e-3, 
			'max_iter': max_iter, 
			'max_eval': max_eval, 
			'tolerance_grad': gtol, 
			'tolerance_change': tol, 
			'history_size': history_size, 
			'line_search_fn': None
		}

	
	def step(self, grad, gnorm, params, line_search_fn, **kwargs):

		# Change to tensor
		tparams = torch.from_numpy(params)
		
		# Calculate stepsize
		lnsearch = getattr(self.lnscheduler, line_search_fn)
		stepsize = lnsearch(gnorm=gnorm, iteration=self.iterno)
		self.optargs['lr'] = stepsize

		# Initialize or set parameters
		self.optargs['params'] = [tparams]
		if self.iterno == 0:
			self.external_optimizer = torch.optim.LBFGS(
				**self.optargs
				) 
		else:
			self.external_optimizer.param_groups = [self.optargs]

		# Define closure
		def closure():
			# Set derivative
			self.external_optimizer._params[0].grad = torch.from_numpy(grad)

			# Set closure variables
			vects = np.asarray(kwargs['atoms'].get_cell())
			potentials = kwargs['potentials']

			# Make sure ions stay in unit cell
			pos_temp = wrap_positions(params[:-3], vects)
			# Update strains
			strains = (params[-3:]-1)+np.identity(3)

			# Calculate energy on current PES point
			energy =0
			for name in potentials:
				if hasattr(potentials[name], 'energy'):
					energy += potentials[name].energy(
						pos_array=pos_temp @ strains.T,
						vects_array=vects @ strains.T)
			loss = torch.tensor(energy)
			return loss

		self.external_optimizer.step(closure=closure)
		self.external_optimizer.zero_grad()
				
        # Add name of used method to list
		self.iterno += 1

		return self.external_optimizer._params[0].detach().numpy()