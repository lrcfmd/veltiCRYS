import numpy as np
from relax.optim.optimizer import Optimizer

class CG(Optimizer):

	def step(self, grad, gnorm, params, line_search_fn, **kwargs):

		# Calculate direction
		if self.iterno == 0:
			new_direction = -grad
		else:
			residual = -np.reshape(grad,
				(grad.shape[0]*grad.shape[1],1))
			last_residual = np.reshape(self.residual, 
				(self.residual.shape[0]*self.residual.shape[1],1))
			last_direction = np.reshape(self.direction, 
				(self.direction.shape[0]*self.direction.shape[1],1))
			beta = residual.T @ (residual-last_residual) / (last_residual.T @ last_residual)
			new_direction = np.reshape(residual + beta*last_direction, params.shape)
   		
		# Calculate stepsize
		lnsearch = getattr(self.lnscheduler, line_search_fn)
		stepsize = lnsearch(gnorm=gnorm, iteration=self.iterno)	
  
		# Perform a step
		params = params + stepsize*new_direction

		# Keep current iteration gradient info
		self.iterno += 1
		self.residual = -grad
		self.direction = new_direction

		return params

