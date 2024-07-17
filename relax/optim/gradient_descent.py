import numpy as np
from relax.optim.optimizer import Optimizer

class GD(Optimizer):

	def step(self, grad, gnorm, params, line_search_fn, **kwargs):

		# Calculate direction
		self.direction = -grad
   
		# Calculate stepsize
		lnsearch = getattr(self.lnscheduler, line_search_fn)
		stepsize = lnsearch(gnorm=gnorm, iteration=self.iterno)	

        # Perform a step
		params = params + stepsize*self.direction

		# Increase iteration number
		self.iterno += 1

		return params

