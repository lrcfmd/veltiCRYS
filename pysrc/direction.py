import numpy as np
import math

def GD(grad, **kwargs):
	"""Gradient Descent updating scheme. Returns
	the direction vector.
	
	Parameters
	----------
	grad : (N+3)x3 array (double)
		Array containing the gradient w.r.t. ion positions
		and strains.
	
	Returns
	-------
	dict[str, array[N(double),M(double)]]

	"""
	return {'Direction': -grad}


def CG(grad, **kwargs):
	"""Conjugate Gradient updating scheme. We are
	using Polak-Ribière updating for beta factor.
	All matrices are reshaped and addressed as vectors.


	Parameters
	----------
	grad : (N+3)x3 array (double)
		Array containing the gradient w.r.t. ion positions
		and strains.
	kwargs['Residual'] : (N+3)x3 array (double)
		The gradient of the previous iteration with opposite sign.
	kwargs['Direction'] : (N+3)x3 array (double)
		The direction vector of the previous iteration.
	
	Returns
	-------
	dict[str, array[N(double),M(double)]]
	
	"""
	residual = -np.reshape(grad,
		(grad.shape[0]*grad.shape[1],1))
	last_residual = np.reshape(kwargs['Residual'], 
		(kwargs['Residual'].shape[0]*kwargs['Residual'].shape[1],1))
	last_direction = np.reshape(kwargs['Direction'], 
		(kwargs['Direction'].shape[0]*kwargs['Direction'].shape[1],1))
	beta = residual.T @ (residual-last_residual) / (last_residual.T @ last_residual)

	ndirection = residual + beta*last_direction
	return {'Direction': np.reshape(ndirection, kwargs['Direction'].shape)}

