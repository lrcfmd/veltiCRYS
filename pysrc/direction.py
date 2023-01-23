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
	kwargs['Residual'] : (N+3)x3 array (double)
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

	
def RMSprop(grad, **kwargs):
	"""RMSprop updating scheme.

	Parameters
	----------
	grad : (N+3)x3 array (double)
		Array containing the gradient w.r.t. ion positions
		and strains.
	kwargs['Square_avg'] : (N+3)x3 array (double)
		The lastly calculated moving average.
	
	Returns
	-------
	dict[str, array[N(double),M(double)]]

	"""
	alpha = 0.1
	eps = 0.1
	centered = False
	momentum = 0

	N = len(kwargs['Positions'])
	grads = [grad[:N], grad[N:]]
	print(len(grads))

	square_avg = [np.zeros(grad.shape) for grad in grads]
	grad_avg = [np.zeros(grad.shape) for grad in grads]

	if 'Momentum' in kwargs:
		momentum_buffer = kwargs['Momentum']
	if 'Alpha' in kwargs:
		alpha = kwargs['alpha']
	if 'Eps' in kwargs:
		eps = kwargs['Eps']
	if 'Centered' in kwargs:
		centered = kwargs['Centered']
	if 'Square_avg' in kwargs:
		square_avg = kwargs['Square_avg']
	if 'Grad_avg' in kwargs:
		grad_avg = kwargs['Grad_avg']

	directions = []
	for g in range(len(grads)):

		square_avg[g] = np.add(alpha*square_avg[g], (1 - alpha)*grads[g]*grads[g])

		if centered:
			grad_avg[g] = np.add( alpha*grad_avg[g] , (1-alpha)*grads[g] )
			avg = np.add(
				np.sqrt(np.add(square_avg[g],-np.multiply(grad_avg[g], grad_avg[g]))), 
				eps
			)
		else:
			avg = np.add(np.sqrt(square_avg[g]), eps)

		if momentum > 0:
			buf = momentum_buffer[g]
			buf = np.multiply(buf,momentum) + np.divide(grads[g],avg)
			directions.append(buf)
		else:
			directions.append(np.divide(grads[g], avg))

	direction = np.concatenate((directions[0], directions[1]), axis=0)

	return {'Square_avg': square_avg, 'Grad_avg': grad_avg, 'Direction': -direction}


def Adam(grad, **kwargs):
	"""Adam updating scheme.

	Parameters
	----------
	grad : (N+3)x3 array (double)
		Array containing the gradient w.r.t. ion positions
		and strains.
	kwargs['Exp_avg'] : (N+3)x3 array (double)
		The lastly calculated moving average of the gradient.
	kwargs['Exp_avg_sq'] : (N+3)x3 array (double)
		The lastly calculated moving average of the 
		squared gradient.
	
	Returns
	-------
	dict[str, array[N(double),M(double)]]

	"""
	exp_avg = np.zeros(grad.shape)
	exp_avg_sq = np.zeros(grad.shape)
	beta1 = 0.9
	beta2 = 0.999
	eps = 1e-8
	iterno = 1

	if 'Beta1' in kwargs:
		beta1 = kwargs['Beta1']
	if 'Beta2' in kwargs:
		beta2 = kwargs['Beta2']
	if 'Iter' in kwargs:
		iterno = kwargs['Iter']+1
	if 'Exp_avg' in kwargs:
		exp_avg = kwargs['Exp_avg']
	if 'Exp_avg_sq' in kwargs:
		exp_avg_sq = kwargs['Exp_avg_sq']
	if 'Eps' in kwargs:
		eps = kwargs['Eps']

	assert(iterno >= 1)
	
	# Decay the first and second moment running average coefficient
	exp_avg = np.add(beta1*exp_avg, (1 - beta1)*grad)    
	exp_avg_sq = np.add(beta2*exp_avg_sq, (1 - beta2)*np.multiply(grad, np.conj(grad)))

	bias_correction1 = 1 - beta1 ** iterno
	bias_correction2 = 1 - beta2 ** iterno

	bias_correction2_sqrt = math.sqrt(bias_correction2)

	denom = np.add((np.sqrt(exp_avg_sq) / bias_correction2_sqrt), eps)
	denom = denom / bias_correction1

	direction = -np.divide(exp_avg, denom)
	return {'Exp_avg': exp_avg, 'Exp_avg_sq': exp_avg_sq, 'Direction': direction}


if __name__=="__main__":

	grad = np.array([[2.], [4.]])
	square_avg = np.array([[0., 0.]])
	grad_avg = np.array([[0., 0.]])

	direction = GD(grad)
	print("Gradient Descent: ",direction['Direction'])

	grad1 = np.array([[2.04], [4.08]])

	direction = CG(grad1,
		direction=direction['Direction'],
		residual=-grad)
	print("Conjugate Gradient: ",direction['Direction'])	

	direction = RMSprop(grad, 
		grad_avg=grad_avg)



