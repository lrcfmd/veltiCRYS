import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from .cubic_regular.cubicmin import cubic_regularization, cubic_minimization
import torch, math

class CubicFit(BaseEstimator, RegressorMixin):
    def __init__(self, L, B, kappa, lr, momentum, dampening, rng):
        self.L = L
        self.kappa = kappa
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.rng = rng
        self.B = B
    def fit(self, params, target):
        self.X_ = params
        self.y_ = target
        self.classes_ = [1]
        # Return the classifier
        return self
    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Gather predictions
        y_pred = []
        
        for group in X:
            # Input validation
            group = check_array(group)
            grad_vec, gnorm, hnorm = group[0], group[1][0], group[2][0]
            hessian = group[3:, :]
            res= None

            initial_vector = torch.zeros(hessian.shape[0])
            optimizer = torch.optim.SGD([initial_vector], lr=self.lr)
            optargs = {'params': [initial_vector], 
                        'lr': self.lr, 
                        'weight_decay': 0,
                        'momentum': self.momentum,
                        'nesterov': True, 
                        'maximize': False,
                        'foreach': None,
                        'dampening': self.dampening,
                        'differentiable': False}

            res, _ = cubic_minimization(grad=grad_vec, gnorm=gnorm, 
                hessian=hessian, hnorm=hnorm, L=self.L, B=self.B, kappa=self.kappa, 
                optimizer=optimizer, tol=0.001, max_iterno=1000, rng=self.rng,
                check=True, **optargs)

            # Calculate cubic regularization function for returned vectors
            reg_value = cubic_regularization(grad_vec, hessian, res[1], self.L)	
            # Check if there is a lowest eigenvector approximation
            if res[2] is not None:
                reg_value_min = res[0]/(2*self.L)*cubic_regularization(
                    grad_vec, hessian, res[2], self.L)
                # Keep the vector that gives smaller regular/tion value
                if reg_value_min<reg_value:
                    reg_value = reg_value_min
            y_pred.append(reg_value)
        return y_pred

def max_raw_log_error(y_true, y_pred):
    return np.max(np.log10(np.abs(y_pred)))

