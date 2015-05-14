'''
Nonlinear optimization by use of Affine DualAveraging

@author: Maximilian Balandat
@date: May 13, 2015
'''

import numpy as np
from .Domains import nBox


class NLoptProblem():
    """ Basic class describing a Nonlinear Optimization problem. """
    
    def __init__(self, domain, objective):
        """ Constructor for the basic problem class. Here objective is a callable that
            provides val and grad methods for computing value and gradient, respectively. """
        if not isinstance(domain, nBox):
            raise Exception('For now only nBoxes are supported!')
        self.domain, self.objective = domain, objective
        
    def run_minimization(self, etas, N, **kwargs):
        """ Runs the minimization of the objective function based on interpreting
            the value/gradient at the current iterate as an affine loss function 
            and applying dual averaging with the Exponential Potential. """
        t, T, = 1, len(etas)
        A = np.zeros((N, self.domain.n))
        actions = [self.domain.sample_uniform(N)]
        bounds = np.array(self.domain.bounds)
        while t<T:
            A += self.objective.grad(actions[-1])
            actions.append(quicksample(bounds, A, etas[t]))
            t += 1
        return actions    
            
            
def quicksample(bounds, A, eta):
    """ Function returning actions sampled from the solution of the Dual Averaging 
        update on an Box with Affine losses, Exponential Potential. """
    C1, C2 = np.exp(-eta*A*bounds[:,0]), np.exp(-eta*A*bounds[:,1])
    Finv = lambda U: -np.log(C1 - (C1-C2)*U)/A/eta
    return Finv(np.random.rand(*A.shape))







