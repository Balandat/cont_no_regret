'''
A collection of LossFunction classes for the Continuous No Regret Problem.  

@author: Maximilian Balandat
@date: Apr 30, 2015
'''

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.linalg import eigh
from ContNoRegret.Distributions import Gaussian
from ContNoRegret.Domains import nBox, UnionOfDisjointnBoxes


class LossFunction(object):
    """ Base class for LossFunctions """        
        
    def val(self, points):
        """ Returns values of the loss function at the specified points """
        raise NotImplementedError
    
    def set_bounds(self, bounds):
        """ Allows to add lower and upper bounds to the function. Helpful for
            finding a lower bound on nustar in the dual averaging algorithm.
            bounds here is a tuple/list with bounds[0] and bounds[1] being the 
            lower and upper bound on the function over the domain. """
        self.bounds = bounds
        
    def val_grid(self, x, y):
        """ Computes the value of the loss on rectangular grid (for now assume 2dim) """
        points = np.array([[a,b] for a in x for b in y])
        return self.val(points).reshape((x.shape[0], y.shape[0]))
    
    def max(self, points):
        """ Returns maximum value of the loss function """
        raise NotImplementedError
        
    def plot(self, points):
        """ Creates a 3D plot of the density via triangulation of
            the density value at the points """
        pltpoints = points[self.domain.iselement(points)]
        vals = self.val(pltpoints)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(pltpoints[:,0], pltpoints[:,1], vals, cmap=cm.jet, linewidth=0.2)
        ax.set_xlabel('$x$'), ax.set_ylabel('$y$'), ax.set_zlabel('loss$(x,y)$')
        ax.set_title('Loss')
        plt.show()
        return fig   
    
    def grad(self, points): 
        """ Returns gradient of the loss function at the specified points """
        raise NotImplementedError
    
    def Hessian(self, points): 
        """ Returns Hessian of the loss function at the specified points """
        raise NotImplementedError
    
    def proj_gradient(self, x, step):
        """ Returns next iterate of a projected gradient step """
        grad_step = x - step*self.grad(x)
        return self.domain.project(grad_step)
    


class GaussianLossFunction(LossFunction):
    """ Loss given by l(s) = M(1 - gamma*exp[-0.5*(s-x)^T Sigma^-1 (s-x)]) """ 
    
    def __init__(self, domain, mean, cov, M):
        self.domain, self.M = domain, M
        self.p = Gaussian(domain, mean, cov)
        self.L = self.computeL()
        self.Cnorm = 1/self.p.density(mean)
    
    def val(self, points):
        return self.M*(1 - self.Cnorm*self.p.density(points))
    
    def min(self):
        """ For now just returns 1-peak, need to do something smarter... """
        return self.M*(1 - self.Cnorm*self.p.density(self.p.mean))
    
    def grad(self, points): 
        return - self.M*self.Cnorm*self.p.grad_density(points)
    
    def Hessian(self, points): 
        return - self.M*self.Cnorm*self.p.Hessian_density(points)
    
    def computeL(self):
        """ Computes a Lipschitz constant for the loss function. """
        lambdamin = eigh(self.p.cov, eigvals=(0,0), eigvals_only=True)
        return self.M*((2*np.pi)**self.domain.n*np.e*np.linalg.det(self.p.cov)*lambdamin)**(-0.5)
    

class AffineLossFunction(LossFunction):
    """ An affine loss function in n dimensions """ 
    
    def __init__(self, domain, a, b):
        """ AffineLossFunction of the form l(s) = <a,s> + b, 
            where a is a vector in R^n and b is a scalar. """
        self.domain = domain
        self.a, self.b = np.array(a), b
        self.desc = 'Affine'

    def val(self, points):
        return np.dot(points, self.a) + self.b
    
    def minmax(self):
        """ Compute the minimum and maximum of the loss function over the domain.
            This assumes that the domain is an nBox. """
        if isinstance(self.domain, nBox):
            vertvals = self.val(self.domain.vertices())
            self.set_bounds([np.min(vertvals), np.max(vertvals)])
            return self.bounds
            return vertvals
        elif isinstance(self.domain, UnionOfDisjointnBoxes):
            vertvals = np.array([self.val(nbox.vertices()) for nbox in self.domain.nboxes])
            self.set_bounds([np.min(vertvals), np.max(vertvals)])
            return self.bounds
        else:
            raise Exception('Sorry, for now only nBoxes are supported for computing minimum and maximum of AffineLossFunctions')

    
    def grad(self, points): 
        return np.repeat(np.array(self.a, ndmin=2), points.shape[0], axis=0)
    
    def __add__(self, affine2):
        """ Add two AffineLossFunction objects (assumes that both functions
            are defined over the same domain. """
        if isinstance(affine2, AffineLossFunction):
            return AffineLossFunction(self.domain, self.a + affine2.a, self.b + affine2.b)
        else:
            raise Exception('So far can only add two affine loss functions!')

 
class PolynomialLossFunction(LossFunction):
    """ A polynomial loss function in n dimensions of arbitrary order, 
        represented in the basis of monomials """ 
    
    def __init__(self, domain, coeffs, exponents):
        """ Construct a PolynomialLossFunction that is the sum of M monomials.
            coeffs is an M-dimensional array containing the coefficients of the
            monomials, and exponents is a list of n-tuples of length M, with 
            the i-th tuple containing the exponents of the n variables in the monomial. 
 
            For example, the polynomial l(x) = 3*x_1^3 + 2*x_1*x_3 + x2^2 + 2.5*x_2*x_3 + x_3^3
            in dimension n=3 is constructed using
                coeffs = [3, 2, 1, 2.5, 1] and
                exponents = [(3,0,0), (1,0,1), (0,2,0), (0,1,1), (0,0,3)] 
        """ 
        self.domain = domain
        self.coeffs, self.exponents = coeffs, exponents
        self.m = len(coeffs)
        self.polydict = {exps:coeff for coeff,exps in zip(coeffs,exponents)}
        self.desc = 'Polynomial'

    def set_bounds(self, bounds):
        """ Allows to add lower and upper bounds to the function. Helpful for
            finding a lower bound on nustar in the dual averaging algorithm.
            bounds here is a tuple/list with bounds[0] and bounds[1] being the 
            lower and upper bound on the function over the domain. """
        self.bounds = bounds

    def val(self, points):
        monoms = np.array([points**exps for exps in self.polydict.keys()]).prod(2)
        return np.sum([monom*coeff for monom,coeff in zip(monoms, self.polydict.values())], axis=0)
    
    def __add__(self, poly2):
        """ Add two PolynomialLossFunction objects (assumes that both polynomials 
            are defined over the same domain. """
        newdict = self.polydict.copy()
        for exps, coeff in poly2.polydict.items():
            try:
                newdict[exps] = newdict[exps] + coeff
            except KeyError:
                newdict[exps] = coeff
        return PolynomialLossFunction(self.domain, newdict.values(), newdict.keys())
        
    
class QuadraticLossFunction(LossFunction):
    """ Loss given by l(s) = 0.5 (s-mu)'Q(s-mu) + c, with Q>0 and c>= 0. 
        This assumes that mu is inside the domain! """ 
    
    def __init__(self, domain, mu, Q, c):
        self.domain, self.mu, self.Q, self.c = domain, mu, Q, c
        self.desc = 'Quadratic'
        # implement computation of Lipschitz constant. Since gradient is 
        # linear, we can just look at the norm on the vertices
#         self.L = self.computeL()
    
    def val(self, points):
        x = points - self.mu
        return 0.5*np.sum(np.dot(x,self.Q)*x, axis=1)  + self.c
    
    def min(self):
        return self.c
    
    def grad(self, points): 
        return np.transpose(np.dot(self.Q, np.transpose(points)))
     
    def Hessian(self, points): 
        return np.array([self.Q,]*points.shape[0])
    
    def __add__(self, quadloss):
        Qtilde = self.Q + quadloss.Q
        btilde = - np.dot(self.mu, self.Q) - np.dot(quadloss.mu, quadloss.Q)
        mutilde = -np.linalg.solve(Qtilde, btilde)
        ctilde = 0.5*(2*self.c + np.dot(self.mu, np.dot(self.Q, self.mu))
                      + 2*quadloss.c + np.dot(quadloss.mu, np.dot(quadloss.Q, quadloss.mu))
                      + np.dot(btilde, mutilde))
        return QuadraticLossFunction(self.domain, mutilde, Qtilde, ctilde)


class CumulativeLoss(LossFunction):
    """ Class for cumulative loss function objects """
    
    def __init__(self, lossfuncs):
        """ Constructor. Here lossfuncs is a list of LossFunction objects """
        # check that domains are the same, then call superclass constructor        
        super(LossFunction, self).__init__(lossfuncs[0].domain)
        self.lossfuncs = lossfuncs
        
    def sample(self, N):
        """ Draw N independent samples from the distribution """
        indices = np.random.choice(len(self.weights), N, p=self.weights/np.sum(self.weights))
        return np.concatenate([self.distributions[index].sample(1) for index in indices])
    
    def add(self, lossfunc):
        """ Adds the loss function to the cumulative loss """
        self.lossfuncs.append(lossfunc)
        
    def val(self, points):
        """ Returns the cumulative loss for each of the points in points """
        return np.array(np.sum(np.array([lossfunc.val(points) for lossfunc in self.lossfuncs]), axis=0), ndmin=1)
    
    def grad(self, points): 
        """ Returns gradient of the cumulative loss function at the specified points """
        # the following looks weird but it allows to vectorize everything.
        return np.sum(np.array([lossfunc.grad(points) for lossfunc in self.lossfuncs]), axis=0)
    
    def Hessian(self, points): 
        """ Returns Hessian of the cumulative loss function at the specified points """       
        return np.sum(np.array([lossfunc.Hessian(points) for lossfunc in self.lossfuncs]), axis=0)
  
    def upperbound_val(self):
        """ Computes the a very crude upper bound of the oveall maximum based on the 
            maxima of the individual loss functions """
        return np.sum(np.array([lossfunc.max() for lossfunc in self.lossfuncs]))
    