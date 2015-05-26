'''
A collection of Distribution classes for the Continuous No Regret Problem.  

@author: Maximilian Balandat, Walid Krichene
@date: Oct 24, 2014
'''

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal
from scipy.linalg import inv


class Distribution(object):
    """ Base class for probability distributions used in Continuous 
        No-Regret algorithms. """
    
    def __init__(self, domain):
        """ Constructor """
        self.domain = domain
    
    def density(self, points):
        """ Returns the density for each of the points in points """
        raise NotImplementedError
        
    def sample(self, N):
        """ Draw N independent samples from the distribution """
        raise NotImplementedError

    def max_density(self):
        """ Computes the maximum of the distribution's pdf """
        raise NotImplementedError
    
    def plot_density(self, points):
        """ Creates a 3D plot of the density via triangulation of
            the density value at the points """
        pltpoints = points[self.domain.iselement(points)]
        vals = self.density(pltpoints)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(pltpoints[:,0], pltpoints[:,1], vals, cmap=cm.jet, linewidth=0.2)
        ax.set_xlabel('$x$'), ax.set_ylabel('$y$'), ax.set_zlabel('pdf$(x,y)$')
        ax.set_title('Density')
        plt.show()
        return fig
    
    def plot_samples(self, samples):
        """ Plots the samples on the x-y plane """
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(samples[:,0], samples[:,1], '*')
        ax.set_xlabel('$x$'), ax.set_ylabel('$y$')
        ax.set_title('{number} samples from the distribution'.format(number=len(samples)))
        plt.show()
        return fig
    
    
class MixtureDistribution(Distribution):
    """ Mixture of probability distributions """
    
    def __init__(self, distributions, weights):
        """ Constructor. Here distributions is a list of Distributions, and weights 
            is a list or numpy array of non-negative weights of the same length """
        # check that domains are the same, then call superclass constructor        
        super(MixtureDistribution, self).__init__(distributions[0].domain)
        # check that weights are non-negative
        if not np.all(np.greater_equal(weights, 0)):
            raise Exception('All weights need to be non-negative!')
        # check that have as many weights as distributions
        if not len(distributions) == len(weights):
            raise Exception('Number of weights must match the number of distributions')
        self.distributions, self.weights = distributions, np.array(weights)
        
    def sample(self, N):
        """ Draw N independent samples from the distribution """
        indices = np.random.choice(len(self.weights), N, p=self.weights/np.sum(self.weights))
        return np.concatenate([self.distributions[index].sample(1) for index in indices])
    
    def append(self, distribution, weight):
        """ Appends the distribution with the given weight """
        # TO DO: check that the domain of distribution and domain of Mixture are the same!
        self.distributions.append(distribution)
        if not np.all(np.greater_equal(weight, 0)):
            raise Exception('Weight needs to be non-negative!')
        self.weights = np.append(self.weights, weight)
        
    def density(self, points):
        """ Returns the density for each of the points in points (unnormalized weights) """
        return np.array(np.sum(np.array([weight*dist.density(points) 
                       for weight, dist in zip(self.weights, self.distributions)]), axis=0), ndmin=1)
    
    def grad_density(self, points): 
        """ Returns gradient of the density at the specified points """
        # the following looks weird but it allows to vectorize everything.
        return np.sum(np.array([weight*dist.grad_density(points) 
                       for weight, dist in zip(self.weights, self.distributions)]), axis=0)
    
    def Hessian_density(self, points): 
        """ Returns Hessian of the density function at the specified points """       
        return np.sum(np.array([weight*dist.Hessian_density(points) 
                       for weight, dist in zip(self.weights, self.distributions)]), axis=0)
  
    def upperbound_density(self):
        """ Computes the a very crude upper bound of the density based on the weighted maxima
            of the individual densities """
        return np.sum(np.array([weight*dist.max_density() for weight, dist 
                                in zip(self.weights, self.distributions)]))
        
        
class ExpifiedMixture(Distribution):
    """ A probability distribution whose density is the (normalized) exponential of 
        the density of a mixture distribution scaled by the learning rate """
    
    def __init__(self, mixdist, eta):
        """ Constructor. Here mixdist is a MixtureDistribution and eta the learning rate """ 
        # check that domains are the same, then call superclass constructor        
        super(ExpifiedMixture, self).__init__(mixdist.domain)
        self.mixdist = mixdist
        self.eta = eta
        
    def sample(self, N, oversample=3):
        """ Implement Rejection sampling method here, potentially add other methods later """
        samples, nreject = self.oversample(N*oversample)
        while samples.shape[0] < N:
            newsamples, newrejects = self.oversample(N*oversample)
            samples = np.append(samples, newsamples[self.domain.iselement(newsamples)], axis=0)
            nreject = nreject + newrejects
        return samples[0:N]#, nreject

    def oversample(self, N):
        """ Implement basic rejection sampling method here, potentially add other methods later """
        # Create a rv that is uniform over the domain
        unif = Uniform(self.domain)
        # Find bounding constant K (VERY crude)
        K = np.exp(self.eta*self.mixdist.upperbound_density())/unif.max_density()
        # Sample uniformly on the domain
        qsamples = unif.sample(N)
        qvals = unif.density(qsamples)
        # sample uniformly on [0,K*qval] for every q-sample
        compvals = np.random.uniform(size=N)*K*qvals
        # reject all qsamples for which compvals are higher than density
        samples = qsamples[compvals <= self.density(qsamples)]
        nreject = N - samples.shape[0]
        return samples, nreject
  
    def density(self, points):
        """ Returns the unnormalized density for each of the points in points """
        return np.array(np.exp(self.eta*self.mixdist.density(points)), ndmin=1)
   
    def grad_density(self, points): 
        """ Returns gradient of the density at the specified points (unnormalized) """
        # not sure how to do this without the list comprehension
        return np.einsum('i,ij->ij', self.density(points), self.eta*self.mixdist.grad_density(points))
    
    def Hessian_density(self, points): 
        """ Returns Hessian of the density function at the specified points (unnormalized) """ 
        mixgrad = self.eta*self.mixdist.grad_density(points)
        outprods = np.einsum('ij...,i...->ij...', mixgrad, mixgrad)
        return np.einsum('i,ijk->ijk', self.density(points), self.mixdist.Hessian_density(points) + outprods)
        
        
        
        
class Gaussian(Distribution):
    """ Gaussian probability distribution on a restricted domain """
      
    def __init__(self, domain, mean, cov):
        """ Constructor. Here domain is a Domain object, and mean 
            and cov are numpy arrays """
        super(Gaussian, self).__init__(domain)
        self.mean, self.cov = mean, cov
        self.rv = multivariate_normal(mean, cov)
    
    def sample(self, N, oversample=2):
        """ Draw N independent samples from the distribution. The parameter oversample
            adjust how much samples are drawn in each iteration, before samples are discarded """
        # For now use a very crude implementation, fix this later!
        samples = self.rv.rvs(N*oversample)
        samples = samples[self.domain.iselement(samples)]
#         nreject = N*oversample - samples.shape[0]
        while samples.shape[0] < N:
#             print(self.rv.cov)
            newsamples = self.rv.rvs(N*oversample)
            samples = np.append(samples, newsamples[self.domain.iselement(newsamples)], axis=0)
        return samples[0:N] #, nreject
    
    def density(self, points):
        """ Returns the density for each of the points in points """
        return self.rv.pdf(points)
           
    def grad_density(self, points): 
        """ Returns gradient of the density at the specified points """
        # the following looks weird but it allows to vectorize everything.
        return - np.transpose(self.density(points)*np.dot(inv(self.cov), 
                                                    np.transpose(points - self.rv.mean)))
    
    def Hessian_density(self, points): 
        """ Returns Hessian of the density function at the specified points """       
        vecs = np.transpose(np.dot(inv(self.cov), 
                                   np.transpose(points - self.rv.mean)))
        # the following einsum call generate the outer product for each of the elements
        vecs = np.array(vecs, ndmin=2) # this is necessary if there is only a single point
        outprods = np.einsum('ij...,i...->ij...', vecs, vecs)
        return - np.transpose(self.density(points)*np.transpose(inv(self.cov) - outprods))
    
    def max_density(self):
        """ Computes the maximum of the distribution's pdf. For a simple
            Gaussian this is just the density at the mean. """
        return self.rv.pdf(self.mean)
        
      
class Uniform(Distribution):
    """ Uniform probability distribution over an arbitrary domain. """
      
    def __init__(self, domain):
        """ Constructor. Here domain is a Domain object """
        super(Uniform, self).__init__(domain)
        self.dens = 1/domain.compute_parameters()[1]
        self.bbox = domain.bbox() # this returns a Rectangle
    
    def sample(self, N, oversample=2):
        """ Draw N independent samples from the uniform distribution over the domain. 
            If rejection sampling is used, the parameter oversample adjust how many 
            samples are drawn in each iteration, before samples are discarded """
        # if we are dealing with a rectangle or a union of disjoint rectangles things are simple
        if isinstance(self.domain, Rectangle) or isinstance(self.domain, UnionOfDisjointRectangles):
            return self.domain.sample_uniform(N)
        # for anything else we'll do a simple rejection sampling (for now!)
        else:
            return self.sample_rej(N, oversample)
            
    def sample_rej(self, N, oversample):
        """ Implements basic rejection sampling method that works for all domains """
#         xsamples = np.random.uniform(low=self.bbox.lb[0], high=self.bbox.ub[0], size=(N,1))
#         ysamples = np.random.uniform(low=self.bbox.lb[1], high=self.bbox.ub[1], size=(N,1))
#         samples = np.concatenate((xsamples, ysamples), axis=1)
#         #samples = samples[self.bbox.iselement(samples)]
#         while samples.shape[0] < N:
#             newsamples = self.rv.rvs(N*oversample)
#             samples = np.append(samples, newsamples[self.domain.iselement(newsamples)], axis=0)
#         return samples[0:N]
        return NotImplementedError   
    
    def density(self, points):
        """ Computes density, this is trivial for the uniform distribution """
        return self.max_density()*np.ones(points.shape[0])
    
    def max_density(self):
        """ Computes maximal density, this is trivial for the uniform distribution """
        return 1/self.domain.volume
    