'''
Some functions for the dual averaging no-regret work

@author: Maximilian Balandat
@date: Apr 24, 2015
'''

import numpy as np


class OmegaPotential(object):
    """ Base class for omega potentials """
        
    def phi(self, u):
        """ Returns phi(u), the value of the zero-potential at the points u"""
        return NotImplementedError
        
    def phi_prime(self, u):
        """ Returns phi'(u), the first derivative of the zero-potential at the points u """
        return NotImplementedError
 
    def phi_double_prime(self, u):
        """ Returns phi''(u), the second derivative of the zero-potential at the points u """
        return NotImplementedError
     
    def phi_inv(self, u):
        """ Returns phi^{-1}(u), the inverse function of the zero-potential at the points u """
        return NotImplementedError
     
    def phi_inv_prime(self, u):
        """ Returns phi^{-1}'(u), the first derivative of the inverse 
            function of the zero-potential at the points u """
        return NotImplementedError
    
    
class ExponentialPotential(OmegaPotential):
    """ The exponential potential, which results in Entropy Dual Averaging """
    
    def __init__(self):
        """ Constructor """
        pass
        
    def phi(self, u):
        """ Returns phi(u), the value of the zero-potential at the points u"""
        return np.exp(u-1)
        
    def phi_prime(self, u):
        """ Returns phi'(u), the first derivative of the zero-potential at the points u """
        return self.phi(u)
 
    def phi_double_prime(self, u):
        """ Returns phi''(u), the second derivative of the zero-potential at the points u """
        return self.phi(u)
     
    def phi_inv(self, u):
        """ Returns phi^{-1}(u), the inverse function of the zero-potential at the points u """
        return 1 + np.log(u)
     
    def phi_inv_prime(self, u):
        """ Returns phi^{-1}'(u), the first derivative of the inverse 
            function of the zero-potential at the points u """
        return 1/u
   
   
class IdentityPotential(OmegaPotential):
    """ The identity potential Phi(x) = x, which results in the Euclidean Projection  """
    
    def __init__(self):
        """ Constructor """
        pass
        
    def phi(self, u):
        """ Returns phi(u), the value of the zero-potential at the points u"""
        return u
        
    def phi_prime(self, u):
        """ Returns phi'(u), the first derivative of the zero-potential at the points u """
        return np.ones_like(u)
 
    def phi_double_prime(self, u):
        """ Returns phi''(u), the second derivative of the zero-potential at the points u """
        return np.zeros_like(u)
     
    def phi_inv(self, u):
        """ Returns phi^{-1}(u), the inverse function of the zero-potential at the points u """
        return u
     
    def phi_inv_prime(self, u):
        """ Returns phi^{-1}'(u), the first derivative of the inverse 
            function of the zero-potential at the points u """
        return np.ones_like(u)
   

class pNormPotential(OmegaPotential):
    """ The potential phi(u) = sgn(u)*|u|**(1/(p-1)) """
    
    def __init__(self, p):
        """ Constructor """
        if (p<=1) or (p>2):
            raise Exception('Need 1 < p <=2 !') 
        self.p = p
        
    def phi(self, u):
        """ Returns phi(u), the value of the pNorm-potential at the points u"""
        return np.sign(u)*np.abs(u)**(1/(self.p - 1))
        
    def phi_prime(self, u):
        """ Returns phi'(u), the first derivative of the pNorm-potential at the points u """
        return np.abs(u)**((2 - self.p)/(self.p - 1))/(self.p - 1)
 
    def phi_double_prime(self, u):
        """ Returns phi''(u), the second derivative of the pNorm-potential at the points u """
        return np.sign(u)*(2 - self.p)/((self.p - 1)**2)*np.abs(u)**((3 - 2*self.p)/(self.p - 2))
     
    def phi_inv(self, u):
        """ Returns phi^{-1}(u), the inverse function of the pNorm-potential at the points u """
        return np.sign(u)*np.abs(u)**(self.p - 1)
     
    def phi_inv_prime(self, u):
        """ Returns phi^{-1}'(u), the first derivative of the inverse 
            function of the pNorm-potential at the points u """
        return (self.p - 1)*np.abs(u)**(self.p - 2)


class CompositeOmegaPotential(OmegaPotential):
    """ A composite omega potential formed by stitching together a
        fractional and a quadratic function """
    
    def __init__(self, gamma=2):
        """ Constructor """
        self.gamma = gamma
        self.c = (gamma-1)**(-1)
        a2 = gamma*(1+gamma)/2
        a1 = gamma - 2*self.c*a2
        a0 = 1 - self.c*a1 - self.c**2*a2
        self.a = np.array([a0, a1, a2])
        
    def phi(self, u):
        """ Returns phi(u), the value of the zero-potential at the points u"""
        return ( (u<self.c)*(self.gamma/(self.gamma-1)-np.minimum(u,self.c))**(-self.gamma) + 
                 (u>=self.c)*(self.a[0]+self.a[1]*np.maximum(u,self.c)+self.a[2]*np.maximum(u,self.c)**2) )
        
    def phi_prime(self, u):
        """ Returns phi'(u), the first derivative of the zero-potential at the points u """
        return ( (u<self.c)*self.gamma*(self.gamma/(self.gamma-1)-np.minimum(u,self.c))**(-(1+self.gamma)) + 
                 (u>=self.c)*(self.a[1]+2*self.a[2]*np.maximum(u,self.c)) )
 
    def phi_double_prime(self, u):
        """ Returns phi''(u), the second derivative of the zero-potential at the points u """
        return ( (u<self.c)*self.gamma*(1+self.gamma)*(self.gamma/(self.gamma-1)-np.minimum(u,self.c))**(-(2+self.gamma)) + 
                 (u>=self.c)*2*self.a[2] )
     
    def phi_inv(self, u):
        """ Returns phi^{-1}(u), the inverse function of the zero-potential at the points u """
        b = self.phi(self.c)
        return ( (u<b)*(self.gamma/(self.gamma-1)-np.minimum(u,b)**(-1/self.gamma)) + 
                 (u>=b)*(-self.a[1]/2/self.a[2]+np.sqrt(self.a[1]**2/4/self.a[2]**2 - (self.a[0]-np.maximum(u,b))/self.a[2])) )
     
    def phi_inv_prime(self, u):
        """ Returns phi^{-1}'(u), the first derivative of the inverse 
            function of the zero-potential at the points u """
        return 1/self.phi_prime(self.phi_inv(u))
    
    
    
    