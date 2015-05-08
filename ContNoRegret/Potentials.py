'''
A collection of omega-potentials for Dual Averaging

@author: Maximilian Balandat
@date: May 8, 2015
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
    
    def __init__(self, desc='ExpPot'):
        """ Constructor """
        self.desc = desc
        
    def phi(self, u):
        """ Returns phi(u), the value of the zero-potential at the points u"""
        return np.exp(u)
        
    def phi_prime(self, u):
        """ Returns phi'(u), the first derivative of the zero-potential at the points u """
        return np.exp(u)
 
    def phi_double_prime(self, u):
        """ Returns phi''(u), the second derivative of the zero-potential at the points u """
        return np.exp(u)
     
    def phi_inv(self, u):
        """ Returns phi^{-1}(u), the inverse function of the zero-potential at the points u """
        return np.log(u)
     
    def phi_inv_prime(self, u):
        """ Returns phi^{-1}'(u), the first derivative of the inverse 
            function of the zero-potential at the points u """
        return 1/u
    
    def isconvex(self):
        """ Returns True if phitilde(u) = max(phi(u), 0) is a convex function. """
        return True
    
    def gen_ccode(self):
        """ Generates a c-code snippet used for fast numerical integration """
        return ['   return exp(-eta*(loss + nu));}']
   
   
class IdentityPotential(OmegaPotential):
    """ The identity potential Phi(x) = x, which results in the Euclidean Projection  """
    
    def __init__(self, desc='IdPot'):
        """ Constructor """
        self.desc = desc
        
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
    
    def isconvex(self):
        """ Returns True if phitilde(u) = max(phi(u), 0) is a convex function. """
        return True
   
    def gen_ccode(self):
        """ Generates a c-code snippet used for fast numerical integration """
        return ['   double z = -eta*(loss + nu);\n',
                '   if(z>0){\n',
                '     return z;}\n',
                '   else{\n',
                '     return 0.0;}\n',
                '   }']


class pNormPotential(OmegaPotential):
    """ The potential phi(u) = sgn(u)*|u|**(1/(p-1)) """
    
    def __init__(self, p, desc='pNormPot'):
        """ Constructor """
        if (p<=1) or (p>2):
            raise Exception('Need 1 < p <=2 !') 
        self.p = p
        self.desc = desc + ', ' + r'$p={{{}}}$'.format(p)
        
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

    def isconvex(self):
        """ Returns True if phitilde(u) = max(phi(u), 0) is a convex function. """
        return True 

    def gen_ccode(self):
        """ Generates a c-code snippet used for fast numerical integration """
        return ['   double z = -eta*(loss + nu);\n',
                '   if(z>0){\n',
                '     return pow(z, {});}}\n'.format(1/(self.p - 1)),
                '   else{\n',
                '     return 0.0;}\n',
                '   }']


class CompositePotential(OmegaPotential):
    """ A composite omega potential formed by stitching together a
        fractional and a quadratic function """
    
    def __init__(self, gamma=2, desc='CompPot'):
        """ Constructor """
        self.gamma = gamma
        self.c = (gamma-1)**(-1)
        a2 = gamma*(1+gamma)/2
        a1 = gamma - 2*self.c*a2
        a0 = 1 - self.c*a1 - self.c**2*a2
        self.a = np.array([a0, a1, a2])
        self.desc = desc + ', ' + r'$\gamma={{{}}}$'.format(gamma)
        
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
    
    def isconvex(self):
        """ Returns True if phitilde(u) = max(phi(u), 0) is a convex function. """
        return True
    
    def gen_ccode(self):
        """ Generates a c-code snippet used for fast numerical integration """
        return ['   double z = -eta*(loss + nu);\n',
                '   if(z<{}){{\n'.format(self.c),
                '     return pow({}-z, -{});}}\n'.format(self.gamma*self.c, self.gamma),
                '   else{\n',
                '     return {} + {}*z + {}*pow(z,2);}}\n'.format(*[a for a in self.a]),
                '   }']
    
    
class ExpPPotential(OmegaPotential):
    """ A potential given by a composition of an exponential and a p norm """
    
    def __init__(self, p, desc='ExpPPot'):
        """ Constructor """
        self.p = p
        self.desc = desc = desc + ', ' + r'$p={{{}}}$'.format(p)
        
    def phi(self, u):
        """ Returns phi(u), the value of the ExpP-potential at the points u"""
        return (u>=1)*u**(1/(self.p-1)) + (u<1)*np.exp((u-1)/(self.p-1))
        
    def phi_prime(self, u):
        """ Returns phi'(u), the first derivative of the zero-potential at the points u """
        return ((u>=1)/(self.p - 1)*u**((2 - self.p)/(self.p - 1)) 
                + (u<1)/(self.p - 1)*np.exp((u - 1)/(self.p - 1)))
 
    def phi_double_prime(self, u):
        """ Returns phi''(u), the second derivative of the zero-potential at the points u """
        return ((u>=1)*(2 - self.p)/(self.p - 1)**2*u**((3 - 2*self.p)/(self.p - 1)) 
                + (u<1)/(self.p - 1)**2*np.exp((u - 1)/(self.p - 1)))
     
    def phi_inv(self, u):
        """ Returns phi^{-1}(u), the inverse function of the zero-potential at the points u """
        return (u>=1)*u**(self.p-1) + (u<1)*(1+(self.p-1)*np.log(u))
     
    def phi_inv_prime(self, u):
        """ Returns phi^{-1}'(u), the first derivative of the inverse 
            function of the zero-potential at the points u """
        raise 1/self.phi_prime(self.phi_inv(u))
    
    def isconvex(self):
        """ Returns True if phitilde(u) = max(phi(u), 0) is a convex function. """
        return True
    
    def gen_ccode(self):
        """ Generates a c-code snippet used for fast numerical integration """
        return ['   double z = -eta*(loss + nu);\n',
                '   if(z>1){\n',
                '     return pow(z, {});}}\n'.format(1/(self.p - 1)),
                '   else{\n',
                '     return exp((z-1)/{});}}\n'.format(self.p - 1),
                '   }']
        

class PExpPotential(OmegaPotential):
    """ A potential given by a composition of a p-norm and 
        an exponential potential """
    
    def __init__(self, p, desc='PExpPot'):
        """ Constructor """
        self.p = p
        self.desc = desc = desc + ', ' + r'$p={{{}}}$'.format(p)
        
    def phi(self, u):
        """ Returns phi(u), the value of the Pexp-potential at the points u"""
        return (u>0)*(u<1)*u**(1/(self.p-1)) + (u>=1)*np.exp((u-1)/(self.p-1))
        
    def phi_prime(self, u):
        """ Returns phi'(u), the first derivative of the zero-potential at the points u """
        return ((u>0)*(u<1)/(self.p-1)*u**((2-self.p)/(self.p-1)) 
                + (u>=1)/(self.p-1)*np.exp((u-1)/(self.p-1)))
 
    def phi_double_prime(self, u):
        """ Returns phi''(u), the second derivative of the zero-potential at the points u """
        return ((u>0)*(u<1)*(2-self.p)/(self.p-1)**2*u**((3-2*self.p)/(self.p-1)) 
                + (u>=1)/(self.p-1)**2*np.exp((u-1)/(self.p-1)))
     
    def phi_inv(self, u):
        """ Returns phi^{-1}(u), the inverse function of the zero-potential at the points u """
        return (u<1)*u**(self.p-1) + (u>=1)*(1+(self.p-1)*np.log(u))
     
    def phi_inv_prime(self, u):
        """ Returns phi^{-1}'(u), the first derivative of the inverse 
            function of the zero-potential at the points u """
        raise 1/self.phi_prime(self.phi_inv(u))
    
    def isconvex(self):
        """ Returns True if phitilde(u) = max(phi(u), 0) is a convex function. """
        return True
    
    def gen_ccode(self):
        """ Generates a c-code snippet used for fast numerical integration """
        return ['   double z = -eta*(loss + nu);\n',
                '   if(z>1){\n',
                '     return exp((z-1)/{});}}\n'.format(self.p - 1),
                '   else if(z>0){\n',
                '     return pow(z, {});}}\n'.format(1/(self.p - 1)),
                '   else{\n',
                '     return 0.0;}\n',
                '   }']
        
    
class HuberPotential(OmegaPotential):
    """ The potential given by the Huber loss function  """
    
    def __init__(self, delta, desc='HuberPot'):
        """ Constructor """
        self.delta = delta
        self.desc = desc = desc + ', ' + r'$\delta={{{}}}$'.format(delta)
        
    def phi(self, u):
        """ Returns phi(u), the value of the Huber-potential at the points u"""
        return (0.5*u**2*((u >= 0) & (u <= self.delta))
                - 0.5*u**2*((u < 0) & (u >= -self.delta)) 
                + self.delta*(u - 0.5*self.delta)*(u > self.delta)
                + self.delta*(u + 0.5*self.delta)*(u < -self.delta))
        
    def phi_prime(self, u):
        """ Returns phi'(u), the first derivative of the zero-potential at the points u """
        return np.abs(u)*(np.abs(u) < self.delta) + self.delta*(np.abs(u) >= self.delta)
 
    def phi_double_prime(self, u):
        """ Returns phi''(u), the second derivative of the zero-potential at the points u """
        return np.sign(u)*(np.abs(u) < self.delta)
     
    def phi_inv(self, u):
        """ Returns phi^{-1}(u), the inverse function of the zero-potential at the points u """
        return (np.sign(u)*np.sqrt(2*np.abs(u))*(np.abs(u) <= 0.5*self.delta**2)
                + (np.sign(u)*self.delta/2 + u/self.delta))
     
    def phi_inv_prime(self, u):
        """ Returns phi^{-1}'(u), the first derivative of the inverse 
            function of the zero-potential at the points u """
        raise 1/self.phi_prime(self.phi_inv(u))
    
    def isconvex(self):
        """ Returns True if phitilde(u) = max(phi(u), 0) is a convex function. """
        return True
    
    def gen_ccode(self):
        """ Generates a c-code snippet used for fast numerical integration """
        return ['   double z = -eta*(loss + nu);\n',
                '   if(z<0){\n',
                '     return 0.0;}\n',
                '   elseif(z<{}){{\n'.format(self.delta),
                '     return 0.5*pow(z,2);}\n',
                '   else{\n',
                '     return {}*(z - 0.5*{});}}\n'.format(self.delta),
                '   }']
        
    
class LogtasticPotential(OmegaPotential):
    """ The logtastic potential function  """
    
    def __init__(self, desc='LogPot'):
        """ Constructor """
        self.desc = desc
        
    def phi(self, u):
        """ Returns phi(u), the value of the Huber-potential at the points u"""
        return (u<0)*np.exp(u) + (u>=0)*(1 + np.log(1 + np.maximum(u, 0)))
        
    def phi_prime(self, u):
        """ Returns phi'(u), the first derivative of the zero-potential at the points u """
        return (u<0)*np.exp(u) + (u>=0)/(1 + np.maximum(u, 0))
 
    def phi_double_prime(self, u):
        """ Returns phi''(u), the second derivative of the zero-potential at the points u """
        return (u<0)*np.exp(u) - (u>=0)/((1 + np.maximum(u, 0))**2)
     
    def phi_inv(self, u):
        """ Returns phi^{-1}(u), the inverse function of the zero-potential at the points u """
        return (u<1)*np.log(u) + (u>1)*(np.exp(u - 1) - 1)
     
    def phi_inv_prime(self, u):
        """ Returns phi^{-1}'(u), the first derivative of the inverse 
            function of the zero-potential at the points u """
        return (u<1)/u + (u>1)*np.exp(u - 1)
    
    def isconvex(self):
        """ Returns True if phitilde(u) = max(phi(u), 0) is a convex function. """
        return False  
    
    def gen_ccode(self):
        """ Generates a c-code snippet used for fast numerical integration """
        return ['   double z = -eta*(loss + nu);\n',
                '   if(z>0){\n',
                '     return 1 + log(1 + z);}\n',
                '   else{\n',
                '     return exp(z);}\n',
                '   }']  