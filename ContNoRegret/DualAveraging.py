'''
Some functions for the dual averaging no-regret work

@author: Maximilian Balandat
@date: May 7, 2015
'''

import numpy as np
import os, ctypes
from _ctypes import dlclose
from subprocess import call
from scipy.optimize import brentq
from scipy.integrate import nquad
from .LossFunctions import PolynomialLossFunction, AffineLossFunction, QuadraticLossFunction
import ContNoRegret


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


class CompositeOmegaPotential(OmegaPotential):
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
    
    
class LogtasticPotential(OmegaPotential):
    """ The logtastic potential function  """
    
    def __init__(self, desc='LogPot'):
        """ Constructor """
        self.desc = desc
        
    def phi(self, u):
        """ Returns phi(u), the value of the Huber-potential at the points u"""
        return (u < 0)*np.exp(u) + (u >= 0)*(1 + np.log(1 + np.maximum(u, 0)))
        
    def phi_prime(self, u):
        """ Returns phi'(u), the first derivative of the zero-potential at the points u """
        return (u < 0)*np.exp(u) + (u >= 0)/(1 + np.maximum(u, 0))
 
    def phi_double_prime(self, u):
        """ Returns phi''(u), the second derivative of the zero-potential at the points u """
        return (u < 0)*np.exp(u) - (u >= 0)/((1 + np.maximum(u, 0))**2)
     
    def phi_inv(self, u):
        """ Returns phi^{-1}(u), the inverse function of the zero-potential at the points u """
        return (u < 1)*np.log(u) + (u > 1)*(np.exp(u - 1) - 1)
     
    def phi_inv_prime(self, u):
        """ Returns phi^{-1}'(u), the first derivative of the inverse 
            function of the zero-potential at the points u """
        return (u < 1)/u + (u > 1)*np.exp(u - 1)
    
    def isconvex(self):
        """ Returns True if phitilde(u) = max(phi(u), 0) is a convex function. """
        return False    
    
    
def nustar_quadratic(dom, gamma, eta, Q, mu, c, nu_prev=1000):
    """ Determines the normalizing nustar for the dual-averaging update. """
    # for speedup, implement function in C in a temporary shared library
    lines = ['#include <math.h>\n',
             'double gam = {};\n'.format(gamma),
             'double eta = {};\n'.format(eta),
             'double Q11 = {};\n'.format(Q[0,0]),
             'double Q12 = {};\n'.format(Q[0,1]),
             'double Q22 = {};\n'.format(Q[1,1]),
             'double c = {};\n'.format(c),
             'double phi(int n, double args[n]){\n',
             '    double x[2];\n'
             '    x[0] = args[0] - {};\n'.format(mu[0]),
             '    x[1] = args[1] - {};\n'.format(mu[1]),
             '    return pow(gam/(gam-1) + eta*(0.5*(Q11*pow(x[0],2.0)+2*Q12*x[0]*x[1]+Q22*pow(x[1],2.0)) + c + args[2]), -gam);}']
    with open('tmpfunc.c', 'w') as file:
        file.writelines(lines)
    call(['gcc', '-shared', '-o', 'tmpfunc.dylib', '-fPIC', 'tmpfunc.c'])
    lib = ctypes.CDLL('tmpfunc.dylib') # Use absolute path to testlib
    lib.phi.restype = ctypes.c_double
    lib.phi.argtypes = (ctypes.c_int, ctypes.c_double)
    # now use this function for integration over the domain
    if isinstance(dom, ContNoRegret.Domains.Rectangle):
        ranges = [[(dom.lb[0], dom.ub[0]), (dom.lb[1], dom.ub[1])]]
    elif isinstance(dom, ContNoRegret.Domains.UnionOfDisjointRectangles):
        ranges = [[(rect.lb[0], rect.ub[0]), (rect.lb[1], rect.ub[1])] for rect in dom.rects]
    else:
        raise Exception('For now domain must be a Rectangle or a UnionOfDisjointRectangles!')
    f = lambda nu: np.sum([nquad(lib.phi, rng, [nu])[0] for rng in ranges]) - 1      
    a = (0.001 - gamma/(gamma-1))/eta - c # this is a lower bound on nustar
    nustar = brentq(f, a, nu_prev)#, full_output=True)
    dlclose(lib._handle) # this is to release the lib, so we can import the new version
    os.remove('tmpfunc.c') # clean up
    os.remove('tmpfunc.dylib') # clean up
    return nustar

def nustar_generic(dom, potential, eta, Lspline, nu_prev=1000):
    """ Determines the normalizing nustar for the dual-averaging update 
        (for now assume problem to be 2-dimensional) """
    # create approximation of the integrand as a function of s1,s2,nu
    phi = lambda s1,s2,nu: potential.phi(-eta*(Lspline(s1,s2) + nu))
    # now use this function for integration over the domain
    if isinstance(dom, ContNoRegret.Domains.Rectangle):
        ranges = [[(dom.lb[0], dom.ub[0]), (dom.lb[1], dom.ub[1])]]
    elif isinstance(dom, ContNoRegret.Domains.UnionOfDisjointRectangles):
        ranges = [[(rect.lb[0], rect.ub[0]), (rect.lb[1], rect.ub[1])] for rect in dom.rects]
    else:
        raise Exception('For now domain must be a Rectangle or a UnionOfDisjointRectangles!')
    f = lambda nu: np.sum([nquad(phi, rng, [nu])[0] for rng in ranges]) - 1
    knots = Lspline.get_knots()
    Lmax = Lspline.ev(knots[0], knots[1]).max()
    a = -1/eta*potential.phi_inv(1/dom.volume) - Lmax
    print('search interval: [{},{}]'.format(a,nu_prev))
    (nustar, res) = brentq(f, a, nu_prev, full_output=True)
    print('iterations: {},  function calls: {}'.format(res.iterations, res.function_calls))
    return nustar


def compute_nustar(dom, potential, eta, Loss, M, nu_prev, eta_prev, t, pid='0', tmpfolder='libs/'):
    """ Determines the normalizing nustar for the dual-averaging update """      
    with open('{}/tmplib{}.c'.format(tmpfolder,pid), 'w') as file:
        file.writelines(generate_ccode(dom, potential, eta, Loss))
    call(['gcc', '-shared', '-o', '{}tmplib{}.dylib'.format(tmpfolder,pid), '-fPIC', '{}tmplib{}.c'.format(tmpfolder,pid)])
    lib = ctypes.CDLL('{}/tmplib{}.dylib'.format(tmpfolder,pid))
    lib.phi.restype = ctypes.c_double
    lib.phi.argtypes = (ctypes.c_int, ctypes.c_double)
    # compute the bounds for the root finding method for mu*
    if potential.isconvex():
        a = eta_prev/eta*nu_prev - M # this is a bound based on convexity of the potential
        b = eta_prev/eta*nu_prev + (eta_prev/eta-1)*t*M
    else:
        a = - Loss.max() - np.max(potential.phi_inv(1/dom.volume)/eta, 0) # this is (coarse) lower bound on nustar
        b = nu_prev + 50 # this is NOT CORRECT - a hack. Need to find conservative bound for nonconvex potentials
    try: 
        if isinstance(dom, ContNoRegret.Domains.nBox) or isinstance(dom, ContNoRegret.Domains.UnionOfDisjointnBoxes):
            if isinstance(dom, ContNoRegret.Domains.nBox):
                ranges = [dom.bounds]
            elif isinstance(dom, ContNoRegret.Domains.UnionOfDisjointnBoxes):
                ranges = [nbox.bounds for nbox in dom.nboxes]
            if isinstance(potential, ExponentialPotential):
                # in this case we don't have to search for nustar, we can find it (semi-)explicitly
                integral = np.sum([nquad(lib.phi, rng, [0])[0] for rng in ranges])
                nustar = np.log(integral)/eta
            else:
                f = lambda nu: np.sum([nquad(lib.phi, rng, [nu])[0] for rng in ranges]) - 1
#                 print('Lmin={}, Lmax={}'.format(Loss.min(), Loss.max()))
#                 print('a={0:.3f}. f(a)={1:.3f}, b={2:.3f}, f(b)={3:.3f}'.format(a, f(a), b, f(b)))
#                 if (f(b)*f(a) > 0):
#                     Loss.plot(dom.sample_uniform(1000))
#                     xs = np.linspace(a-5, b+5, 50)
#                     plt.plot(xs, np.array([f(x) for x in xs]))
#                     plt.show()
                success = False
                while not success:
                    try:
                        nustar = brentq(f, a, b)
                        success = True
                    except ValueError:
                        a, b = a - 20, b + 20
                        print('WARINING: PROCESS {} HAS ENCOUNTERED f(a)!=f(b)!'.format(pid))
        elif isinstance(dom, ContNoRegret.Domains.DifferenceOfnBoxes):
            if isinstance(potential, ExponentialPotential):
                # in this case we don't have to search for nustar, we can find it (semi-)explicitly
                integral = (nquad(lib.phi, dom.outer.bounds, [0])[0] 
                            - np.sum([nquad(lib.phi, nbox.bounds, [0])[0] for nbox in dom.inner]))
                nustar = np.log(integral)/eta
            else:
                f = lambda nu: (nquad(lib.phi, dom.outer.bounds, [nu])[0] 
                                - np.sum([nquad(lib.phi, nbox.bounds, [nu])[0] for nbox in dom.inner]) - 1)
                success = False
                while not success:
                    try:
                        nustar = brentq(f, a, b)
                        success = True
                    except ValueError:
                        a, b = a - 20, b + 20
                        print('WARINING: PROCESS {} HAS ENCOUNTERED f(a)!=f(b)!'.format(pid))
        else:
            raise Exception('For now, domain must be an nBox or a UnionOfDisjointnBoxes!') 
        return nustar
    finally: 
        dlclose(lib._handle) # this is to release the lib, so we can import the new version
        os.remove('{}/tmplib{}.c'.format(tmpfolder,pid)) # clean up
        os.remove('{}/tmplib{}.dylib'.format(tmpfolder,pid)) # clean up


def generate_ccode(dom, potential, eta, Loss):
    """ Generates the c source code that is complied and used for faster numerical 
        integration (using ctypes). Hard-codes known parameters (except s and nu) as
        literals and returns a list of strings that are the lines of a C source file. """
    header = ['#include <math.h>\n\n',
              'double eta = {};\n'.format(eta)]
    if isinstance(Loss, AffineLossFunction):
        affine = ['double a[{}] = {{{}}};\n'.format(dom.n, ','.join(str(a) for a in Loss.a)),
                  'double phi(int n, double args[n]){\n',
                  '   double nu = *(args + {});\n'.format(dom.n),
                  '   int i;\n',
                  '   double loss = {};\n'.format(Loss.b),
                  '   for (i=0; i<{}; i++){{\n'.format(dom.n),
                  '     loss += a[i]*(*(args + i));\n',
                  '     }\n']
        header = header + affine
    elif isinstance(Loss, PolynomialLossFunction):
        poly = ['double c[{}] = {{{}}};\n'.format(Loss.m, ','.join(str(coeff) for coeff in Loss.coeffs)),
                'double e[{}] = {{{}}};\n\n'.format(Loss.m*dom.n, ','.join(str(xpnt) for xpntgrp in Loss.exponents for xpnt in xpntgrp)),
                'double phi(int n, double args[n]){\n',
                '   double nu = *(args + {});\n'.format(dom.n),
                '   int i,j;\n',
                '   double mon;\n',  
                '   double loss = 0.0;\n',
                '   for (i=0; i<{}; i++){{\n'.format(Loss.m),
                '     mon = 1.0;\n',
                '     for (j=0; j<{}; j++){{\n'.format(dom.n),
                '       mon = mon*pow(args[j], e[i*{}+j]);\n'.format(dom.n),
                '       }\n',
                '     loss += c[i]*mon;\n',
                '     }\n']
        header = header + poly
    elif isinstance(Loss, QuadraticLossFunction):
        quad = ['double Q[{}][{}] = {{{}}};\n'.format(dom.n, dom.n, ','.join(str(q) for row in Loss.Q for q in row)),
                'double mu[{}] = {{{}}};\n'.format(dom.n, ','.join(str(m) for m in Loss.mu)),
                'double c = {};\n\n'.format(Loss.c),
                'double phi(int n, double args[n]){\n',
                '   double nu = *(args + {});\n'.format(dom.n),
                '   int i,j;\n',
                '   double loss = c;\n',
                '   for (i=0; i<{}; i++){{\n'.format(dom.n),
                '     for (j=0; j<{}; j++){{\n'.format(dom.n),
                '       loss += Q[i][j]*(args[i]-mu[i])*(args[j]-mu[j]);\n',
                '       }\n',
                '     }\n']  
        header = header + quad        
    if isinstance(potential, ExponentialPotential):   
        return header + ['   return exp(-eta*(loss + nu));}']
    elif isinstance(potential, IdentityPotential):
        return header + ['   double z = -eta*(loss + nu);\n',
                         '   if(z>0){\n',
                         '     return z;}\n',
                         '   else{\n',
                         '     return 0.0;}\n',
                         '   }']
    elif isinstance(potential, CompositeOmegaPotential):
        omega_pot = ['   double z = -eta*(loss + nu);\n',
                     '   if(z<{}){{\n'.format(potential.c),
                     '     return pow({}-z, -{});}}\n'.format(potential.gamma*potential.c, potential.gamma),
                     '   else{\n',
                     '     return {} + {}*z + {}*pow(z,2);}}\n'.format(*[a for a in potential.a]),
                     '   }']
        return header + omega_pot
    elif isinstance(potential, pNormPotential):
        pNorm_pot = ['   double z = -eta*(loss + nu);\n',
                     '   if(z>0){\n',
                     '     return pow(z, {});}}\n'.format(1/(potential.p - 1)),
                     '   else{\n',
                     '     return 0.0;}\n',
                     '   }']
        return header + pNorm_pot
    elif isinstance(potential, LogtasticPotential):
        pNorm_pot = ['   double z = -eta*(loss + nu);\n',
                     '   if(z>0){\n',
                     '     return 1 + log(1 + z);}\n',
                     '   else{\n',
                     '     return exp(z);}\n',
                     '   }']
        return header + pNorm_pot 
    elif isinstance(potential, HuberPotential):
        Huber_pot = ['   double z = -eta*(loss + nu);\n',
                     '   if(z<0){\n',
                     '     return 0.0;}\n',
                     '   elseif(z<{}){{\n'.format(Loss.delta),
                     '     return 0.5*pow(z,2);}\n',
                     '   else{\n',
                     '     return {}*(z - 0.5*{});}}\n'.format(Loss.delta),
                     '   }']
        return header + Huber_pot
