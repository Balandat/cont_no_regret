'''
Core functionality for the dual averaging no-regret work

@author: Maximilian Balandat
@date: May 8, 2015
'''

import numpy as np
import os, ctypes
from _ctypes import dlclose
from subprocess import call
from scipy.optimize import brentq
from scipy.integrate import nquad
from .LossFunctions import PolynomialLossFunction, AffineLossFunction, QuadraticLossFunction
from .Potentials import (ExponentialPotential, IdentityPotential, pNormPotential, ExpPPotential,
                         PExpPotential, CompositePotential, HuberPotential, LogtasticPotential) 
import ContNoRegret


def compute_nustar(dom, potential, eta, Loss, M, nu_prev, eta_prev, t, 
                   pid='0', tmpfolder='libs/'):
    """ Determines the normalizing nustar for the dual-averaging update """      
    with open('{}/tmplib{}.c'.format(tmpfolder,pid), 'w') as file:
        file.writelines(generate_ccode(dom, potential, eta, Loss))
    call(['gcc', '-shared', '-o', '{}tmplib{}.dylib'.format(tmpfolder,pid), 
          '-fPIC', '{}tmplib{}.c'.format(tmpfolder,pid)])
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
    return header + Loss.gen_ccode() + potential.gen_ccode()
    


#######################################################################
# The functions below are deprecated and to be removed at a later stage
#######################################################################

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



