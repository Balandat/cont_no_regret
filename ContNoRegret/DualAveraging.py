'''
Core functionality for the dual averaging no-regret work

@author: Maximilian Balandat
@date: May 19, 2015
'''

import numpy as np
import os, ctypes, uuid
from _ctypes import dlclose
from subprocess import call
from scipy.optimize import brentq
from scipy.integrate import nquad
from .Domains import nBox, UnionOfDisjointnBoxes, DifferenceOfnBoxes
from .Potentials import ExponentialPotential, pNormPotential, IdentityPotential


def compute_nustar(dom, potential, eta, Loss, M, nu_prev, eta_prev, t, 
                   pid='0', tmpfolder='libs/'):
    """ Determines the normalizing nustar for the dual-averaging update """ 
    tmpfile = '{}{}'.format(tmpfolder, str(uuid.uuid4()))
    with open(tmpfile+'.c', 'w') as file:
        file.writelines(generate_ccode(dom, potential, eta, Loss)) 
    call(['gcc', '-shared', '-o', tmpfile+'.dylib', '-fPIC', tmpfile+'.c'])
    lib = ctypes.CDLL(tmpfile+'.dylib')
    lib.f.restype = ctypes.c_double
    lib.f.argtypes = (ctypes.c_int, ctypes.c_double)
    # compute the bounds for the root finding method for mu*
    if potential.isconvex():
        a = eta_prev/eta*nu_prev - M # this is a bound based on convexity of the potential
        b = eta_prev/eta*nu_prev + (eta_prev/eta-1)*t*M
    else:
        a = - Loss.max() - np.max(potential.phi_inv(1/dom.volume)/eta, 0) # this is (coarse) lower bound on nustar
        b = nu_prev + 50 # this is NOT CORRECT - a hack. Need to find conservative bound for nonconvex potentials
    try:                         
        if isinstance(dom, nBox) or isinstance(dom, UnionOfDisjointnBoxes):
            if isinstance(dom, nBox):
                ranges = [dom.bounds]
            elif isinstance(dom, UnionOfDisjointnBoxes):
                ranges = [nbox.bounds for nbox in dom.nboxes]
            if isinstance(potential, ExponentialPotential):
                # in this case we don't have to search for nustar, we can find it (semi-)explicitly
                integral = np.sum([nquad(lib.f, rng, [0])[0] for rng in ranges])
                nustar = np.log(integral)/eta
            else:
                if isinstance(potential, IdentityPotential):
                    grid = dom.grid(500000)
                    f = lambda nu: np.sum(potential.phi(-eta*(Loss.val(grid)+nu)))/len(grid)
                else:
                    f = lambda nu: np.sum([nquad(lib.f, rng, args=[nu], 
                                                 opts=[{'epsabs':1.49e-4, 'epsrel':1.49e-3}]*dom.n)[0] 
                                           for rng in ranges]) - 1
                success = False
                while not success:
                    try:
                        nustar, r = brentq(f, a, b, full_output=True)
                        print(r.root, r.iterations, r.function_calls, r.converged, r.flag)
                        success = True
                    except ValueError:
                        print('WARINING: PROCESS {} HAS ENCOUNTERED f(a)!=f(b)!'.format(pid))
                        a, b = a - 20, b + 20
            
        elif isinstance(dom, DifferenceOfnBoxes):
            if isinstance(potential, ExponentialPotential):
                # in this case we don't have to search for nustar, we can find it (semi-)explicitly
                integral = (nquad(lib.f, dom.outer.bounds, [0])[0] 
                            - np.sum([nquad(lib.f, nbox.bounds, args=[0],
                                      opts=[{'epsabs':1.49e-4, 'epsrel':1.49e-3}]*dom.n)[0] for nbox in dom.inner]))
                nustar = np.log(integral)/eta
            else:
                if isinstance(potential, IdentityPotential):
                    grid = dom.grid(500000)
                    f = lambda nu: np.sum(potential.phi(-eta*(Loss.val(grid)+nu)))/len(grid)
                else:
                    f = lambda nu: (nquad(lib.f, dom.outer.bounds, [nu], opts=[{'epsabs':1.49e-4, 'epsrel':1.49e-3}])[0] 
                                    - np.sum([nquad(lib.f, nbox.bounds, [nu])[0] for nbox in dom.inner]) - 1)
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
        try:
            os.remove(tmpfile+'.c') # clean up
            os.remove(tmpfile+'.dylib') # clean up
        except FileNotFoundError: pass


def generate_ccode(dom, potential, eta, Loss):
    """ Generates the c source code that is complied and used for faster numerical 
        integration (using ctypes). Hard-codes known parameters (except s and nu) as
        literals and returns a list of strings that are the lines of a C source file. """
    header = ['#include <math.h>\n\n',
              'double eta = {};\n'.format(eta)]
    return header + Loss.gen_ccode() + potential.gen_ccode()



    
