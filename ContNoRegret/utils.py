'''
Collection of utilities to analyze Continuous No-Regret algorithms

@author: Maximilian Balandat, Walid Krichene
@date Dec 4, 2014
'''

import ctypes
from _ctypes import dlclose
import numpy as np
import os
from subprocess import call
from matplotlib import pyplot as plt
from scipy.linalg import orth, eigh
from scipy.stats import uniform, gamma, linregress
from scipy.optimize import brentq
from scipy.integrate import nquad, dblquad
from scipy.special import gamma as Gamma
from .Domains import Rectangle, UnionOfDisjointRectangles
from .LossFunctions import PolynomialLossFunction
from .DualAveraging import ExponentialPotential, IdentityPotential, CompositeOmegaPotential
import ContNoRegret
from scipy.interpolate.fitpack2 import RectBivariateSpline


# def create_random_Sigmas(n, N, lambdamin, dist):
#     """ Creates N random nxn covariance matrices s.t. the Lipschitz 
#         constants are uniformly bounded by L. Here dist is a 'frozen' 
#         scipy.stats probability distribution supported on R+"""
#     # compute lower bound for eigenvalues
# #     lambdamin = 2
# #     gammamax = np.sqrt(lambdamin*np.e*L**2)
#     Sigmas = []
#     for i in range(N):
#         # create random orthonormal matrix
#         V = orth(np.random.uniform(size=(n,n)))   
#         # create n random eigenvalues from the distribution and shift them by lambdamin
#         lambdas = lambdamin + dist.rvs(n)
#         Sigma = np.zeros((n,n))
#         for lbda, v in zip(lambdas, V):
#             Sigma = Sigma + lbda*np.outer(v,v)
#         Sigmas.append(Sigma)
#     return np.array(Sigmas)

def create_random_gammas(covs, Lbnd, dist=uniform()):
    """ Creates a random scaling factor gamma for each of the covariance matrices
        in the array like 'covs', based on the Lipschitz bound L. Here dist is a 'frozen' 
        scipy.stats probability distribution supported on [0,1] """
    # compute upper bound for scaling
    gammas = np.zeros(covs.shape[0])
    for i,cov in enumerate(covs):
        lambdamin = eigh(cov, eigvals=(0,0), eigvals_only=True)
        gammamax = np.sqrt(lambdamin*np.e)*Lbnd
        gammas[i] = gammamax*dist.rvs(1)
    return gammas


def create_random_Sigmas(n, N, L, M, dist=gamma(2, scale=2)):
    """ Creates N random nxn covariance matrices s.t. the Lipschitz 
        constants are uniformly bounded by Lbnd. Here dist is a 'frozen' 
        scipy.stats probability distribution supported on R+"""
    # compute lower bound for eigenvalues
    lambdamin = ((2*np.pi)**n*np.e*L**2/M**2)**(-1.0/(n+1))
    Sigmas = []
    for i in range(N):
        # create random orthonormal matrix
        V = orth(np.random.uniform(size=(n,n)))   
        # create n random eigenvalues from the distribution and shift them by lambdamin
        lambdas = lambdamin + dist.rvs(n)
        Sigma = np.zeros((n,n))
        for lbda, v in zip(lambdas, V):
            Sigma = Sigma + lbda*np.outer(v,v)
        Sigmas.append(Sigma)
    return np.array(Sigmas)


def create_random_Q(domain, mu, L, M, dist=uniform()):
    """ Creates random nxn covariance matrix s.t. the Lipschitz constant of the resulting 
        quadratic function is bounded Lbnd. Here M is the uniform bound on the maximal loss 
        and dist is a 'frozen' scipy.stats probability  distribution supported on [0,1] """
    n = domain.n
    # compute upper bound for eigenvalues
    Dmu = domain.compute_Dmu(mu)
    lambdamax = np.min((L/Dmu, 2*M/Dmu**2))
    # create random orthonormal matrix
    V = orth(np.random.uniform(size=(n,n)))   
    # create n random eigenvalues from the distribution  dist and 
    # scale them by lambdamax
    lambdas = lambdamax*dist.rvs(n)
    Q = np.zeros((n,n))
    for lbda, v in zip(lambdas, V):
        Q = Q + lbda*np.outer(v,v)
    return Q

def create_random_Cs(covs, dist=uniform()):
    """ Creates a random offset C for each of the covariance matrices in the 
        array-like 'covs'. Here dist is a 'frozen' scipy.stats probability 
        distribution supported on [0,1] """
    C = np.zeros(covs.shape[0])
    for i,cov in enumerate(covs):
        pmax = ((2*np.pi)**covs.shape[1]*np.linalg.det(cov))**(-0.5)
        C[i] = np.random.uniform(low=pmax, high=1+pmax)
    return C

# def compute_C(volS, n):
#     """ Computes the constant C = vol(S)/vol(B_1), where B_1 is the unit ball in R^n """
#     volB1 = np.pi**(n/2)/Gamma(n/2+1)
#     return volS/volB1

def compute_etaopt(dom, M, T):
    """ Computes the optimal learning rate for known time horizon T"""
    return np.sqrt(8*(dom.n*np.log(T) - np.log(dom.v))/T)/M
    
def regret_bound_const(dom, eta, T, L, M):
    """ Computes the bound for the time-average regret for constant learning rates """
    diameter = dom.compute_parameters()[0]
    return M**2*eta/8 + L*diameter/T + (dom.n*np.log(T) - np.log(dom.v))/eta/T
       
def regret_bounds(dom, theta, alpha, L, M, Tmax, algo='hedge'):
    """ Computes vector of regret bounds for t=1,...,Tmax """
    diameter = dom.compute_parameters()[0]
    t = np.arange(Tmax) + 1
    if algo == 'hedge':
        return (M**2*theta/8/(1-alpha)/(t**alpha) + L*diameter/t 
                        + (dom.n*np.log(t) - np.log(dom.v))/(theta*t**(1-alpha)))
    elif algo == 'greedy':
        return diameter**2/2/(t**(1-alpha)) + L**2/2/(1-alpha)/(t**alpha)
       
def estimate_loglog_slopes(tsavg_regret, N):
    """ Estimates slope, intercept and r_value of the asymptotic log-log plot
        for each element f tsavg_regert, using the N last data points """
    slopes, intercepts, r_values = [], [], []
    for regret in tsavg_regret:
        T = np.arange(len(regret)-N, len(regret))
        Y = regret[len(regret)-N:]
        slope, intercept, r_value, p_value, std_err = linregress(np.log(T), np.log(Y))
        slopes.append(slope), intercepts.append(intercept), r_values.append(r_value)
    return np.array(slopes), np.array(intercepts), np.array(r_values)
  

def plot_results(results, offset=500, filename=None, show=True):
        """ Plots and shows or saves (or both) the simulation results """
        # set up figures
        plt.figure(1)
        plt.title('cumulative regret, {} losses'.format(results[0].desc))
        plt.xlabel('t')
        plt.figure(2)
        plt.title('time-avg. cumulative regret, {} losses'.format(results[0].desc))
        plt.xlabel('t')
        plt.figure(3)
        plt.title(r'log time-avg. cumulative regret, {} losses'.format(results[0].desc))
        plt.xlabel('t')   
        # and now plot, depending on what data is there
        for result in results:
            if result.etaopts:
                for i,(T,eta) in enumerate(result.etaopts.items()):
                    plt.figure(1)
                    lavg = plt.plot(result.regs_etaopts['savg'][i][0:T], linewidth=2.0, 
                                    label=result.algo+r' $\eta_{{opt}}(T={0:.1e}) = {1:.3f}$'.format(T, eta), rasterized=True)
                    plt.plot(np.arange(T,result.problem.T), result.regs_etaopts['savg'][i][T:], '--', 
                             color=lavg[0].get_color(), linewidth=2, rasterized=True)
                    plt.fill_between(np.arange(result.problem.T), result.regs_etaopts['perc_10'][i], 
                                     result.regs_etaopts['perc_90'][i], color=lavg[0].get_color(), alpha=0.2, rasterized=True)
                    plt.figure(2)
                    ltavg = plt.plot(np.arange(offset,T), result.regs_etaopts['tsavg'][i][offset:T], linewidth=2.0, 
                                     label=result.algo+r' $\eta_{{opt}}(T={0:.1e}) = {1:.3f}$'.format(T, eta), rasterized=True)
                    plt.plot(np.arange(T,result.problem.T), result.regs_etaopts['tsavg'][i][T:], '--', 
                             color=ltavg[0].get_color(), linewidth=2, rasterized=True)
                    plt.fill_between(np.arange(offset,result.problem.T), result.regs_etaopts['tavg_perc_10'][i][offset:], 
                                     result.regs_etaopts['tavg_perc_90'][i][offset:], color=ltavg[0].get_color(), alpha=0.2, rasterized=True)
                    plt.xlim((0, result.problem.T))
                    plt.figure(3)
                    llogtavg = plt.plot(np.arange(1,result.problem.T+1), result.regs_etaopts['tsavg'][i], 
                                        linewidth=2.0, label=result.algo+r' $\eta_{{opt}}(T={0:.1e}) = {1:.3f}$'.format(T, eta), rasterized=True)
                    plt.fill_between(np.arange(1,result.problem.T+1), result.regs_etaopts['tavg_perc_10'][i], 
                                    result.regs_etaopts['tavg_perc_90'][i], color=llogtavg[0].get_color(), alpha=0.2, rasterized=True)
                    plt.plot(np.arange(1,result.problem.T+1), result.regs_etaopts['tsavgbnd'][i], '--', 
                             color=llogtavg[0].get_color(), linewidth=2, rasterized=True)
            if result.etas:
                for i,eta in enumerate(result.etas):
                    plt.figure(1)
                    lavg = plt.plot(result.regs_etas['savg'][i], linewidth=2.0, label=result.algo+r' $\eta = {0:.3f}$'.format(eta), rasterized=True)
                    plt.fill_between(np.arange(result.problem.T), result.regs_etas['perc_10'][i], 
                                     result.regs_etas['perc_90'][i], color=lavg[0].get_color(), alpha=0.15, rasterized=True)
                    plt.figure(2)
                    ltavg = plt.plot(np.arange(offset,result.problem.T), result.regs_etas['tsavg'][i][offset:], 
                                     linewidth=2.0, label=result.algo+r'$\eta = {0:.3f}$'.format(eta), rasterized=True)
                    plt.fill_between(np.arange(offset,result.problem.T), result.regs_etas['tavg_perc_10'][i][offset:], 
                                     result.regs_etas['tavg_perc_90'][i][offset:], color=ltavg[0].get_color(), alpha=0.15, rasterized=True)
                    plt.xlim((0, result.problem.T))
                    plt.figure(3)
                    llogtavg = plt.plot(np.arange(1,result.problem.T+1), result.regs_etas['tsavg'][i], linewidth=2.0, label=result.algo+r' $\eta = {0:.3f}$'.format(eta), rasterized=True)
                    plt.fill_between(np.arange(1,result.problem.T+1), result.regs_etas['tavg_perc_10'][i], 
                                     result.regs_etas['tavg_perc_90'][i], color=llogtavg[0].get_color(), alpha=0.15, rasterized=True) 
                    plt.plot(np.arange(1,result.problem.T+1), result.regs_etas['tsavgbnd'][i], '--', color=llogtavg[0].get_color(), linewidth=2, rasterized=True)     
            if result.alphas:
                for i,alpha in enumerate(result.alphas):
                    plt.figure(1)
                    lavg = plt.plot(result.regs_alphas['savg'][i], linewidth=2.0,
                             label=result.algo+r' $\eta_t = {0} \cdot t^{{{1}}}$'.format(result.thetas[i], -alpha), rasterized=True)
                    plt.fill_between(np.arange(result.problem.T), result.regs_alphas['perc_10'][i], 
                                     result.regs_alphas['perc_90'][i], color=lavg[0].get_color(), alpha=0.15, rasterized=True)
                    plt.figure(2)
                    ltavg = plt.plot(np.arange(result.problem.T)[offset:], result.regs_alphas['tsavg'][i][offset:], linewidth=2.0, 
                             label=result.algo+r' $\eta_t = {0} \cdot t^{{{1}}}$'.format(result.thetas[i], -alpha), rasterized=True)
                    plt.fill_between(np.arange(offset,result.problem.T), result.regs_alphas['tavg_perc_10'][i][offset:], 
                                     result.regs_alphas['tavg_perc_90'][i][offset:], color=ltavg[0].get_color(), alpha=0.15, rasterized=True)
                    plt.xlim((0, result.problem.T)) 
                    plt.figure(3)
                    lltsavg = plt.plot(np.arange(1,result.problem.T+1), result.regs_alphas['tsavg'][i], linewidth=2.0, 
                                         label=result.algo+r' $\eta_t = {0} \cdot t^{{{1}}}$'.format(result.thetas[i], -alpha), rasterized=True)
                    plt.fill_between(np.arange(1,result.problem.T+1), result.regs_alphas['tavg_perc_10'][i], 
                                    result.regs_alphas['tavg_perc_90'][i], color=lltsavg[0].get_color(), alpha=0.15, rasterized=True)
                    plt.plot(np.arange(1,result.problem.T+1), result.regs_alphas['tsavgbnd'][i], '--', color=lltsavg[0].get_color(), linewidth=2.0, rasterized=True) 
        # make plots pretty and show legend
        plt.figure(1)
        plt.legend(loc='upper left', prop={'size':13}, frameon=False) 
        plt.figure(2)
        plt.legend(loc='upper right', prop={'size':13}, frameon=False) 
        plt.figure(3)
        plt.yscale('log'), plt.xscale('log')
        plt.legend(loc='upper right', prop={'size':13}, frameon=False) 
        if filename:
            plt.figure(1)
            plt.savefig(filename + '_{}_cumloss.pdf'.format(results[0].desc), bbox_inches='tight', dpi=300)
            plt.figure(2)
            plt.savefig(filename + '_{}_tavgloss.pdf'.format(results[0].desc), bbox_inches='tight', dpi=300)
            plt.figure(3)
            plt.savefig(filename + '_{}_loglogtavgloss.pdf'.format(results[0].desc), bbox_inches='tight', dpi=300)
        if show:
            plt.show()
            

def parse_regrets(reg_results, regrets, prob, theta, alpha, algo='hedge'):
    """ Function that computes some aggregate information from the raw regret 
        samples in the list 'regrets' """ 
    reg_results['savg'].append(np.average(regrets, axis=0))
    reg_results['perc_10'].append(np.percentile(regrets, 10, axis=0))
    reg_results['perc_90'].append(np.percentile(regrets, 90, axis=0))
    reg_results['tsavg'].append(reg_results['savg'][-1]/(1+np.arange(prob.T)))
    reg_results['tavg_perc_10'].append(reg_results['perc_10'][-1]/(1+np.arange(prob.T)))
    reg_results['tavg_perc_90'].append(reg_results['perc_90'][-1]/(1+np.arange(prob.T)))
    reg_results['tsavgbnd'].append(regret_bounds(prob.domain, theta, alpha, 
                                          prob.Lbnd, prob.M, prob.T, algo=algo))
    return reg_results
        


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
#     def ftest():
#         return f(-1)
#     nbr = 10
#     print('Time passed for {} numerical integrations: {}'.format(nbr, timeit.timeit(ftest, number=nbr)))
    a = (0.001 - gamma/(gamma-1))/eta - c # this is a lower bound on nustar
    nustar = brentq(f, a, nu_prev)#, full_output=True)
    dlclose(lib._handle) # this is to release the lib, so we can import the new version
    os.remove('tmpfunc.c') # clean up
    os.remove('tmpfunc.dylib') # clean up
    return nustar

#     phi = lambda s1,s2,nu: (gamma/(gamma-1) + eta*(0.5*np.dot(np.dot(Q,[s1-mu[0],s2-mu[1]]),[s1-mu[0],s2-mu[1]]) + c + nu))**(-gamma)    
#     ranges = [(dom.lb[0], dom.ub[0]), (dom.lb[1], dom.ub[1])]
#     f = lambda nu: nquad(phi, ranges, args=[nu])[0] - 1
#     a = (0.001 - gamma/(gamma-1))/eta - c
#     return brentq(f, a, nu_prev)#, full_output=True)

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
#     def ftest():
#         return f(-1)
#     nbr = 10
#     print('Time passed for {} numerical integrations: {}'.format(nbr, timeit.timeit(ftest, number=nbr)))
    knots = Lspline.get_knots()
    Lmax = Lspline.ev(knots[0], knots[1]).max()
    a = -1/eta*potential.phi_inv(1/dom.volume) - Lmax
    print('search interval: [{},{}]'.format(a,nu_prev))
    (nustar, res) = brentq(f, a, nu_prev, full_output=True)
    print('iterations: {},  function calls: {}'.format(res.iterations, res.function_calls))
    return nustar


def nustar_polynomial(dom, potential, eta, Loss, nu_prev=1000):
    """ Determines the normalizing nustar for the dual-averaging 
        update for polynomial loss functions """
    if isinstance(dom, ContNoRegret.Domains.nBox):
        ranges = [dom.bounds]
    elif isinstance(dom, ContNoRegret.Domains.UnionOfDisjointnBoxes):
        ranges = [nbox.bounds for nbox in dom.nboxes]
    else:
        raise Exception('For now, domain must be an nBox or a UnionOfDisjointnBoxes!')
    with open('libs/tmplib.c', 'w') as file:
        file.writelines(generate_ccode(dom, potential, eta, Loss))
    call(['gcc', '-shared', '-o', 'libs/tmplib.dylib', '-fPIC', 'libs/tmplib.c'])
    lib = ctypes.CDLL('libs/tmplib.dylib')
    lib.phi.restype = ctypes.c_double
    lib.phi.argtypes = (ctypes.c_int, ctypes.c_double)
    f = lambda nu: np.sum([nquad(lib.phi, rng, [nu])[0] for rng in ranges]) - 1
    a = -Loss.bounds[1] - potential.phi_inv(1/dom.volume)/eta # this is (coarse) lower bound on nustar
    nustar = brentq(f, a, nu_prev)
    dlclose(lib._handle) # this is to release the lib, so we can import the new version
    os.remove('libs/tmplib.c') # clean up
    os.remove('libs/tmplib.dylib') # clean up
    return nustar


def generate_ccode(dom, potential, eta, Loss):
    """ Generates the c source code that is complied and used for faster numerical 
        integration (using ctypes). Hard-codes known parameters (except s and nu) as
        literals and returns a list of strings that are the lines of a C source file. """
    header = ['#include <math.h>\n\n',
              'double eta = {};\n'.format(eta),
              'double c[{}] = {{{}}};\n'.format(Loss.m, ','.join(str(coeff) for coeff in Loss.coeffs)),
              'double e[{}] = {{{}}};\n\n'.format(Loss.m*dom.n, ','.join(str(xpnt) for xpntgrp in Loss.exponents for xpnt in xpntgrp))]
    poly = ['double phi(int n, double args[n]){\n',
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
    if isinstance(potential, ExponentialPotential):   
        return header + poly + ['   return exp(-eta*(loss + nu));}']
    elif isinstance(potential, IdentityPotential):
        return header + poly + ['   return -eta*(loss + nu);}']
    elif isinstance(potential, CompositeOmegaPotential):
        omega_pot = ['   double z = -eta*(loss + nu);\n',
                     '   if(z<{}){{\n'.format(potential.c),
                     '     return pow({}-z, -{});}}\n'.format(potential.gamma*potential.c, potential.gamma),
                     '   else{\n',
                     '     return {} + {}*z + {}*pow(z,2);}}\n'.format(*[a for a in potential.a]),
                     '   }']
        return header + poly + omega_pot

