'''
Basic Algorithms for the Continuous No-Regret Problem.

@author: Maximilian Balandat
@date Apr 30, 2015
'''

import numpy as np
from .Distributions import Uniform, Gaussian, MixtureDistribution, ExpifiedMixture
from .LossFunctions import QuadraticLossFunction, AffineLossFunction, PolynomialLossFunction
from .Domains import UnionOfDisjointRectangles
from .utils import compute_etaopt, parse_regrets
from .DualAveraging import nustar_quadratic, nustar_generic, compute_nustar
from scipy.stats import linregress
from scipy.interpolate import RectBivariateSpline
from cvxopt import solvers, matrix
  

class ContNoRegretProblem(object):
    """ Basic class describing a Continuous No-Regret problem. This implementation for now
        assumes that the loss functions have "distr" class variable that is a Distribution object """
    
    def __init__(self, domain, lossfuncs, Lbnd, M, desc='nodesc'):
        """ Constructor for the basic problem class. Here lossfuncs is a 
            list of loss LossFunction objects. """
        self.domain, self.Lbnd, self.M = domain, Lbnd, M
        self.lossfuncs = lossfuncs
        self.T = len(lossfuncs)
        self.optaction, self.optval = None, None
        self.desc = desc
            
    def compute_regrets(self, losses):    
        """ Computes the regrets (for each time step) for the given array of loss sequences """
        if not self.optval: 
            self.optval = self.compute_optimum()
        return (np.sum(losses, axis=1) - self.optval)/self.T
    
    def regret_bound(self):
        """ Computes the bound on the time-average regret """
        return NotImplementedError
        # self.Lbnd*self.domain.diameter/self.T + np.sqrt(self.domain.n/2.0*np.log(self.domain.diameter/self.domain.epsilon*self.T)/self.T)
    
    def cumulative_loss(self, points):
        """ Computes the cumulative loss at the given points """
        loss = np.zeros((points.shape[0], 1))
        for lossfunc in self.lossfuncs:
            loss = loss + lossfunc.val(points)
        return loss
    
    def compute_optimum(self):
        """ Computes the optimal decision in hindsight and the associated overall loss """
        raise NotImplementedError
        
    def compute_eta_opt(self):
        """ Computes the optimal learning rate eta """
        return NotImplementedError
        #np.sqrt(8*self.domain.n*np.log(self.domain.diameter/self.domain.epsilon*self.T)/self.T)

    def plot_regrets(self, savg_regret_opt, tsavg_regret_opt, 
                     savg_regret, tsavg_regret, alphas, thetas):
        """ Plots the results of a simulation of algorithm """
        raise NotImplementedError

    def run_simulation(self, N, algo, Ngrid=100000, label='nolabel', **kwargs):
        """ Runs the no-regret algorithm for different parameters and returns the
            results as a 'Result' object. Accepts optimal constant rates in the 
            dictionary 'etaopts', constant rates in the array-like 'etas', and 
            time-varying rates with parameters in the array-like 'alphas', 'thetas' """
        result_args = {}
        if 'etaopts' in kwargs:
            regs_etaopts = {'savg':[], 'tsavg':[], 'tsavgbnd':[], 'perc_10':[], 
                            'perc_90':[], 'tavg_perc_10':[], 'tavg_perc_90':[]}
            for T,eta in kwargs['etaopts'].items():
                print('Simulating eta={}'.format(eta))
                regrets = self.simulate(N, etas=eta*np.ones(self.T), algo=algo, Ngrid=Ngrid, **kwargs)[2]
                parse_regrets(regs_etaopts, regrets, self, eta, 0, algo)
                result_args['regs_etaopts'] = regs_etaopts
        else:
            regs_etaopts = None
        if 'etas' in kwargs:
            regs_etas = {'savg':[], 'tsavg':[], 'tsavgbnd':[], 'perc_10':[], 
                         'perc_90':[], 'tavg_perc_10':[], 'tavg_perc_90':[]}
            for eta in kwargs['etas']:
                print('Simulating eta={}'.format(eta))
                regrets = self.simulate(N, etas=eta*np.ones(self.T), algo=algo, Ngrid=Ngrid, **kwargs)[2]
                parse_regrets(regs_etas, regrets, self, eta, 0, algo)
                result_args['etas'] = kwargs['etas']
                result_args['regs_etas'] = regs_etas
        else:
            regs_etas = None
        if 'alphas' in kwargs:
            regs_alphas = {'savg':[], 'tsavg':[], 'tsavgbnd':[], 'perc_10':[], 
                           'perc_90':[], 'tavg_perc_10':[], 'tavg_perc_90':[]}
            for alpha,theta in zip(kwargs['alphas'], kwargs['thetas']): # run for Nloss different sequences of loss functions
                print('Simulating alpha={}, theta={}'.format(alpha, theta))
                regrets = self.simulate(N, etas=theta*(1+np.arange(self.T))**(-alpha), algo=algo, Ngrid=Ngrid, **kwargs)[2]
                parse_regrets(regs_alphas, regrets, self, theta, alpha, algo)
                result_args['alphas'] = kwargs['alphas']
                result_args['thetas'] = kwargs['thetas']
                result_args['regs_alphas'] = regs_alphas
        else:
            regs_alphas = None
        # now return the results as a Results object
        return Results(self, label=label, **result_args)
    
    
    def simulate(self, N, etas='opt', algo='DA', Ngrid=100000, **kwargs):
        """ Simulates the result of running the No-Regret algorithm (N times).
            Returns a list of sequences of decisions and associated losses, one for each run. 
            The grid is used for computing both the regret and the actions! """
        if etas == 'opt': # use optimal choice of etas
            etas = compute_etaopt(self.domain, self.M, self.T)*np.ones(self.T)
        if algo in ['ONS', 'FAL', 'EWOO']:
            alpha = kwargs['alpha']
            beta = 0.5*np.minimum(1/4/self.Lbnd/self.domain.diameter, alpha)
            epsilon = 1/beta**2/self.domain.diameter**2
            if algo == 'EWOO':
                Nexp = kwargs['Nexp']
        elif algo == 'DA_Quad':
            gamma = kwargs['gamma']
        elif algo == 'DA_generic':
            zero_pot = kwargs['potential']
            ngrid = kwargs['ngrid']
            # create x and y values for uniform grid
            bbox = self.domain.bbox()
            x = np.linspace(bbox.lb[0], bbox.ub[0], ngrid[0])
            y = np.linspace(bbox.lb[1], bbox.ub[1], ngrid[1])
        elif algo == 'DA':
            pot = kwargs['potential']
        actions, losses, cumloss, regrets = [], [], [], []
        gridpoints = self.domain.grid(Ngrid)
        approxL = np.zeros(gridpoints.shape[0])
        for t, lossfunc in enumerate(self.lossfuncs): 
            if t % 25 == 0:
                print(str(kwargs['pid']) + ': ', t)
            if algo == 'hedge':
                etaL = etas[t]*approxL
                # to ensure numerical stability, we normalize the weights (since we will compute the exponential,
                # additive constants will cancel out in computing the normalization!
                weights = np.exp(-(etaL - np.min(etaL))) 
                action = gridpoints[np.random.choice(weights.shape[0], size=N, p=weights/np.sum(weights))]
            elif algo == 'greedy':
                if t == 0:
                    action = self.domain.sample_uniform(N) # pick arbitrary action in the first step, may as well sample
                else:
                    action = self.lossfuncs[t-1].proj_gradient(actions[-1], etas[t]) # do a projected gradient step
            elif algo == 'ONS': # Online Newton Step
                if t == 0: 
                    action = self.domain.sample_uniform(N) # pick arbitrary action in the first step, may as well sample
                    grad = lossfunc.grad(action)
                    A = np.einsum('ij...,i...->ij...', grad, grad) + epsilon*np.array([np.eye(2),]*N)
                    Ainv = np.array([np.linalg.inv(mat) for mat in A])
                else:
                    points = actions[-1] - np.einsum('ijk...,ik...->ij...', Ainv, grad)/beta
                    action, dist = self.domain.gen_project(points, A)
                    grad = lossfunc.grad(action)
                    A = A + np.einsum('ij...,i...->ij...', grad, grad)
                    z = np.einsum('ijk...,ik...->ij...', Ainv, grad)
                    Ainv = Ainv - np.einsum('ij...,i...->ij...', z, z)/(1 + np.einsum('ij,ij->i',points,z))[:,np.newaxis,np.newaxis]
            elif algo == 'FAL':
                if t == 0: 
                    action = self.domain.sample_uniform(N) # pick arbitrary action in the first step, may as well sample
                    grad = lossfunc.grad(action)
                    A = np.einsum('ij...,i...->ij...', grad, grad)
                    b = grad*(np.einsum('ij,ij->i', grad, action) - 1/beta)[:,np.newaxis]
                    Ainv = np.array([np.linalg.pinv(mat) for mat in A]) # so these matrices are singular... what's the issue?
                else:
                    points = np.einsum('ijk...,ik...->ij...', Ainv, b) 
                    action, dist = self.domain.gen_project(points, A)
                    grad = lossfunc.grad(action)
                    A = A + np.einsum('ij...,i...->ij...', grad, grad)
                    b = b + grad*(np.einsum('ij,ij->i', grad, action) - 1/beta)[:,np.newaxis]
                    z = np.einsum('ijk...,ik...->ij...', Ainv, grad)
                    Ainv = Ainv - np.einsum('ij...,i...->ij...', z, z)/(1 + np.einsum('ij,ij->i',points,z))[:,np.newaxis,np.newaxis]
            elif algo == 'EWOO': 
                if t == 0:
                    if not self.domain.isconvex():
                        raise Exception('EWOO algorithm only makes sense if the domain is convex!')
                    action = self.domain.sample_uniform(N)
                else:
                    # if the domain is convex we can just approximate the expectation through sampling.
                    etaL = kwargs['alpha']*approxL
                    # to ensure numerical stability, we normalize the weights (since we will compute the exponential,
                    # additive constants will cancel out in computing the normalization!
                    weights = np.exp(-(etaL - np.min(etaL)))
                    exp_approx = np.sum(gridpoints[np.random.choice(weights.shape[0], size=Nexp, p=weights/np.sum(weights))], axis=0)/Nexp
                    # the action is the same for all samples after the first observation....
                    action = np.array([exp_approx,]*N)
            elif algo == 'DA_Quad':
                if t == 0:
                    if not isinstance(lossfunc, QuadraticLossFunction):
                        raise Exception('For now only accept quadratic losses for dual-averaging algorithm!')
                    Qcpd, bcpd, ccpd = np.zeros(lossfunc.Q.shape), np.zeros(lossfunc.Q.shape[0]), 0.0
                    nustar = 1000 # this is needed for warm-starting the intervals of root-finder
                    action = Uniform(self.domain).sample(N)
                else:
                    mu = -np.linalg.solve(Qcpd, bcpd)
                    nustar = nustar_quadratic(self.domain, gamma, etas[t], Qcpd, mu, ccpd, nustar+0.15)
                    L = QuadraticLossFunction(self.domain, mu, Qcpd, ccpd)
                    weights = (gamma/(gamma-1) + etas[t]*(L.val(gridpoints) + nustar))**(-gamma)
                    action = gridpoints[np.random.choice(weights.shape[0], size=N, p=weights/np.sum(weights))]
                # update matrices
                Qcpd = Qcpd + lossfunc.Q
                bcpd = bcpd - np.dot(lossfunc.Q, lossfunc.mu)
                ccpd = ccpd + 0.5*(2*lossfunc.c + np.dot(lossfunc.mu, np.dot(lossfunc.Q, lossfunc.mu))) 
            elif algo == 'DA_generic':
                if t == 0:
                    nustar = 1000 # this is needed for warm-starting the intervals of root-finder
                    Lgrid = lossfunc.val_grid(x,y)
                    action = Uniform(self.domain).sample(N)
                else:
                    # create a spline approximation of the integrand
                    Lspline = RectBivariateSpline(x,y,Lgrid)
                    nustar = nustar_generic(self.domain, zero_pot, etas[t], Lspline, nustar+0.15)
                    weights = zero_pot.phi(-etas[t]*(approxL + nustar))
                    action = gridpoints[np.random.choice(weights.shape[0], size=N, p=weights/np.sum(weights))]
                    Lgrid = Lgrid + lossfunc.val_grid(x,y)
            elif algo == 'DA':
                if t == 0:
                    if isinstance(lossfunc, AffineLossFunction):
                        cumLoss = AffineLossFunction(self.domain, np.zeros(self.domain.n), 0)
                    elif isinstance(lossfunc, QuadraticLossFunction):
                        cumLoss = QuadraticLossFunction(self.domain, np.zeros(self.domain.n),
                                                        np.zeros(self.domain.n, self.domain.n), 0)
                    elif isinstance(lossfunc, PolynomialLossFunction):
                        cumLoss = PolynomialLossFunction(self.domain, [0], [0]*self.domain.n)
                    else:
                        raise Exception('For now DualAveraging allows only Affine, Quadratic or Polynomial loss functions')
                    cumLoss.set_bounds([0, 0])
                    # compute  warm-starting the intervals of root-finder
                    nustar = -1/etas[t]*pot.phi_inv(1/self.domain.volume)
                    action = self.domain.sample_uniform(N)
                else:
                    # create a spline approximation of the integrand
                    cumLoss = cumLoss + lossfunc
                    nustar = compute_nustar(self.domain, pot, etas[t], cumLoss, nustar, id=kwargs['pid'])
                    weights = np.maximum(pot.phi(-etas[t]*(approxL + nustar)), 0)
#                     print(np.max(weights)/np.average(weights))
                    action = gridpoints[np.random.choice(weights.shape[0], size=N, p=weights/np.sum(weights))]
            # now store the actions, losses, etc.
            actions.append(action)
            loss = lossfunc.val(action)
            losses.append(loss)
            if t == 0:
                cumloss.append(loss)
            else:
                cumloss.append(cumloss[-1] + loss)
            # compute and append regret -- resort to gridding for now
            approxL += lossfunc.val(gridpoints)
            optval = np.min(approxL)
            regrets.append(cumloss[-1] - optval)
        return np.transpose(np.array(actions), (1,0,2)), np.transpose(np.array(losses)), np.transpose(np.array(regrets))
    
    
    
class GaussianNoRegretProblem(ContNoRegretProblem):
    """ Continuous No-Regret problem for Gaussian loss functions """
    
    def __init__(self, domain, lossfuncs, Lbnd, M):
        """ Constructor for the basic problem class. Here lossfuncs is a 
            list of loss LossFunction objects. """
        super(GaussianNoRegretProblem, self).__init__(domain, lossfuncs, Lbnd, M)
        self.desc = 'Gaussian'
         
    def simulate_nogrid(self, N, etas='opt', Ngrid=100000):
        """ Simulates the result of running the No-Regret algorithm (N times).
            Returns a list of sequences of decisions and associated losses, one for each run.
            The grid here is only used to compute the optimal decisions, not the actions """
        if etas == 'opt': # use optimal choice of etas
            etas = compute_etaopt(self.domain, self.M, self.T)*np.ones(self.T)
        actions, losses, cumloss, regrets = [], [], [], []
        gridpoints = self.domain.grid(Ngrid)
        approxL = np.zeros(gridpoints.shape[0])
        for t, lossfunc in enumerate(self.lossfuncs): 
            print(t)
            if t == 0: # start off with a uniform draw from the domain, create mixture distr
                action = Uniform(self.domain).sample(N)
                mixdistr = MixtureDistribution([lossfunc.p], [1])
            else: # after first observation sample from mixture distribution
                action = ExpifiedMixture(mixdistr, etas[t]).sample(N)
                mixdistr.append(lossfunc.p, 1)
            actions.append(action)
            loss = lossfunc.val(action)
            losses.append(loss)
            if t == 0:
                cumloss.append(loss)
            else:
                cumloss.append(cumloss[-1] + loss)
            # compute and append regret -- resort to gridding for now
            approxL += lossfunc.val(gridpoints)
            optval = np.min(approxL)
            regrets.append(cumloss[-1] - optval)
        return np.transpose(np.array(actions), (1,0,2)), np.transpose(np.array(losses)), np.transpose(np.array(regrets))
        
        
class QuadraticNoRegretProblem(ContNoRegretProblem):
    """ Continuous No-Regret problem for Quadratic loss functions """
    
    def __init__(self, domain, quadlosses, Lbnd, M):
        """ Constructor for the basic problem class. Here lossfuncs is a 
            list of loss LossFunction objects. """
        self.domain, self.Lbnd, self.M = domain, Lbnd, M
        self.lossfuncs = quadlosses
        self.T = len(quadlosses)
        self.optaction, self.optval = None, None
        self.desc = 'Quadratic'
         
    def simulate_nogrid(self, N, etas='opt', Ngrid=None):
        """ Simulates the result of running the No-Regret algorithm (N times).
            Returns a list of sequences of decisions and associated losses, one for each run. """
        if etas == 'opt': # use optimal choice of etas
            etas = compute_etaopt(self.domain, self.M, self.T)*np.ones(self.T)
        actions, losses, cumloss, regrets = [], [], [], []
        Qcpd, bcpd, ccpd = np.zeros(self.lossfuncs[0].Q.shape), np.zeros(self.lossfuncs[0].Q.shape[0]), 0.0
        Qtilde, btilde = np.zeros(self.lossfuncs[0].Q.shape), np.zeros(self.lossfuncs[0].Q.shape[0])
        mutilde = None
        for t, lossfunc in enumerate(self.lossfuncs): 
            print(t)
            if t == 0: # start off with a uniform draw from the domain, create mixture distr
                action = Uniform(self.domain).sample(N)
            else: # after first observation sample from mixture distribution
                action = Gaussian(self.domain, mutilde, np.linalg.inv(Qtilde)).sample(N)
            # update matrices
            Qcpd = Qcpd + lossfunc.Q
            bcpd = bcpd - np.dot(lossfunc.Q, lossfunc.mu)
            ccpd = ccpd + 0.5*(2*lossfunc.c + np.dot(lossfunc.mu, np.dot(lossfunc.Q, lossfunc.mu)))
            Qtilde, btilde, ctilde = etas[t]*Qcpd, etas[t]*bcpd, etas[t]*ccpd
            mutilde = -np.linalg.solve(Qtilde, btilde)
            # append action and loss, this could be dome more efficiently
            actions.append(action)
            loss = lossfunc.val(action)
            losses.append(loss)
            if t == 0:
                cumloss.append(loss)
            else:
                cumloss.append(cumloss[-1] + loss)
            # compute and append regret
            optval = self.compute_minimum(Qcpd, bcpd, ccpd)
            regrets.append(cumloss[-1] - optval)
        return np.transpose(np.array(actions), (1,0,2)), np.transpose(np.array(losses)), np.transpose(np.array(regrets))        
    
    def compute_minimum(self, Q, b, c):
        mucpd = -np.linalg.solve(Q, b)
        if np.all(self.domain.iselement(np.array([mucpd], ndmin=2))):
            return c - 0.5*np.dot(b.T, np.linalg.solve(Q, b))
        else:
            if isinstance(self.domain, UnionOfDisjointRectangles):
                print('CVXOPT CALLED!')
                minval = np.Inf
                solvers.options['show_progress'] = False
                Qcvx = matrix(Q, tc='d')
                bcvx = matrix(b, tc='d')
                Gcvx = matrix([[1,-1,0,0], [0,0,1,-1]], tc='d')
                for rect in self.domain.rects:
                    hcvx = matrix([rect.ub[0], -rect.lb[0], rect.ub[1], -rect.lb[1]], tc='d')
                    res = solvers.qp(Qcvx, bcvx, Gcvx, hcvx)
                    minval = min(minval, res['primal objective'])
                print('CVXOPT SUCCESSFUL with return value {} and c= {}!'.format(minval,c))
                return minval + c
            else:
                raise NotImplementedError('Minimization on general non-convex sets not implemented yet')
    
    def compute_optimum(self):
        """ Computes the optimal decision in hindsight and the associated overall loss """
        # the minimum is achieved at the compound mean and the value is the compound offset
        Qcpd = np.sum(np.array([lossfunc.Q for lossfunc in self.lossfuncs]), axis=0)
        bcpd = -np.sum(np.array([np.dot(lossfunc.Q, lossfunc.mu) for lossfunc in self.lossfuncs]), axis=0)
        ccpd = 0.5*np.sum(np.array([2*lossfunc.c + np.dot(lossfunc.mu.T, np.dot(lossfunc.Q, lossfunc.mu)) 
                                    for lossfunc in self.lossfuncs]), axis=0)      
        return ccpd - 0.5*np.dot(bcpd.T, np.linalg.solve(Qcpd, bcpd))
    
    

class Results(object):
    """ Class for 'result' objects that contain simulation results generated by ContNoRegretProblems """
    
    def __init__(self, problem, **kwargs):
        self.problem = problem
        self.etaopts, self.regs_etaopts = kwargs.get('etaopts'), kwargs.get('regs_etaopts')
        self.etas, self.regs_etas = kwargs.get('etas'), kwargs.get('regs_etas')
        self.alphas, self.thetas, self.regs_alphas = kwargs.get('alphas'), kwargs.get('thetas'), kwargs.get('regs_alphas')
        self.label = kwargs.get('label')
            
    def estimate_loglog_slopes(self, N=1000):
        """ Estimates slope, intercept and r_value of the asymptotic log-log plot
        for each element f tsavg_regert, using the N last data points """
        slopes, slopes_bnd = {}, {}
        if self.etaopts:
            slopes['etaopts'] = self.loglog_slopes(self.regs_etaopts['tsavg'], N)
#             slopes_bnd['etaopts'] = self.loglog_slopes(self.regs_etaopts['tsavgbnd'], N)
        if self.etas:   
            slopes['etas'] = self.loglog_slopes(self.regs_etas['tsavg'], N)
#             slopes_bnd['etas'] = self.loglog_slopes(self.regs_etas['tsavgbnd'], N)
        if self.alphas:
            slopes['alphas'] = self.loglog_slopes(self.regs_alphas['tsavg'], N)
#             slopes_bnd['alphas'] = self.loglog_slopes(self.regs_alphas['tsavgbnd'], N)
        return slopes #, slopes_bnd
        
    def loglog_slopes(self, regrets, N): 
        slopes = []
        for regret in regrets:
            T = np.arange(len(regret)-N, len(regret))
            Y = regret[len(regret)-N:]
            slope = linregress(np.log(T), np.log(Y))[0]
            slopes.append(slope)
        return slopes
    
    