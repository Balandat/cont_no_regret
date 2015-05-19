'''
Basic Algorithms for the Continuous No-Regret Problem.

@author: Maximilian Balandat
@date May 10, 2015
'''

import numpy as np
from .LossFunctions import ZeroLossFunction, AffineLossFunction, ctypes_integrate
from .utils import compute_etaopt 
from .DualAveraging import compute_nustar
from .NLopt import quicksample
from .Domains import nBox, UnionOfDisjointnBoxes, DifferenceOfnBoxes
from .Potentials import ExponentialPotential
from scipy.stats import linregress
  

class ContNoRegretProblem(object):
    """ Basic class describing a Continuous No-Regret problem. """
    
    def __init__(self, domain, lossfuncs, L, M, desc='nodesc'):
        """ Constructor for the basic problem class. Here lossfuncs 
            is a list of loss LossFunction objects. """
        self.domain, self.L, self.M = domain, L, M
        self.lossfuncs = lossfuncs
        self.T = len(lossfuncs)
        self.optaction, self.optval = None, None
        self.desc = desc
        self.data = []
        if domain.n == 2:
            self.pltpoints = self.create_pltpoints(1000)
                   
    def cumulative_loss(self, points):
        """ Computes the cumulative loss at the given points """
        loss = np.zeros((points.shape[0], 1))
        for lossfunc in self.lossfuncs:
            loss = loss + lossfunc.val(points)
        return loss
    
    def create_pltpoints(self, Nplot):
        """ Create a number of points used for plotting the evolution of
            the density function for the DA algorithm """
        if self.domain.n != 2:
            return None
        if isinstance(self.domain, nBox):
            return [self.domain.grid(Nplot)]
        elif isinstance(self.domain, UnionOfDisjointnBoxes):
            weights = np.array([nbox.volume for nbox in self.domain.nboxes])/self.domain.volume
            return [nbox.grid(np.ceil(weight*Nplot)) for nbox,weight in zip(self.domain.nboxes, weights)]
        elif isinstance(self.domain, DifferenceOfnBoxes):
            if len(self.domain.inner) > 1:
                raise Exception('Can only create pltpoints for DifferenceOfnBoxes with single box missing!')
            bnds_inner, bnds_outer = self.domain.inner[0].bounds, self.domain.outer.bounds
            nboxes = [nBox([bnds_outer[0], [bnds_inner[1][1], bnds_outer[1][1]]]),
                      nBox([bnds_outer[0], [bnds_outer[1][0], bnds_inner[1][0]]]),
                      nBox([[bnds_outer[0][0], bnds_inner[0][0]], bnds_inner[1]]),
                      nBox([[bnds_inner[0][1], bnds_outer[0][1]], bnds_inner[1]])]
            weights = np.array([nbox.volume for nbox in nboxes])/self.domain.volume
            return [nbox.grid(np.ceil(weight*Nplot)) for nbox,weight in zip(nboxes, weights)]
            

    def run_simulation(self, N, algo, Ngrid=100000, label='nolabel', **kwargs):
        """ Runs the no-regret algorithm for different parameters and returns the
            results as a 'Result' object. Accepts optimal constant rates in the 
            dictionary 'etaopts', constant rates in the array-like 'etas', and 
            time-varying rates with parameters in the array-like 'alphas', 'thetas' """
        result_args = {}
        if algo == 'GP':
            regs_GP = {'savg':[], 'tsavg':[], 'tsavgbnd':[], 'perc_10':[], 
                       'perc_90':[], 'tavg_perc_10':[], 'tavg_perc_90':[]}
            print('Simulating GP, rate eta_t=t^(-0.5)')
            regrets = self.simulate(N, etas=(1+np.arange(self.T))**(-0.5), algo=algo, Ngrid=Ngrid, **kwargs)[2]
            self.parse_regrets(regs_GP, regrets)
            self.regret_bound(regs_GP, algo, alpha=0.5)
            result_args['regs_{}'.format(algo)] = regs_GP
        elif algo == 'OGD':
            regs_OGD = {'savg':[], 'tsavg':[], 'tsavgbnd':[], 'perc_10':[], 
                        'perc_90':[], 'tavg_perc_10':[], 'tavg_perc_90':[]}
            theta = 1/kwargs['H']
            print('Simulating OGD, rate eta_t={0:.2f}t^(-1)'.format(theta))
            regrets = self.simulate(N, etas=theta/(1+np.arange(self.T)), algo=algo, Ngrid=Ngrid, **kwargs)[2]
            self.parse_regrets(regs_OGD, regrets)
            self.regret_bound(regs_OGD, algo, H=kwargs['H'])
            result_args['regs_{}'.format(algo)] = regs_OGD
        elif algo == 'DA':
            pot = kwargs['potential']
            reg_info = {'savg':[], 'tsavg':[], 'tsavgbnd':[], 'perc_10':[], 
                        'perc_90':[], 'tavg_perc_10':[], 'tavg_perc_90':[]}
            if  kwargs.get('opt_rate') == True:
                if isinstance(pot, ExponentialPotential):
                    theta = np.sqrt((pot.c_omega*(self.domain.n-np.log(self.domain.v)) 
                                     + pot.d_omega*self.domain.v)/2/self.M**2)
                    alpha = None
                    print('Simulating {0}, {1} (HEDGE), opt. rate '.format(algo, pot.desc) + 
                          'eta_t={0:.3f} sqrt(log t/t)'.format(theta))
                    etas = theta*np.sqrt(np.log(1+np.arange(self.T)+1)/(1+np.arange(self.T)))
                else:
                    try:
                        M = pot.M
                    except AttributeError:
                        M = self.M
                    alpha, theta = pot.alpha_opt(self.domain.n), pot.theta_opt(self.domain, M)
                    print('Simulating {0}, {1}, opt. rate '.format(algo, pot.desc) + 
                          'eta_t={0:.3f}t^(-{1:.3f})$'.format(theta, alpha))
                    etas = theta*(1+np.arange(self.T))**(-alpha)
                regrets = self.simulate(N, etas=etas, algo=algo, Ngrid=Ngrid, **kwargs)[2]
                self.parse_regrets(reg_info, regrets) 
                self.regret_bound(reg_info, algo, alpha=alpha, theta=theta, potential=pot)
                result_args['regs_DAopt'] = reg_info
            if 'etaopts' in kwargs:
                regs_etaopts = {'savg':[], 'tsavg':[], 'tsavgbnd':[], 'perc_10':[], 
                                'perc_90':[], 'tavg_perc_10':[], 'tavg_perc_90':[]}
                for T,eta in kwargs['etaopts'].items():
                    if algo == 'DA':
                        print('Simulating {0}, {1}, opt. constant rate eta_t={2:.3f}'.format(algo, pot.desc, eta))
                    else:
                        print('Simulating {0}, opt. constant rate eta_t={1:.3f}'.format(algo, eta))
                    regrets = self.simulate(N, etas=eta*np.ones(self.T), algo=algo, Ngrid=Ngrid, **kwargs)[2]
                    self.parse_regrets(regs_etaopts, regrets)
                    result_args['regs_etaopts'] = regs_etaopts
            if 'etas' in kwargs:
                regs_etas = {'savg':[], 'tsavg':[], 'tsavgbnd':[], 'perc_10':[], 
                             'perc_90':[], 'tavg_perc_10':[], 'tavg_perc_90':[]}
                for eta in kwargs['etas']:
                    if algo == 'DA':
                        print('Simulating {0}, {1}, constant rate eta={2:.3f}'.format(algo, kwargs['potential'].desc, eta))
                    else:
                        print('Simulating {0}, constant rate eta={1:.3f}'.format(algo, eta))
                    regrets = self.simulate(N, etas=eta*np.ones(self.T), algo=algo, Ngrid=Ngrid, **kwargs)[2]
                    self.parse_regrets(regs_etas, regrets)
                    result_args['etas'] = kwargs['etas']
                    result_args['regs_etas'] = regs_etas
            if 'alphas' in kwargs:
                regs_alphas = {'savg':[], 'tsavg':[], 'tsavgbnd':[], 'perc_10':[], 
                               'perc_90':[], 'tavg_perc_10':[], 'tavg_perc_90':[]}
                for alpha,theta in zip(kwargs['alphas'], kwargs['thetas']): # run for Nloss different sequences of loss functions
                    if algo == 'DA':
                        print('Simulating {0}, {1}, decaying rate with alpha={2:.3f}, theta={3}'.format(algo, kwargs['potential'].desc, alpha, theta))
                    else:
                        print('Simulating {0}, decaying rate with alpha={1:.3f}, theta={2}'.format(algo, alpha, theta))
                    regrets = self.simulate(N, etas=theta*(1+np.arange(self.T))**(-alpha), algo=algo, Ngrid=Ngrid, **kwargs)[2]
                    self.parse_regrets(regs_alphas, regrets)
                    self.regret_bound(regs_alphas, algo, alpha=alpha, theta=theta, potential=kwargs['potential'])
                    result_args['alphas'] = kwargs['alphas']
                    result_args['thetas'] = kwargs['thetas']
                    result_args['regs_alphas'] = regs_alphas
        else:
            regs_norate = {'savg':[], 'tsavg':[], 'tsavgbnd':[], 'perc_10':[], 
                           'perc_90':[], 'tavg_perc_10':[], 'tavg_perc_90':[]}
            print('Simulating {0}, exp-concavity parameter alpha={1:.3f}'.format(algo, kwargs['alpha']))
            regrets = self.simulate(N, algo=algo, Ngrid=Ngrid, **kwargs)[2]
            self.parse_regrets(regs_norate, regrets)
            self.regret_bound(regs_norate, algo, **kwargs)
            result_args['regs_{}'.format(algo)] = regs_norate
        return Results(self, label=label, algo=algo, **result_args)
    
    
    def simulate(self, N, etas='opt', algo='DA', Ngrid=100000, **kwargs):
        """ Simulates the result of running the No-Regret algorithm (N times).
            Returns a list of sequences of decisions and associated losses, one for each run. 
            The grid is used for computing both the regret and the actions! """
        if etas == 'opt': # use optimal choice of etas
            etas = compute_etaopt(self.domain, self.M, self.T)*np.ones(self.T)
        if algo == 'DA':
            pot = kwargs['potential']
        if algo in ['ONS', 'FTAL', 'EWOO']:
            alpha = kwargs['alpha']
            beta = 0.5*np.minimum(1/4/self.L/self.domain.diameter, alpha)
            epsilon = 1/beta**2/self.domain.diameter**2
        # set up some data structures for keeping record            
        actions, losses, cumloss, regrets = [], [], [], []    
        gridpoints = self.domain.grid(Ngrid)
        approxL = np.zeros(gridpoints.shape[0])
        cumLossFunc = ZeroLossFunction(self.domain)
        # now run the iterations
        for t, lossfunc in enumerate(self.lossfuncs): 
            if t  == 0:
                print('pid {}: Starting...'.format(kwargs['pid']))
            elif t % 25 == 0:
                print('pid {}: t={}'.format(kwargs['pid'], t))
            if algo in ['GP', 'OGD']: # GP and OGD are the same except for the rates
                if t == 0:
                    action = self.domain.sample_uniform(N) # pick arbitrary action in the first step, may as well sample
                else:
                    action = self.lossfuncs[t-1].proj_gradient(actions[-1], etas[t]) # do a projected gradient step
            elif algo == 'DA': # Our very own Dual Averaging algorithm
                if t == 0:
                    # compute nustar for warm-starting the intervals of root-finder
                    nustar = -1/etas[t]*pot.phi_inv(1/self.domain.volume)
                    action = self.domain.sample_uniform(N)
                    try:
                        self.data.append([np.ones(pltpoints.shape[0])/self.domain.volume for pltpoints in self.pltpoints])
                    except AttributeError: pass
                else:
                    if (isinstance(cumLossFunc, AffineLossFunction) and isinstance(pot, ExponentialPotential) and
                        isinstance(self.domain, nBox) or isinstance(self.domain, UnionOfDisjointnBoxes)):
                        action = quicksample(np.array(self.domain.bounds), np.repeat(np.array(cumLossFunc.a, ndmin=2), N, axis=0), etas[t])
                    else:
                        nustar = compute_nustar(self.domain, pot, etas[t], cumLossFunc, self.M, nustar, 
                                                etas[t-1], t, pid=kwargs['pid'], tmpfolder=kwargs['tmpfolder'])
                        weights = np.maximum(pot.phi(-etas[t]*(approxL + nustar)), 0)
                        # let us plot the probability distribution
                        action = gridpoints[np.random.choice(weights.shape[0], size=N, p=weights/np.sum(weights))]
                    try:
                        self.data.append([np.maximum(pot.phi(-etas[t]*(cumLossFunc.val(pltpoints) + nustar)), 0) for pltpoints in self.pltpoints])
                    except AttributeError: pass
            elif algo == 'ONS': # Hazan's Online Newton Step
                if t == 0: 
                    action = self.domain.sample_uniform(N) # pick arbitrary action in the first step, may as well sample
                    grad = lossfunc.grad(action)
                    A = np.einsum('ij...,i...->ij...', grad, grad) + epsilon*np.array([np.eye(2),]*N)
                    Ainv = np.array([np.linalg.inv(mat) for mat in A])
                else:
                    points = actions[-1] - np.einsum('ijk...,ik...->ij...', Ainv, grad)/beta
                    action = self.domain.gen_project(points, A)
                    grad = lossfunc.grad(action)
                    A = A + np.einsum('ij...,i...->ij...', grad, grad)
                    z = np.einsum('ijk...,ik...->ij...', Ainv, grad)
                    Ainv = Ainv - np.einsum('ij...,i...->ij...', z, z)/(1 + np.einsum('ij,ij->i',grad,z))[:,np.newaxis,np.newaxis]
            elif algo == 'FTAL':
                if t == 0: 
                    action = self.domain.sample_uniform(N) # pick arbitrary action in the first step, may as well sample
                    grad = lossfunc.grad(action)
                    A = np.einsum('ij...,i...->ij...', grad, grad)
                    b = grad*(np.einsum('ij,ij->i', grad, action) - 1/beta)[:,np.newaxis]
                    Ainv = np.array([np.linalg.pinv(mat) for mat in A]) # so these matrices are singular... what's the issue?
                else:
                    points = np.einsum('ijk...,ik...->ij...', Ainv, b) 
                    action = self.domain.gen_project(points, A)
                    grad = lossfunc.grad(action)
                    A = A + np.einsum('ij...,i...->ij...', grad, grad)
                    b = b + grad*(np.einsum('ij,ij->i', grad, action) - 1/beta)[:,np.newaxis]
                    # the following uses the matrix inversion lemma for 
                    # efficient computation the update of Ainv
                    z = np.einsum('ijk...,ik...->ij...', Ainv, grad)
                    Ainv = Ainv - np.einsum('ij...,i...->ij...', z, z)/(1 + np.einsum('ij,ij->i',grad,z))[:,np.newaxis,np.newaxis]
            elif algo == 'EWOO': 
                if t == 0:
                    if not self.domain.isconvex():
                        raise Exception('EWOO algorithm only makes sense if the domain is convex!')
                    action = self.domain.sample_uniform(N)
                else:
                    if isinstance(self.domain, nBox):
                        ranges = [self.domain.bounds]
                    elif isinstance(self.domain, UnionOfDisjointnBoxes):
                        ranges = [nbox.bounds for nbox in self.domain.nboxes]
                    else:
                        raise Exception('For now, domain must be an nBox or a UnionOfDisjointnBoxes!') 
                    action_ewoo = action_EWOO(cumLossFunc, alpha, ranges, tmpfolder=kwargs['tmpfolder'])
                    action = np.array([action_ewoo,]*N)
            
            # now store the actions, losses, etc.
            actions.append(action)
            loss = lossfunc.val(action)
            losses.append(loss)
            if t == 0:
                cumloss.append(loss)
                cumLossFunc = lossfunc
            else:
                cumloss.append(cumloss[-1] + loss)
                cumLossFunc = cumLossFunc + lossfunc
            # compute and append regret -- resort to gridding for now
            approxL += lossfunc.val(gridpoints)
            optval = np.min(approxL)
            regrets.append(cumloss[-1] - optval)
        return np.transpose(np.array(actions), (1,0,2)), np.transpose(np.array(losses)), np.transpose(np.array(regrets))
    
    
    def parse_regrets(self, reg_results, regrets):
        """ Function that computes some aggregate information from the 
            raw regret samples in the list 'regrets' """ 
        reg_results['savg'].append(np.average(regrets, axis=0))
        reg_results['perc_10'].append(np.percentile(regrets, 10, axis=0))
        reg_results['perc_90'].append(np.percentile(regrets, 90, axis=0))
        reg_results['tsavg'].append(reg_results['savg'][-1]/(1+np.arange(self.T)))
        reg_results['tavg_perc_10'].append(reg_results['perc_10'][-1]/(1+np.arange(self.T)))
        reg_results['tavg_perc_90'].append(reg_results['perc_90'][-1]/(1+np.arange(self.T)))
        return reg_results
    
    
    def regret_bound(self, reg_results, algo, **kwargs):
        """ Computes the regret bound for the ContNoRegret Problem. """
        t = 1 + np.arange(self.T)
        n, D, L = self.domain.n, self.domain.diameter, self.L
        if algo == 'DA':
            pot, v = kwargs['potential'], self.domain.v
            alpha, theta = kwargs['alpha'], kwargs['theta']
            if isinstance(pot, ExponentialPotential):
                reg_bnd = self.M*np.sqrt(8*(pot.c_omega*(n-np.log(v)) + pot.d_omega*v))*np.sqrt(np.log(t+1)/t) + L*D/t
            else:
                lpsi, p_dualnorm = pot.l_psi()
                C, epsilon = pot.bounds_asymp()
                try:
                    M = pot.M
                except AttributeError:
                    M = self.M
                reg_bnd = (M**2*theta/lpsi/(1-alpha)*t**(-alpha)
                           + (L*D + C/theta*v**(-epsilon))*t**(-(1-alpha)/(1+n*epsilon)))
        elif algo == 'GP':
            # for now assume eta_t = t**(-0.5)
            reg_bnd = (D**2/2 + L**2)*t**(-0.5) - L**2/2/t
        elif algo == 'OGD':
            reg_bnd = L**2/2/kwargs['H']*(1+np.log(t))/t
        elif algo == 'ONS':
            reg_bnd = 5*(1/kwargs['alpha'] + L*D)*n*np.log(t+1)/t
        elif algo == 'FTAL':
            reg_bnd = 64*(1/kwargs['alpha'] + L*D)*n*(1+np.log(t))/t
        elif algo == 'EWOO':
            reg_bnd = 1/kwargs['alpha']*n*(1+np.log(t+1))/t
        else:
            raise NotImplementedError
        reg_results['tsavgbnd'].append(reg_bnd)
        
        
class Results(object):
    """ Class for 'result' objects that contain simulation results 
        generated by ContNoRegretProblems """
    
    def __init__(self, problem, **kwargs):
        self.problem = problem
        self.label = kwargs.get('label')
        self.algo = kwargs.get('algo')
        if self.algo == 'DA':
            try: self.regs_norate = kwargs['regs_DAopt']
            except: KeyError
            try: self.etaopts, self.regs_etaopts = kwargs['etaopts'], kwargs['regs_etaopts'] 
            except KeyError: pass
            try: self.etas, self.regs_etas = kwargs['etas'], kwargs['regs_etas']
            except KeyError: pass
            try: self.alphas, self.thetas, self.regs_alphas = kwargs['alphas'], kwargs['thetas'], kwargs['regs_alphas']
            except KeyError: pass
        else:
            self.regs_norate = kwargs['regs_{}'.format(self.algo)]
        self.slopes, self.slopes_bnd = self.estimate_loglog_slopes()
            
    def estimate_loglog_slopes(self, N=125):
        """ Estimates slope, intercept and r_value of the asymptotic log-log plot
        for each element f tsavg_regert, using the N last data points """
        slopes, slopes_bnd = {}, {}
#         if N < self.problem.T:
#             N = np.floor(self.problem.T/2)
        try:
            slopes['etaopts'] = self.loglog_slopes(self.regs_etaopts['tsavg'], N)
            slopes_bnd['etaopts'] = self.loglog_slopes(self.regs_etaopts['tsavgbnd'], N)
        except AttributeError: pass
        try: 
            slopes['etas'] = self.loglog_slopes(self.regs_etas['tsavg'], N)
            slopes_bnd['etas'] = self.loglog_slopes(self.regs_etas['tsavgbnd'], N)
        except AttributeError: pass
        try:
            slopes['alphas'] = self.loglog_slopes(self.regs_alphas['tsavg'], N)
            slopes_bnd['alphas'] = self.loglog_slopes(self.regs_alphas['tsavgbnd'], N)
        except AttributeError: pass
        try:
            slopes['{}'.format(self.algo)] = self.loglog_slopes(self.regs_norate['tsavg'], N)
            slopes_bnd['{}'.format(self.algo)] = self.loglog_slopes(self.regs_norate['tsavgbnd'], N)
        except AttributeError: pass
        return slopes, slopes_bnd
        
    def loglog_slopes(self, regrets, N): 
        slopes = []
        for regret in regrets:
            T = np.arange(len(regret)-N, len(regret))
            Y = regret[len(regret)-N:]
            slope = linregress(np.log(T), np.log(Y))[0]
            slopes.append(slope)
        return slopes
                
            
    
    
def action_EWOO(cumLossFunc, alpha, ranges, tmpfolder='libs/'):
    """ Function for computing the (single) action of the EWOO algorithm """
    header = ['#include <math.h>\n\n',
              'double alpha = {};\n'.format(alpha)]
    func = cumLossFunc.gen_ccode()
    ccode = header + func + ['   return exp(-alpha*loss);\n',
                             '   }'] 
    integr = ctypes_integrate(ccode, ranges, tmpfolder)
    actions = []
    for i in range(cumLossFunc.domain.n):
        footer = ['   return args[{}]*exp(-alpha*loss);\n'.format(i),
                  '   }']  
        ccode = header + func + footer
        actions.append(ctypes_integrate(ccode, ranges, tmpfolder)/integr)
    return np.array(actions)

