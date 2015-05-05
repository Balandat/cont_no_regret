'''
Collection of utilities to analyze Continuous No-Regret algorithms

@author: Maximilian Balandat, Walid Krichene
@date Dec 4, 2014
'''

import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import orth, eigh
from scipy.stats import uniform, gamma, linregress
from .LossFunctions import AffineLossFunction, QuadraticLossFunction


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


def random_AffineLosses(dom, L, T, d=2):
    """ Creates T random L-Lipschitz AffineLossFunction over domain dom,
        and returns uniform bound M. For now sample the a-vector uniformly
        from the n-ball. Uses random samples of Beta-like distributions as 
        described in:
        'R. Harman and V. Lacko. On decompositional algorithms for uniform 
        sampling from n-spheres and n-balls. Journal of Multivariate 
        Analysis, 101(10):2297 – 2304, 2010.'
    """
    lossfuncs, Ms = [], []
    asamples = sample_Bnrd(dom.n, L, d, T)
    for a in asamples:
        lossfunc = AffineLossFunction(dom, a, 0)
        lossmin, lossmax = lossfunc.minmax()
        lossfunc.b = - lossmin
        lossfunc.set_bounds([0, lossmax-lossmin])
        lossfuncs.append(lossfunc)
        Ms.append(lossfunc.bounds[1]) 
    return lossfuncs, np.max(Ms)

def sample_Bnrd(n, r, d, N):
    """ Draw N independent samples from the B_n(r,d) distribution
        discussed in:
        'R. Harman and V. Lacko. On decompositional algorithms for uniform 
        sampling from n-spheres and n-balls. Journal of Multivariate 
        Analysis, 101(10):2297 – 2304, 2010.'
    """
    Bsqrt = np.sqrt(np.random.beta(n/2, d/2, size=N))
    X = np.random.randn(N, n)
    normX = np.linalg.norm(X, 2, axis=1)
    S = X/normX[:, np.newaxis]
    return r*Bsqrt[:, np.newaxis]*S


def random_QuadraticLosses(dom, mus, L, M, dist=uniform()):
    """ Creates T random L-Lipschitz PolynomialLossFunctions of degree 2
        over the domain dom, uniformly bounded (in infinity norm) by M.
    """
    Qs = [create_random_Q(dom, mu, L, M, dist) for mu in mus] 
    return [QuadraticLossFunction(dom, mu, Q, 0) for mu,Q in zip(mus, Qs)]


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
       
def regret_bounds(dom, theta, alpha, L, M, Tmax, algo):
    """ Computes vector of regret bounds for t=1,...,Tmax """
    diameter = dom.compute_parameters()[0]
    t = np.arange(Tmax) + 1
    if algo == 'hedge':
        return (M**2*theta/8/(1-alpha)/(t**alpha) + L*diameter/t 
                        + (dom.n*np.log(t) - np.log(dom.v))/(theta*t**(1-alpha)))
    else:
        raise NotImplementedError
#         if algo == 'greedy':
#         return diameter**2/2/(t**(1-alpha)) + L**2/2/(1-alpha)/(t**alpha)
       
def DA_regret_bounds(dom, theta, alpha, L, M, Tmax, potential):
    raise NotImplementedError
       
    
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
  

def plot_results(results, offset=500, path=None, show=True):
        """ Plots and shows or saves (or both) the simulation results """
        # set up figures
        plt.figure(1)
        plt.title('cumulative regret, {} losses'.format(results[0].problem.lossfuncs[0].desc))
        plt.xlabel('t')
        plt.figure(2)
        plt.title('time-avg. cumulative regret, {} losses'.format(results[0].problem.lossfuncs[0].desc))
        plt.xlabel('t')
        plt.figure(3)
        plt.title(r'log time-avg. cumulative regret, {} losses'.format(results[0].problem.lossfuncs[0].desc))
        plt.xlabel('t')   
        # and now plot, depending on what data is there
        for result in results:
            if result.etaopts:
                for i,(T,eta) in enumerate(result.etaopts.items()):
                    plt.figure(1)
                    lavg = plt.plot(result.regs_etaopts['savg'][i][0:T], linewidth=2.0, 
                                    label=result.label+r' $\eta_{{opt}}(T={0:.1e}) = {1:.3f}$'.format(T, eta), rasterized=True)
                    plt.plot(np.arange(T,result.problem.T), result.regs_etaopts['savg'][i][T:], '--', 
                             color=lavg[0].get_color(), linewidth=2, rasterized=True)
                    plt.fill_between(np.arange(result.problem.T), result.regs_etaopts['perc_10'][i], 
                                     result.regs_etaopts['perc_90'][i], color=lavg[0].get_color(), alpha=0.2, rasterized=True)
                    plt.figure(2)
                    ltavg = plt.plot(np.arange(offset,T), result.regs_etaopts['tsavg'][i][offset:T], linewidth=2.0, 
                                     label=result.label+r' $\eta_{{opt}}(T={0:.1e}) = {1:.3f}$'.format(T, eta), rasterized=True)
                    plt.plot(np.arange(T,result.problem.T), result.regs_etaopts['tsavg'][i][T:], '--', 
                             color=ltavg[0].get_color(), linewidth=2, rasterized=True)
                    plt.fill_between(np.arange(offset,result.problem.T), result.regs_etaopts['tavg_perc_10'][i][offset:], 
                                     result.regs_etaopts['tavg_perc_90'][i][offset:], color=ltavg[0].get_color(), alpha=0.2, rasterized=True)
                    plt.xlim((0, result.problem.T))
                    plt.figure(3)
                    llogtavg = plt.plot(np.arange(1,result.problem.T+1), result.regs_etaopts['tsavg'][i], 
                                        linewidth=2.0, label=result.label+r' $\eta_{{opt}}(T={0:.1e}) = {1:.3f}$'.format(T, eta), rasterized=True)
                    plt.fill_between(np.arange(1,result.problem.T+1), result.regs_etaopts['tavg_perc_10'][i], 
                                    result.regs_etaopts['tavg_perc_90'][i], color=llogtavg[0].get_color(), alpha=0.2, rasterized=True)
#                     plt.plot(np.arange(1,result.problem.T+1), result.regs_etaopts['tsavgbnd'][i], '--', 
#                              color=llogtavg[0].get_color(), linewidth=2, rasterized=True)
            if result.etas:
                for i,eta in enumerate(result.etas):
                    plt.figure(1)
                    lavg = plt.plot(result.regs_etas['savg'][i], linewidth=2.0, label=result.label+r' $\eta = {0:.3f}$'.format(eta), rasterized=True)
                    plt.fill_between(np.arange(result.problem.T), result.regs_etas['perc_10'][i], 
                                     result.regs_etas['perc_90'][i], color=lavg[0].get_color(), alpha=0.15, rasterized=True)
                    plt.figure(2)
                    ltavg = plt.plot(np.arange(offset,result.problem.T), result.regs_etas['tsavg'][i][offset:], 
                                     linewidth=2.0, label=result.label+r'$\eta = {0:.3f}$'.format(eta), rasterized=True)
                    plt.fill_between(np.arange(offset,result.problem.T), result.regs_etas['tavg_perc_10'][i][offset:], 
                                     result.regs_etas['tavg_perc_90'][i][offset:], color=ltavg[0].get_color(), alpha=0.15, rasterized=True)
                    plt.xlim((0, result.problem.T))
                    plt.figure(3)
                    llogtavg = plt.plot(np.arange(1,result.problem.T+1), result.regs_etas['tsavg'][i], linewidth=2.0, label=result.label+r' $\eta = {0:.3f}$'.format(eta), rasterized=True)
                    plt.fill_between(np.arange(1,result.problem.T+1), result.regs_etas['tavg_perc_10'][i], 
                                     result.regs_etas['tavg_perc_90'][i], color=llogtavg[0].get_color(), alpha=0.15, rasterized=True) 
#                     plt.plot(np.arange(1,result.problem.T+1), result.regs_etas['tsavgbnd'][i], '--', color=llogtavg[0].get_color(), linewidth=2, rasterized=True)     
            if result.alphas:
                for i,alpha in enumerate(result.alphas):
                    plt.figure(1)
                    lavg = plt.plot(result.regs_alphas['savg'][i], linewidth=2.0,
                             label=result.label+r' $\eta_t = {0} \cdot t^{{{1}}}$'.format(result.thetas[i], -alpha), rasterized=True)
                    plt.fill_between(np.arange(result.problem.T), result.regs_alphas['perc_10'][i], 
                                     result.regs_alphas['perc_90'][i], color=lavg[0].get_color(), alpha=0.15, rasterized=True)
                    plt.figure(2)
                    ltavg = plt.plot(np.arange(result.problem.T)[offset:], result.regs_alphas['tsavg'][i][offset:], linewidth=2.0, 
                             label=result.label+r' $\eta_t = {0} \cdot t^{{{1}}}$'.format(result.thetas[i], -alpha), rasterized=True)
                    plt.fill_between(np.arange(offset,result.problem.T), result.regs_alphas['tavg_perc_10'][i][offset:], 
                                     result.regs_alphas['tavg_perc_90'][i][offset:], color=ltavg[0].get_color(), alpha=0.15, rasterized=True)
                    plt.xlim((0, result.problem.T)) 
                    plt.figure(3)
                    lltsavg = plt.plot(np.arange(1,result.problem.T+1), result.regs_alphas['tsavg'][i], linewidth=2.0, 
                                         label=result.label+r' $\eta_t = {0} \cdot t^{{{1}}}$'.format(result.thetas[i], -alpha), rasterized=True)
                    plt.fill_between(np.arange(1,result.problem.T+1), result.regs_alphas['tavg_perc_10'][i], 
                                    result.regs_alphas['tavg_perc_90'][i], color=lltsavg[0].get_color(), alpha=0.15, rasterized=True)
#                     plt.plot(np.arange(1,result.problem.T+1), result.regs_alphas['tsavgbnd'][i], '--', color=lltsavg[0].get_color(), linewidth=2.0, rasterized=True) 
        # make plots pretty and show legend
        plt.figure(1)
        plt.legend(loc='upper left', prop={'size':13}, frameon=False) 
        plt.figure(2)
        plt.legend(loc='upper right', prop={'size':13}, frameon=False) 
        plt.figure(3)
        plt.yscale('log'), plt.xscale('log')
        plt.legend(loc='upper right', prop={'size':13}, frameon=False) 
        if path:
            descriptor = '{}_{}_'.format(results[0].problem.desc,
                                         results[0].problem.lossfuncs[0].desc)
            plt.figure(1)
            plt.savefig(path + descriptor + 'cumloss.pdf', bbox_inches='tight', dpi=300)
            plt.figure(2)
            plt.savefig(path + descriptor + 'tavgloss.pdf', bbox_inches='tight', dpi=300)
            plt.figure(3)
            plt.savefig(path + descriptor + 'loglogtavgloss.pdf', bbox_inches='tight', dpi=300)
        if show:
            plt.show()
            

def parse_regrets(reg_results, regrets, prob, theta, alpha, algo):
    """ Function that computes some aggregate information from the raw regret 
        samples in the list 'regrets' """ 
    reg_results['savg'].append(np.average(regrets, axis=0))
    reg_results['perc_10'].append(np.percentile(regrets, 10, axis=0))
    reg_results['perc_90'].append(np.percentile(regrets, 90, axis=0))
    reg_results['tsavg'].append(reg_results['savg'][-1]/(1+np.arange(prob.T)))
    reg_results['tavg_perc_10'].append(reg_results['perc_10'][-1]/(1+np.arange(prob.T)))
    reg_results['tavg_perc_90'].append(reg_results['perc_90'][-1]/(1+np.arange(prob.T)))
#     reg_results['tsavgbnd'].append(regret_bounds(prob.domain, theta, alpha, 
#                                           prob.Lbnd, prob.M, prob.T, algo=algo))
    return reg_results


def CNR_worker(prob, *args, **kwargs):
    """ Helper function for wrapping class methods to allow for easy 
        use of the multiprocessing package for parallel computing """
    return prob.run_simulation(*args, **kwargs)

