'''
Collection of utilitiy functions to analyze Continuous No-Regret algorithms

@author: Maximilian Balandat
@date May 6, 2015
'''

import numpy as np
import pickle, os
from matplotlib import pyplot as plt
from scipy.stats import linregress 
from .Domains import nBox


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
  

def plot_results(results, offset=500, directory=None, show=True):
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
            if result.algo in ['DA', 'OGD']:
                try:
                    for i,(T,eta) in enumerate(result.etaopts.items()):
                        plt.figure(1)
                        lavg = plt.plot(result.regs_etaopts['savg'][i][0:T], linewidth=2.0, 
                                        label=result.label+' '+r' $\eta_{{opt}}(T={0:.1e}) = {1:.3f}$'.format(T, eta), rasterized=True)
                        plt.plot(np.arange(T,result.problem.T), result.regs_etaopts['savg'][i][T:], '--', 
                                 color=lavg[0].get_color(), linewidth=2, rasterized=True)
                        plt.fill_between(np.arange(result.problem.T), result.regs_etaopts['perc_10'][i], 
                                         result.regs_etaopts['perc_90'][i], color=lavg[0].get_color(), alpha=0.2, rasterized=True)
                        plt.figure(2)
                        ltavg = plt.plot(np.arange(offset,T), result.regs_etaopts['tsavg'][i][offset:T], linewidth=2.0, 
                                         label=result.label+' '+r' $\eta_{{opt}}(T={0:.1e}) = {1:.3f}$'.format(T, eta), rasterized=True)
                        plt.plot(np.arange(T,result.problem.T), result.regs_etaopts['tsavg'][i][T:], '--', 
                                 color=ltavg[0].get_color(), linewidth=2, rasterized=True)
                        plt.fill_between(np.arange(offset,result.problem.T), result.regs_etaopts['tavg_perc_10'][i][offset:], 
                                         result.regs_etaopts['tavg_perc_90'][i][offset:], color=ltavg[0].get_color(), alpha=0.2, rasterized=True)
                        plt.xlim((0, result.problem.T))
                        plt.figure(3)
                        llogtavg = plt.plot(np.arange(1,result.problem.T+1), result.regs_etaopts['tsavg'][i], 
                                            linewidth=2.0, label=result.label+' '+r' $\eta_{{opt}}(T={0:.1e}) = {1:.3f}$'.format(T, eta), rasterized=True)
                        plt.fill_between(np.arange(1,result.problem.T+1), result.regs_etaopts['tavg_perc_10'][i], 
                                        result.regs_etaopts['tavg_perc_90'][i], color=llogtavg[0].get_color(), alpha=0.2, rasterized=True)
    #                     plt.plot(np.arange(1,result.problem.T+1), result.regs_etaopts['tsavgbnd'][i], '--', 
    #                              color=llogtavg[0].get_color(), linewidth=2, rasterized=True)
                except AttributeError: pass
                try:
                    for i,eta in enumerate(result.etas):
                        plt.figure(1)
                        lavg = plt.plot(result.regs_etas['savg'][i], linewidth=2.0, label=result.label+' '+r' $\eta = {0:.3f}$'.format(eta), rasterized=True)
                        plt.fill_between(np.arange(result.problem.T), result.regs_etas['perc_10'][i], 
                                         result.regs_etas['perc_90'][i], color=lavg[0].get_color(), alpha=0.15, rasterized=True)
                        plt.figure(2)
                        ltavg = plt.plot(np.arange(offset,result.problem.T), result.regs_etas['tsavg'][i][offset:], 
                                         linewidth=2.0, label=result.label+' '+r'$\eta = {0:.3f}$'.format(eta), rasterized=True)
                        plt.fill_between(np.arange(offset,result.problem.T), result.regs_etas['tavg_perc_10'][i][offset:], 
                                         result.regs_etas['tavg_perc_90'][i][offset:], color=ltavg[0].get_color(), alpha=0.15, rasterized=True)
                        plt.xlim((0, result.problem.T))
                        plt.figure(3)
                        llogtavg = plt.plot(np.arange(1,result.problem.T+1), result.regs_etas['tsavg'][i], linewidth=2.0, 
                                            label=result.label+' '+r' $\eta = {0:.3f}$'.format(eta), rasterized=True)
                        plt.fill_between(np.arange(1,result.problem.T+1), result.regs_etas['tavg_perc_10'][i], 
                                         result.regs_etas['tavg_perc_90'][i], color=llogtavg[0].get_color(), alpha=0.15, rasterized=True) 
    #                     plt.plot(np.arange(1,result.problem.T+1), result.regs_etas['tsavgbnd'][i], '--', color=llogtavg[0].get_color(), linewidth=2, rasterized=True)     
                except AttributeError: pass
                try:
                    for i,alpha in enumerate(result.alphas):
                        plt.figure(1)
                        lavg = plt.plot(result.regs_alphas['savg'][i], linewidth=2.0,
                                 label=result.label+' '+r' $\eta_t = {0} \cdot t^{{{1}}}$'.format(result.thetas[i], -alpha), rasterized=True)
                        plt.fill_between(np.arange(result.problem.T), result.regs_alphas['perc_10'][i], 
                                         result.regs_alphas['perc_90'][i], color=lavg[0].get_color(), alpha=0.15, rasterized=True)
                        plt.figure(2)
                        ltavg = plt.plot(np.arange(result.problem.T)[offset:], result.regs_alphas['tsavg'][i][offset:], linewidth=2.0, 
                                 label=result.label+' '+r' $\eta_t = {0} \cdot t^{{{1}}}$'.format(result.thetas[i], -alpha), rasterized=True)
                        plt.fill_between(np.arange(offset,result.problem.T), result.regs_alphas['tavg_perc_10'][i][offset:], 
                                         result.regs_alphas['tavg_perc_90'][i][offset:], color=ltavg[0].get_color(), alpha=0.15, rasterized=True)
                        plt.xlim((0, result.problem.T)) 
                        plt.figure(3)
                        lltsavg = plt.plot(np.arange(1,result.problem.T+1), result.regs_alphas['tsavg'][i], linewidth=2.0, 
                                             label=result.label+' '+r' $\eta_t = {0} \cdot t^{{{1}}}$'.format(result.thetas[i], -alpha), rasterized=True)
                        plt.fill_between(np.arange(1,result.problem.T+1), result.regs_alphas['tavg_perc_10'][i], 
                                        result.regs_alphas['tavg_perc_90'][i], color=lltsavg[0].get_color(), alpha=0.15, rasterized=True)
                        plt.plot(np.arange(1,result.problem.T+1), result.regs_alphas['tsavgbnd'][i], '--', color=lltsavg[0].get_color(), 
                                 linewidth=2.0, rasterized=True) 
                except AttributeError: pass
            else:
                plt.figure(1)
                lavg = plt.plot(result.regs_norate['savg'][i], linewidth=2.0, label=result.label, rasterized=True)
                plt.fill_between(np.arange(result.problem.T), result.regs_norate['perc_10'][i], 
                                 result.regs_norate['perc_90'][i], color=lavg[0].get_color(), alpha=0.15, rasterized=True)
                plt.figure(2)
                ltavg = plt.plot(np.arange(result.problem.T)[offset:], result.regs_norate['tsavg'][i][offset:], 
                                 linewidth=2.0, label=result.label, rasterized=True)
                plt.fill_between(np.arange(offset,result.problem.T), result.regs_norate['tavg_perc_10'][i][offset:], 
                                 result.regs_norate['tavg_perc_90'][i][offset:], color=ltavg[0].get_color(), alpha=0.15, rasterized=True)
                plt.xlim((0, result.problem.T)) 
                plt.figure(3)
                lltsavg = plt.plot(np.arange(1,result.problem.T+1), result.regs_norate['tsavg'][i], linewidth=2.0, 
                                   label=result.label, rasterized=True)
                plt.fill_between(np.arange(1,result.problem.T+1), result.regs_norate['tavg_perc_10'][i], 
                                result.regs_norate['tavg_perc_90'][i], color=lltsavg[0].get_color(), alpha=0.15, rasterized=True)
                     
        # make plots pretty and show legend
        plt.figure(1)
        plt.legend(loc='upper left', prop={'size':13}, frameon=False) 
        plt.figure(2)
        plt.legend(loc='upper right', prop={'size':13}, frameon=False) 
        plt.figure(3)
        plt.yscale('log'), plt.xscale('log')
        plt.legend(loc='upper right', prop={'size':13}, frameon=False) 
        if directory:
            os.makedirs(directory+'figures/', exist_ok=True) # this could probably use a safer implementation  
            filename = '{}{}_{}_'.format(directory+'figures/', results[0].problem.desc, results[0].problem.lossfuncs[0].desc)
            plt.figure(1)
            plt.savefig(filename + 'cumloss.pdf', bbox_inches='tight', dpi=300)
            plt.figure(2)
            plt.savefig(filename + 'tavgloss.pdf', bbox_inches='tight', dpi=300)
            plt.figure(3)
            plt.savefig(filename + 'loglogtavgloss.pdf', bbox_inches='tight', dpi=300)
        if show:
            plt.show()
            

def save_results(results, directory):
    """ Serializes a results object for persistent storage using the pickle module. """ 
    os.makedirs(directory, exist_ok=True) # this could probably use a safer implementation
    for result in results:
        try:
            del result.problem.pltpoints, result.problem.data
        except AttributeError:
            pass
    pigglname = '{}{}_{}.piggl'.format(directory, results[0].problem.desc, 
                                       results[0].problem.lossfuncs[0].desc)    
    with open(pigglname, 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)     
 
            
def circular_tour(domain, N):
    """ Returns a sequence of N points that wander around in a circle
        in the domain. Used for understanding various learning rates. """
    if isinstance(domain, nBox):
        if domain.n != 2:
            raise Exception('For now circular_tour only works in dimension 2')
        center = np.array([0.5*(bnd[0]+bnd[1]) for bnd in domain.bounds])
        halfaxes = np.array([0.75*0.5*(bnd[1]-bnd[0]) for bnd in domain.bounds])
        return np.array([center[0] + halfaxes[0]*np.cos(np.linspace(0,2*np.pi,N)), 
                         center[1] + halfaxes[1]*np.sin(np.linspace(0,2*np.pi,N))]).T 
    else:
        raise Exception('For now circular_tour only works on nBoxes')
        

def CNR_worker(prob, *args, **kwargs):
    """ Helper function for wrapping class methods to allow for easy 
        use of the multiprocessing package for parallel computing """
    return prob.run_simulation(*args, **kwargs)

