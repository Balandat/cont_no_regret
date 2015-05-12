'''
Numerical example for the Continuous No-Regret Algorithm with quadratic loss functions

@author: Maximilian Balandat, Walid Krichene
@date: Nov 24, 2014
'''

import numpy as np
from ContNoRegret.Domains import Rectangle, UnionOfDisjointRectangles
from ContNoRegret.Distributions import Uniform
from ContNoRegret.LossFunctions import QuadraticLossFunction
from ContNoRegret.HedgeAlgorithm import QuadraticNoRegretProblem
from matplotlib import pyplot as plt
from ContNoRegret.utils import compute_etaopt, regret_bound_const, create_random_Q,\
    regret_bounds

# set up some basic parameters
T = 5000
M = 5.0
Lbnd = 10.0 # Uniform bound on the Lipschitz constant
N = 250

# just a simple rectangle
dom = Rectangle([-1.0, 1.0], [-1.0, 1.0])

# # two rectangles next to each other, shifted vertically
# rect1 = Rectangle([-1.0, 0.0], [-2.0, 1.0])
# rect2 = Rectangle([0.0, 1.0], [1.0, 2.0])
# dom = UnionOfDisjointRectangles([rect1, rect2])

# # a C
# rect1 = Rectangle([-1.0, 1.0], [1.0, 2.0])
# rect2 = Rectangle([-1.0, 0.0], [-1.0, 1.0])
# rect3 = Rectangle([-1.0, 1.0], [-2.0, -1.0])
# dom = UnionOfDisjointRectangles([rect1, rect2, rect3])

etaopt = compute_etaopt(dom, M, T)
Rbnd_const = regret_bound_const(dom, etaopt, T, Lbnd, M)
print('Time-average regret bound (T known): {0:.4f}'.format(Rbnd_const))
print('Optimal constant learning rate (T known): {0:.4f}'.format(etaopt))

# create random means, uniformly over the domain
mus = Uniform(dom).sample(T)
# create random Q matrices, based on the Lipschitz bound and the uniform bound M
Qs = [create_random_Q(dom, mu, Lbnd, M) for mu in mus]
# create list of loss functions
lossfuncs = [QuadraticLossFunction(dom, mu, Q, 0.0) for mu,Q in zip(mus,Qs)]
# Create quadratic problem
quadprob = QuadraticNoRegretProblem(dom, lossfuncs, Lbnd, M)
    
# first run the algorithm for the best fixed eta for known T
actions_opt, losses_opt, regrets_opt = quadprob.simulate(N, etas='opt')
savg_regret_opt = np.average(regrets_opt, axis=0)
tsavg_regret_opt = savg_regret_opt/(1+np.arange(T))

tavg_regrets, savg_regret, tsavg_regret = [], [], []
tavg_bounds, const_bounds = [], []

# run the problem for different constant etas
etas_const = [0.125, 0.25, 0.5]
for eta in etas_const: # run for Nloss different sequences of loss functions
    actions, losses, regrets = quadprob.simulate(N, etas=eta*np.ones(T))
    savg = np.average(regrets, axis=0)
    savg_regret.append(savg)
    tsavg_regret.append(savg/(1+np.arange(T)))
    const_bounds.append(regret_bound_const(dom, eta, T, Lbnd, M))
    
# run the problem for different alphas and thetas
alphas = [0.125, 0.2, 0.3]
thetas = [1, 1, 1]
for alpha,theta in zip(alphas, thetas): # run for Nloss different sequences of loss functions
    actions, losses, regrets = quadprob.simulate(N, etas=theta*(1+np.arange(T))**(-alpha))
    savg = np.average(regrets, axis=0)
    savg_regret.append(savg)
    tsavg_regret.append(savg/(1+np.arange(T)))
    tavg_bounds.append(regret_bounds(dom, theta, alpha, Lbnd, M, T))

# plot the regrets for the optimal constant learning rate
f1 = plt.figure(1)
plt.title('Expected Cumulative Regret (sample avg. from {} runs)'.format(N))
plt.xlabel('t')
plt.plot(savg_regret_opt, linewidth=2.0)
f2 = plt.figure(2)
plt.title('Time-average expected Cumulative Regret (sample avg. from {} runs)'.format(N))
plt.xlabel('t')
offset = 250
plt.plot(np.arange(T)[offset:], tsavg_regret_opt[offset:], linewidth=2.0)
plt.plot([0,T-1], [Rbnd_const, Rbnd_const], 'k:')
labels= [r'$\eta_t = \eta^{opt}(T) = $' + '{0:.3f}'.format(etaopt)]
f3 = plt.figure(3)
plt.title(r'log time-avg. expected Cumulative Regret (sample avg. from {} runs)'.format(N))
plt.xlabel('log t')
plt.loglog(tsavg_regret_opt, linewidth=2.0)

# plot the regrets for the other learning rates
for i,eta in enumerate(etas_const):
    labels.append(r'$\eta = {0}$'.format(eta))
    plt.figure(1)
    plt.plot(savg_regret[i], '--', linewidth=2.0)
    plt.figure(2)
    plt.plot(np.arange(T)[offset:], tsavg_regret[i][offset:], '--', linewidth=2.0)
    plt.plot([0,T-1], [const_bounds[i], const_bounds[i]], ':', linewidth=1.5)
    plt.xlim((0, T))
    plt.figure(3)
    plt.loglog(tsavg_regret[i], '--', linewidth=2.0)
    
for i,alpha in enumerate(alphas):
    j = len(etas_const) - 1 + i
    labels.append(r'$\eta_t = {0} \cdot t^{{{1}}}$'.format(theta, -alpha))
    plt.figure(1)
    plt.plot(savg_regret[j], '--', linewidth=2.0)
    plt.figure(2)
    plt.plot(np.arange(T)[offset:], tsavg_regret[j][offset:], '--', linewidth=2.0)
    plt.plot(np.arange(T)[offset:], tavg_bounds[i][offset:], '--', linewidth=2.0)
    plt.xlim((0, T))
    plt.figure(3)
    plt.loglog(tsavg_regret[j], '--', linewidth=2.0)
    
# make plots pretty and show legend
plt.figure(1)
plt.legend(labels, loc='upper left') 
plt.figure(2)
labels2 = labels.copy()
labels2.insert(1, 'regret bound for $\eta^{opt}$')
plt.legend(labels2, loc='upper right') 
plt.figure(3)
plt.legend(labels, loc='lower left') 
plt.show()

