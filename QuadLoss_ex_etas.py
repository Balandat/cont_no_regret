'''
Numerical example for the Continuous No-Regret Algorithm with 
Quadratic loss functions

@author: Maximilian Balandat, Walid Krichene
@date: Dec 20, 2014
'''

from ContNoRegret.Domains import S
from ContNoRegret.Distributions import Uniform
from ContNoRegret.LossFunctions import QuadraticLossFunction
from ContNoRegret.HedgeAlgorithm import QuadraticNoRegretProblem
from ContNoRegret.utils import create_random_Q, compute_etaopt, plot_results

# set up some basic parameters
T = 10000
M = 10.0
Lbnd = 5.0 # Uniform bound on the Lipschitz constant
N = 2500

# domain is an 'S'
dom = S()

# create random means, uniformly over the domain
mus = Uniform(dom).sample(T)
# create random Q matrices, based on the Lipschitz bound and the uniform bound M
Qs = [create_random_Q(dom, mu, Lbnd, M) for mu in mus]
# create list of loss functions
lossfuncs = [QuadraticLossFunction(dom, mu, Q, 0.0) for mu,Q in zip(mus,Qs)]
# Create quadratic problem
quadprob = QuadraticNoRegretProblem(dom, lossfuncs, Lbnd, M)

# run the problem for different constant rates etas
etaopts = {}
Ts = [2500, 7500]
for T in Ts:
    etaopts[T] = compute_etaopt(dom, M, T)
etas = [0.1, 0.2]

results = quadprob.run_simulation(N, etaopts=etaopts, etas=etas, Ngrid=200000)    
plot_results(results, offset=1000, filename='figures/Quad_etas_S')
slopes, slopes_bnd = results.estimate_loglog_slopes()

