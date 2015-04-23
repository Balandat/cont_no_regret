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
from ContNoRegret.utils import create_random_Q, plot_results

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

# run the problem for different alphas and thetas
alphas = [0.125, 0.2, 0.3, 0.4]
thetas = [0.25, 0.25, 0.25, 0.25]  

results = quadprob.run_simulation(N, etas=None, alphas=alphas, thetas=thetas)    
plot_results(results, offset=1000, filename='figures/Quad_alphas_S')
slopes, slopes_bnd = results.estimate_loglog_slopes()

