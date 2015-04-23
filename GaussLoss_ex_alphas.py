'''
Numerical example for the Continuous No-Regret Algorithm with quadratic loss functions

@author: Maximilian Balandat, Walid Krichene
@date: Dec 20, 2014
'''

from ContNoRegret.Domains import S
from ContNoRegret.Distributions import Uniform
from ContNoRegret.LossFunctions import GaussianLossFunction
from ContNoRegret.HedgeAlgorithm import GaussianNoRegretProblem
from scipy.stats import expon
from ContNoRegret.utils import create_random_Sigmas, plot_results

# set up some basic parameters
T = 10000
M = 10.0
Lbnd = 5.0 # Uniform bound on the Lipschitz constant
N = 5000
Ngrid = 200000

# # just a simple rectangle
# dom = Rectangle([-1.0, 1.0], [-1.0, 1.0])

# S domain
dom = S()

# create random means, uniformly over the domain (change that later to also be around the domain
mus = Uniform(dom).sample(T)
# create random covariance matrices, based on the Lipschitz bound and the uniform bound M
covs = create_random_Sigmas(dom.n, T, Lbnd, M, expon())
# create list of loss functions (for now: Ignore uniform bound M!)
lossfuncs = [GaussianLossFunction(dom, mu, cov, M) for mu,cov in zip(mus,covs)]
# create the problem object
gaussprob = GaussianNoRegretProblem(dom, lossfuncs, Lbnd, M)
    
# run the problem for different alphas and thetas
alphas = [0.125, 0.2, 0.3, 0.4]
thetas = [0.25, 0.25, 0.25, 0.25]
    
results = gaussprob.run_simulation(N, etas=None, alphas=alphas, thetas=thetas)    
plot_results(results, offset=1000, filename='figures/Gauss_alphas_S')
slopes, slopes_bnd = results.estimate_loglog_slopes()

