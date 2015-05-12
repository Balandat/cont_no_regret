'''
Numerical example for the Continuous No-Regret Algorithm with quadratic loss functions

@author: Maximilian Balandat, Walid Krichene
@date: Dec 20, 2014
'''

from ContNoRegret.Domains import S
from ContNoRegret.Distributions import Uniform
from ContNoRegret.LossFunctions import GaussianLossFunction
from ContNoRegret.HedgeAlgorithm import GaussianNoRegretProblem
from ContNoRegret.utils import create_random_Sigmas, compute_etaopt, plot_results
from scipy.stats import expon

# set up some basic parameters
T = 10000
M = 10.0
Lbnd = 5.0 # Uniform bound on the Lipschitz constant
N = 2500
Ngrid = 200000

# # just a simple rectangle
# dom = Rectangle([-1.0, 1.0], [-1.0, 1.0])

# domain S
dom = S() 

# create random means, uniformly over the domain
mus = Uniform(dom).sample(T)
# create random covariance matrices, based on the Lipschitz bound and the uniform bound M
covs = create_random_Sigmas(dom.n, T, Lbnd, M, expon())
# create list of loss functions (for now: Ignore uniform bound M!)
lossfuncs = [GaussianLossFunction(dom, mu, cov, M) for mu,cov in zip(mus,covs)]
# Create gauss problem for the largest horizon
gaussprob = GaussianNoRegretProblem(dom, lossfuncs, Lbnd, M)

# run the problem for different constant rates etas
etaopts = {}
Ts = [2500, 7500]
for T in Ts:
    etaopts[T] = compute_etaopt(dom, M, T)
etas = [0.1, 0.2]

results = gaussprob.run_simulation(N, etaopts=etaopts, etas=etas, Ngrid=Ngrid)    
plot_results(results, offset=1000, filename='figures/Gauss_etas_S')
slopes, slopes_bnd = results.estimate_loglog_slopes()
