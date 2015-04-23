'''
Simulation of Dual Averaging for generic losses

@author: Maximilian Balandat, Walid Krichene
@date: Mar 5, 2015
'''

from ContNoRegret.Domains import Rectangle, S
from ContNoRegret.Distributions import Uniform
from ContNoRegret.LossFunctions import GaussianLossFunction
from ContNoRegret.HedgeAlgorithm import GaussianNoRegretProblem
from ContNoRegret.utils import plot_results, create_random_Sigmas
from ContNoRegret.DualAveraging import ExponentialZeroPotential, CompositeZeroPotential
from scipy.stats import expon

# set up some basic parameters
T = 100
M = 10.0
Lbnd = 5.0 # Uniform bound on the Lipschitz constant
N = 2500
Ngrid = 250000

# start with just a simple rectangle
dom = Rectangle([-1.0, 1.0], [-1.0, 1.0])

# # we can also do something more compliated. yay!
# dom = S()

# create random means, uniformly over the domain
mus = Uniform(dom).sample(T)
covs = create_random_Sigmas(dom.n, T, Lbnd, M, expon())
lossfuncs = [GaussianLossFunction(dom, mu, cov, M) for mu,cov in zip(mus,covs)]
gaussprob = GaussianNoRegretProblem(dom, lossfuncs, Lbnd, M)

# run the problem for eta_t = t^-0.5
thetas = [1]
alphas = [0.5]

# zero_pot = ExponentialZeroPotential()
zero_pot = CompositeZeroPotential(gamma=1.5)

results = [gaussprob.run_simulation(N, alphas=alphas, thetas=thetas, Ngrid=Ngrid, algo='DA_generic', 
                                    potential=zero_pot, ngrid=(100,100))]
plot_results(results, offset=25, filename='figures/DA_generic')

