'''
Comparison of the Hedge algorithm with the simple greedy projection algorithm from Zinkevich

@author: Maximilian Balandat, Walid Krichene
@date: Feb 15, 2015
'''

from ContNoRegret.Domains import Rectangle, S
from ContNoRegret.Distributions import Uniform
from ContNoRegret.LossFunctions import GaussianLossFunction, QuadraticLossFunction
from ContNoRegret.HedgeAlgorithm import GaussianNoRegretProblem, QuadraticNoRegretProblem
from ContNoRegret.utils import compute_etaopt, plot_results, create_random_Q, create_random_Sigmas
from scipy.stats import expon

# set up some basic parameters
T = 10000
M = 10.0
Lbnd = 5.0 # Uniform bound on the Lipschitz constant
N = 2000
Ngrid = 250000

# # start with just a simple rectangle
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

# create random means, uniformly over the domain
mus_Q = Uniform(dom).sample(T)
# create random Q matrices, based on the Lipschitz bound and the uniform bound M
Qs = [create_random_Q(dom, mu, Lbnd, M) for mu in mus]
# create list of loss functions
lossfuncs = [QuadraticLossFunction(dom, mu, Q, 0.0) for mu,Q in zip(mus,Qs)]
# Create quadratic problem
quadprob = QuadraticNoRegretProblem(dom, lossfuncs, Lbnd, M)

# run the problem for eta_t = t^-0.5
thetas = [1, 1]
alphas = [0.3, 0.5]

results_gauss = [gaussprob.run_simulation(N, alphas=alphas, thetas=thetas, Ngrid=Ngrid, algo='hedge'),
                 gaussprob.run_simulation(N, alphas=alphas, thetas=thetas, Ngrid=Ngrid, algo='greedy')]
results_quad = [quadprob.run_simulation(N, alphas=alphas, thetas=thetas, Ngrid=Ngrid, algo='hedge'),
                quadprob.run_simulation(N, alphas=alphas, thetas=thetas, Ngrid=Ngrid, algo='greedy')]
plot_results(results_gauss, offset=1000, filename='figures/Comparison_Gauss_S')
plot_results(results_quad, offset=1000, filename='figures/Comparison_Quad_S')

