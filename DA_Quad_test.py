'''
Comparison of the Hedge algorithm with the simple greedy projection algorithm from Zinkevich

@author: Maximilian Balandat, Walid Krichene
@date: Feb 15, 2015
'''

from ContNoRegret.Domains import Rectangle, S
from ContNoRegret.Distributions import Uniform
from ContNoRegret.LossFunctions import QuadraticLossFunction
from ContNoRegret.HedgeAlgorithm import QuadraticNoRegretProblem
from ContNoRegret.utils import plot_results, create_random_Q

# set up some basic parameters
T = 7500
M = 10.0
Lbnd = 5.0 # Uniform bound on the Lipschitz constant
N = 2500
Ngrid = 250000

# start with just a simple rectangle
# dom = Rectangle([-1.0, 1.0], [-1.0, 1.0])

# we can also do something more compliated. yay!
dom = S()

# create random means, uniformly over the domain
mus = Uniform(dom).sample(T)
# create random Q matrices, based on the Lipschitz bound and the uniform bound M
Qs = [create_random_Q(dom, mu, Lbnd, M) for mu in mus]
# create list of loss functions
lossfuncs = [QuadraticLossFunction(dom, mu, Q, 0.0) for mu,Q in zip(mus,Qs)]
# Create quadratic problem
quadprob = QuadraticNoRegretProblem(dom, lossfuncs, Lbnd, M)

# run the problem for eta_t = t^-0.5
thetas = [1]
alphas = [0.5]

results = [quadprob.run_simulation(N, alphas=alphas, thetas=thetas, Ngrid=Ngrid, algo='DA_Quad', gamma=1.5),
           quadprob.run_simulation(N, alphas=alphas, thetas=thetas, Ngrid=Ngrid, algo='DA_Quad', gamma=1.75),
           quadprob.run_simulation(N, alphas=alphas, thetas=thetas, Ngrid=Ngrid, algo='DA_Quad', gamma=2.5),
           quadprob.run_simulation(N, alphas=alphas, thetas=thetas, Ngrid=Ngrid, algo='DA_Quad', gamma=5)]
#            quadprob.run_simulation(N, alphas=alphas, thetas=thetas, Ngrid=Ngrid, algo='hedge')]
plot_results(results, offset=500, filename='figures/DA_Quad_betas')

