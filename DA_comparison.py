'''
Comparison of Dual Averaging algorithms

@author: Maximilian Balandat
@date: May 5, 2015
'''

# Set up infrastructure and basic problem parameters
import multiprocessing as mp
import numpy as np
from ContNoRegret.Domains import nBox, UnionOfDisjointnBoxes
from ContNoRegret.LossFunctions import AffineLossFunction
from ContNoRegret.NoRegretAlgos import ContNoRegretProblem
from ContNoRegret.utils import random_AffineLosses, random_QuadraticLosses, plot_results, CNR_worker
from ContNoRegret.DualAveraging import ExponentialPotential, IdentityPotential, CompositeOmegaPotential, pNormPotential, LogtasticPotential

# this is the location of the folder for the results
results_path = '/Users/balandat/Documents/Code/Continuous_No-Regret/results/figures/'
desc = 'DA_Comparison'

T = 500 # Time horizon
M = 10.0 # Uniform bound on the funciton (L-infinity norm)
Lbnd = 5.0 # Uniform bound on the Lipschitz constant
N = 2500 # Number of parallel algorithm instances
Ngrid = 250000 # Number of gridpoints for the sampling step
dom = nBox([(-1,1), (-1,1)])#, (-1,1)]) # domain is a unit ball (1-norm) in 2D
# dom = UnionOfDisjointnBoxes([nBox([(-1,0), (-1,0)]), nBox([(0,1), (0,1)])])

# Now create some random loss functions
# d=2 means sampling the a vector uniformly at random from {x : ||x||_2<L}}
# lossfuncs, M = random_AffineLosses(dom, Lbnd, T, d=2)

epsilon = 0.3
mus = ((1-epsilon)*(0.5 + 0.5*np.array([np.sin(np.linspace(0,2*np.pi,T)), np.cos(np.linspace(0,2*np.pi,T))])).T 
       + epsilon*dom.sample_uniform(T))
lossfuncs = random_QuadraticLosses(dom, mus, Lbnd, M)

# create Continuous No-Regret problem
prob = ContNoRegretProblem(dom, lossfuncs, Lbnd, M, desc=desc)

# choose learning rate parameters
thetas = [1]
alphas = [0.5]

# potentials
potentials = [ExponentialPotential(), CompositeOmegaPotential(gamma=2), LogtasticPotential(), pNormPotential(1.25)]
# potentials = [pNormPotential(1.25), pNormPotential(1.5), pNormPotential(1.75), pNormPotential(2.0)]

# the following runs fine if the script is the __main__ method, but crashes when running from ipython
pool = mp.Pool(processes=4)

kwargs = [{'alphas':alphas, 'thetas':thetas, 'Ngrid':Ngrid, 'potential':pot, 
           'pid':i, 'label':pot.desc} for i,pot in enumerate(potentials)]
processes = [pool.apply_async(CNR_worker, (prob, N, 'DA'), kwarg) for kwarg in kwargs]
results = [process.get() for process in processes]

plot_results(results, offset=100, path=results_path)     

