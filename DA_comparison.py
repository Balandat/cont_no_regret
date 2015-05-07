'''
Comparison of Dual Averaging algorithms

@author: Maximilian Balandat
@date: May 7, 2015
'''

# Set up infrastructure and basic problem parameters
import multiprocessing as mp
import numpy as np
import datetime, os
from ContNoRegret.Domains import nBox, UnionOfDisjointnBoxes, DifferenceOfnBoxes
from ContNoRegret.LossFunctions import AffineLossFunction, QuadraticLossFunction, random_AffineLosses, random_QuadraticLosses
from ContNoRegret.NoRegretAlgos import ContNoRegretProblem
from ContNoRegret.utils import CNR_worker, plot_results, save_results
from ContNoRegret.DualAveraging import ExponentialPotential, IdentityPotential, CompositeOmegaPotential, pNormPotential, LogtasticPotential, HuberPotential

# this is the location of the folder for the results
results_path = '/Users/balandat/Documents/Code/Continuous_No-Regret/results/'
desc = 'DA_Comparison'
tmpfolder = '/Volumes/tmp/' # if possible, choose this to be a RamDisk

# before running the computation, read this file so we can later save a copy in the results folder
with open(__file__, 'r') as f:
    thisfile = f.read()

T = 500 # Time horizon
M = 10.0 # Uniform bound on the function (L-infinity norm)
Lbnd = 5.0 # Uniform bound on the Lipschitz constant
N = 2500 # Number of parallel algorithm instances
Ngrid = 250000 # Number of gridpoints for the sampling step
dom = nBox([(-1,1), (-1,1)]) #, (-1,1)]) # domain is a unit ball (1-norm) in 2D
# dom = UnionOfDisjointnBoxes([nBox([(-1,0), (-1,0)]), nBox([(0,1), (0,1)])])
# dom = DifferenceOfnBoxes(nBox([(-1,1), (-1,1)]), [nBox([(-0.5,0.5), (-0.5,0.5)])])

# Now create some random loss functions
# d=2 means sampling the a vector uniformly at random from {x : ||x||_2<L}}
lossfuncs, M = random_AffineLosses(dom, Lbnd, T, d=2)

mus_random = dom.sample_uniform(T)
# epsilon = 0.3
# mus = ((1-epsilon)*(0.5 + 0.5*np.array([np.sin(np.linspace(0,2*np.pi,T)), np.cos(np.linspace(0,2*np.pi,T))])).T 
#        + epsilon*mus_random)
# lossfuncs, Mnew = random_QuadraticLosses(dom, mus_random, Lbnd, M, pd=False)

# print(M, Mnew)
# normbounds = {'1': [lossfunc.norm(1) for lossfunc in lossfuncs],
#               '2': [lossfunc.norm(2) for lossfunc in lossfuncs]}

# create Continuous No-Regret problem
prob = ContNoRegretProblem(dom, lossfuncs, Lbnd, M, desc=desc)

# choose learning rate parameters
thetas = [1]
alphas = [0.5]

# potentials
potentials = [ExponentialPotential(), CompositeOmegaPotential(gamma=2), CompositeOmegaPotential(gamma=4),
              pNormPotential(1.25), pNormPotential(1.75)]
# potentials = [ExponentialPotential(), CompositeOmegaPotential(gamma=2), CompositeOmegaPotential(gamma=4), 
#               pNormPotential(1.25), pNormPotential(1.75), LogtasticPotential()]
# potentials = [pNormPotential(1.25), pNormPotential(1.5), pNormPotential(1.75), pNormPotential(2.0)]


# the following runs fine if the script is the __main__ method, but crashes when running from ipython
pool = mp.Pool(processes=7)

kwargs = [{'alphas':alphas, 'thetas':thetas, 'Ngrid':Ngrid, 'potential':pot, 
           'pid':i, 'tmpfolder':tmpfolder, 'label':pot.desc} for i,pot in enumerate(potentials)]
processes = [pool.apply_async(CNR_worker, (prob, N, 'DA'), kwarg) for kwarg in kwargs]
results = [process.get() for process in processes]


# create a time stamp for unambiguously naming the results folder
timenow = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') 
results_directory = '{}{}/'.format(results_path, timenow)
os.makedirs(results_directory, exist_ok=True) # this could probably use a safer implementation  

# store the previously read-in contents of this file in the results folder
with open(results_directory+str(__file__), 'w') as f:
    f.write(thisfile)

# plot results and save a persistent copy (pickled) of the detailed results
plot_results(results, offset=100, directory=results_directory)
save_results(results, directory=results_directory)  



