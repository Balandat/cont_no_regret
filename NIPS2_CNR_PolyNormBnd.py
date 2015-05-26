# Set up infrastructure and basic problem parameters
import matplotlib as mpl
mpl.use('Agg')
import multiprocessing as mp
import numpy as np
import datetime, os
from ContNoRegret.NoRegretAlgos import ContNoRegretProblem
from ContNoRegret.Domains import nBox, UnionOfDisjointnBoxes, DifferenceOfnBoxes, unitbox, hollowbox
from ContNoRegret.LossFunctions import random_PolynomialLosses, random_AffineLosses, random_QuadraticLosses, PolynomialLossFunction
from ContNoRegret.NoRegretAlgos import ContNoRegretProblem
from ContNoRegret.utils import CNR_worker, plot_results, save_results, circular_tour
from ContNoRegret.animate import save_animations
from ContNoRegret.Potentials import ExponentialPotential, IdentityPotential, pNormPotential
from ContNoRegret.loss_params import *

# this is the location of the folder for the results
results_path = '/Users/balandat/Documents/Code/Continuous_No-Regret/results/'
desc = 'NIPS2_CNR_PolyNormBounds'
tmpfolder = '/Volumes/tmp/' # if possible, choose this to be a RamDisk

# some flags for keeping a record of the simulation parameters
save_res = True
show_plots = False
save_anims = False
show_anims = False

T = 500 # Time horizon
L = 5.0 # Uniform bound on the Lipschitz constant
N = 2500 # Number of parallel algorithm instances
Ngrid = 300000 # Number of gridpoints for the sampling step

dom = unitbox(2)

# before running the computation, read this file so we can later save a copy in the results folder
with open(__file__, 'r') as f:
    thisfile = f.read()
        
# bootstrap loss functions by sampling from the list of stored functions
idx = np.random.choice(len(coeffs2), T)
coeffs = [coeffs2[i] for i in idx]
exponents = [exponents2[i] for i in idx]

lossfuncs = [PolynomialLossFunction(dom, coeff, expo) for coeff,expo in zip(coeffs2,exponents2)]
Minf, M2 = np.max(inf_norms2), np.max(two_norms2)

# create Continuous No-Regret problem
prob = ContNoRegretProblem(dom, lossfuncs, L, Minf, desc='PolyNormBounds')
    
# Select a number of potentials for the Dual Averaging algorithm
potentials = [ExponentialPotential(), pNormPotential(1+nus2[0], M=Minf), pNormPotential(1+nus2[0], M=M2)] 
#[ExponentialPotential(), pNormPotential(1.05, M=Minf), pNormPotential(2, M=M2)]

  
# the following runs fine if the script is the __main__ method, but crashes when running from ipython
pool = mp.Pool(processes=mp.cpu_count()-1)
processes = []

DAkwargs = [{'opt_rate':True, 'Ngrid':Ngrid, 'potential':pot, 'pid':i, 
             'tmpfolder':tmpfolder, 'label':'norm_'+pot.desc} for i,pot in enumerate(potentials)]
processes += [pool.apply_async(CNR_worker, (prob, N, 'DA'), kwarg) for kwarg in DAkwargs]

# wait for the processes to finish an collect the results
results = [process.get() for process in processes]
 
# plot results and/or save a persistent copy (pickled) of the detailed results
timenow = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')# create a time stamp for unambiguously naming the results folder
results_directory = '{}{}/'.format(results_path, timenow)
   
if save_res:   
    os.makedirs(results_directory, exist_ok=True) # this could probably use a safer implementation
    plot_results(results, 100, results_directory, show_plots)
    if save_anims:
        save_animations(results, 10, results_directory, show_anims)  
    save_results(results, results_directory)  
    # store the previously read-in contents of this file in the results folder
    with open(results_directory+str(__file__), 'w') as f:
        f.write(thisfile)
else:
    plot_results(results, offset=100)


