# Set up infrastructure and basic problem parameters
import multiprocessing as mp
import numpy as np
import datetime, os
from ContNoRegret.NoRegretAlgos import ContNoRegretProblem
from ContNoRegret.Domains import nBox, UnionOfDisjointnBoxes, DifferenceOfnBoxes, unitbox, hollowbox
from ContNoRegret.LossFunctions import random_PolynomialLosses, random_AffineLosses, random_QuadraticLosses, PolynomialLossFunction
from ContNoRegret.NoRegretAlgos import ContNoRegretProblem
from ContNoRegret.utils import CNR_worker, plot_results, save_results, circular_tour
from ContNoRegret.animate import save_animations
from ContNoRegret.Potentials import (ExponentialPotential, IdentityPotential, pNormPotential, CompositePotential,
                                        ExpPPotential, PExpPotential, HuberPotential, LogtasticPotential, FractionalLinearPotential)

from ContNoRegret.loss_params import *

# this is the location of the folder for the results
results_path = '/Users/balandat/Documents/Code/Continuous_No-Regret/results/'
desc = 'NIPS2_CNR_PolyNormBounds'
tmpfolder = '/Volumes/tmp/' # if possible, choose this to be a RamDisk

# some flags for keeping a record of the simulation parameters
save_res = True
show_plots = True
save_anims = False
show_anims = False

coeffs = coeffs + coeffs
exponents = exponents + exponents

T = len(coeffs) # Time horizon
L = 5.0 # Uniform bound on the Lipschitz constant
N = 2500 # Number of parallel algorithm instances
Ngrid = 500000 # Number of gridpoints for the sampling step

dom = unitbox(3)
nus = [0.05, 1]

# before running the computation, read this file so we can later save a copy in the results folder
with open(__file__, 'r') as f:
    thisfile = f.read()

# lossfuncs = []
# while len(lossfuncs) < T:
#     tmpfuncs = np.array(random_PolynomialLosses(dom, 10, M, L, 4, [0,1,2,3,4]))
#     normbounds = {nu: np.array([lossfunc.norm(2/nu, tmpfolder=tmpfolder) for lossfunc in tmpfuncs]) for nu in nus}
#     Ms = {nu: np.array(normbounds[nu]) for nu in nus}
#     for i in range(len(normbounds)):
#         if normbounds[nus[0]][i]/normbounds[nus[1]][i] > 5:
#             lossfuncs.append(tmpfuncs[i])
        

lossfuncs = [PolynomialLossFunction(dom, coeff, expo) for coeff,expo in zip(coeffs,exponents)]
Minf, M2 = np.max(inf_norms), np.max(two_norms)

# create Continuous No-Regret problem
prob = ContNoRegretProblem(dom, lossfuncs, L, Minf, desc='PolyNormBounds')
    
# Select a number of potentials for the Dual Averaging algorithm
potentials = [ExponentialPotential(), pNormPotential(1.05, M=Minf), pNormPotential(2, M=M2)] 
#[ExponentialPotential(), pNormPotential(1.05, M=Minf), pNormPotential(2, M=M2)]

  
# the following runs fine if the script is the __main__ method, but crashes when running from ipython
pool = mp.Pool(processes=mp.cpu_count()-1)
processes = []

DAkwargs = [{'opt_rate':True, 'Ngrid':Ngrid, 'potential':pot, 'pid':i, 
             'tmpfolder':tmpfolder, 'label':pot.desc} for i,pot in enumerate(potentials)]
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



