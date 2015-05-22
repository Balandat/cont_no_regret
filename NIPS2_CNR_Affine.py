'''
Comparison of Continuous No-Regret Algorithms for the 2nd NIPS paper

@author: Maximilian Balandat
@date: May 22, 2015
'''

# Set up infrastructure and basic problem parameters
import multiprocessing as mp
import numpy as np
import datetime, os
from ContNoRegret.Domains import nBox, UnionOfDisjointnBoxes, DifferenceOfnBoxes, unitbox, hollowbox
from ContNoRegret.LossFunctions import AffineLossFunction, QuadraticLossFunction, random_AffineLosses, random_QuadraticLosses
from ContNoRegret.NoRegretAlgos import ContNoRegretProblem
from ContNoRegret.utils import CNR_worker, plot_results, save_results, circular_tour
from ContNoRegret.animate import save_animations
from ContNoRegret.Potentials import (ExponentialPotential, IdentityPotential, pNormPotential, CompositePotential,
                                        ExpPPotential, PExpPotential, HuberPotential, LogtasticPotential, FractionalLinearPotential)

# this is the location of the folder for the results
results_path = '/home/max/Documents/CNR_results/'
desc = 'NIPS2_CNR_Affine'
tmpfolder = '/media/tmp/' # if possible, choose this to be a RamDisk

# some flags for keeping a record of the simulation parameters
save_res = True
show_plots = False
save_anims = False
show_anims = False

T = 500 # Time horizon
M = 10.0 # Uniform bound on the function (in the dual norm)
Lbnd = 15.0 # Uniform bound on the Lipschitz constant
N = 2500 # Number of parallel algorithm instances
Ngrid = 250000 # Number of gridpoints for the sampling step
dom = unitbox(2)

# before running the computation, read this file so we can later save a copy in the results folder
with open(__file__, 'r') as f:
    thisfile = f.read()
    
# Now create some random loss functions
# d=2 means sampling the a vector uniformly at random from {x : ||x||_2<L}}
lossfuncs, M = random_AffineLosses(dom, Lbnd, T, d=2)

# compute bounds on the norms
normbounds = {'{}'.format(p): [lossfunc.norm(p, tmpfolder=tmpfolder) for lossfunc in lossfuncs] for p in [1,2,np.Infinity]}
normmax = {key:np.max(val) for key,val in normbounds.items()}
print(normmax)
  
# create Continuous No-Regret problem
prob = ContNoRegretProblem(dom, lossfuncs, Lbnd, M, desc=desc)
  
# Select a number of potentials for the Dual Averaging algorithm
potentials = [ExponentialPotential(), FractionalLinearPotential(1.5), FractionalLinearPotential(2), FractionalLinearPotential(3)]#, pNormPotential(1.25), pNormPotential(1.75)]
  
# the following runs fine if the script is the __main__ method, but crashes when running from ipython
pool = mp.Pool(processes=mp.cpu_count()-1)
processes = []

DAkwargs = [{'opt_rate':True, 'Ngrid':Ngrid, 'potential':pot, 'pid':i, 
             'tmpfolder':tmpfolder, 'label':pot.desc} for i,pot in enumerate(potentials)]
processes += [pool.apply_async(CNR_worker, (prob, N, 'DA'), kwarg) for kwarg in DAkwargs]
  
# GPkwargs = {'Ngrid':Ngrid, 'pid':len(processes), 'tmpfolder':tmpfolder, 'label':'GP'}
# processes.append(pool.apply_async(CNR_worker, (prob, N, 'GP'), GPkwargs))

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

