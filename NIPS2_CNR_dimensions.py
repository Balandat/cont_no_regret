'''
Comparison of Continuous No-Regret Algorithms for the 2nd NIPS paper

@author: Maximilian Balandat
@date: May 22, 2015
'''

# Set up infrastructure and basic problem parameters
import matplotlib as mpl
mpl.use('Agg') # this is needed when running on a linux server over terminal
import multiprocessing as mp
import numpy as np
import datetime, os
from ContNoRegret.Domains import nBox, UnionOfDisjointnBoxes, DifferenceOfnBoxes, unitbox, hollowbox
from ContNoRegret.LossFunctions import QuadraticLossFunction, random_QuadraticLosses
from ContNoRegret.NoRegretAlgos import ContNoRegretProblem
from ContNoRegret.utils import CNR_worker, plot_results, save_results, circular_tour
from ContNoRegret.animate import save_animations
from ContNoRegret.Potentials import (ExponentialPotential, IdentityPotential, pNormPotential, CompositePotential,
                                        ExpPPotential, PExpPotential, HuberPotential, LogtasticPotential, FractionalLinearPotential)

# this is the location of the folder for the results
results_path = '/home/max/Documents/CNR_results/'
desc = 'NIPS2_CNR_dimensions'
tmpfolder = '/media/tmp/' # if possible, choose this to be a RamDisk

# some flags for keeping a record of the simulation parameters
save_res = True
show_plots = False
save_anims = False
show_anims = False

T = 5000 # Time horizon
M = 10.0 # Uniform bound on the function (in the dual norm)
L = 5.0 # Uniform bound on the Lipschitz constant
N = 2500 # Number of parallel algorithm instances
Ngrid = 1000000 # Number of gridpoints for the sampling step
H = 0.1 # strict convexity parameter (lower bound on evals of Q)

doms = [unitbox(n+2) for n in range(3)]

# before running the computation, read this file so we can later save a copy in the results folder
with open(__file__, 'r') as f:
    thisfile = f.read()

problems = []

# loop over the domains with different dimension
for dom in doms:

    # Now create some random loss functions
    mus = dom.sample_uniform(T)
    lossfuncs, Mnew, lambdamax = random_QuadraticLosses(dom, mus, L, M, pd=True, H=H)
    alpha_ec = H/dom.diameter/lambdamax
    #M2 = np.max([lossfunc.norm(2, tmpfolder=tmpfolder) for lossfunc in lossfuncs])

    # create the problem
    problems.append(ContNoRegretProblem(dom, lossfuncs, L, Mnew, desc=desc))
  
# Select a couple of potentials for the Dual Averaging algorithm
potentials = [ExponentialPotential(), pNormPotential(1.5)]
  
# the following runs fine if the script is the __main__ method, but crashes when running from ipython
pool = mp.Pool(processes=mp.cpu_count()-1)
processes = []

DAkwargs = [{'opt_rate':True, 'Ngrid':Ngrid, 'potential':pot, 'pid':i, 
             'tmpfolder':tmpfolder, 'label':pot.desc} for i,pot in enumerate(potentials)]
processes += [pool.apply_async(CNR_worker, (prob, N, 'DA'), kwarg) for kwarg in DAkwargs for prob in problems]
     
GPkwargs = {'Ngrid':Ngrid, 'pid':len(processes), 'tmpfolder':tmpfolder, 'label':'GP'}
processes += [pool.apply_async(CNR_worker, (prob, N, 'GP'), GPkwargs) for prob in problems]
     
OGDkwargs = {'H':H, 'Ngrid':Ngrid, 'pid':len(processes), 'tmpfolder':tmpfolder, 'label':'OGD'}
processes += [pool.apply_async(CNR_worker, (prob, N, 'OGD'), OGDkwargs) for prob in problems]
    
ONSkwargs = {'alpha':alpha_ec, 'Ngrid':Ngrid, 'pid':len(processes), 'tmpfolder':tmpfolder, 'label':'ONS'}
processes += [pool.apply_async(CNR_worker, (prob, N, 'ONS'), ONSkwargs) for prob in problems] 
   
FTALkwargs = {'alpha':alpha_ec, 'Ngrid':Ngrid, 'pid':len(processes), 'tmpfolder':tmpfolder, 'label':'FTAL'}
processes += [pool.apply_async(CNR_worker, (prob, N, 'FTAL'), FTALkwargs) for prob in problems]
  
EWOOkwargs = {'alpha':alpha_ec, 'Ngrid':Ngrid, 'pid':len(processes), 'tmpfolder':tmpfolder, 'label':'EWOO'}
processes += [pool.apply_async(CNR_worker, (prob, N, 'EWOO'), EWOOkwargs) for prob in problems]

# wait for the processes to finish an collect the results
results = [process.get() for process in processes]
  
# plot results and/or save a persistent copy (pickled) of the detailed results
timenow = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')# create a time stamp for unambiguously naming the results folder
results_directory = '{}{}/'.format(results_path, timenow)
  
if save_res:   
    os.makedirs(results_directory, exist_ok=True) # this could probably use a safer implementation
#    plot_results(results, 100, results_directory, show_plots)
#    if save_anims:
#        save_animations(results, 10, results_directory, show_anims)  
    save_results(results, results_directory)  
    # store the previously read-in contents of this file in the results folder
    with open(results_directory+str(__file__), 'w') as f:
        f.write(thisfile)
#else:
#    plot_results(results, offset=100)

