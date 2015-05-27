'''
Comparison of Dual Averaging algorithms

@author: Maximilian Balandat
@date: May 8, 2015
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
results_path = '/Users/balandat/Documents/Code/Continuous_No-Regret/results/'
desc = 'DA_Comparison'
tmpfolder = '/Volumes/tmp/' # if possible, choose this to be a RamDisk
save_res = True
show_plots = False
create_anims = True
show_anims = False

# before running the computation, read this file so we can later save a copy in the results folder
with open(__file__, 'r') as f:
    thisfile = f.read()

T = 5000 # Time horizon
M = 10.0 # Uniform bound on the function (L-infinity norm)
Lbnd = 5.0 # Uniform bound on the Lipschitz constant
N = 2500 # Number of parallel algorithm instances
Ngrid = 250000 # Number of gridpoints for the sampling step
dom = unitbox(2)
# dom = UnionOfDisjointnBoxes([nBox([(-1,0), (-1,0)]), nBox([(0,1), (0,1)])])
# dom = DifferenceOfnBoxes(nBox([(-1,1), (-1,1)]), [nBox([(-0.5,0.5), (-0.5,0.5)])])

# Now create some random loss functions
# d=2 means sampling the a vector uniformly at random from {x : ||x||_2<L}}
lossfuncs, M = random_AffineLosses(dom, Lbnd, T, d=2)

# mus = circular_tour(dom, T)
# mus_random = dom.sample_uniform(T)
# epsilon = 0.4
# mus = ((1-epsilon)*mus + epsilon*mus_random)
# lossfuncs, Mnew = random_QuadraticLosses(dom, mus, Lbnd, M, pd=True)

# testfunc = QuadraticLossFunction(dom, [0,0], np.array([[1,0],[0,1]]), 0)
# c = testfunc.min()
# lossfuncs = [QuadraticLossFunction(dom, [0,0], np.array([[1,0],[0,1]]), -c) for t in range(T)]

# print(M, Mnew)
# normbounds = {'1': [lossfunc.norm(1) for lossfunc in lossfuncs],
#               '2': [lossfunc.norm(2) for lossfunc in lossfuncs]}

# create Continuous No-Regret problem
prob = ContNoRegretProblem(dom, lossfuncs, Lbnd, M, desc=desc)

# choose learning rate parameters
thetas = [1]
alphas = [0.5]

# potentials
potentials = [ExponentialPotential(), pNormPotential(1.25), pNormPotential(1.5), pNormPotential(1.75)]#, CompositePotential(2)]
# , pNormPotential(1.5), ExpPPotential(1.5), PExpPotential(1.5),
#               CompositePotential(2), CompositePotential(4)]
# , CompositePotential(gamma=2), CompositePotential(gamma=4),
#               pNormPotential(1.25), pNormPotential(1.5), pNormPotential(1.75)]
# potentials = [ExponentialPotential(), CompositePotential(gamma=2), CompositePotential(gamma=4),
#               pNormPotential(1.25), pNormPotential(1.75)]
# potentials = [ExponentialPotential(), CompositePotential(gamma=2), CompositePotential(gamma=4), 
#               pNormPotential(1.25), pNormPotential(1.75), LogtasticPotential()]
# potentials = [pNormPotential(1.25), pNormPotential(1.5), pNormPotential(1.75), pNormPotential(2.0)]


# the following runs fine if the script is the __main__ method, but crashes when running from ipython
pool = mp.Pool(processes=mp.cpu_count()-1)

DAkwargs = [{'alphas':[pot.alpha_opt(dom.n)], 'thetas':thetas, 'Ngrid':Ngrid, 'potential':pot, 
           'pid':i, 'tmpfolder':tmpfolder, 'label':pot.desc} for i,pot in enumerate(potentials)]
processes = [pool.apply_async(CNR_worker, (prob, N, 'DA'), kwarg) for kwarg in DAkwargs]

# GPkwargs = {'alphas':alphas, 'thetas':thetas, 'Ngrid':Ngrid, 'pid':len(processes), 'tmpfolder':tmpfolder, 'label':'GP'}
# processes.append(pool.apply_async(CNR_worker, (prob, N, 'GP'), GPkwargs))
#
# OGDkwargs = {'alphas':alphas, 'thetas':thetas, 'Ngrid':Ngrid, 'pid':len(processes), 'tmpfolder':tmpfolder, 'label':'OGD'}
# processes.append(pool.apply_async(CNR_worker, (prob, N, 'OGD'), OGDkwargs))
# 
# ONSkwargs = {'alpha':0.1, 'Ngrid':Ngrid, 'pid':len(processes), 'tmpfolder':tmpfolder, 'label':'ONS'}
# processes.append(pool.apply_async(CNR_worker, (prob, N, 'ONS'), ONSkwargs))
#
# FTALkwargs = {'alpha':0.1, 'Ngrid':Ngrid, 'pid':len(processes), 'tmpfolder':tmpfolder, 'label':'ONS'}
# processes.append(pool.apply_async(CNR_worker, (prob, N, 'FTAL'), FTALkwargs))
# 
# EWOOkwargs = {'alpha':0.1, 'Ngrid':Ngrid, 'pid':len(processes), 'tmpfolder':tmpfolder, 'label':'EWOO'}
# processes.append(pool.apply_async(CNR_worker, (prob, N, 'EWOO'), EWOOkwargs))

# wait for the processes to finish an collect the results
results = [process.get() for process in processes]


# plot results and/or save a persistent copy (pickled) of the detailed results
if save_res:
    # create a time stamp for unambiguously naming the results folder
    timenow = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') 
    results_directory = '{}{}/'.format(results_path, timenow)
    os.makedirs(results_directory, exist_ok=True) # this could probably use a safer implementation  
    plot_results(results, 100, results_directory, show_plots)
    save_animations(results, 10, results_directory, show_anims)  
    save_results(results, results_directory)  
    # store the previously read-in contents of this file in the results folder
    with open(results_directory+str(__file__), 'w') as f:
        f.write(thisfile)
else:
    plot_results(results, offset=100)



