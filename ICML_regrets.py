'''
Illustration of densities for the ICML 2015 talk

@author: Maximilian Balandat
@date: Jun 20, 2015
'''

# Set up infrastructure and basic problem parameters
import matplotlib as mpl
# mpl.use('Agg') # this is needed when running on a linux server over terminal
import multiprocessing as mp
import numpy as np
import datetime, os
import pickle
from ContNoRegret.Domains import hollowbox, unitbox
from ContNoRegret.LossFunctions import random_QuadraticLosses, random_AffineLosses, random_PolynomialLosses
from ContNoRegret.NoRegretAlgos import ContNoRegretProblem
from ContNoRegret.utils import CNR_worker, plot_results, save_results
from ContNoRegret.animate import save_animations_ICML
from ContNoRegret.Potentials import ExponentialPotential

# this is the location of the folder for the results
# results_path = '/home/max/Documents/CNR_results/'
results_path = '/Users/balandat/Documents/Code/Continuous_No-Regret/results/'
desc = 'ICML_regrets'
tmpfolder = '/Volumes/tmp/' # if possible, choose this to be a RamDisk

# some flags for keeping a record of the simulation parameters
save_res = True
show_plots = False
save_anims = False
show_anims = False

T = 500 # Time horizon
M = 10.0 # Uniform bound on the function (in the dual norm)
L = 5.0 # Uniform bound on the Lipschitz constant
N = 2500 # Number of parallel algorithm instances
Ngrid = 500000 # Number of gridpoints for the sampling step

# before running the computation, read this file so we can later save a copy in the results folder
with open(__file__, 'r') as f:
    thisfile = f.read()

dom = hollowbox(3)
epsilon = 0.2
# mus = (1-epsilon)*dom.vertices()[0] + epsilon*dom.sample_uniform(T)
mus = dom.sample_uniform(T)

problems = []
prob_descs = ['Affine', 'Quadratic', 'Polynomial']

# Affine Losses
lossfuncs, M = random_AffineLosses(dom, L, T, d=2)
problems.append(ContNoRegretProblem(dom, lossfuncs, L, M, desc=desc))
# Quadratic Losses
lossfuncs, Mnew, lambdamax = random_QuadraticLosses(dom, mus, L, M, pd=True)
problems.append(ContNoRegretProblem(dom, lossfuncs, L, Mnew, desc=desc))
# Polynomial Losses
lossfuncs = random_PolynomialLosses(dom, T, M, L, 4, [1,2,3,4])
problems.append(ContNoRegretProblem(dom, lossfuncs, L, M, desc=desc))

# only do Hedge
pot = ExponentialPotential()

# the following runs fine if the script is the __main__ method, but crashes when running from ipython
pool = mp.Pool(4)
processes = []

for problem, probdesc in zip(problems, prob_descs): 
    processes.append(pool.apply_async(CNR_worker, (problem, N,'DA'), {'opt_rate':True, 'Ngrid':Ngrid, 
				     'potential':pot, 'pid':len(processes), 'tmpfolder':tmpfolder, 'label':probdesc}))
pool.close()
# collect the results
results = [process.get() for process in processes]
pool.join()


# plot results and/or save a persistent copy (pickled) of the detailed results
timenow = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') # create a time stamp
results_directory = '{}{}/'.format(results_path, timenow)
  
if save_res:   
    os.makedirs(results_directory, exist_ok=True) # this could probably use a safer implementation
    # plot_results(results, 100, results_directory, show_plots)
    save_results(results, results_directory)  
    # store the previously read-in contents of this file in the results folder
    with open(results_directory+str(__file__), 'w') as f:
        f.write(thisfile)
else:
    plot_results(results, offset=100)
