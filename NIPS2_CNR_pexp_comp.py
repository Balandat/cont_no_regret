'''
Comparison of Continuous No-Regret Algorithms for the 2nd NIPS paper

@author: Maximilian Balandat
@date: May 25, 2015
'''

# Set up infrastructure and basic problem parameters
import matplotlib as mpl
mpl.use('Agg') # this is needed when running on a linux server over terminal
import multiprocessing as mp
import numpy as np
import datetime, os
from ContNoRegret.Domains import unitbox
from ContNoRegret.LossFunctions import random_QuadraticLosses
from ContNoRegret.NoRegretAlgos import ContNoRegretProblem
from ContNoRegret.utils import CNR_worker, plot_results, save_results, circular_tour
from ContNoRegret.animate import save_animations
from ContNoRegret.Potentials import ExponentialPotential, ExpPPotential, pExpPotential

# this is the location of the folder for the results
results_path = '/Users/balandat/Documents/Code/Continuous_No-Regret/results/'
desc = 'NIPS2_CNR_pexp_comparison'
tmpfolder = '/Volumes/tmp/' # if possible, choose this to be a RamDisk

# some flags for keeping a record of the simulation parameters
save_res = True
show_plots = False
save_anims = False
show_anims = False

T = 500 # Time horizon
M = 10.0 # Uniform bound on the function (in the dual norm)
L = 5.0 # Uniform bound on the Lipschitz constant
N = 5 # Number of parallel algorithm instances
Ngrid = 250000 # Number of gridpoints for the sampling step
dom = unitbox(2)

# before running the computation, read this file so we can later save a copy in the results folder
with open(__file__, 'r') as f:
    thisfile = f.read()

ps = [1.25, 1.5, 1.75]
gammas = [1, 3, 10]

# Now create some random loss functions
mus = dom.sample_uniform(T)
lossfuncs, Mnew, lambdamax = random_QuadraticLosses(dom, mus, L, M, pd=True)
prob = ContNoRegretProblem(dom, lossfuncs, L, Mnew, desc=desc)
# Ms_dual = {p : np.max([lossfunc.norm(2/(p-1), tmpfolder=tmpfolder) for lossfunc in lossfuncs]) for p in ps}
# 
# # Select a couple of potentials for the Dual Averaging algorithm
# potentials = [ExponentialPotential()]
# for p in ps:
#     for gamma in gammas:
#         potentials += [pExpPotential(p, gamma=gamma, M=Ms_dual[p]), 
#                        ExpPPotential(p, gamma=gamma, M=Ms_dual[p])]

# the following runs fine if the script is the __main__ method, but crashes when running from ipython
pool = mp.Pool(processes=mp.cpu_count()-1)
processes = []

processes.append(pool.apply_async(CNR_worker, (prob,N,'Greedy'), {'Ngrid':Ngrid, 'pid':len(processes), 'label':'Greedy'}))

# for pot in potentials:
#     processes.append(pool.apply_async(CNR_worker, (prob,N,'DA'), {'opt_rate':True, 'Ngrid':Ngrid,
# 				  'potential':pot, 'pid':len(processes), 'tmpfolder':tmpfolder, 'label':pot.desc, 
#                   'results_path':results_path})) 

# wait for the processes to finish an collect the results
results = [process.get() for process in processes]

# read the results from file
#results = []
#for rfile in resultfiles:
#    with open(rfile, 'rb') as f:
#        results.append(pickle.load(f))
#

# plot results and/or save a persistent copy (pickled) of the detailed results
timenow = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')# create a time stamp for unambiguously naming the results folder
results_directory = '{}{}/'.format(results_path, timenow)
   
if save_res:   
    os.makedirs(results_directory, exist_ok=True) # this could probably use a safer implementation
#     plot_results(results, 100, results_directory, show_plots)
#     if save_anims:
#         save_animations(results, 10, results_directory, show_anims)  
    save_results(results, results_directory)  
    # store the previously read-in contents of this file in the results folder
    with open(results_directory+str(__file__), 'w') as f:
        f.write(thisfile)
# else:
#     plot_results(results, offset=100)

