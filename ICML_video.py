'''
Illustration of densities for the ICML 2015 talk

@author: Maximilian Balandat
@date: Jun 20, 2015
'''

# Set up infrastructure and basic problem parameters
import matplotlib as mpl
mpl.use('Agg') # this is needed when running on a linux server over terminal
import multiprocessing as mp
import numpy as np
import datetime, os
import pickle
from ContNoRegret.Domains import S
from ContNoRegret.LossFunctions import random_QuadraticLosses
from ContNoRegret.NoRegretAlgos import ContNoRegretProblem
from ContNoRegret.utils import CNR_worker, plot_results, save_results
from ContNoRegret.animate import save_animations_ICML
from ContNoRegret.Potentials import ExponentialPotential

# this is the location of the folder for the results
# results_path = '/home/max/Documents/CNR_results/'
results_path = '/Users/balandat/Documents/Code/Continuous_No-Regret/results/'
desc = 'ICML_video'
tmpfolder = '/Volumes/tmp/' # if possible, choose this to be a RamDisk

# some flags for keeping a record of the simulation parameters
save_res = True
show_plots = False
save_anims = False
show_anims = False

T = 10000 # Time horizon
M = 10.0 # Uniform bound on the function (in the dual norm)
L = 5.0 # Uniform bound on the Lipschitz constant
N = 2500 # Number of parallel algorithm instances
Ngrid = 500000 # Number of gridpoints for the sampling step

# before running the computation, read this file so we can later save a copy in the results folder
with open(__file__, 'r') as f:
    thisfile = f.read()

dom = S()
epsilon = 0.2
# mus = (1-epsilon)*np.array([[1.5, 2.5]]*T) + epsilon*dom.sample_uniform(T)
mus = np.array([3*np.random.rand(T), 2+np.random.rand(T)]).T

lossfuncs, Mnew, lambdamax = random_QuadraticLosses(dom, mus, L, M, pd=True) 
Mnew = np.max([lossfunc.max() for lossfunc in lossfuncs])
problem = ContNoRegretProblem(dom, lossfuncs, L, Mnew, desc=desc)
pot = ExponentialPotential()

theta = np.sqrt((pot.c_omega*(dom.n - np.log(dom.v)) 
                 + pot.d_omega*dom.v)/2/Mnew**2)
theta_scalings = [1, 3]
plt_titles = [r'$\eta_t = {{{0:.2f}}}$'.format(theta*scal) + r' $\sqrt{\frac{\log t}{t}}$' for scal in theta_scalings]

# the following runs fine if the script is the __main__ method, but crashes when running from ipython
pool = mp.Pool(4)
processes = []

for scal in theta_scalings:
    etas = scal*theta*np.sqrt(np.log(1+np.arange(T)+1)/(1+np.arange(T)))
    processes.append(pool.apply_async(CNR_worker, (problem, N,'DA'), {'opt_rate':False, 'Ngrid':Ngrid, 
				     'potential':pot, 'pid':len(processes), 'tmpfolder':tmpfolder, 'etas':etas, 
                     'label':'v={0:.2f}, '.format(problem.domain.v)+pot.desc, 'animate':[]})) 
pool.close()
# collect the sesults
results = [process.get() for process in processes]
pool.join()


# plot results and/or save a persistent copy (pickled) of the detailed results
timenow = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') # create a time stamp for unambiguously naming the results folder
results_directory = '{}{}/'.format(results_path, timenow)
  
if save_res:   
    os.makedirs(results_directory, exist_ok=True) # this could probably use a safer implementation
    # plot_results(results, 100, results_directory, show_plots)
    if save_anims:
        save_animations_ICML(results[0], results[1], 10, results_directory+'/animation.mp4', 
            show_anims, titles=plt_titles, figsize=(8,4))  
    save_results(results, results_directory)  
    # store the previously read-in contents of this file in the results folder
    with open(results_directory+str(__file__), 'w') as f:
        f.write(thisfile)
else:
    plot_results(results, offset=100)

