'''
Second numerical example for the No-Regret work

@author: Maximilian Balandat, Walid Krichene
@date: Oct 9, 2014
'''


# from ContNoRegret import Domain, GaussianLossFunction, Problem
#  
# # define some parameters
# T = 0
# 
# # Create a domain
# S = Domain(parameters)
# 
# # Create a list of loss functions
# losses = []
# for t in range(T):
#     losses.append(LossFunction(parameters))
#     
# # Create a Continuous No-Regret Problem object
# prob = Problem(S, losses)
# 
# # Simulate the No-Regret algorithm and create some statistics

import numpy as np
from ContNoRegret import *
from matplotlib import pyplot as plt

dom = Rectangle([-1, 1], [-1, 1])
T = 10
# create random means, uniformly over the domain

# create random covariance matrices, using a priory bound 
points = np.array([-1, -1]) + 2*np.random.rand(1000,2)
means = [np.array([-0.5, -0.5]), np.array([0.5, 0.5])]
covs = [0.1*np.array([[1, 0.25], [0.25, 1]]), 0.015*np.array([[2, 0.15], [0.15, 1]])]

gaussies = [Gaussian(rect, mean, cov) for mean,cov in zip(means,covs)]
weights = np.array([0.3, 0.7])
sumgauss = MixtureDistribution(gaussies, weights) 
expifiedsumgauss = ExpifiedMixture(sumgauss)

sumgauss.plot_density(points)
expifiedsumgauss.plot_density(points)

N = 1000
samples = sumgauss.sample(N)
expsamples, expreject = expifiedsumgauss.sample(N)
sumgauss.plot_samples(samples)
expifiedsumgauss.plot_samples(expsamples)

print('Rejected for exp-sampling: {}%'.format(expreject/(expreject+N)*100))

# unif = Uniform(rect)
# print(unif.max_density())

# testpoints = np.array([[-0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
# print(sumgauss.density(testpoints))
# print(expifiedsumgauss.density(testpoints))
# print(sumgauss.grad_density(testpoints))
# print(expifiedsumgauss.grad_density(testpoints))
# print(sumgauss.Hessian_density(testpoints))
# print(expifiedsumgauss.Hessian_density(testpoints))
# 
# print(sumgauss.upperbound_density())
  
# print(gaussies[0].density(testpoints))
# print(gaussies[0].grad_density(testpoints))
# print(gaussies[0].Hessian_density(testpoints))
# 
# print(expifiedsumgauss.density(testpoints))
# print(expifiedsumgauss.grad_density(testpoints))