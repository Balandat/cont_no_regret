'''
Script for visualizing some potential functions

@author: Maximilian Balandat
@date: May 6, 2015
'''

from ContNoRegret.DualAveraging import *
import matplotlib.pyplot as plt

potentials = [ExponentialPotential(), CompositeOmegaPotential(gamma=2), CompositeOmegaPotential(gamma=4), 
              pNormPotential(1.25), pNormPotential(1.75), LogtasticPotential()]

u = np.linspace(-10, 20, 10000)
vals = [np.maximum(pot.phi(u), 0) for pot in potentials]

plt.figure()
for pot, val in zip(potentials, vals):
    plt.semilogy(u, val, label=pot.desc)
plt.legend(loc='upper left')
plt.show()

