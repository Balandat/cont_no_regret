'''
Some test on the fractional 0-potential

@author: Maximilian Balandat
@date: Feb 24, 2015
'''

import numpy as np
import matplotlib.pyplot as plt

betas = [1.05, 1.1, 1.2, 1.3, 1.5, 2, 3, 5]
ubs = [beta/(beta-1) for beta in betas]

f = plt.figure()
for beta,ub in zip(betas, ubs):
    u = np.linspace(-10, 0.975*ub, 500)
    phi = lambda u: (ub - u)**(-beta)
    plt.plot(u, phi(u), label=r'$\beta$='+'{0}'.format(beta))
plt.plot(np.linspace(-10,10,500), np.exp(u), 'r--', linewidth=2, label='exp')
plt.ylim((0,7.5))
plt.xlim((-10,10))
plt.legend(loc='upper left')
plt.show()