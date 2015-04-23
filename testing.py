'''
Created on Feb 24, 2015

@author: balandat
'''

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from ContNoRegret.Domains import S
from ContNoRegret.Distributions import Uniform
from ContNoRegret.utils import create_random_Sigmas
from ContNoRegret.LossFunctions import GaussianLossFunction
from scipy.stats import expon
from scipy.interpolate import SmoothBivariateSpline, LSQBivariateSpline

# def compute_constants(gamma):
#     c = (gamma-1)**(-1)
#     a2 = gamma*(1+gamma)/2
#     a1 = gamma - 2*c*a2
#     a0 = 1 - c*a1 - c**2*a2
#     return c, np.array([a0, a1, a2])
# 
# def phi(u, gamma):
#     c,a = compute_constants(gamma)
#     return ( (u<c)*(gamma/(gamma-1)-np.minimum(u,c))**(-gamma) + 
#              (u>=c)*(a[0]+a[1]*np.maximum(u,c)+a[2]*np.maximum(u,c)**2) )
# 
# def phi_prime(u, gamma):
#     c,a = compute_constants(gamma)
#     return (u<c)*gamma*(gamma/(gamma-1)-np.minimum(u,c))**(-(1+gamma)) + (u>=c)*(a[1]+2*a[2]*np.maximum(u,c))
# 
# def phi_double_prime(u, gamma):
#     c,a = compute_constants(gamma)
#     return (u<c)*gamma*(1+gamma)*(gamma/(gamma-1)-np.minimum(u,c))**(-(2+gamma)) + (u>=c)*2*a[2]
# 
# def phi_inv(u, gamma):
#     c,a = compute_constants(gamma)
#     b = phi(c, gamma)
#     return ( (u<b)*(gamma/(gamma-1)-np.minimum(u,b)**(-1/gamma)) + 
#              (u>=b)*(-a[1]/2/a[2]+np.sqrt(a[1]**2/4/a[2]**2 - (a[0]-np.maximum(u,b))/a[2])) )
# 
# def phi_inv_prime(u, gamma):
#     return 1/phi_prime(phi_inv(u, gamma))
# 
# 
# # Plot some functions
# gammas = [1.25, 1.5, 1.75, 2, 3]
# u = np.linspace(-1.5,5,10000)
# v = np.linspace(0.001,10,10000)
# f,axs = plt.subplots(3,1)
# axs[0].plot(u, np.exp(u-1))
# axs[1].plot(u, np.exp(u-1))
# axs[2].plot(u, np.exp(u-1))
# for gamma in gammas:
#     axs[0].plot(u, phi(u,gamma))
#     axs[1].plot(u, phi_prime(u,gamma))
#     axs[2].plot(u, phi_double_prime(u,gamma))
# plt.show()



# for gamma in gammas:
#     # gamma = 1.5    
#     ctilde = gamma/(gamma-1)
#     a2 = 0.5*gamma*(1+gamma)/((ctilde-1)**(2+gamma))
#     a1 = gamma/((ctilde-1)**(1+gamma)) - 2*a2
#     a0 = 1/((ctilde-1)**gamma) - a1 - a2
#     
#     def phi(u):  
#         return (u<1)*(ctilde-np.minimum(u,1))**(-gamma) + (u>=1)*(a0+a1*np.maximum(u,1)+a2*np.maximum(u,1)**2)
#     
#     def phiprime(u):
#         return (u<1)*gamma*(ctilde-np.minimum(u,1))**(-(1+gamma)) + (u>=1)*(a1+2*a2*np.maximum(u,1))
#     
#     def phiinv(u):
#         return (u<1)*(ctilde-np.minimum(u,1)**(-1/gamma)) + (u>=1)*(-a1/2/a2+np.sqrt(a1**2/4/a2**2 - (a0-np.maximum(u,1))/a2))    
#     
#     def phiinvprime(u):
#         return 1/phiprime(phiinv(u))
#     #     return (u<1)/gamma*u**(-1+1/gamma) + (u>=1)*(a1**2-4*a2*(a0-np.maximum(u,1)))**(-1/2)
#     
#     
#     # fig2, (ax2, ax3) = plt.subplots(2, 1)
#     # fig3, ax4 = plt.subplots(1)
#     
#     ax1.plot(u, phi(u))#, u, np.exp(u-1))
#     # v = np.linspace(0.001, 5, 10000)
#     # ax2.plot(v, phiinv(v), v, 1+np.log(v))
#     # ax3.plot(v, phiinvprime(v), v, 1/v)
#     # ax4.plot(v, phiinvprime(v)-1/(3*v))
#     # print(np.min(phiinvprime(v)-1/(3+v))
# plt.show()
    

    