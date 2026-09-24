# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 15:45:20 2026

@author: David McKeagney
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy.integrals.quadrature import gauss_laguerre
import scipy.special as sc
#%%
#Important functions 
# These functions together perform Gauss Laguerre quadrature to numerically integrate the Fano cross sections to get a photoionization rate for CR modelling
def Lorentzian(x,beta,E_r,E_0,Gamma):
    #x is an array of energies
    #beta is an array of inverse temperatures
    #E_r,E_0,Gamma are are scalar parameters
    delta=(beta*Gamma/2)
    x_r=E_r-E_0
    x_r=beta*x_r
    return 1/(delta**2 + (x-x_r)**2)

def Integral_Moments(n, i, j, beta,E_r,E_0,Gamma):
    #i is the number of points included in the quadrature
    #j is the degree of precision of the floating point variables
    #n is the n-th order moment of the integral
    x_i,w_i=gauss_laguerre(i, j)
    x_i=np.array(x_i)
    w_i=np.array(w_i)
    Integral_points= w_i*(x_i**n)*Lorentzian(x_i, beta, E_r, E_0, Gamma)
    Integral=np.sum(Integral_points)
    return Integral

def Convergence_test(i, j, beta, E_r, E_0, Gamma):
    n_array = np.arange(0, 3)

    Moment_array = np.array([
        Integral_Moments(n, i, j, beta, E_r, E_0, Gamma)
        for n in n_array
    ])

    delta = beta * Gamma / 2
    x_r = beta * (E_r - E_0)

    return (
        (delta**2 + x_r**2) * Moment_array[0]
        - 2*x_r * Moment_array[1]
        + Moment_array[2]
    )
def PhotoIonizationRate(beta,E_r,E_0,Gamma,q):
    #Factor In front of the integral is not included and needs to be added later, the factor is all the constants in front of the plank function
    n_array= np.arange(0,5)
    
    
    Moment_array=np.zeros((len(n_array),len(beta)))
    
    for i,n in enumerate(n_array):
        for j,b in enumerate(beta):
            Moment_array[i,j]+= (b**(-n+1))*Integral_Moments(n, 8, 20, b, E_r, E_0, Gamma)
    #Moment_array=np.array([(beta**(-n+1))*Integral_Moments(n,8,20,beta,E_r,E_0,Gamma) for n in n_array])
    
    x_r=E_r-E_0
    
    C= 0.5*q*Gamma - x_r
    
    D= C + E_0
    
    Mom_sum_zero= ((C*E_0)**2)*Moment_array[0]
    Mom_sum_one= 2*D*C*E_0*Moment_array[1]
    Mom_sum_two= (2*E_0 + D**2)*Moment_array[2]
    Mom_sum_three= 2*D*Moment_array[3]
    Mom_sum_four=Moment_array[4]
    
    PI_rate=np.exp(-beta*E_0)*(Mom_sum_four + Mom_sum_one + Mom_sum_two + Mom_sum_zero + Mom_sum_three)
    
    h=4.135667696e-15
    c=299792458
    A=8*np.pi/((h**3)*(c**3))
    return A*PI_rate
def CollisionalIonization(beta,E_0,n_elec):
    a=4.5e-14
    me=9.1093897E-31
    pre_fac=(n_elec*a/E_0)*np.sqrt(beta*8*(me)**3/np.pi)
    return pre_fac*sc.expi(beta*E_0)
    
    
    
#%%
num_points=np.arange(2,15)
results=np.array([Convergence_test(n, 20, 1/2, 80.23, 20.203, 0.27) for n in num_points]).astype(float)
#%%
plt.plot(num_points,results)
plt.xlabel('Number of points')
#%%
T_range=np.arange(0.1,2.05,0.05)
beta=1/T_range
#%%
PI_rates=PhotoIonizationRate(beta, 80.23, 20.203, 0.27, 2.13)
#%%
plt.plot(T_range,PI_rates)
plt.xlabel('T (eV)')
plt.ylabel('PI_rate (s^-1)')
plt.yscale('log')
