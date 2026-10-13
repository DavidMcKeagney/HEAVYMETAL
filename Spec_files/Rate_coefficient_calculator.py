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
def Lorentzian(x,T,E_r,E_0,Gamma):
    #x is an array of energies
    #beta is an array of inverse temperatures
    #E_r,E_0,Gamma are are scalar parameters
    T_eff=2*T/Gamma
    x_r=E_r-E_0
    delta=2*x_r/Gamma
    return 1/(1 + ((T_eff)*x-delta)**2)

def Integral_Moments(n, i, j, T,E_r,E_0,Gamma,k):
    #i is the number of points included in the quadrature
    #j is the degree of precision of the floating point variables
    #n is the n-th order moment of the integral
    #the k**n is to account for the fact that x_i/k is defined 
    x_i,w_i=gauss_laguerre(i, j)
    x_i=np.array(x_i)/k
    w_i=np.array(w_i)
    
    n = np.asarray(n).reshape(-1, 1, 1)
    T = np.asarray(T).reshape(1, -1, 1)

    x_i = np.asarray(x_i).reshape(1, 1, -1)
    w_i = np.asarray(w_i).reshape(1, 1, -1)
    
    Integral_points= (k**n)*w_i*(x_i**n)*Lorentzian(x_i, T, E_r, E_0, Gamma)
    Integral=np.sum(Integral_points,axis=-1)
    return Integral

def Convergence_test(Moment_array, T, E_r, E_0, Gamma,k):
    n_array = np.arange(0, 3)

    #Moment_array = np.array([
    #    Integral_Moments(n, i, j, beta, E_r, E_0, Gamma)
    #    for n in n_array
    #])


    x_r = E_r - E_0
    delta=2*x_r/Gamma
    T_eff= 2*T/Gamma
    return (
        (delta**2 + 1) * Moment_array[0]
        - (2*delta/k)*T_eff * Moment_array[1]
        + ((T_eff/k)**2)*Moment_array[2]
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
    
    A=x_r-q*Gamma/2
    
    B=E_0-A
    
    C=A*E_0
    
    Mom_sum_zero= ((C)**2)*Moment_array[0]
    Mom_sum_one= -2*B*C*Moment_array[1]
    Mom_sum_two= (-2*C + B**2)*Moment_array[2]
    Mom_sum_three= 2*B*Moment_array[3]
    Mom_sum_four=Moment_array[4]
    
    PI_rate=np.exp(-beta*E_0)*(Mom_sum_four + Mom_sum_one + Mom_sum_two + Mom_sum_zero + Mom_sum_three)
    
    h=4.135667696e-15
    c=299792458
    A=8*np.pi/((h**3)*(c**3))
    return A*PI_rate
def CollisionalIonizationRate(beta,E_0,n_elec,n_e):
    #a=4.5e-14
    #me=9.1093897E-31
    #kT_joule=(1/beta)*1.602176634e-19
    #beta0=1/kT_joule
    #pre_fac=(a*n_elec/E_0)*np.sqrt(beta0*8/(me*np.pi))
    a = 4.5e-14                 # cm^2 eV^2
    me = 9.1093837e-28          # g
    eV_to_erg = 1.602176634e-12 # erg/eV
    pre_fac=(a*n_elec/E_0)*np.sqrt(8*eV_to_erg*beta/(np.pi*me))
    return n_e*pre_fac*sc.exp1(beta*E_0)
    
def PIONRATE(T,tol1,tol2, E_r, E_0, Gamma, q):
    R_array=np.zeros(len(T))
    for n in range(1,101):
        if n==1:
            R_n_1=np.zeros(len(T))
        else:
            R_n_1=R_array
        tol_init=1
        n_moments=np.arange(0,5)
        while tol_init>tol1:
            grid_num=1
            Ints=Integral_Moments(n_moments[:,None], grid_num, 20, T[None,:], E_r, E_0, Gamma, n)
            Convergence_val=Convergence_test(Ints[:3], T, E_r, E_0, Gamma,n)
            tol_init=np.max(np.abs(Convergence_val-1))
            grid_num+=1
        
        x_r=E_r-E_0
        
        A=x_r-q*Gamma/2
        
        B=E_0-A
        
        C=A*E_0
        
        Mom_sum_zero= ((C)**2)*(T/n)*Ints[0]
        Mom_sum_one= -2*B*C*((T/n)**2)*Ints[1]
        Mom_sum_two= (-2*C + B**2)*((T/n)**3)*Ints[2]
        Mom_sum_three= 2*B*((T/n)**4)*Ints[3]
        Mom_sum_four=((T/n)**5)*Ints[4]
        
        R_n=np.exp(-beta*E_0)*(Mom_sum_four + Mom_sum_one + Mom_sum_two + Mom_sum_zero + Mom_sum_three).astype(float)
        
        h=4.135667696e-15
        c=299792458
        A=8*np.pi/((h**3)*(c**3))
        
        R_array += R_n
        if np.max(np.abs(R_n_1-R_array))<tol2:
            return A*R_array, n
    return R_array
    
    
#%%
num_points=np.arange(2,15)
results=np.array([Convergence_test(n, 20, 1/2, 80.23, 20.203, 0.27) for n in num_points]).astype(float)
#%%
plt.plot(num_points,results)
plt.xlabel('Number of points')
#%%
T_range=np.arange(0.1,5.05,0.05)
beta=1/T_range
#%%
PI_rates=PhotoIonizationRate(beta, 80.23, 20.203, 0.27, 2.13)
#%%
plt.plot(T_range,PI_rates)
plt.xlabel('T (eV)')
plt.ylabel('PI_rate (s^-1)')
plt.yscale('log')
#%%
EII_rates=CollisionalIonizationRate(beta, 20.203, 2, 1e21)
#%%
plt.plot(T_range,EII_rates)
plt.plot(T_range,PI_rates)
plt.xlabel('T (eV)')
plt.ylabel('EII_rate (s^-1)')
plt.yscale('log')
#%%
n_moments=np.arange(0,5)
Test_ints=Integral_Moments(n_moments[:,None], 5, 20, T_range[None,:], 80.23, 20.203, 0.27, 3).astype(float)

#%%
Conv_vals=Convergence_test(Test_ints, T_range, 80.23, 20.203, 0.27, 3).astype(float)
#%%
plt.plot(T_range,Test_ints[0])
#plt.xlim(0.2,1) 
#%%
plt.plot(T_range,np.mean(Conv_vals-1))
#%%
PI_rates,n=PIONRATE(T_range, 1e-16, 1e-16, 80.23, 20.203, 0.27, 2.13)
#%%
plt.plot(T_range,PI_rates)