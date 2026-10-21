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
def CollisionalIonizationRate(T, params):
    tol1, tol2, E_r, E_0, Gamma, q, D, g_i, g_j, n_elec,W = params
    

    #a=4.5e-14
    #me=9.1093897E-31
    #kT_joule=(1/beta)*1.602176634e-19
    #beta0=1/kT_joule
    #pre_fac=(a*n_elec/E_0)*np.sqrt(beta0*8/(me*np.pi))
    a = 4.5e-14                 # cm^2 eV^2
    me = 9.1093837e-28          # g
    eV_to_erg = 1.602176634e-12 # erg/eV
    pre_fac=(a*n_elec/E_0)*np.sqrt(8*eV_to_erg/(T*np.pi*me))
    return pre_fac*sc.exp1(E_0/T)


def PIONRATE(T, params):
    tol1, tol2, E_r, E_0, Gamma, q, D, g_i, g_j, n_elec,W = params

    #E_r is resonance position
    #Gamma is the width
    #E_0 is the ionization potential
    #q is the q parameter
    #D is the dipole matrix elements

    Cross_sec_fac=1.7106*E_r*(D**2)/((q**2)*Gamma)*1e-22

    R_array=np.zeros(len(T))
    for n in range(1,101):
        if n==1:
            R_n_1=np.zeros(len(T))
        else:
            R_n_1=R_array
        tol_init=1
        n_moments=np.arange(0,5)
        grid_num=1
        while tol_init>tol1:
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

        R_n=np.exp(-E_0/T)*(Mom_sum_four + Mom_sum_one + Mom_sum_two + Mom_sum_zero + Mom_sum_three).astype(float)

        h=4.135667696e-15
        c=299792458
        A=8*np.pi/((h**3)*(c**2))

        R_array += R_n
        if np.max(np.abs(R_n_1-R_array))<tol2:
            return W*Cross_sec_fac*A*R_array
    return W*R_array


def RadRecom(T, params):
    tol1, tol2, E_r, E_0, Gamma, q, D, g_i, g_j, n_elec,W = params
    Temp = T

    Cross_sec_fac=1.7106*E_r*(D**2)/((q**2)*Gamma)*1e-22
    #Factor In front of the integral is not included and needs to be added later, the factor is all the constants in front of the plank function
    n_array= np.arange(0,5)

    Moment_array=np.zeros((len(n_array),len(T)))

    for i,n in enumerate(n_array):
        for j,T in enumerate(Temp):
            Moment_array[i,j]+= (T**(n+1))*Integral_Moments(n, 8, 20, T, E_r, E_0, Gamma,1)
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

    RR_rate=Cross_sec_fac*1.30967e10*(g_i/g_j)*(Temp**(-3/2))*(Mom_sum_four + Mom_sum_one + Mom_sum_two + Mom_sum_zero + Mom_sum_three)

    return RR_rate


def recom_3b(T, params):
    tol1, tol2, E_r, E_0, Gamma, q, D, g_i, g_j, n_elec,W = params
    

    h = 6.62607015e-34
    me = 9.10938371e-31
    eV = 1.602176634e-19

    lam = h / np.sqrt(2 * np.pi * me * (T * eV)) # returns units of meters
    lam=100*lam

    saha_fac=0.5*(g_i/g_j)*(lam**3)*np.exp(E_0/T)

    return saha_fac*CollisionalIonizationRate(T, params)

def f(T,ne,T_r,params):
    tol1, tol2, E_r, E_0, Gamma, q, D, g_i, g_j, n_elec,W = params
    
    Delta=RadRecom(T, params)/(ne*recom_3b(T, params)+RadRecom(T, params))
    
    h = 6.62607015e-34
    me = 9.10938371e-31
    eV = 1.602176634e-19

    lam = h / np.sqrt(2 * np.pi * me * (T * eV)) # returns units of meters
    lam=100*lam
    
    big_fac=((g_i/g_j)*(ne*Delta*CollisionalIonizationRate(T, params)+Delta*PIONRATE(T_r, params))/(W*RadRecom(T_r, params)))**(2/3)
    
    T_Rad=(h**2/(2*np.pi*me))*big_fac
    
    return T_Rad
    
    
def Rad_Temps(T_e, n_e, tol, max_it, params):

    T_e = np.asarray(T_e, dtype=float)
    n_e = np.asarray(n_e, dtype=float)

    # Generate every temperature-density combination
    T_grid, ne_grid = np.broadcast_arrays(
        T_e[:, None],
        n_e[None, :]
    )

    shape = T_grid.shape

    # Flatten to 1D for compatibility with rate functions
    T_flat = T_grid.ravel()
    ne_flat = ne_grid.ravel()

    # Initial radiation temperature for every combination
    T_r = np.ones_like(T_flat)

    for n in range(max_it):

        # Evaluate all temperature-density combinations
        T_r_new = np.asarray(
            f(T_flat, ne_flat, T_r, params)
        )

        # Check convergence across the entire grid
        tol_init = np.abs(T_r_new - T_r)

        if np.all(tol_init <= tol):
            return T_r_new.reshape(shape)

        T_r = T_r_new

    # Return None if any combination fails to converge
    return None
    


            
    
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
PI_rates,n=PIONRATE(T_range, 1e-16, 1e-16, 79.87, 9.226, 0.28, 2.50,0.45)
#%%
plt.plot(T_range,PI_rates)
#%%
RadRates=RadRecom(T_range, 79.87, 9.226, 0.28, 2.50, 1, 1,0.45)
#%%
plt.plot(T_range,1e21*RadRates)
#%%
recom_3b_Au=recom_3b(T_range, 9.226, 2, 1e21, 1, 1)
#%%
ne_grid=np.arange(1,10,0.1)*1e18
T_range=np.arange(0.1,5.05,0.05)
params_test=(1e-16,1e-16,79.87, 9.226, 0.28, 2.50, 0.45,1,1,2,0.5)
#%%
T_radiation=Rad_Temps(T_range, ne_grid, 1e-8, 500, params_test)