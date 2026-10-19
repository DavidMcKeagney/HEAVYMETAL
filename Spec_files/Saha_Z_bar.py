# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:52:49 2026

@author: David McKeagney
"""

import numpy as np
import matplotlib.pyplot as plt
#%%

#def Saha(ne, E_i, g_i, E_ii, g_ii, E_I_eV, kT_eV):
#    h = 6.62607015e-34
#    me = 9.10938371e-31
#    eV = 1.602176634e-19

#    lam = h / np.sqrt(2 * np.pi * me * (kT_eV * eV))

#    Z_i = PartitionFunction(E_i, g_i, np.atleast_1d(kT_eV))
#    Z_ii = PartitionFunction(E_ii, g_ii, np.atleast_1d(kT_eV))

#    return (2 / (ne * lam**3)) * (Z_ii / Z_i) * np.exp(-E_I_eV / kT_eV)
c=299792458
me=9.1093897E-31
esp0=8.854187817E-12
ec=1.60217733E-19
wavelen = 1064*1e-9 # laser wavelength
frac_ne= 1
ne=4*(np.pi**2)*(c**2)*me*esp0/((ec**2)*(wavelen**2))*(1E-6)*frac_ne # electron density

# Temperature info
Te_start=0.01
Te_step=0.01
Te_end=1000
Te=np.arange(Te_start, Te_end+Te_step, Te_step)

# Element info
Atomic_num=5
IPs_AU = [9.225554,20.203,30.0,45.0,60.0,74.0]
IPs_Hf=[6.825070,14.61,22.55,33.370,68.37,98]
g_AU=[2,1,6,9,10,9]

#def PartitionFunction(E_arr,g_arr,Temp):
#    tot_part_func=np.zeros((len(E_arr),len(Temp)))
#    for i,T in enumerate(Temp):
#        tot_part_func[:,i] += g_arr*np.exp(-E_arr/T)
#    
#    return np.sum(tot_part_func,axis=0)
#%%


def PartitionFunction(E_arr, g_arr, Temp):
    E_arr = np.asarray(E_arr)
    g_arr = np.asarray(g_arr)
    Temp = np.atleast_1d(Temp)

    tot_part_func = np.zeros((len(E_arr), len(Temp)))

    for i, T in enumerate(Temp):
        tot_part_func[:, i] = g_arr * np.exp(-E_arr / T)

    return np.sum(tot_part_func, axis=0)


def Saha(ne, E_i, g_i, E_ii, g_ii, 
         E_ion, g_higher, kT_eV):
    
    n_e=ne*1e6
    h = 6.62607015e-34
    me = 9.10938371e-31
    eV = 1.602176634e-19

    kT_eV = np.atleast_1d(kT_eV)

    # Thermal de Broglie wavelength
    lam = h / np.sqrt(2 * np.pi * me * kT_eV * eV)

    # Partition functions for neutral and singly ionised
    Z_i = PartitionFunction(E_i, g_i, kT_eV)
    Z_ii = PartitionFunction(E_ii, g_ii, kT_eV)

    # Higher stages: ground-state approximation
    Z_higher = [
        np.full(len(kT_eV), g)
        for g in g_higher
    ]

    # All partition functions
    Z = np.array([Z_i, Z_ii] + Z_higher)

    # Saha ratios between consecutive stages
    ratios = []

    for i, chi in enumerate(E_ion):
        ratio = (
            2 / (ne * lam**3)
            * Z[i+1] / Z[i]
            * np.exp(-chi / kT_eV)
        )
        ratios.append(ratio)

    ratios = np.array(ratios)

    # Relative populations, starting from neutral
    populations = np.ones((len(Z), len(kT_eV)))

    for i in range(1, len(Z)):
        populations[i] = populations[i-1] * ratios[i-1]

    # Normalise to get ionisation fractions
    fractions = populations / np.sum(populations, axis=0)

    return fractions
#%%
n = np.ones((int(len(Te)), int(Atomic_num+1)))
frac = np.zeros((int(len(Te)), int(Atomic_num+1)))
nt = np.zeros(int(len(Te)))

for i in np.arange(0,Atomic_num+1):
    if i>0:
        n[:,i]=Saha(1e16, g_AU[i-1], g_AU[i], IPs_AU[i-1], Te)*n[:,i-1]

nt=np.sum(n,1)

for i in range (0,Atomic_num+1):
    frac[:,i]=n[:,i]/nt

Z_bar_Au_LTE=np.dot(frac,np.arange(0,Atomic_num+1))
#%%
IPs_AU = [9.225554,20.203,30.0,45.0,60.0]

g_Au_I=np.array([2,6,4,2,4,6])
E_Au_I=np.array([0,1.1358412,2.6576250,4.6319246,5.1050014,5.2276115])

g_Au_II=np.array([1,7,5,3,5,9])
E_Au_II=np.array([0,1.8646693,2.1871576,3.4425152,3.6725668,5.0187254])

g_higher_AU=np.array([6,9,10,9])

#%%
plt.plot(Te,frac)
plt.xlim(0,3)
#%%
plt.plot(Te,Z_bar_Au_LTE)
plt.xlim(0,3.5)
#%%
T_vals=np.arange(0.01,5.01,0.01)
#%%
U_Au_I=PartitionFunction(E_Au_I, g_Au_I, T_vals) 
U_Au_II=PartitionFunction(E_Au_II, g_Au_II, T_vals)   
#%%
plt.plot(T_vals,U_Au_II/U_Au_I)
#%%
Au_frac=Saha(1e21, E_Au_I, g_Au_I, E_Au_II, g_Au_II, IPs_AU, g_higher_AU, T_vals)
#%%
plt.plot(T_vals,Au_frac.T)
plt.xlabel('Temperature')
plt.ylabel('Ion Fraction')
plt.legend(['Au I', 'Au II', 'Au III', 'Au IV', 'Au V','Au VI'])
#%%
plt.plot(T_vals,np.sum(Au_frac.T,axis=1))