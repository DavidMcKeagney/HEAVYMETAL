# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:52:49 2026

@author: David McKeagney
"""

import numpy as np
import matplotlib.pyplot as plt
#%%
def Saha(ne, g_i, g_ii, E_I_eV, kT_eV):
    h = 6.62607015e-34
    me = 9.10938371e-31
    eV = 1.602176634e-19

    lam = h / np.sqrt(2*np.pi*me*(kT_eV * eV))

    return (2/(ne*lam**3)) * (g_ii/g_i) * np.exp(-E_I_eV/kT_eV)
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
plt.plot(Te,frac)
plt.xlim(0,3)
#%%
plt.plot(Te,Z_bar_Au_LTE)
plt.xlim(0,3.5)
    