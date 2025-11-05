# -*- coding: utf-8 -*-
"""
Created on Mon May  5 14:01:26 2025

@author: padmin
"""

import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({'font.size': 22})
plt.rcParams["figure.figsize"] = (15,10) 
#%%
## Physical constants
c=299792458
me=9.1093897E-31
esp0=8.854187817E-12
ec=1.60217733E-19
#%%
# Electron density info
wavelen = 1064*1e-9 # laser wavelength
frac_ne= 1; # fraction of critical density
ne=4*(np.pi**2)*(c**2)*me*esp0/((ec**2)*(wavelen**2))*(1E-6)*frac_ne # electron density

# Temperature info
Te_start=0.01
Te_step=0.01
Te_end=1000
Te=np.arange(Te_start, Te_end+Te_step, Te_step)

# Element info
Atomic_num=5
IPs = (9.225554,20.203,30.0,45.0,60.0,74.0)
#%%
# Arrays
n = np.ones((int(len(Te)), int(Atomic_num+1)))
S = np.zeros((int(len(Te)), int(Atomic_num+1)))
alphaR = np.zeros((int(len(Te)), int(Atomic_num+1)))
alpha3b = np.zeros((int(len(Te)), int(Atomic_num+1)))
nt = np.zeros(int(len(Te)))
frac = np.zeros((int(len(Te)), int(Atomic_num+1)))

# Rates
for i in range (0,Atomic_num+1):
    S[:,i]=(((9E-6)*((Te/IPs[i])**(1/2)))/((IPs[i]**(3/2))*(4.88+(Te/IPs[i]))))*np.exp(-IPs[i]/Te)
    alphaR[:,i]= (5.2E-14)*((IPs[i]/Te)**(1/2))*(i)*(0.429+(0.5*np.log(IPs[i]/Te)+(0.469*((Te/IPs[i])**(1/2)))))
    alpha3b[:,i]=((2.97E-27))/((Te*(IPs[i]**2))*(4.88+(Te/IPs[i])))
    if i>0:
        # n[:,0] already set to 1, calculate rest with rates above
        n[:,i]=(S[:,i-1]/(alphaR[:,i]+ne*alpha3b[:,i]))*n[:,i-1]

# Calculate totals from sum        
nt=np.sum(n,1)

for i in range (0,Atomic_num+1):
    frac[:,i]=n[:,i]/nt
    
plt.plot(Te, frac)
plt.legend(["Au I","Au II","Au III","Au IV","Au V","Au VI"])
plt.xlim([0.1, 20])
plt.ylim([0.01, 1])
plt.xlabel('T_e [eV]')
plt.ylabel('N_i/N_t')
#%% Recombination timescales in nano-seconds 

timescale=1/(alpha3b+ne*alphaR)*10**9
#%%
plt.plot(Te,timescale[:,1],label='Au II recombination timescale')
plt.plot(Te,timescale[:,2],label='Au III recombination timescale')
plt.xlim(0.1,8)
plt.ylim(0,12)
plt.xlabel('Te [eV]')
plt.ylabel(' timescale [ns]')
plt.legend()


