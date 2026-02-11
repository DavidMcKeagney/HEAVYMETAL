# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 14:34:43 2026

@author: David McKeagney
"""

import numpy as np
import random 
import matplotlib.pyplot as plt
#%%
def discrete_probability(E,x,Gamma):
    return (Gamma/(2*np.pi))*(1/((E-x)**2 + 0.25*(Gamma**2)))
def Photon_dist_gauss(E,mu,sig):
    return (1/(sig*np.sqrt(2*np.pi)))*np.exp(- 0.5*((E-mu)/sig)**2)

def BacklightModel(N,mu,sig,x,Gamma):
    Spectrometer_Energy=[] #Defines the energies recorded after the photon interacts with the atom
    Spectrometer_Counts=[] #Defines the number of counts of each photon energy recorded by the spectrometer
    Backlight_Energy=[] # Defines the energies in the photon prob distribution of the backlight
    for i in range(N):
        E=random.gauss(mu,sig)
        E=round(E,2) # Randomly samples a photon energy from the backlight distrubution
        p=discrete_probability(E, x, Gamma) # Probability of the interaction with the atom 
        if len(Backlight_Energy)==0: # Records the backlight energies and probabilities 
            Backlight_Energy.append(E)
        else:
            if E not in Backlight_Energy:
                Backlight_Energy.append(E)
        if p<1/(np.pi*Gamma): # If the photon doesn't interact with the atom
            if len(Spectrometer_Energy)==0:
                Spectrometer_Counts.append(1)
                Spectrometer_Energy.append(E)
            else:
                if E not in Spectrometer_Energy:
                    Spectrometer_Energy.append(E)
                    Spectrometer_Counts.append(1)
                else:
                    j=Spectrometer_Energy.index(E)
                    Spectrometer_Counts[j]+=1
    Spectrometer_Energy=np.array(Spectrometer_Energy)
    Spectrometer_Counts=np.array(Spectrometer_Counts)
    Spectrometer_prob=1/(np.sum(Spectrometer_Counts))*Spectrometer_Counts
    return np.array(Backlight_Energy), Spectrometer_Energy , Spectrometer_prob
#%%
Results=BacklightModel(1000000, 4, 0.5, 3.8, 0.2)
Backlight_prob=1/60*Photon_dist_gauss(Results[0], 4, 0.5)
#%%
plt.scatter(Results[1],Results[2])
plt.scatter(Results[0],Backlight_prob)            
        