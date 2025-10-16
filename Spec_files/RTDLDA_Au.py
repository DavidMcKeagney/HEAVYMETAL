# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 11:55:06 2025

@author: David McKeagney
"""

import numpy as np
import matplotlib.pyplot as plt 
import function_library_phd
#%% AU I RTDLDA
AuI=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuI.dat',dtype=float)
AuI_5d96s2_1_5=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuI_5d96s2_1_5.dat',dtype=float)
AuI_5d96s2_2_5=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuI_5d96s2_2_5.dat',dtype=float)
#%% AU I RTDLDA New Energies 
AuI_new_energies=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuI_new_energies.dat',dtype=float)
AuI_5d96s2_1_5_new_energies=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuI_5d96s2_1_5_new_energies.dat',dtype=float)
AuI_5d96s2_2_5_new_energies=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuI_5d96s2_2_5_new_energies.dat',dtype=float)
#%% AuII RTDLDA
AuII=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuII.dat',dtype=float)
AuII_5d96s_2_5=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuII_5d96s_2_5.dat',dtype=float)
AuII_5d96s_1_5=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuII_5d96s_1_5.dat',dtype=float)
AuII_5d86s2_1_5=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuII_5d86s2_1_5.dat',dtype=float)
AuII_5d86s2_2_5=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuII_5d86s2_2_5.dat',dtype=float)
AuII_5d86s2_2_5_1_5=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuII_5d86s2_2_5_1_5.dat',dtype=float)
#%% AuII RTDLDA New Energies
AuII_new_energies=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuII_new_energies.dat',dtype=float)
AuII_5d96s_2_5_new_energies=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuII_5d96s_2_5_new_energies.dat',dtype=float)
AuII_5d96s_1_5_new_energies=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuII_5d96s_1_5_new_energies.dat',dtype=float)
AuII_5d86s2_1_5_new_energies=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuII_5d86s2_1_5_new_energies.dat',dtype=float)
AuII_5d86s2_2_5_new_energies=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuII_5d86s2_2_5_new_energies.dat',dtype=float)
AuII_5d86s2_2_5_1_5_new_energies=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuII_5d86s2_2_5_1_5_new_energies.dat',dtype=float)
#%% Summing up all the intensities of Au I
AuI_energies=AuI[:,0]
AuI_intensities=AuI[:,1]+AuI_5d96s2_1_5[:,1]+AuI_5d96s2_2_5[:,1]
#%% Summing up all the intensities of Au I New Energies
AuI_energies_new_energies=AuI_new_energies[:,0]
AuI_intensities_new_energies=AuI_new_energies[:,1]+AuI_5d96s2_1_5_new_energies[:,1]+AuI_5d96s2_2_5_new_energies[:,1]
#%% Fixing the array sizes for Au II
AuII_fix=np.zeros((3,2))
AuII_5d96s_1_5_fix=np.zeros((1,2))
AuII_5d86s2_1_5_fix=np.zeros((8,2))
AuII_5d86s2_2_5_fix=np.zeros((4,2))
AuII_5d86s2_2_5_1_5_fix=np.zeros((6,2))
#%% Fixing the array sizes for Au II New Energies
AuII_fix_new_energies=np.zeros((2,2))
AuII_5d96s_1_5_fix_new_energies=np.zeros((4,2))
AuII_5d86s2_1_5_fix_new_energies=np.zeros((6,2))
AuII_5d86s2_2_5_fix_new_energies=np.zeros((6,2))
AuII_5d86s2_2_5_1_5_fix_new_energies=np.zeros((5,2))
AuII_5d96s_2_5_fix_new_energies=np.zeros((4,2))
#%% temp fix for Au II maybe try interpolating later 
AuII=np.concatenate((AuII_fix,AuII),axis=0)
AuII_5d96s_1_5=np.concatenate((AuII_5d96s_1_5_fix,AuII_5d96s_1_5),axis=0)
AuII_5d86s2_2_5=np.concatenate((AuII_5d86s2_2_5_fix,AuII_5d86s2_2_5),axis=0)
AuII_5d86s2_1_5=np.concatenate((AuII_5d86s2_1_5_fix,AuII_5d86s2_1_5),axis=0)
AuII_5d86s2_2_5_1_5=np.concatenate((AuII_5d86s2_2_5_1_5_fix,AuII_5d86s2_2_5_1_5),axis=0)
#%% temp fix for Au II maybe try interpolating later 
AuII_new_energies=np.concatenate((AuII_fix_new_energies,AuII_new_energies),axis=0)
AuII_5d96s_1_5_new_energies=np.concatenate((AuII_5d96s_1_5_fix_new_energies,AuII_5d96s_1_5_new_energies),axis=0)
AuII_5d96s_2_5_new_energies=np.concatenate((AuII_5d96s_2_5_fix_new_energies,AuII_5d96s_2_5_new_energies),axis=0)
AuII_5d86s2_2_5_new_energies=np.concatenate((AuII_5d86s2_2_5_fix_new_energies,AuII_5d86s2_2_5_new_energies),axis=0)
AuII_5d86s2_1_5_new_energies=np.concatenate((AuII_5d86s2_1_5_fix_new_energies,AuII_5d86s2_1_5_new_energies),axis=0)
AuII_5d86s2_2_5_1_5_new_energies=np.concatenate((AuII_5d86s2_2_5_1_5_fix_new_energies,AuII_5d86s2_2_5_1_5_new_energies),axis=0)
#%%
AuII_intensity=[]
for E in AuII_5d96s_2_5[:,0]:
    a=0
    if len(AuII_5d96s_1_5[:,0][AuII_5d96s_1_5[:,0]==E])>0:
        a+=AuII_5d96s_1_5[:,1][AuII_5d96s_1_5[:,0]==E]
    elif len(AuII[:,0][AuII[:,0]==E])>0:
        a+=AuII[:,1][AuII[:,0]==E]
    elif len(AuII_5d86s2_2_5[:,0][AuII_5d86s2_2_5[:,0]==E])>0:
        a+=AuII_5d86s2_2_5[:,1][AuII_5d86s2_2_5[:,0]==E]
    elif len(AuII_5d86s2_1_5[:,0][AuII_5d86s2_1_5[:,0]==E]):
        a+=AuII_5d86s2_1_5[:,1][AuII_5d86s2_1_5[:,0]==E]
    elif len(AuII_5d86s2_2_5_1_5[:,0][AuII_5d86s2_2_5_1_5[:,0]==E]):
        a+=AuII_5d86s2_2_5_1_5[:,1][AuII_5d86s2_2_5_1_5[:,0]==E]
    AuII_intensity.append(a+AuII_5d96s_2_5[:,1][AuII_5d96s_2_5[:,0]==E])
AuII_intensity=np.array(AuII_intensity)[:,0]
#%%
AuII_intensity_new_energies=[]
for E in AuII_5d96s_2_5[:,0]:
    a=0
    if len(AuII_5d96s_1_5_new_energies[:,0][AuII_5d96s_1_5_new_energies[:,0]==E])>0:
        a+=AuII_5d96s_1_5_new_energies[:,1][AuII_5d96s_1_5_new_energies[:,0]==E]
    elif len(AuII_new_energies[:,0][AuII_new_energies[:,0]==E])>0:
        a+=AuII_new_energies[:,1][AuII_new_energies[:,0]==E]
    elif len(AuII_5d86s2_2_5_new_energies[:,0][AuII_5d86s2_2_5_new_energies[:,0]==E])>0:
        a+=AuII_5d86s2_2_5_new_energies[:,1][AuII_5d86s2_2_5_new_energies[:,0]==E]
    elif len(AuII_5d86s2_1_5_new_energies[:,0][AuII_5d86s2_1_5_new_energies[:,0]==E]):
        a+=AuII_5d86s2_1_5_new_energies[:,1][AuII_5d86s2_1_5_new_energies[:,0]==E]
    elif len(AuII_5d86s2_2_5_1_5_new_energies[:,0][AuII_5d86s2_2_5_1_5_new_energies[:,0]==E]):
        a+=AuII_5d86s2_2_5_1_5_new_energies[:,1][AuII_5d86s2_2_5_1_5_new_energies[:,0]==E]
    elif len(AuII_5d96s_2_5_new_energies[:,0][AuII_5d96s_2_5_new_energies[:,0]==E])>0:
        a+=AuII_5d96s_2_5_new_energies[:,1][AuII_5d96s_2_5_new_energies[:,0]==E]
    AuII_intensity_new_energies.append(a)
AuII_intensity_new_energies=np.array(AuII_intensity_new_energies)[:,0]
#%% Summing up all the intensities of Au II
AuII_energies=AuII_5d96s_2_5[:,0]
#AuII_intensities=AuII[:,1]+AuII_5d96s_1_5[:,1]+AuII_5d96s_2_5[:,1]+AuII_5d86s2_1_5[:,1]+AuII_5d86s2_2_5[:,1]+AuII_5d86s2_2_5_1_5[:,1]
#%%
Intensity_500ns=0.75*AuI_intensities + 0.25*AuII_intensity
#%%
Eric_data_500ns=np.loadtxt('C:/Users/Padmin/OneDrive/Desktop/Eric_data_500ns.txt',dtype=float).T
Intensity_500ns_exp=Eric_data_500ns[1][np.logical_and(Eric_data_500ns[0]>=75,Eric_data_500ns[0]<=130)]
Energy=Eric_data_500ns[0][np.logical_and(Eric_data_500ns[0]>=75,Eric_data_500ns[0]<=130)]
#%%
#plt.plot(AuI_energies+ np.repeat(0.8,len(AuI_energies)),AuI_intensities,label='Au I')
#plt.plot(AuII_energies+ np.repeat(0.8,len(AuI_energies)),AuII_intensity,label='Au II')
plt.plot(AuI_energies+ np.repeat(0.8,len(AuI_energies)),0.75*AuI_intensities+0.25*AuII_intensity,label='75% AuI, 25% AuII')
plt.plot(Energy,120*Intensity_500ns_exp,label='500ns')
#plt.plot(AuI_energies+np.repeat(0.8,len(AuI_energies)),Intensity_500ns,label='75% AuI, 25% AuII')
plt.xlabel('Energy')
plt.ylabel('Intensity')
plt.grid(True)
plt.xlim(75,120)
plt.legend()
#%%
Au_II_5d86s2_2_5_energy=AuII_5d86s2_2_5[:,0]
Au_II_5d86s2_1_5_energy=AuII_5d86s2_1_5[:,0]
Au_II_5d86s2_2_5_1_5_energy=AuII_5d86s2_2_5_1_5[:,0]
AuII_5d96s_1_5_energy=AuII_5d96s_1_5[:,0]
AuII_5d96s_2_5_energy=AuII_5d96s_2_5[:,0]
AuII_energies=AuII[:,0]
#%%
#plt.plot(Au_II_5d86s2_1_5_energy,AuII_5d86s2_1_5)
#plt.plot(Au_II_5d86s2_2_5_energy,AuII_5d86s2_2_5)
#plt.plot(Au_II_5d86s2_2_5_1_5_energy,AuII_5d86s2_2_5_1_5)
plt.plot(AuII_energies + np.repeat(0.8,len(AuII_energies)),AuII[:,1],label='5d10')
plt.plot(AuII_5d96s_1_5_energy + np.repeat(0.8,len(AuII_5d96s_1_5_energy)),AuII_5d96s_1_5,label='5d96s j=1.5')
plt.plot(AuII_5d96s_2_5_energy+np.repeat(0.8,len(AuII_5d96s_2_5_energy)),AuII_5d96s_2_5,label='5d96s j=2.5')
plt.plot(Energy,120*Intensity_500ns_exp,label='500ns')
plt.xlabel('Energy')
plt.ylabel('Intensity')
plt.grid(True)
plt.xlim(75,105)
plt.legend()
#%%
#plt.plot(AuI_energies_new_energies,AuI_intensities_new_energies,label='AuI New Energies')
#plt.plot(AuI_energies,AuI_intensities,label='AuI Old Energies',ls='--')
#plt.plot(AuI_energies+np.repeat(0.8,len(AuII_energies)),0.65*AuI_intensities_new_energies+0.35*AuII_intensity_new_energies,label='65% AuI, 35% AuII')
plt.plot(Energy,100*Intensity_500ns_exp,label='500ns')
plt.plot(AuII_5d96s_1_5_new_energies[:,0],AuII_5d96s_1_5_new_energies[:,1])
#plt.plot(AuII_5d96s_2_5_new_energies[:,0],AuII_5d96s_2_5_new_energies[:,1])
#plt.plot(AuII_new_energies[:,0],AuII_new_energies[:,1])
#plt.plot(AuII_5d86s2_1_5_new_energies[:,0],AuII_5d86s2_1_5_new_energies[:,1])
#plt.plot(AuII_5d86s2_2_5_new_energies[:,0],AuII_5d86s2_2_5_new_energies[:,1])
#plt.plot(AuII_5d86s2_2_5_1_5_new_energies[:,0],AuII_5d86s2_1_5_new_energies[:,1])
#plt.plot(AuII_energies,AuII_intensity_new_energies,label='AuII New Energies')
#plt.plot(AuII_energies,AuII_intensity,label='AuII Old Energies',ls='--')
plt.legend()
plt.xlabel('Energies (eV)')
plt.ylabel('Intensity')
