# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 11:55:06 2025

@author: David McKeagney
"""

import numpy as np
import matplotlib.pyplot as plt 
import function_library_phd as flp
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
#%% Au II RTDLDA 4f channels 
AuII_4f_4f125d106s2_2_5=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuII_4f125d106s2_2_5.dat',dtype=float)
AuII_4f_4f125d106s2_3_5=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuII_4f125d106s2_3_5.dat',dtype=float)
AuII_4f_4f125d106s2_2_5_3_5=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuII_4f125d106s2_2_5_3_5.dat',dtype=float)
AuII_4f135d96s2_2_5_1_5=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuII_4f135d96s2_2_5_1_5.dat',dtype=float)
AuII_4f135d96s2_2_5_2_5=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuII_4f135d96s2_2_5_2_5.dat',dtype=float)
AuII_4f135d96s2_3_5_1_5=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuII_4f135d96s2_3_5_1_5.dat',dtype=float)
AuII_4f135d96s2_3_5_2_5=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuII_4f135d96s2_3_5_2_5.dat',dtype=float)
AuII_4f135d106s_2_5=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuII_4f135d106s_2_5.dat',dtype=float)
AuII_4f135d106s_3_5=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuII_4f135d106s_3_5.dat',dtype=float)
#%% Au I RTDLDA 4f channels
AuI_4f135d106s2_2_5=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuI_4f135d106s2_2_5.dat',dtype=float)
AuI_4f135d106s2_3_5=np.loadtxt('C:/Users/Padmin/OneDrive/Documents/GitHub/HEAVYMETAL/RTDLDA_Files/AuI_4f135d106s2_3_5.dat',dtype=float)
#%% Summing up all the intensities of Au I
AuI_energies=AuI[:,0]
AuI_intensities=AuI[:,1]+AuI_5d96s2_1_5[:,1]+AuI_5d96s2_2_5[:,1]+AuI_4f135d106s2_2_5[30:,1]+AuI_4f135d106s2_3_5[30:,1]
#%% Summing up all the intensities of Au I New Energies
AuI_energies_new_energies=AuI_new_energies[:,0]
AuI_intensities_new_energies=AuI_new_energies[:,1]+AuI_5d96s2_1_5_new_energies[:,1]+AuI_5d96s2_2_5_new_energies[30:,1]+AuI_4f135d106s2_2_5[30:,1]+AuI_4f135d106s2_3_5[30:,1]
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
for E in AuI_5d96s2_1_5[:,0]:
    a=0
    if len(AuII_5d96s_1_5_new_energies[:,0][AuII_5d96s_1_5_new_energies[:,0]==E])>0:
        a+=AuII_5d96s_1_5_new_energies[:,1][AuII_5d96s_1_5_new_energies[:,0]==E]
    if len(AuII_new_energies[:,0][AuII_new_energies[:,0]==E])>0:
        a+=AuII_new_energies[:,1][AuII_new_energies[:,0]==E]
    if len(AuII_5d86s2_2_5_new_energies[:,0][AuII_5d86s2_2_5_new_energies[:,0]==E])>0:
        a+=AuII_5d86s2_2_5_new_energies[:,1][AuII_5d86s2_2_5_new_energies[:,0]==E]
    if len(AuII_5d86s2_1_5_new_energies[:,0][AuII_5d86s2_1_5_new_energies[:,0]==E]>0):
        a+=AuII_5d86s2_1_5_new_energies[:,1][AuII_5d86s2_1_5_new_energies[:,0]==E]
    if len(AuII_5d86s2_2_5_1_5_new_energies[:,0][AuII_5d86s2_2_5_1_5_new_energies[:,0]==E])>0:
        a+=AuII_5d86s2_2_5_1_5_new_energies[:,1][AuII_5d86s2_2_5_1_5_new_energies[:,0]==E]
    if len(AuII_5d96s_2_5_new_energies[:,0][AuII_5d96s_2_5_new_energies[:,0]==E])>0:
        a+=AuII_5d96s_2_5_new_energies[:,1][AuII_5d96s_2_5_new_energies[:,0]==E]
    if len(AuII_4f135d106s_2_5[:,0][AuII_4f135d106s_2_5[:,0]==E])>0:
        a+=AuII_4f135d106s_2_5[:,1][AuII_4f135d106s_2_5[:,0]==E]
    if len(AuII_4f135d106s_3_5[:,0][AuII_4f135d106s_3_5[:,0]==E])>0:
        a+=AuII_4f135d106s_3_5[:,1][AuII_4f135d106s_3_5[:,0]==E]
    if len(AuII_4f_4f125d106s2_2_5[:,0][AuII_4f_4f125d106s2_2_5[:,0]==E])>0:
        a+=AuII_4f_4f125d106s2_2_5[:,1][AuII_4f_4f125d106s2_2_5[:,0]==E]
    if len(AuII_4f_4f125d106s2_3_5[:,0][AuII_4f_4f125d106s2_3_5[:,0]==E])>0:
        a+=AuII_4f_4f125d106s2_3_5[:,1][AuII_4f_4f125d106s2_3_5[:,0]==E]
    if len(AuII_4f_4f125d106s2_2_5_3_5[:,0][AuII_4f_4f125d106s2_2_5_3_5[:,0]==E])>0:
        a+=AuII_4f_4f125d106s2_2_5_3_5[:,1][AuII_4f_4f125d106s2_2_5_3_5[:,0]==E]
    if len(AuII_4f135d96s2_2_5_2_5[:,0][AuII_4f135d96s2_2_5_2_5[:,0]==E])>0:
        a+=AuII_4f135d96s2_2_5_2_5[:,1][AuII_4f135d96s2_2_5_2_5[:,0]==E]
    if len(AuII_4f135d96s2_2_5_1_5[:,0][AuII_4f135d96s2_2_5_1_5[:,0]==E])>0:
        a+=AuII_4f135d96s2_2_5_1_5[:,1][AuII_4f135d96s2_2_5_1_5[:,0]==E]
    if len(AuII_4f135d96s2_3_5_1_5[:,0][AuII_4f135d96s2_3_5_1_5[:,0]==E])>0:
        a+=AuII_4f135d96s2_3_5_1_5[:,1][AuII_4f135d96s2_3_5_1_5[:,0]==E]
    if len(AuII_4f135d96s2_3_5_2_5[:,0][AuII_4f135d96s2_3_5_2_5[:,0]==E])>0:
        a+=AuII_4f135d96s2_3_5_2_5[:,1][AuII_4f135d96s2_3_5_2_5[:,0]==E]
    AuII_intensity_new_energies.append(a)
#AuII_intensity_new_energies=np.array(AuII_intensity_new_energies)
AuII_intensity_new_energies=np.array(AuII_intensity_new_energies)[:,0]
#%% Summing up all the intensities of Au II
AuII_energies=AuI_5d96s2_1_5[:,0]
#AuII_intensities=AuII[:,1]+AuII_5d96s_1_5[:,1]+AuII_5d96s_2_5[:,1]+AuII_5d86s2_1_5[:,1]+AuII_5d86s2_2_5[:,1]+AuII_5d86s2_2_5_1_5[:,1]
#%%
Intensity_500ns=0.75*AuI_intensities + 0.25*AuII_intensity
#%%
Eric_data_500ns=np.loadtxt('C:/Users/Padmin/OneDrive/Desktop/Eric_data_500ns.txt',dtype=float).T
Eric_data_450ns=np.loadtxt('C:/Users/Padmin/OneDrive/Desktop/Eric_data_450ns.txt',dtype=float).T
Intensity_500ns_exp=Eric_data_500ns[1][np.logical_and(Eric_data_500ns[0]>=75,Eric_data_500ns[0]<=130)]
Intensity_450ns_exp=Eric_data_450ns[1][np.logical_and(Eric_data_450ns[0]>=75,Eric_data_450ns[0]<=130)]
Energy=Eric_data_500ns[0][np.logical_and(Eric_data_500ns[0]>=75,Eric_data_500ns[0]<=130)]
#%%
plt.plot(AuI_energies+ np.repeat(0.8,len(AuI_energies)),AuI_intensities,label='Au I')
plt.plot(AuII_energies+ np.repeat(0.8,len(AuI_energies)),AuII_intensity,label='Au II')
#plt.plot(AuI_energies+ np.repeat(0.8,len(AuI_energies)),AuI_intensities,label='AuI')
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
plt.plot(Energy,550*Intensity_500ns_exp,label='500ns')
#plt.plot(AuII_5d96s_1_5_new_energies[:,0],AuII_5d96s_1_5_new_energies[:,1])
#plt.plot(AuII_5d96s_2_5_new_energies[:,0],AuII_5d96s_2_5_new_energies[:,1])
#plt.plot(AuII_new_energies[:,0],AuII_new_energies[:,1])
#plt.plot(AuII_5d86s2_1_5_new_energies[:,0],AuII_5d86s2_1_5_new_energies[:,1])
#plt.plot(AuII_5d86s2_2_5_new_energies[:,0],AuII_5d86s2_2_5_new_energies[:,1])
#plt.plot(AuII_5d86s2_2_5_1_5_new_energies[:,0],AuII_5d86s2_1_5_new_energies[:,1])
plt.plot(AuII_energies +np.repeat(0.8,len(AuII_energies)),AuII_intensity_new_energies,label='AuII New Energies')
#plt.plot(AuII_energies,AuII_intensity,label='AuII Old Energies',ls='--')
plt.legend()
plt.xlabel('Energies (eV)')
plt.ylabel('Intensity')
#%%
# Computes moving average
def MovingAverage(window_size,array):
    ws=window_size

    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []

    # Loop through the array to consider
    # every window of size 3
    while i < len(array) - ws + 1:
      
        # Store elements from i to i+window_size
        # in list to get the current window
        window = array[i : i + ws]

        # Calculate the average of current window
        window_average = sum(window) / window_size
        
        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)
        
        # Shift window to right by one position
        i += 1
    return moving_averages
#%%
AuII_moving_avg= MovingAverage(5, AuII_intensity_new_energies)
AuI_moving_avg= MovingAverage(5, AuI_intensities_new_energies)
AuI_energies_moving_avg= MovingAverage(5, AuI_energies_new_energies)
AuII_energies_moving_avg= MovingAverage(5, AuII_energies)
#%%
#plt.plot(AuII_energies_moving_avg,AuII_moving_avg,label='AuII moving average')
#plt.plot(AuI_energies_moving_avg+ np.repeat(0.8,len(AuI_energies_moving_avg)),AuI_moving_avg,label='AuI moving average')
plt.plot(Energy,265*Intensity_500ns_exp,label='500ns')
#plt.plot(Energy,250*Intensity_450ns_exp,label='450ns')
plt.plot(AuI_energies_moving_avg+ np.repeat(2.8,len(AuI_energies_moving_avg)),0.95*np.array(AuI_moving_avg) + 0.05*np.array(AuII_moving_avg),label='95% AuI, 5% AuII')
plt.legend()
plt.xlabel('Energy [eV]')
plt.ylabel('Averaged Cross Section [mb]')
