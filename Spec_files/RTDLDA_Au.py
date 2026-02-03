# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 11:55:06 2025

@author: David McKeagney
"""

import numpy as np
import matplotlib.pyplot as plt 
import function_library_phd as flp

plt.rcParams.update({'font.size': 22})
plt.rcParams["figure.figsize"] = (15,10)
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
AuI_energies_moving_avg= MovingAverage(5, AuI_energies_new_energies)[:497]
AuII_energies_moving_avg= MovingAverage(5, AuII_energies)
#%% Channel averaging 
AuII_chan_avg=np.array(AuII_moving_avg)[20:]
Au_I_chan_avg=np.array(AuI_moving_avg)[:497]

#%%
#plt.plot(AuII_energies_moving_avg,AuII_moving_avg,label='AuII moving average')
#plt.plot(AuI_energies_moving_avg+ np.repeat(0.8,len(AuI_energies_moving_avg)),AuI_moving_avg,label='AuI moving average')
#plt.plot(Energy,265*Intensity_500ns_exp,label='500ns')
#plt.plot(Energy,250*Intensity_450ns_exp,label='450ns')
plt.plot(AuI_energies_moving_avg,0.95*np.array(AuI_moving_avg) + 0.05*np.array(AuII_moving_avg),label='95% AuI, 5% AuII')
plt.plot(AuI_energies_moving_avg,0.9*np.array(AuI_moving_avg) + 0.1*np.array(AuII_moving_avg),label='90% AuI, 10% AuII')
plt.plot(AuI_energies_moving_avg,0.8*np.array(AuI_moving_avg) + 0.2*np.array(AuII_moving_avg),label='80% AuI, 20% AuII')
plt.plot(AuI_energies_moving_avg,0.75*np.array(AuI_moving_avg) + 0.25*np.array(AuII_moving_avg),label='75% AuI, 25% AuII')
plt.plot(AuI_energies_moving_avg,0.65*np.array(AuI_moving_avg) + 0.35*np.array(AuII_moving_avg),label='65% AuI, 35% AuII')
#plt.plot(AuI_energies_moving_avg,0.05*np.array(AuI_moving_avg) + 0.95*np.array(AuII_moving_avg),label='5% AuI, 95% AuII')
#plt.vlines(94.9,0,120,colors='red',linestyles='--')
#plt.vlines(98.5,0,120,colors='red',linestyles='--')
#plt.vlines(97,0,120,colors='red',linestyles='--')
#plt.vlines(99.7,0,120,colors='blue',linestyles='--')
#plt.vlines(96.4,0,120,colors='red',linestyles='--')
plt.legend(fontsize=18)
#plt.grid()
plt.xlim(78,99.5)
#plt.ylim(35,130)
plt.xlabel('Energy [eV]')
plt.ylabel('Averaged Cross Section [mb]')
#%%
plt.plot(AuI_energies_moving_avg,0.7*Au_I_chan_avg + 0.3*AuII_chan_avg,label='70% AuI, 30% AuII')
plt.plot(AuI_energies_moving_avg,0.6*Au_I_chan_avg + 0.4*AuII_chan_avg,label='60% AuI, 40% AuII')
plt.plot(AuI_energies_moving_avg,0.35*Au_I_chan_avg + 0.65*AuII_chan_avg,label='35% AuI, 65% AuII')
plt.plot(AuI_energies_moving_avg,0.05*Au_I_chan_avg + 0.95*AuII_chan_avg,label='5% AuI, 95% AuII')
plt.xlabel('Energy [eV]')
plt.ylabel('Smoothed cross sections')
plt.legend()
plt.xlim(78,99.5)
#%%
def epsilon(x,Er,gamma):
    return (x-Er)*2/gamma
def Fano(x,Er,q,gamma):
     return (q+epsilon(x,Er,gamma))**2/(1+epsilon(x,Er,gamma)**2)
#%%
Fano1=Fano(np.array(AuI_energies_moving_avg),82.8314+1.4, 2.9, 0.26989)
Fano2=Fano(np.array(AuI_energies_moving_avg),79.1645+1.4, 2.5, 0.28415)
Fano3=Fano(np.array(AuI_energies_moving_avg),81.2532+1.4, 2.73, 0.26989)
plt.plot(np.array(AuI_energies_moving_avg)+ np.repeat(0.7,len(AuI_energies_moving_avg)),0.9*(np.array(AuI_moving_avg)+Fano1+Fano2+Fano3) + 0.1*np.array(AuII_moving_avg),label='90% AuI, 10% AuII')
plt.plot(Energy,270*Intensity_500ns_exp,label='500ns')
plt.legend()
#%%
AuII_intensity_new_energies_4f=[]
for E in AuI_5d96s2_1_5[:,0]:
    a=0
    if len(AuII_4f135d96s2_2_5_2_5[:,0][AuII_4f135d96s2_2_5_2_5[:,0]==E])>0:
        a+=AuII_4f135d96s2_2_5_2_5[:,1][AuII_4f135d96s2_2_5_2_5[:,0]==E]
    if len(AuII_4f135d96s2_2_5_1_5[:,0][AuII_4f135d96s2_2_5_1_5[:,0]==E])>0:
        a+=AuII_4f135d96s2_2_5_1_5[:,1][AuII_4f135d96s2_2_5_1_5[:,0]==E]
    if len(AuII_4f135d96s2_3_5_1_5[:,0][AuII_4f135d96s2_3_5_1_5[:,0]==E])>0:
        a+=AuII_4f135d96s2_3_5_1_5[:,1][AuII_4f135d96s2_3_5_1_5[:,0]==E]
    if len(AuII_4f135d96s2_3_5_2_5[:,0][AuII_4f135d96s2_3_5_2_5[:,0]==E])>0:
        a+=AuII_4f135d96s2_3_5_2_5[:,1][AuII_4f135d96s2_3_5_2_5[:,0]==E]
    if len(AuII_4f135d106s_2_5[:,0][AuII_4f135d106s_2_5[:,0]==E])>0:
        a+=AuII_4f135d106s_2_5[:,1][AuII_4f135d106s_2_5[:,0]==E]
    if len(AuII_4f135d106s_3_5[:,0][AuII_4f135d106s_3_5[:,0]==E])>0:
        a+=AuII_4f135d106s_3_5[:,1][AuII_4f135d106s_3_5[:,0]==E]
    AuII_intensity_new_energies_4f.append(a)

AuII_intensity_new_energies_4f=np.array(AuII_intensity_new_energies_4f)[:,0]
#%%
AuII_intensity_new_energies_5d=[]
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
    AuII_intensity_new_energies_5d.append(a)


AuII_intensity_new_energies_5d=np.array(AuII_intensity_new_energies_5d)[:,0]
#%%
AuI_energies_new_energies=AuI_new_energies[:,0]
AuI_intensities_new_energies_5d=AuI_new_energies[:,1]+AuI_5d96s2_1_5_new_energies[:,1]+AuI_5d96s2_2_5_new_energies[30:,1]
#%%
AuI_intensities_new_energies_4f=AuI_4f135d106s2_2_5[30:,1]+AuI_4f135d106s2_3_5[30:,1]
#%%
plt.plot(AuI_5d96s2_2_5[:,0],AuII_intensity_new_energies_4f)
#%%
plt.plot(AuI_5d96s2_2_5[:,0],AuII_intensity_new_energies_5d)
plt.plot(AuI_energies,AuI_intensities_new_energies_5d)
#%%
AuII_moving_avg_4f=MovingAverage(5, AuII_intensity_new_energies_4f)[16:]
AuII_moving_avg_5d=MovingAverage(5, AuII_intensity_new_energies_5d)[16:]
AuI_moving_avg_5d=MovingAverage(5, AuI_intensities_new_energies_5d)[:501]
AuI_moving_avg_4f=MovingAverage(5, AuI_intensities_new_energies_4f)[:501]
AuI_energies_moving_avg_5d=MovingAverage(5, AuI_energies_new_energies)[:501]
#%%
plt.plot(AuI_energies_moving_avg_5d,AuII_moving_avg_5d)
plt.plot(AuI_energies_moving_avg_5d,AuI_moving_avg_5d)
#%%
plt.plot(AuI_energies_moving_avg_5d,0.95*(1/3)*np.array(AuI_moving_avg_5d)+0.95*0.5*np.array(AuI_moving_avg_4f)+0.05*(1/6)*(np.array(AuII_moving_avg_5d)+np.array(AuII_moving_avg_4f)),label='95% Au I, 5% Au II')
plt.plot(AuI_energies_moving_avg_5d,0.9*(1/3)*np.array(AuI_moving_avg_5d)+0.9*0.5*np.array(AuI_moving_avg_4f)+0.1*(1/6)*(np.array(AuII_moving_avg_5d)+np.array(AuII_moving_avg_4f)),label='90% Au I, 10% Au II')
plt.plot(AuI_energies_moving_avg_5d,0.8*(1/3)*np.array(AuI_moving_avg_5d)+0.8*0.5*np.array(AuI_moving_avg_4f)+0.2*(1/6)*(np.array(AuII_moving_avg_5d)+np.array(AuII_moving_avg_4f)),label='80% Au I, 20% Au II')
plt.plot(AuI_energies_moving_avg_5d,0.75*(1/3)*np.array(AuI_moving_avg_5d)+0.75*0.5*np.array(AuI_moving_avg_4f)+0.25*(1/6)*(np.array(AuII_moving_avg_5d)+np.array(AuII_moving_avg_4f)),label='75% Au I, 25% Au II')
plt.plot(AuI_energies_moving_avg_5d,0.65*(1/3)*np.array(AuI_moving_avg_5d)+0.65*0.5*np.array(AuI_moving_avg_4f)+0.35*(1/6)*(np.array(AuII_moving_avg_5d)+np.array(AuII_moving_avg_4f)),label='65% Au I, 35% Au II')
plt.legend()