# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 11:23:20 2025

@author: padmin
"""

import numpy as np
import matplotlib.pyplot as plt
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
Eric_data_500ns=np.loadtxt('C:/Users/David McKeagney/Downloads/Eric_data_500ns.txt',dtype=float).T
#Eric_data_500ns=np.loadtxt('C:/Users/Padmin/Downloads/Eric_data_500ns.txt',dtype=float).T
Intensity_500ns=Eric_data_500ns[1][np.logical_and(Eric_data_500ns[0]>=78,Eric_data_500ns[0]<=100)]
Energy_500ns=Eric_data_500ns[0][np.logical_and(Eric_data_500ns[0]>=78,Eric_data_500ns[0]<=100)]
#%%
Au_I_J_2_5_3_5=[]
Au_I_J_2_5_2_5=[]
Au_I_J_1_5_2_5=[]
with open('C:\\Users\David McKeagney\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.I.J=2.5-3.5.sigma') as file:
    for lines in file:
        Au_I_J_2_5_3_5.append(lines.split())
with open('C:\\Users\David McKeagney\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.I.J=2.5-2.5.sigma') as file:
    for lines in file:
        Au_I_J_2_5_2_5.append(lines.split())
with open('C:\\Users\David McKeagney\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.I.J=1.5-2.5.sigma') as file:
    for lines in file:
        Au_I_J_1_5_2_5.append(lines.split())
#%%
Au_II_J_1_2=[]
Au_II_J_2_2=[]
Au_II_J_2_3=[]
Au_II_J_3_3=[]
Au_II_J_3_4=[]
Au_II_J_4_4=[]
Au_II_J_4_5=[]
with open('C:\\Users\David McKeagney\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=1.0-2.0.sigma') as file:
    for lines in file:
        Au_II_J_1_2.append(lines.split())
with open('C:\\Users\David McKeagney\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=2.0-2.0.sigma') as file:
    for lines in file:
        Au_II_J_2_2.append(lines.split())
with open('C:\\Users\David McKeagney\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=2.0-3.0.sigma') as file:
    for lines in file:
        Au_II_J_2_3.append(lines.split())
with open('C:\\Users\David McKeagney\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=3.0-3.0.sigma') as file:
    for lines in file:
        Au_II_J_3_3.append(lines.split())
with open('C:\\Users\David McKeagney\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=3.0-4.0.sigma') as file:
    for lines in file:
        Au_II_J_3_4.append(lines.split())
with open('C:\\Users\David McKeagney\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=4.0-4.0.sigma') as file:
    for lines in file:
        Au_II_J_4_4.append(lines.split())
with open('C:\\Users\David McKeagney\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=4.0-5.0.sigma') as file:
    for lines in file:
        Au_II_J_4_5.append(lines.split())

#%%
Au_I_J_1_5_2_5=np.array(Au_I_J_1_5_2_5[4:]).astype(float)
Au_I_J_2_5_2_5=np.array(Au_I_J_2_5_2_5[4:]).astype(float)
Au_I_J_2_5_3_5=np.array(Au_I_J_2_5_3_5[4:]).astype(float)
#%%
Energy=Au_I_J_1_5_2_5[:,0]
#%%
Au_I_J_1_5_2_5=np.exp(-(2.97397/0.5))*Au_I_J_1_5_2_5[:,1]
Au_I_J_2_5_2_5=np.exp(-(1.3931/0.5))*Au_I_J_2_5_2_5[:,1]
Au_I_J_2_5_3_5=np.exp(-(1.3931/0.5))*Au_I_J_2_5_3_5[:,1]
#%%
Au_I_J_1_5_2_5_nlte=0.02*Au_I_J_1_5_2_5[:,1]
Au_I_J_2_5_2_5_nlte=0.25*Au_I_J_2_5_2_5[:,1]-np.repeat(0.3,len(Au_I_J_2_5_2_5))
Au_I_J_2_5_3_5_nlte=0.025*Au_I_J_2_5_3_5[:,1]
#%%
Au_II_J_1_2=np.array(Au_II_J_1_2[4:]).astype(float)
Au_II_J_2_2=np.array(Au_II_J_2_2[4:]).astype(float)
Au_II_J_2_3=np.array(Au_II_J_2_3[4:]).astype(float)
Au_II_J_3_3=np.array(Au_II_J_3_3[4:]).astype(float)
Au_II_J_3_4=np.array(Au_II_J_3_4[4:]).astype(float)
Au_II_J_4_4=np.array(Au_II_J_4_4[4:]).astype(float)
Au_II_J_4_5=np.array(Au_II_J_4_5[4:]).astype(float)
#%%
Au_II_J_1_2=np.exp(-(3.4011/0.5))*Au_II_J_1_2[:,1]
Au_II_J_2_2=np.exp(-(2.1513/0.5))*Au_II_J_2_2[:,1]
Au_II_J_2_3=np.exp(-(2.1513/0.5))*Au_II_J_2_3[:,1]
Au_II_J_3_3=np.exp(-(1.7873/0.5))*Au_II_J_3_3[:,1]
Au_II_J_3_4=np.exp(-(1.7873/0.5))*Au_II_J_3_4[:,1]
Au_II_J_4_4=np.exp(-(5.2425/0.5))*Au_II_J_4_4[:,1]
Au_II_J_4_5=np.exp(-(5.2425/0.5))*Au_II_J_4_5[:,1]
#%%
total_cross_sections_AuI=Au_I_J_1_5_2_5+Au_I_J_2_5_2_5+Au_I_J_2_5_3_5
#%%
total_cross_sections_AuI_nlte=Au_I_J_1_5_2_5_nlte+Au_I_J_2_5_2_5_nlte+Au_I_J_2_5_3_5_nlte
#%%
total_cross_sections_AuII=Au_II_J_1_2+Au_II_J_2_2+Au_II_J_2_3+Au_II_J_3_3+Au_II_J_3_4+Au_II_J_4_4+Au_II_J_4_5
#%%
tcs_mov_avg_AuI=MovingAverage(5,total_cross_sections_AuI)
tcs_mov_avg_AuII=MovingAverage(5,total_cross_sections_AuII)
moving_avg_energy=MovingAverage(5, Energy)
#%%
#plt.plot(Energy+np.repeat(1.4,len(Energy)),Au_I_J_1_5_2_5_nlte,label='J:1.5-2.5')
#plt.plot(Energy+np.repeat(1.4,len(Energy)),Au_I_J_2_5_2_5_nlte,label='J:2.5-2.5')
#plt.plot(Energy+np.repeat(1.4,len(Energy)),Au_I_J_2_5_3_5_nlte,label='J:2.5-3.5')
plt.plot(Energy+np.repeat(0.7,len(Energy)),total_cross_sections_AuII,label='total AuII')
#plt.plot(Energy,Au_II_J_1_2[:,1],label='J:1-2')
#plt.plot(Energy,Au_II_J_2_2[:,1],label='J:2-2')
#plt.plot(Energy,Au_II_J_2_3[:,1],label='J:2-3')
#plt.plot(Energy,Au_II_J_3_3[:,1],label='J:3-3')
#plt.plot(Energy,Au_II_J_3_4[:,1],label='J:3-4')
#plt.plot(Energy,Au_II_J_4_4[:,1],label='J:4-4')
#plt.plot(Energy,Au_II_J_4_5[:,1],label='J:4-5')
#plt.plot(moving_avg_energy,tcs_mov_avg_AuI)
#plt.plot(moving_avg_energy,tcs_mov_avg_AuII)
#plt.plot(Energy+np.repeat(1.4,len(Energy)),total_cross_sections_AuI_nlte,label='AuI total')
plt.plot(Energy_500ns,4*Intensity_500ns-np.repeat(0.85,len(Intensity_500ns)),label='Exp 500ns')
#plt.plot(Energy,0.87*total_cross_sections_AuI_nlte + 0.13*total_cross_sections_AuII)
plt.legend()
plt.xlabel('Energy [eV]')
plt.ylabel('Cross section [mb]')