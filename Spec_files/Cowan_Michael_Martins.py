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

def TotalCrossSection(list_of_arrays,list_of_weights):
    energy_vals=np.arange(75.7,90.1,0.1)
    Intensity_vals=np.zeros(len(energy_vals))
    for a,sigma in enumerate(list_of_arrays):
        for b,E in enumerate(energy_vals):
            if len(sigma[:,0][sigma[:,0]==E])>0:
                c=sigma[:,1][sigma[:,0]==E]
                Intensity_vals[b]+=c[0]*list_of_weights[a]
    return Intensity_vals,energy_vals

def TotalCrossSection2(list_of_arrays,list_of_weights):
    energy_vals=np.arange(75.7,90.1,0.1)
    Intensity_vals=np.zeros(len(energy_vals))

    for sigma, w in zip(list_of_arrays, list_of_weights):
        for i, E in enumerate(energy_vals):
            mask = np.isclose(sigma[:, 0], E, atol=1e-6)
            if np.any(mask):
                Intensity_vals[i] += sigma[mask, 1][0] * w

    return Intensity_vals, energy_vals
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
Au_I_J_1_5_2_5[:,0]+=np.repeat(1.4,len(Au_I_J_1_5_2_5[:,0]))
Au_I_J_2_5_2_5[:,0]+=np.repeat(1.4,len(Au_I_J_2_5_2_5[:,0]))
Au_I_J_2_5_3_5[:,0]+=np.repeat(1.4,len(Au_I_J_2_5_3_5[:,0]))
#%%
Au_I_J_1_5_2_5_int=Au_I_J_1_5_2_5[:,1]
Au_I_J_2_5_2_5_int=Au_I_J_2_5_2_5[:,1]
Au_I_J_2_5_3_5_int=Au_I_J_2_5_3_5[:,1]
#%%
Au_II_J_1_2=np.array(Au_II_J_1_2[4:]).astype(float)
Au_II_J_2_2=np.array(Au_II_J_2_2[4:]).astype(float)
Au_II_J_2_3=np.array(Au_II_J_2_3[4:]).astype(float)
Au_II_J_3_3=np.array(Au_II_J_3_3[4:]).astype(float)
Au_II_J_3_4=np.array(Au_II_J_3_4[4:]).astype(float)
Au_II_J_4_4=np.array(Au_II_J_4_4[4:]).astype(float)
Au_II_J_4_5=np.array(Au_II_J_4_5[4:]).astype(float)
#%%
Au_II_J_1_2[:,0]+=np.repeat(0.7,len(Au_II_J_1_2[:,0]))
Au_II_J_2_2[:,0]+=np.repeat(0.7,len(Au_II_J_2_2[:,0]))
Au_II_J_2_3[:,0]+=np.repeat(0.7,len(Au_II_J_2_3[:,0]))
Au_II_J_3_3[:,0]+=np.repeat(0.7,len(Au_II_J_3_3[:,0]))
Au_II_J_3_4[:,0]+=np.repeat(0.7,len(Au_II_J_3_4[:,0]))
Au_II_J_4_4[:,0]+=np.repeat(0.7,len(Au_II_J_4_4[:,0]))
Au_II_J_4_5[:,0]+=np.repeat(0.7,len(Au_II_J_4_5[:,0]))
Energy=Au_II_J_1_2[:143,0]
#%%
Au_I_J_1_5_2_5_int=Au_I_J_1_5_2_5_int[:136]
Au_I_J_2_5_2_5_int=Au_I_J_2_5_2_5_int[:136]
Au_I_J_2_5_3_5_int=Au_I_J_2_5_3_5_int[:136]
#%%
stand_in=np.zeros(7)
Au_I_J_1_5_2_5_int=np.concatenate((stand_in,Au_I_J_1_5_2_5_int))
Au_I_J_2_5_2_5_int=np.concatenate((stand_in,Au_I_J_2_5_2_5_int))
Au_I_J_2_5_3_5_int=np.concatenate((stand_in,Au_I_J_2_5_3_5_int))
#%%
#list_of_arrays_AuI=[Au_I_J_1_5_2_5,Au_I_J_2_5_2_5,Au_I_J_2_5_3_5]
#list_of_weights_AuI=[0.1,0.1,0.1]
#list_of_arrays_AuII=[Au_II_J_1_2,Au_II_J_2_2,Au_II_J_2_3,Au_II_J_3_3,Au_II_J_3_4,Au_II_J_4_4,Au_II_J_4_5]
#list_of_weights_AuII=[np.exp(-(3.4011/0.5)),np.exp(-(2.1513/0.5)),np.exp(-(2.1513/0.5)),np.exp(-(1.7873/0.5)),np.exp(-(1.7873/0.5)),np.exp(-(5.2425/0.5)),np.exp(-(5.2425/0.5))]
#%%
Au_II_J_1_2=Au_II_J_1_2[:,1]
Au_II_J_2_2=Au_II_J_2_2[:,1]
Au_II_J_2_3=Au_II_J_2_3[:,1]
Au_II_J_3_3=Au_II_J_3_3[:,1]
Au_II_J_3_4=Au_II_J_3_4[:,1]
Au_II_J_4_4=Au_II_J_4_4[:,1]
Au_II_J_4_5=Au_II_J_4_5[:,1]
#%%
#Au_II_J_1_2=np.exp(-(3.4011/2.44))*Au_II_J_1_2[:,1]
#Au_II_J_2_2=np.exp(-(2.1513/2.44))*Au_II_J_2_2[:,1]
#Au_II_J_2_3=np.exp(-(2.1513/2.44))*Au_II_J_2_3[:,1]
#Au_II_J_3_3=np.exp(-(1.7873/2.44))*Au_II_J_3_3[:,1]
#Au_II_J_3_4=np.exp(-(1.7873/2.44))*Au_II_J_3_4[:,1]
#Au_II_J_4_4=np.exp(-(5.2425/2.44))*Au_II_J_4_4[:,1]
#Au_II_J_4_5=np.exp(-(5.2425/2.44))*Au_II_J_4_5[:,1]
#%%
Au_II_J_1_2_int=Au_II_J_1_2[:143]
Au_II_J_2_2_int=Au_II_J_2_2[:143]
Au_II_J_2_3_int=Au_II_J_2_3[:143]
Au_II_J_3_3_int=Au_II_J_3_3[:143]
Au_II_J_3_4_int=Au_II_J_3_4[:143]
Au_II_J_4_4_int=Au_II_J_4_4[:143]
Au_II_J_4_5_int=Au_II_J_4_5[:143]
#%%
total_cross_sections_AuI=Au_I_J_1_5_2_5_int+Au_I_J_2_5_2_5_int+Au_I_J_2_5_3_5_int
total_cross_sections_AuII=Au_II_J_1_2_int+Au_II_J_2_2_int+Au_II_J_2_3_int+Au_II_J_3_3_int+Au_II_J_3_4_int+Au_II_J_4_4_int+Au_II_J_4_5_int
#%%
#plt.plot(Energy,Au_I_J_1_5_2_5_int,label='J:1.5-2.5')
#plt.plot(Energy,Au_I_J_2_5_2_5_int,label='J:2.5-2.5')
#plt.plot(Energy,Au_I_J_2_5_3_5_int,label='J:2.5-3.5')
#plt.plot(Energy,total_cross_sections_AuII,label='total AuII')
#plt.plot(Energy,Au_II_J_1_2_int,label='J:1-2')
#plt.plot(Energy,Au_II_J_2_2_int,label='J:2-2')
#plt.plot(Energy,Au_II_J_2_3_int,label='J:2-3')
#plt.plot(Energy,Au_II_J_3_3_int,label='J:3-3')
#plt.plot(Energy,Au_II_J_3_4_int,label='J:3-4')
#plt.plot(Energy,Au_II_J_4_4_int,label='J:4-4')
#plt.plot(Energy,Au_II_J_4_5_int,label='J:4-5')
#plt.plot(moving_avg_energy,tcs_mov_avg_AuI)
#plt.plot(moving_avg_energy,tcs_mov_avg_AuII)
#plt.plot(Energy,total_cross_sections_AuI,label='AuI total')
plt.plot(Energy_500ns,170*Intensity_500ns-np.repeat(40,len(Intensity_500ns)),label='Exp 500ns')
#plt.plot(Energy,Au_II_J_2_3_int+0.85*Au_II_J_3_4_int+0.3*Au_I_J_2_5_3_5_int)
plt.plot(Energy,0.6*Au_I_J_1_5_2_5_int+0.6*0.85*Au_II_J_1_2_int+0.3*Au_II_J_4_5_int+Au_II_J_2_3_int+0.85*Au_II_J_3_4_int+0.3*Au_I_J_2_5_3_5_int)
#plt.plot(Energy,0.775*total_cross_sections_AuI + 0.225*total_cross_sections_AuII,label='22.5% AuII, 77.5% AuI')
plt.xlim(78,90)
plt.legend()
plt.xlabel('Energy [eV]')
plt.ylabel('Cross section [mb]')