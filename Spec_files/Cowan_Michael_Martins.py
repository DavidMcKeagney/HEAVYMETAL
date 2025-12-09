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
def epsilon(x,Er,gamma):
    return (x-Er)*2/gamma
def Fano(x,Er,q,gamma):
     return (q+epsilon(x,Er,gamma))**2/(1+epsilon(x,Er,gamma)**2)
#%%
#Eric_data_500ns=np.loadtxt('C:/Users/David McKeagney/Downloads/Eric_data_500ns.txt',dtype=float).T
#Eric_data_300ns=np.loadtxt('C:/Users/David McKeagney/Downloads/Eric_data_300ns.txt',dtype=float).T
Eric_data_300ns=np.loadtxt('C:/Users/Padmin/Downloads/Eric_data_300ns.txt',dtype=float).T
Eric_data_500ns=np.loadtxt('C:/Users/Padmin/Downloads/Eric_data_500ns.txt',dtype=float).T
Intensity_300ns=Eric_data_300ns[1][np.logical_and(Eric_data_300ns[0]>=78,Eric_data_300ns[0]<=100)]
Energy_300ns=Eric_data_300ns[0][np.logical_and(Eric_data_300ns[0]>=78,Eric_data_300ns[0]<=100)]
Intensity_500ns=Eric_data_500ns[1][np.logical_and(Eric_data_500ns[0]>=78,Eric_data_500ns[0]<=100)]
Energy_500ns=Eric_data_500ns[0][np.logical_and(Eric_data_500ns[0]>=78,Eric_data_500ns[0]<=100)]
#%%
Au_I_J_2_5_3_5=[]
Au_I_J_2_5_2_5=[]
Au_I_J_1_5_2_5=[]
with open('C:\\Users\Padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.I.J=2.5-3.5.sigma') as file:
    for lines in file:
        Au_I_J_2_5_3_5.append(lines.split())
with open('C:\\Users\Padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.I.J=2.5-2.5.sigma') as file:
    for lines in file:
        Au_I_J_2_5_2_5.append(lines.split())
with open('C:\\Users\Padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.I.J=1.5-2.5.sigma') as file:
    for lines in file:
        Au_I_J_1_5_2_5.append(lines.split())
#%%
Au_I_J_1_5_2_5_new=[]
Au_I_J_2_5_2_5_new=[]
Au_I_J_2_5_3_5_new=[]
with open('C:\\Users\Padmin\Downloads\Au.I.J=1.5-2.5_new.sigma') as file:
    for lines in file:
        Au_I_J_1_5_2_5_new.append(lines.split())
with open('C:\\Users\Padmin\Downloads\Au.I.J=2.5-2.5_new.sigma') as file:
    for lines in file:
        Au_I_J_2_5_2_5_new.append(lines.split())
with open('C:\\Users\Padmin\Downloads\Au.I.J=2.5-3.5_new.sigma') as file:
    for lines in file:
        Au_I_J_2_5_3_5_new.append(lines.split())

#%%
Au_II_J_1_2=[]
Au_II_J_2_2=[]
Au_II_J_2_3=[]
Au_II_J_3_3=[]
Au_II_J_3_4=[]
Au_II_J_4_4=[]
Au_II_J_4_5=[]
with open('C:\\Users\Padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=1.0-2.0.sigma') as file:
    for lines in file:
        Au_II_J_1_2.append(lines.split())
with open('C:\\Users\Padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=2.0-2.0.sigma') as file:
    for lines in file:
        Au_II_J_2_2.append(lines.split())
with open('C:\\Users\Padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=2.0-3.0.sigma') as file:
    for lines in file:
        Au_II_J_2_3.append(lines.split())
with open('C:\\Users\Padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=3.0-3.0.sigma') as file:
    for lines in file:
        Au_II_J_3_3.append(lines.split())
with open('C:\\Users\Padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=3.0-4.0.sigma') as file:
    for lines in file:
        Au_II_J_3_4.append(lines.split())
with open('C:\\Users\Padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=4.0-4.0.sigma') as file:
    for lines in file:
        Au_II_J_4_4.append(lines.split())
with open('C:\\Users\Padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=4.0-5.0.sigma') as file:
    for lines in file:
        Au_II_J_4_5.append(lines.split())

#%%
Au_I_J_1_5_2_5=np.array(Au_I_J_1_5_2_5[4:]).astype(float)
Au_I_J_2_5_2_5=np.array(Au_I_J_2_5_2_5[4:]).astype(float)
Au_I_J_2_5_3_5=np.array(Au_I_J_2_5_3_5[4:]).astype(float)
#%%
Au_I_J_1_5_2_5_new=np.array(Au_I_J_1_5_2_5_new[4:]).astype(float)
Au_I_J_2_5_2_5_new=np.array(Au_I_J_2_5_2_5_new[4:]).astype(float)
Au_I_J_2_5_3_5_new=np.array(Au_I_J_2_5_3_5_new[4:]).astype(float)
#%%
Au_I_J_1_5_2_5[:,0]+=np.repeat(1.4,len(Au_I_J_1_5_2_5[:,0]))
Au_I_J_2_5_2_5[:,0]+=np.repeat(1.4,len(Au_I_J_2_5_2_5[:,0]))
Au_I_J_2_5_3_5[:,0]+=np.repeat(1.4,len(Au_I_J_2_5_3_5[:,0]))
#%%
Au_I_J_1_5_2_5_new[:,0]+=np.repeat(1.4,len(Au_I_J_1_5_2_5_new[:,0]))
Au_I_J_2_5_2_5_new[:,0]+=np.repeat(1.4,len(Au_I_J_2_5_2_5_new[:,0]))
Au_I_J_2_5_3_5_new[:,0]+=np.repeat(1.4,len(Au_I_J_2_5_3_5_new[:,0]))
#%%
Au_I_J_1_5_2_5_int=0.1*np.exp(-2.97397/1.88)*Au_I_J_1_5_2_5[:,1]
Au_I_J_2_5_2_5_int=0.1*np.exp(-1.3931/1.88)*Au_I_J_2_5_2_5[:,1]
Au_I_J_2_5_3_5_int=0.1*np.exp(-1.3931/1.88)*Au_I_J_2_5_3_5[:,1]
#%%
Au_I_J_1_5_2_5_new_int=0.1*np.exp(-2.97397/1.88)*Au_I_J_1_5_2_5_new[:,1]
Au_I_J_2_5_2_5_new_int=0.1*np.exp(-1.3931/1.88)*Au_I_J_2_5_2_5_new[:,1]
Au_I_J_2_5_3_5_new_int=0.1*np.exp(-1.3931/1.88)*Au_I_J_2_5_3_5_new[:,1]
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
Au_I_J_1_5_2_5_new_int=Au_I_J_1_5_2_5_new_int[:136]
Au_I_J_2_5_2_5_new_int=Au_I_J_2_5_2_5_new_int[:136]
Au_I_J_2_5_3_5_new_int=Au_I_J_2_5_3_5_new_int[:136]
#%%
stand_in=np.zeros(7)
Au_I_J_1_5_2_5_int=np.concatenate((stand_in,Au_I_J_1_5_2_5_int))
Au_I_J_2_5_2_5_int=np.concatenate((stand_in,Au_I_J_2_5_2_5_int))
Au_I_J_2_5_3_5_int=np.concatenate((stand_in,Au_I_J_2_5_3_5_int))
#%%
Au_I_J_1_5_2_5_new_int=np.concatenate((stand_in,Au_I_J_1_5_2_5_new_int))
Au_I_J_2_5_2_5_new_int=np.concatenate((stand_in,Au_I_J_2_5_2_5_new_int))
Au_I_J_2_5_3_5_new_int=np.concatenate((stand_in,Au_I_J_2_5_3_5_new_int))
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
#Au_II_J_1_2=np.exp(-(3.4011/1.67))*Au_II_J_1_2[:,1]
#Au_II_J_2_2=np.exp(-(2.1513/1.67))*Au_II_J_2_2[:,1]
#Au_II_J_2_3=np.exp(-(2.1513/1.67))*Au_II_J_2_3[:,1]
#Au_II_J_3_3=np.exp(-(1.7873/1.67))*Au_II_J_3_3[:,1]
#Au_II_J_3_4=np.exp(-(1.7873/1.67))*Au_II_J_3_4[:,1]
#Au_II_J_4_4=np.exp(-(5.2425/1.67))*Au_II_J_4_4[:,1]
#Au_II_J_4_5=np.exp(-(5.2425/1.67))*Au_II_J_4_5[:,1]
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
total_cross_sections_AuII=Au_II_J_2_2_int+Au_II_J_2_3_int+Au_II_J_3_3_int+Au_II_J_3_4_int+Au_II_J_4_4_int+Au_II_J_4_5_int
#%%
total_cross_sections_AuI_new=Au_I_J_1_5_2_5_new_int+Au_I_J_2_5_2_5_new_int+Au_I_J_2_5_3_5_new_int
#%%
Fano_3_4=Fano(Energy, (79.4882-1.7873+0.7), 2.66, 0.28568)
Fano_2_3=Fano(Energy,(79.1685-2.15125+0.7),2.62, 0.28694)
Fano_2_3_2=Fano(Energy, (82.8240-2.15125+0.7),3.03,0.27548)
#%%
Fano_plot1=Fano(Energy, 82.8314+1.4, 2.9, 0.26989)
Fano_plot2=Fano(Energy, 79.1645+1.4, 2.5, 0.28415)
Fano_plot3=Fano(Energy, 81.2532+1.4, 2.73, 0.26989)
#%%
total_cross_sections_AuI_sr=Fano_plot1+Fano_plot2+Fano_plot3
#%%
#plt.plot(Energy, total_cross_sections_AuI_sr,label='Single Fano')
#plt.plot(Energy, total_cross_sections_AuI,label='Michaels Calculated cross sections')
#plt.plot(Energy,Au_I_J_1_5_2_5_int,label='J:1.5-2.5')
#plt.plot(Energy,Au_I_J_2_5_2_5_int,label='J:2.5-2.5')
#plt.plot(Energy,Au_I_J_2_5_3_5_int,label='J:2.5-3.5')
#plt.plot(Energy,Au_I_J_1_5_2_5_new_int,label='J:1.5-2.5 new')
#plt.plot(Energy,Au_I_J_2_5_2_5_new_int,label='J:2.5-2.5 new')
#plt.plot(Energy,Au_I_J_2_5_3_5_new_int,label='J:2.5-3.5 new')
#plt.plot(Energy,total_cross_sections_AuII,label='total AuII')
plt.plot(Energy,0.005*Au_II_J_1_2_int+np.repeat(1.1,len(Energy)),label='J:1-2')
#plt.plot(Energy,Au_II_J_2_2_int,label='J:2-2')
#plt.plot(Energy,Fano_3_4)
#plt.plot(Energy,Fano_2_3)
#plt.plot(Energy,Fano_2_3_2)
#plt.plot(Energy,0.009*Fano_plot1+np.repeat(0.24,len(Energy)),label='J:2.5-2.5')
#plt.plot(Energy,0.008*Fano_plot2+np.repeat(0.28,len(Energy)),label='J:2.5-3.5')
#plt.plot(Energy,0.01*Fano_plot3+np.repeat(0.25,len(Energy)),label='J:1.5-2.5')
#plt.plot(Energy,1/170*Au_II_J_2_3_int+np.repeat(0.23,len(Energy)),label='J:2-3')
#plt.plot(Energy,Au_II_J_3_3_int,label='J:3-3')
#plt.plot(Energy,1/170*Au_II_J_3_4_int+np.repeat(0.23,len(Energy)),label='J:3-4')
#plt.plot(Energy, 1/170*(Au_II_J_2_3_int+Au_II_J_3_4_int)+np.repeat(0.23,len(Energy)),label='J:3-4+2-3')
#plt.plot(Energy,Au_II_J_4_4_int,label='J:4-4')
plt.plot(Energy,0.005*Au_II_J_4_5_int+np.repeat(1.1,len(Energy)),label='J:4-5')
#plt.plot(Energy,Au_II_J_4_5_int+Au_II_J_1_2_int,label='J: 1-2 + 4-5')
#plt.plot(moving_avg_energy,tcs_mov_avg_AuI)
#plt.plot(moving_avg_energy,tcs_mov_avg_AuII)
#plt.plot(Energy,total_cross_sections_AuI,label='AuI total')
plt.plot(Energy_300ns,Intensity_300ns,label='300ns')
#plt.plot(Energy_300ns,15*Intensity_300ns-np.repeat(14,len(Intensity_300ns)))
#plt.plot(Energy_500ns,Intensity_500ns,label='Exp 500ns')
#plt.plot(Energy,Au_II_J_2_3_int+0.85*Au_II_J_3_4_int+0.3*Au_I_J_2_5_3_5_int)
#plt.plot(Energy,0.8*0.4*Au_I_J_1_5_2_5_int+0.8*0.4*0.85*Au_II_J_1_2_int+0.8*0.4*0.3*Au_II_J_4_5_int+0.6*Au_II_J_2_3_int+0.6*0.85*Au_II_J_3_4_int+0.6*0.3*Au_I_J_2_5_3_5_int)
#plt.plot(Energy,0.875*total_cross_sections_AuI_sr + 0.125*total_cross_sections_AuII,label='12.5% AuII, 87.5% AuI')
plt.xlim(78,90)
plt.legend()
plt.title('Au II Fano Features 300ns ')
plt.xlabel('Energy [eV]')
plt.ylabel('Cross section [mb]')
#plt.ylim(0.2,0.4)