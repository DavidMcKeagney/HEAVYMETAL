# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 11:23:20 2025

@author: padmin
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})
plt.rcParams["figure.figsize"] = (15,10)
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
Eric_data_500ns=np.loadtxt('C:/Users/David McKeagney/Downloads/Eric_data_500ns.txt',dtype=float).T
Eric_data_300ns=np.loadtxt('C:/Users/David McKeagney/Downloads/Eric_data_300ns.txt',dtype=float).T
#Eric_data_300ns=np.loadtxt('C:/Users/Padmin/Downloads/Eric_data_300ns.txt',dtype=float).T
#Eric_data_500ns=np.loadtxt('C:/Users/Padmin/Downloads/Eric_data_500ns.txt',dtype=float).T
Intensity_300ns=Eric_data_300ns[1][np.logical_and(Eric_data_300ns[0]>=78,Eric_data_300ns[0]<=100)]
Energy_300ns=Eric_data_300ns[0][np.logical_and(Eric_data_300ns[0]>=78,Eric_data_300ns[0]<=100)]
Intensity_500ns=Eric_data_500ns[1][np.logical_and(Eric_data_500ns[0]>=78,Eric_data_500ns[0]<=100)]
Energy_500ns=Eric_data_500ns[0][np.logical_and(Eric_data_500ns[0]>=78,Eric_data_500ns[0]<=100)]
#%%
Au_I_J_2_5_3_5_f=[]
Au_I_J_2_5_2_5_f=[]
Au_I_J_1_5_2_5_f=[]
with open('C:\\Users\David McKeagney\Downloads\Au.I.J=2.5-3.5.sigmaf') as file:
    for lines in file:
        Au_I_J_2_5_3_5_f.append(lines.split())
with open('C:\\Users\David McKeagney\Downloads\Au.I.J=2.5-2.5.sigmaf') as file:
    for lines in file:
        Au_I_J_2_5_2_5_f.append(lines.split())
with open('C:\\Users\David McKeagney\Downloads\Au.I.J=1.5-2.5.sigmaf') as file:
    for lines in file:
        Au_I_J_1_5_2_5_f.append(lines.split())


Au_I_J_1_5_2_5_f=np.array(Au_I_J_1_5_2_5_f[4:]).astype(float)
Au_I_J_2_5_2_5_f=np.array(Au_I_J_2_5_2_5_f[4:]).astype(float)
Au_I_J_2_5_3_5_f=np.array(Au_I_J_2_5_3_5_f[4:]).astype(float)

Au_I_J_1_5_2_5_f[:,0]+=np.repeat(1.4,len(Au_I_J_1_5_2_5_f[:,0]))
Au_I_J_2_5_2_5_f[:,0]+=np.repeat(1.4,len(Au_I_J_2_5_2_5_f[:,0]))
Au_I_J_2_5_3_5_f[:,0]+=np.repeat(1.4,len(Au_I_J_2_5_3_5_f[:,0]))

Au_I_J_1_5_2_5_int_f=Au_I_J_1_5_2_5_f[:,1]
Au_I_J_2_5_2_5_int_f=Au_I_J_2_5_2_5_f[:,1]
Au_I_J_2_5_3_5_int_f=Au_I_J_2_5_3_5_f[:,1]

Au_I_J_1_5_2_5_int_f=Au_I_J_1_5_2_5_int_f[:136]
Au_I_J_2_5_2_5_int_f=Au_I_J_2_5_2_5_int_f[:136]
Au_I_J_2_5_3_5_int_f=Au_I_J_2_5_3_5_int_f[:136]

stand_in=np.zeros(7)
Au_I_J_1_5_2_5_int_f=np.concatenate((stand_in,Au_I_J_1_5_2_5_int_f))
Au_I_J_2_5_2_5_int_f=np.concatenate((stand_in,Au_I_J_2_5_2_5_int_f))
Au_I_J_2_5_3_5_int_f=np.concatenate((stand_in,Au_I_J_2_5_3_5_int_f))

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
Au_I_J_1_5_2_5_new=[]
Au_I_J_2_5_2_5_new=[]
Au_I_J_2_5_3_5_new=[]
with open('C:\\Users\David McKeagney\Downloads\Au.I.J=1.5-2.5_new.sigma') as file:
    for lines in file:
        Au_I_J_1_5_2_5_new.append(lines.split())
with open('C:\\Users\David McKeagney\Downloads\Au.I.J=2.5-2.5_new.sigma') as file:
    for lines in file:
        Au_I_J_2_5_2_5_new.append(lines.split())
with open('C:\\Users\David McKeagney\Downloads\Au.I.J=2.5-3.5_new.sigma') as file:
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
Au_I_J_1_5_2_5_int=Au_I_J_1_5_2_5[:,1]
Au_I_J_2_5_2_5_int=Au_I_J_2_5_2_5[:,1]
Au_I_J_2_5_3_5_int=Au_I_J_2_5_3_5[:,1]
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
def bckg_500(x):
    return -0.005*x+0.692
def bckg_300(x):
    return -0.011*x+1.936

bck_500ns=bckg_500(Energy)
bck_300ns=bckg_300(Energy)
#%%
New_Fano_plot_1_5_2_5=0.008*Fano_plot3+bck_500ns
New_Fano_plot_2_5_3_5=0.007*Fano_plot2+bck_500ns
New_Fano_plot_2_5_2_5=0.009*Fano_plot1+bck_500ns
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
#plt.plot(Energy,Au_I_J_1_5_2_5_int_f,label='J: 1.5-2.5 sigmaf')
#plt.plot(Energy,Au_I_J_2_5_2_5_int_f,label='J: 2.5-2.5 sigmaf')
#plt.plot(Energy,Au_I_J_2_5_3_5_int_f,label='J: 2.5-3.5 sigma')
#plt.plot(Energy,Fano_plot1,label='J: 2.5-2.5 Fano')
#plt.plot(Energy,Fano_plot2,label='J: 2.5-3.5 Fano')
#plt.plot(Energy,Fano_plot3,label='J: 1.5-2.5 Fano')
plt.plot(Energy,0.005*Au_II_J_1_2_int+np.repeat(1.05,len(Energy)),label='J:1-2')
#plt.plot(Energy,Au_II_J_2_2_int,label='J:2-2')
#plt.plot(Energy,Fano_3_4)
#plt.plot(Energy,Fano_2_3)
#plt.plot(Energy,Fano_2_3_2)
#plt.plot(Energy,0.009*Fano_plot1+np.repeat(0.24,len(Energy)),label='J:2.5-2.5')
#plt.plot(Energy,0.008*Fano_plot2+np.repeat(0.28,len(Energy)),label='J:2.5-3.5')
#plt.plot(Energy,0.01*Fano_plot3+np.repeat(0.25,len(Energy)),label='J:1.5-2.5')
#plt.plot(Energy,1/38*Au_II_J_2_3_int+np.repeat(0.95,len(Energy)),label='J:2-3')
#plt.plot(Energy,Au_II_J_3_3_int,label='J:3-3')
#plt.plot(Energy,1/38*Au_II_J_3_4_int+np.repeat(0.95,len(Energy)),label='J:3-4')
#plt.plot(Energy, 1/38*(Au_II_J_2_3_int+Au_II_J_3_4_int)+np.repeat(0.95,len(Energy)),label='J:Summed')
#plt.plot(Energy,Au_II_J_4_4_int,label='J:4-4')
plt.plot(Energy,0.005*Au_II_J_4_5_int+np.repeat(1.05,len(Energy)),label='J:4-5')
plt.plot(Energy,0.005*(Au_II_J_4_5_int+Au_II_J_1_2_int)+np.repeat(1.05,len(Energy)),label='J:Summed')
#plt.plot(moving_avg_energy,tcs_mov_avg_AuI)
#plt.plot(moving_avg_energy,tcs_mov_avg_AuII)
#plt.plot(Energy,total_cross_sections_AuI,label='AuI total')
plt.plot(Energy_300ns,Intensity_300ns,label='300ns',color='black')
#plt.plot(Energy_300ns,15*Intensity_300ns-np.repeat(14,len(Intensity_300ns)))
#plt.plot(Energy,New_Fano_plot_1_5_2_5,label='J:1.5-2.5')
#plt.plot(Energy,New_Fano_plot_2_5_2_5,label='J:2.5-2.5')
#plt.plot(Energy,New_Fano_plot_2_5_3_5,label='J:2.5-3.5')
#plt.plot(Energy_500ns,Intensity_500ns,label='500ns',color='black')
#plt.plot(Energy,Au_II_J_2_3_int+0.85*Au_II_J_3_4_int+0.3*Au_I_J_2_5_3_5_int)
#plt.plot(Energy,0.8*0.4*Au_I_J_1_5_2_5_int+0.8*0.4*0.85*Au_II_J_1_2_int+0.8*0.4*0.3*Au_II_J_4_5_int+0.6*Au_II_J_2_3_int+0.6*0.85*Au_II_J_3_4_int+0.6*0.3*Au_I_J_2_5_3_5_int)
#plt.plot(Energy,0.875*total_cross_sections_AuI_sr + 0.125*total_cross_sections_AuII,label='12.5% AuII, 87.5% AuI')
plt.xlim(78,86)
#plt.ylim(0.21,0.39)
plt.legend()
#plt.title('Au II Fano Features 300ns ')
plt.xlabel('Energy [eV]')
plt.ylabel('Absorbance')
#plt.ylim(0.91,1.69)
#%%
def Plotting_Fano(x,a,b):
    return a*x + b

y1=0.009*Fano_plot1+np.repeat(0.24,len(Energy))
y2=0.008*Fano_plot2+np.repeat(0.28,len(Energy))
y3=0.01*Fano_plot3+np.repeat(0.25,len(Energy))

fig,ax1=plt.subplots()
ax1.plot(Energy,Fano_plot1,label='J:2.5-2.5')
ax1.plot(Energy,Fano_plot2,label='J:2.5-3.5')
ax1.plot(Energy,Fano_plot3,label='J:1.5-2.5')
ax1.set_xlabel('Energy [eV]')
ax1.set_ylabel('Cross Section [mb]')
ax1.tick_params(axis='y')
#ax1.set_ylim(0.25,0.3)
ax1.set_xlim(78,86)

ax2=ax1.twinx()
ax2.plot(Energy_500ns,Intensity_500ns,color='black',label='500ns')
ax2.set_ylabel('log(I/I_0)')
ax2.tick_params(axis='y')
ax2.set_ylim(0.23,0.39)


fig.legend(loc='upper right')
plt.show()
#%%
fig,axes=plt.subplots(3,figsize=(9, 10))
#fig.subplots_adjust(hspace=0.4, wspace=0.7)
axes[0].plot(Energy,0.009*Fano_plot1+np.repeat(0.24,len(Energy)),label='J:2.5-2.5')
axes[0].plot(Energy,0.008*Fano_plot2+np.repeat(0.28,len(Energy)),label='J:2.5-3.5')
axes[0].plot(Energy,0.01*Fano_plot3+np.repeat(0.25,len(Energy)),label='J:1.5-2.5')
axes[0].plot(Energy_500ns,Intensity_500ns,label='500ns',color='black')
axes[1].plot(Energy,1/38*Au_II_J_2_3_int+np.repeat(0.95,len(Energy)),label='J:2-3')
axes[1].plot(Energy,1/38*Au_II_J_3_4_int+np.repeat(0.95,len(Energy)),label='J:3-4')
axes[1].plot(Energy, 1/38*(Au_II_J_2_3_int+Au_II_J_3_4_int)+np.repeat(0.95,len(Energy)),label='J:Summed')
axes[1].plot(Energy_300ns,Intensity_300ns,label='300ns',color='black')
axes[2].plot(Energy,0.005*Au_II_J_1_2_int+np.repeat(1.05,len(Energy)),label='J:1-2')
axes[2].plot(Energy,0.005*Au_II_J_4_5_int+np.repeat(1.05,len(Energy)),label='J:4-5')
axes[2].plot(Energy,0.005*(Au_II_J_4_5_int+Au_II_J_1_2_int)+np.repeat(1.05,len(Energy)),label='J:Summed')
axes[2].plot(Energy_300ns,Intensity_300ns,label='300ns',color='black')
#for ax in axes.flat:
#    ax.set(xlabel='Energy [eV]', ylabel='Absorbance')
axes[1].set_ylabel('Absorbance',fontsize='x-large')
axes[2].set_xlabel('Energy [eV]', fontsize='x-large')
for ax in axes.flat:
    ax.label_outer()
    
for ax in axes.flat:
    ax.set_xlim(78,86)
for ax in axes.flat:
    ax.legend(fontsize=12)

axes[0].set_ylim(0.2,0.43)
axes[0].set_yticks([0.25,0.35],minor=True)
axes[1].set_ylim(0.9,1.67)
axes[1].set_yticks([1,1.3,1.6])
axes[1].set_yticks([1.15,1.45],minor=True)
axes[2].set_ylim(0.9,1.67)
axes[2].set_yticks([1,1.3,1.6])
axes[2].set_yticks([1.15,1.45],minor=True)
#axes[1].set_yticks([0.9,1.2],minor=True)
#axes[2].set_yticks([0.9,1.2],minor=True)
for ax in axes.flat:
    ax.set_xticks([79,81,83,85],minor=True)
for ax in axes.flat:
    ax.tick_params(axis='both',which='minor',length=3)
    
labels=['a)','b)','c)']
    
for ax, label in zip(axes, labels):
    # Place label in upper-left corner of each subplot
    ax.text(
        0.02, 0.95, label, 
        transform=ax.transAxes,  # Coordinates relative to axes
        fontsize=14, fontweight='bold',
        va='top', ha='left'
    )

#axes[0].set_xlim(78,86)
#axes[1].set_xlim(78,86)
#axes[2].set_xlim(78,86)
#axes[0].legend()
#axes[1].legend()
#axes[2].legend()
#%%
scaled_shifted_4_5=0.005*Au_II_J_4_5_int+np.repeat(1.05,len(Energy))
scaled_shifted_1_2=0.005*Au_II_J_1_2_int+np.repeat(1.05,len(Energy))
np.savetxt('C:\\Users\David McKeagney\Desktop\Data Availabilty- Fano Resonances\Theoretical Data\Au II\J=4-5.dat',np.transpose([Energy,Au_II_J_4_5_int,scaled_shifted_4_5]))
#%%
np.savetxt('C:\\Users\David McKeagney\Desktop\Data Availabilty- Fano Resonances\Theoretical Data\Au II\J=1-2.dat',np.transpose([Energy,Au_II_J_1_2_int,scaled_shifted_1_2]))
#%%
scaled_shifted_2_3=1/38*Au_II_J_2_3_int+np.repeat(0.95,len(Energy))    
scaled_shifted_3_4=1/38*Au_II_J_3_4_int+np.repeat(0.95,len(Energy))
np.savetxt('C:\\Users\David McKeagney\Desktop\Data Availabilty- Fano Resonances\Theoretical Data\Au II\J=2-3.dat',np.transpose([Energy,Au_II_J_2_3_int,scaled_shifted_2_3])) 
np.savetxt('C:\\Users\David McKeagney\Desktop\Data Availabilty- Fano Resonances\Theoretical Data\Au II\J=3-4.dat',np.transpose([Energy,Au_II_J_3_4_int,scaled_shifted_3_4]))
#%%
scaled_shifted_2_5_2_5=0.009*Fano_plot1+np.repeat(0.24,len(Energy))
scaled_shifted_2_5_3_5=0.008*Fano_plot2+np.repeat(0.28,len(Energy))
scaled_shifted_1_5_2_5=0.01*Fano_plot3+np.repeat(0.25,len(Energy))
np.savetxt('C:\\Users\David McKeagney\Desktop\Data Availabilty- Fano Resonances\Theoretical Data\Au I\J=2.5-2.5.dat',np.transpose([Energy,Fano_plot1,scaled_shifted_2_5_2_5]))
np.savetxt('C:\\Users\David McKeagney\Desktop\Data Availabilty- Fano Resonances\Theoretical Data\Au I\J=2.5-3.5.dat',np.transpose([Energy,Fano_plot2,scaled_shifted_2_5_3_5]))
np.savetxt('C:\\Users\David McKeagney\Desktop\Data Availabilty- Fano Resonances\Theoretical Data\Au I\J=1.5-2.5.dat',np.transpose([Energy,Fano_plot3,scaled_shifted_1_5_2_5]))