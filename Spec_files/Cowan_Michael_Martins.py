# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 11:23:20 2025

@author: padmin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
def Gaussian(x,FWHM,mu):
    Sigma=FWHM/(2*np.sqrt(2*np.log(2)))
    return 1/(Sigma*np.sqrt(2*np.pi))*(np.exp(-(x-mu)**2/(2*(Sigma)**2)))
#%%
Eric_data_500ns=np.loadtxt('C:/Users/David McKeagney/Downloads/Eric_data_500ns.txt',dtype=float).T
Eric_data_300ns=np.loadtxt('C:/Users/David McKeagney/Downloads/Eric_data_300ns.txt',dtype=float).T
Eric_data_250ns=np.loadtxt('C:/Users/David McKeagney/Downloads/Eric_data_250ns.txt',dtype=float).T
#Eric_data_300ns=np.loadtxt('C:/Users/Padmin/Downloads/Eric_data_300ns.txt',dtype=float).T
#Eric_data_500ns=np.loadtxt('C:/Users/Padmin/Downloads/Eric_data_500ns.txt',dtype=float).T
Intensity_300ns=Eric_data_300ns[1][np.logical_and(Eric_data_300ns[0]>=78,Eric_data_300ns[0]<=100)]
Energy_300ns=Eric_data_300ns[0][np.logical_and(Eric_data_300ns[0]>=78,Eric_data_300ns[0]<=100)]
Intensity_500ns=Eric_data_500ns[1][np.logical_and(Eric_data_500ns[0]>=78,Eric_data_500ns[0]<=100)]
Energy_500ns=Eric_data_500ns[0][np.logical_and(Eric_data_500ns[0]>=78,Eric_data_500ns[0]<=100)]
Intensity_250ns=Eric_data_250ns[1][np.logical_and(Eric_data_250ns[0]>=78,Eric_data_250ns[0]<=100)]
Energy_250ns=Eric_data_250ns[0][np.logical_and(Eric_data_250ns[0]>=78,Eric_data_250ns[0]<=100)]
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
Au_II_J_1_2=np.exp(-(3.4011/3))*Au_II_J_1_2[:,1]
Au_II_J_2_2=np.exp(-(2.1513/3))*Au_II_J_2_2[:,1]
Au_II_J_2_3=np.exp(-(2.1513/3))*Au_II_J_2_3[:,1]
Au_II_J_3_3=np.exp(-(1.7873/3))*Au_II_J_3_3[:,1]
Au_II_J_3_4=np.exp(-(1.7873/3))*Au_II_J_3_4[:,1]
Au_II_J_4_4=np.exp(-(5.2425/3))*Au_II_J_4_4[:,1]
Au_II_J_4_5=np.exp(-(5.2425/3))*Au_II_J_4_5[:,1]
#%%
Au_II_J_1_2_int=Au_II_J_1_2[:143]
Au_II_J_2_2_int=Au_II_J_2_2[:143]
Au_II_J_2_3_int=Au_II_J_2_3[:143]
Au_II_J_3_3_int=Au_II_J_3_3[:143]
Au_II_J_3_4_int=Au_II_J_3_4[:143]
Au_II_J_4_4_int=Au_II_J_4_4[:143]
Au_II_J_4_5_int=Au_II_J_4_5[:143]
#%%
#total_cross_sections_AuI=Au_I_J_1_5_2_5_int+Au_I_J_2_5_2_5_int+Au_I_J_2_5_3_5_int
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
    return -0.003*x+0.432
def bckg_300(x):
    return -0.015*x+2.006

bck_500ns=bckg_500(Energy)
bck_300ns=bckg_300(Energy)

#%%
New_Fano_plot_1_5_2_5=0.008*Fano_plot3+bck_500ns
New_Fano_plot_2_5_3_5=0.007*Fano_plot2+bck_500ns
New_Fano_plot_2_5_2_5=0.009*Fano_plot1+bck_500ns
#%%
def SynthSpec(a,b):
    return a*(Au_II_J_2_3_int+Au_II_J_3_4_int)+b*(Au_II_J_1_2_int+Au_II_J_4_5_int) +bck_300ns
#%%
Test_spec=SynthSpec(0.038,0.007)
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
#plt.plot(Energy,1/38*Au_II_J_2_2_int,label='J:2-2')
#plt.plot(Energy,Fano_3_4)
#plt.plot(Energy,Fano_2_3)
#plt.plot(Energy,Fano_2_3_2)
#plt.plot(Energy,0.009*Fano_plot1+np.repeat(0.24,len(Energy)),label='J:2.5-2.5')
#plt.plot(Energy,0.008*Fano_plot2+np.repeat(0.28,len(Energy)),label='J:2.5-3.5')
#plt.plot(Energy,0.01*Fano_plot3+np.repeat(0.25,len(Energy)),label='J:1.5-2.5')
#plt.plot(Energy,1/38*Au_II_J_2_3_int+np.repeat(0.95,len(Energy)),label='J:2-3')
#plt.plot(Energy,1/38*Au_II_J_3_3_int+np.repeat(1.05,len(Energy)),label='J:3-3')
#plt.plot(Energy,1/38*Au_II_J_3_4_int+np.repeat(0.95,len(Energy)),label='J:3-4')
#plt.plot(Energy, 1/38*(Au_II_J_2_3_int+Au_II_J_3_4_int)+np.repeat(0.95,len(Energy)),label='J:Summed')
#plt.plot(Energy,0.005*Au_II_J_4_4_int+np.repeat(1.05,len(Energy)),label='J:4-4')
plt.plot(Energy,0.005*Au_II_J_4_5_int+np.repeat(1.05,len(Energy)),label='J:4-5')
plt.plot(Energy,0.005*(Au_II_J_4_5_int+Au_II_J_1_2_int)+np.repeat(1.05,len(Energy)),label='J:Summed')
#plt.plot(moving_avg_energy,tcs_mov_avg_AuI)
#plt.plot(moving_avg_energy,tcs_mov_avg_AuII)
#plt.plot(Energy,total_cross_sections_AuI,label='AuI total')
#plt.plot(Energy,Test_spec-np.repeat(0.05,len(Energy)),label='synth spec')
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
#%%
Au_III_J_0_5_0_5=[]
Au_III_J_0_5_1_5=[]
Au_III_J_1_5_1_5=[]
Au_III_J_1_5_2_5=[]
Au_III_J_2_5_2_5=[]
Au_III_J_2_5_3_5=[]
Au_III_J_3_5_3_5=[]
Au_III_J_3_5_4_5=[]
Au_III_J_4_5_4_5=[]
Au_III_J_4_5_5_5=[]
Au_III_J_5_5_5_5=[]
with open('C:\\Users\David McKeagney\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.III.J=0.5-0.5.sigma') as file:
    for lines in file:
        Au_III_J_0_5_0_5.append(lines.split())
with open('C:\\Users\David McKeagney\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.III.J=0.5-1.5.sigma') as file:
    for lines in file:
        Au_III_J_0_5_1_5.append(lines.split())
with open('C:\\Users\David McKeagney\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.III.J=1.5-1.5.sigma') as file:
    for lines in file:
        Au_III_J_1_5_1_5.append(lines.split())
with open('C:\\Users\David McKeagney\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.III.J=1.5-2.5.sigma') as file:
    for lines in file:
        Au_III_J_1_5_2_5.append(lines.split())
with open('C:\\Users\David McKeagney\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.III.J=2.5-2.5.sigma') as file:
    for lines in file:
        Au_III_J_2_5_2_5.append(lines.split())
with open('C:\\Users\David McKeagney\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.III.J=2.5-3.5.sigma') as file:
    for lines in file:
        Au_III_J_2_5_3_5.append(lines.split())
with open('C:\\Users\David McKeagney\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.III.J=3.5-3.5.sigma') as file:
    for lines in file:
        Au_III_J_3_5_3_5.append(lines.split())
with open('C:\\Users\David McKeagney\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.III.J=3.5-4.5.sigma') as file:
    for lines in file:
        Au_III_J_3_5_4_5.append(lines.split())
with open('C:\\Users\David McKeagney\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.III.J=4.5-4.5.sigma') as file:
    for lines in file:
        Au_III_J_4_5_4_5.append(lines.split())
with open('C:\\Users\David McKeagney\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.III.J=4.5-5.5.sigma') as file:
    for lines in file:
        Au_III_J_4_5_5_5.append(lines.split())
with open('C:\\Users\David McKeagney\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.III.J=5.5-5.5.sigma') as file:
    for lines in file:
        Au_III_J_5_5_5_5.append(lines.split())
#%%
Au_III_J_0_5_0_5=np.array(Au_III_J_0_5_0_5[4:]).astype(float)
Au_III_J_0_5_1_5=np.array(Au_III_J_0_5_1_5[4:]).astype(float)
Au_III_J_1_5_1_5=np.array(Au_III_J_1_5_1_5[4:]).astype(float)
Au_III_J_1_5_2_5=np.array(Au_III_J_1_5_2_5[4:]).astype(float)
Au_III_J_2_5_2_5=np.array(Au_III_J_2_5_2_5[4:]).astype(float)
Au_III_J_2_5_3_5=np.array(Au_III_J_2_5_3_5[4:]).astype(float)
Au_III_J_3_5_3_5=np.array(Au_III_J_3_5_3_5[4:]).astype(float)
Au_III_J_3_5_4_5=np.array(Au_III_J_3_5_4_5[4:]).astype(float)
Au_III_J_4_5_4_5=np.array(Au_III_J_4_5_4_5[4:]).astype(float)
Au_III_J_4_5_5_5=np.array(Au_III_J_4_5_5_5[4:]).astype(float)
Au_III_J_5_5_5_5=np.array(Au_III_J_5_5_5_5[4:]).astype(float)
Energy_AuIII=Au_III_J_0_5_0_5[:,0]
#%%
plt.plot(Energy_AuIII,np.exp(-(5.8297+0.8390)/5.4)*Au_III_J_0_5_0_5[:,1],label='AuIII: J=0.5 - 0.5')
plt.plot(Energy_AuIII,np.exp(-(5.8297+0.8390)/5.4)*Au_III_J_0_5_1_5[:,1],label='AuIII: J=0.5 - 1.5')
plt.plot(Energy_AuIII,np.exp(-(0.7801+0.8390)/5.4)*Au_III_J_1_5_1_5[:,1],label='AuIII: J=1.5 - 1.5')
plt.plot(Energy_AuIII,np.exp(-(0.7801+0.8390)/5.4)*Au_III_J_1_5_2_5[:,1],label='AuIII: J=1.5 - 2.5')
plt.plot(Energy_AuIII,Au_III_J_2_5_2_5[:,1],label='AuIII: J=2.5 - 2.5')
plt.plot(Energy_AuIII,Au_III_J_2_5_3_5[:,1],label='AuIII: J=2.5 - 3.5')
plt.plot(Energy_AuIII,np.exp(-(3.5370+0.8390)/5.4)*Au_III_J_3_5_3_5[:,1],label='AuIII: J=3.5 - 3.5')
plt.plot(Energy_AuIII,np.exp(-(3.5370+0.8390)/5.4)*Au_III_J_3_5_4_5[:,1],label='AuIII: J=3.5 - 4.5')
plt.plot(Energy_AuIII,np.exp(-(2.8096+0.8390)/5.4)*Au_III_J_4_5_4_5[:,1],label='AuIII: J=4.5 - 4.5')
plt.plot(Energy_AuIII,np.exp(-(2.8096+0.8390)/5.4)*Au_III_J_4_5_5_5[:,1],label='AuIII: J=4.5 - 5.5')
plt.plot(Energy_AuIII,np.exp(-(13.2294+0.8390)/5.4)*Au_III_J_5_5_5_5[:,1],label='AuIII: J=5.5 - 5.5')
plt.legend()
#%%
Au_III_J_0_5_0_5_w=np.exp(-(5.8297+0.8390)/3)*Au_III_J_0_5_0_5[:,1]
Au_III_J_0_5_1_5_w=np.exp(-(5.8297+0.8390)/3)*Au_III_J_0_5_1_5[:,1]
Au_III_J_1_5_1_5_w=np.exp(-(0.7801+0.8390)/3)*Au_III_J_1_5_1_5[:,1]
Au_III_J_1_5_2_5_w=np.exp(-(0.7801+0.8390)/3)*Au_III_J_1_5_2_5[:,1]
Au_III_J_3_5_3_5_w=np.exp(-(3.5370+0.8390)/3)*Au_III_J_3_5_3_5[:,1]
Au_III_J_3_5_4_5_w=np.exp(-(3.5370+0.8390)/3)*Au_III_J_3_5_4_5[:,1]
Au_III_J_4_5_4_5_w=np.exp(-(2.8096+0.8390)/3)*Au_III_J_4_5_4_5[:,1]
Au_III_J_4_5_5_5_w=np.exp(-(2.8096+0.8390)/3)*Au_III_J_4_5_5_5[:,1]
Au_III_J_5_5_5_5_w=np.exp(-(13.2294+0.8390)/3)*Au_III_J_5_5_5_5[:,1]
#%%
AuIII_tot_T_3_1=Au_III_J_0_5_0_5_w+Au_III_J_0_5_1_5_w+Au_III_J_1_5_1_5_w+Au_III_J_1_5_2_5_w+Au_III_J_2_5_2_5[:,1]+Au_III_J_2_5_3_5[:,1]+Au_III_J_3_5_3_5_w+Au_III_J_3_5_4_5_w+Au_III_J_4_5_4_5_w+Au_III_J_4_5_5_5_w+Au_III_J_5_5_5_5_w
#%%
Au_I_J_2_5_3_5_int=np.exp(-1.3931/3)*Au_I_J_2_5_3_5_int_f
Au_I_J_2_5_2_5_int=np.exp(-1.3931/3)*Au_I_J_2_5_2_5_int_f
Au_I_J_1_5_2_5_int=np.exp(-2.97397/3)*Au_I_J_1_5_2_5_int_f
AuIII_tot_T_3_1_int=AuIII_tot_T_3_1[:143]
#plt.plot(Energy,0.669*total_cross_sections_AuII)
#plt.plot(Energy_300ns,18*Intensity_300ns-np.repeat(13.8,len(Intensity_300ns)),label='300ns')
#plt.plot(Energy,0.044*AuIII_tot_T_3_1_int)
plt.plot(Energy+np.repeat(0.1,len(Energy)),0.11*(Au_I_J_2_5_3_5_int+Au_I_J_2_5_2_5_int+Au_I_J_1_5_2_5_int)+0.78*total_cross_sections_AuII+0.11*AuIII_tot_T_3_1_int,label='78% AuII 11% Au I 11% Au III')
plt.xlabel('Energy [eV]')
plt.ylabel('Cross Section [Mb]')
plt.legend()
plt.xlim(78,86)
#%%
def InstrumentalBroadening(E,x_vals, sigma, CS):
    int_vals=np.arange(-x_vals,x_vals,0.01)
    Broad_CS=np.zeros(len(E))
    for a,i in enumerate(E):
        for b,j in enumerate(int_vals):
            Broad_CS[a]+=CS[a]*1/(np.sqrt(2*np.pi)*sigma)*np.exp(-((i-j)/sigma)**2)*0.01
    return Broad_CS
#%%
Broad_AuII=InstrumentalBroadening(Energy, 1000, 0.05, total_cross_sections_AuII)
#%%
plt.plot(Energy,Broad_AuII,label='Guassian Broadening')
plt.plot(Energy,total_cross_sections_AuII,label='No Broadening')
plt.legend()
#%%
Test_Synth_Spectra_300ns=0.0*(Au_I_J_2_5_3_5_int+Au_I_J_2_5_2_5_int+Au_I_J_1_5_2_5_int)+0.5*total_cross_sections_AuII+0.5*AuIII_tot_T_3_1_int
def LinearBckSynthSpec(X,a,b,c):
    CS,E = X
    return a*CS + b*E +c
#%%
#popt,cov=curve_fit(LinearBckSynthSpec, (Test_Synth_Spectra_300ns, Energy_300ns), Intensity_300ns)
#Energy=Energy[23:]
Test_Synth_Spectra_300ns=Test_Synth_Spectra_300ns[23:]

indices=[np.abs(Energy_300ns - v).argmin() for v in Energy]
sub_Energy_300ns=Energy_300ns[indices]
mask = np.isin(Eric_data_300ns[0, :], sub_Energy_300ns)
sub_Intensity_300ns = Eric_data_300ns[1, mask]
#%%
popt,cov=curve_fit(LinearBckSynthSpec, (Test_Synth_Spectra_300ns, sub_Energy_300ns), sub_Intensity_300ns,p0=[1/15,-0.01,1.7])
#%%
SynthSpec_LinBack=1/15*Test_Synth_Spectra_300ns+bck_300ns
#%%
plt.plot(Energy,SynthSpec_LinBack,label='75% AuII, 22% Au I, 3% Au III')
plt.plot(Energy_250ns,Intensity_250ns,label='250ns')
#plt.plot(Energy_300ns,Intensity_300ns,label='300ns')
#plt.plot(Energy_500ns,Intensity_500ns,label='500ns')
plt.xlim(78,86)
plt.legend()
plt.xlabel('Energy [eV]')
plt.ylabel('Absorbance [Arb.]')
#%%
shift_vals=np.arange(-0.5,0.5,0.1)
Gaussian_kernal=Gaussian(shift_vals, 0.2, 0)/(np.sum(Gaussian(shift_vals, 0.2, 0))*0.01)
L=len(Gaussian_kernal)+len(Test_Synth_Spectra_300ns)-1
Conv_spec_300ns=np.fft.ifft(np.fft.fft(Gaussian_kernal,L)*np.fft.fft(Test_Synth_Spectra_300ns,L)).real*0.1
start = (len(Gaussian_kernal) - 1) // 2
conv_same = Conv_spec_300ns[start:start + len(Test_Synth_Spectra_300ns)]
plt.plot(Energy,1/120*conv_same+bck_300ns)
plt.plot(Energy_300ns,Intensity_300ns)
plt.xlim(78.2,86)


