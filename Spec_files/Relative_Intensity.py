# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 12:06:51 2025

@author: David McKeagney
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import function_library_phd as flp

plt.rcParams.update({'font.size': 22})
plt.rcParams["figure.figsize"] = (15,10)
#%%
au_1_6_spec=[]
with open('C:/Users/David McKeagney/Desktop/au.sub.1.6.spec') as file4:
    for lines in file4:
        if len(lines.split())>17:
            au_1_6_spec.append(lines.split())

au_1_5_spec=[]
with open('C:/Users/David McKeagney/Desktop/au.sub.1.5.spec') as file4:
    for lines in file4:
        if len(lines.split())>17:
            au_1_5_spec.append(lines.split())
            
au_1_7_spec=[]
with open('C:/Users/David McKeagney/Desktop/au.sub.1.7.spec') as file4:
    for lines in file4:
        if len(lines.split())>17:
            au_1_7_spec.append(lines.split())

au_1_8_spec=[]
with open('C:/Users/David McKeagney/Desktop/au.sub.1.8.spec') as file4:
    for lines in file4:
        if len(lines.split())>17:
            au_1_8_spec.append(lines.split())
            
au_1_9_spec=[]
with open('C:/Users/David McKeagney/Desktop/au.sub.1.9.spec') as file4:
    for lines in file4:
        if len(lines.split())>17:
            au_1_9_spec.append(lines.split())
au_2_0_spec=[]
with open('C:/Users/David McKeagney/Desktop/au.sub.2.0.spec') as file4:
    for lines in file4:
        if len(lines.split())>17:
            au_2_0_spec.append(lines.split())
au1_1_5_spec=[]
with open('C:/Users/David McKeagney/Desktop/au1.sub.1.5.spec') as file1:
    for lines in file1:
        if len(lines.split())>17:
            au1_1_5_spec.append(lines.split())
au1_1_5_spec=np.array(au1_1_5_spec)[1:,:]
au_1_6_spec=np.array(au_1_6_spec)[1:,:]
au_1_7_spec=np.array(au_1_7_spec)[1:,:]
au_1_5_spec=np.array(au_1_5_spec)[1:,:]
au_1_8_spec=np.array(au_1_8_spec)[1:,:]
au_1_9_spec=np.array(au_1_9_spec)[1:,:]
au_2_0_spec=np.array(au_2_0_spec)[1:,:]
#%% T 1.6
au_1_6_spec_1=au_1_6_spec[au_1_6_spec[:,8]=='1']
au_1_6_spec_5=au_1_6_spec[au_1_6_spec[:,8]=='5']
au_1_6_spec=np.concatenate((au_1_6_spec_1,au_1_6_spec_5),axis=0)
dE_4f_6d_au_1_6=au_1_6_spec[:,11].astype(float)
gf_4f_6d_au_1_6=np.exp(au_1_6_spec[:,15].astype(float))
gamma_4f_6d_au_1_6=au_1_6_spec[:,16].astype(float)*1e-3
#%% T 1.7
au_1_7_spec_1=au_1_7_spec[au_1_7_spec[:,8]=='1']
au_1_7_spec_5=au_1_7_spec[au_1_7_spec[:,8]=='5']
au_1_7_spec=np.concatenate((au_1_7_spec_1,au_1_7_spec_5),axis=0)
dE_4f_6d_au_1_7=au_1_7_spec[:,11].astype(float)
gf_4f_6d_au_1_7=np.exp(au_1_7_spec[:,15].astype(float))
gamma_4f_6d_au_1_7=au_1_7_spec[:,16].astype(float)*1e-3
#%% T 1.8
au_1_8_spec_1=au_1_8_spec[au_1_8_spec[:,8]=='1']
au_1_8_spec_5=au_1_8_spec[au_1_8_spec[:,8]=='5']
au_1_8_spec=np.concatenate((au_1_8_spec_1,au_1_8_spec_5),axis=0)
dE_4f_6d_au_1_8=au_1_8_spec[:,11].astype(float)
gf_4f_6d_au_1_8=np.exp(au_1_8_spec[:,15].astype(float))
gamma_4f_6d_au_1_8=au_1_8_spec[:,16].astype(float)*1e-3
#%% T 1.9
au_1_9_spec_1=au_1_9_spec[au_1_9_spec[:,8]=='1']
au_1_9_spec_5=au_1_9_spec[au_1_9_spec[:,8]=='5']
au_1_9_spec=np.concatenate((au_1_9_spec_1,au_1_9_spec_5),axis=0)
dE_4f_6d_au_1_9=au_1_9_spec[:,11].astype(float)
gf_4f_6d_au_1_9=np.exp(au_1_9_spec[:,15].astype(float))
gamma_4f_6d_au_1_9=au_1_9_spec[:,16].astype(float)*1e-3

#%% T 1.5
au_1_5_spec_1=au_1_5_spec[au_1_5_spec[:,8]=='1']
au_1_5_spec_5=au_1_5_spec[au_1_5_spec[:,8]=='5']
au_1_5_spec=np.concatenate((au_1_5_spec_1,au_1_5_spec_5),axis=0)
dE_4f_6d_au_1_5=au_1_5_spec[:,11].astype(float)
gf_4f_6d_au_1_5=np.exp(au_1_5_spec[:,15].astype(float))
gamma_4f_6d_au_1_5=au_1_5_spec[:,16].astype(float)*1e-3
#%% T 2.0
au_2_0_spec_1=au_2_0_spec[au_2_0_spec[:,8]=='1']
au_2_0_spec_5=au_2_0_spec[au_2_0_spec[:,8]=='5']
au_2_0_spec=np.concatenate((au_2_0_spec_1,au_2_0_spec_5),axis=0)
dE_4f_6d_au_2_0=au_2_0_spec[:,11].astype(float)
gf_4f_6d_au_2_0=np.exp(au_2_0_spec[:,15].astype(float))
gamma_4f_6d_au_2_0=au_2_0_spec[:,16].astype(float)*1e-3
#%%
au1_1_5_spec_1=au1_1_5_spec[au1_1_5_spec[:,8]=='1']
au1_1_5_spec_5=au1_1_5_spec[au1_1_5_spec[:,8]=='5']
au1_1_5_spec=np.concatenate((au1_1_5_spec_1,au1_1_5_spec_5),axis=0)
dE_4f_6d_au1_1_5=au1_1_5_spec[:,11].astype(float)
gf_4f_6d_au1_1_5=np.exp(au1_1_5_spec[:,15].astype(float))
gamma_4f_6d_au1_1_5=au1_1_5_spec[:,16].astype(float)*1e-3
#%% Covolving the lines for the different temperatures 
E_vals=np.arange(88,120,0.001)
conv_au_1_6=flp.ConvolvingFunc(0, E_vals, dE_4f_6d_au_1_6, gf_4f_6d_au_1_6, 0.05,gamma_4f_6d_au_1_6, 3)
conv_au_1_5=flp.ConvolvingFunc(0, E_vals, dE_4f_6d_au_1_5, gf_4f_6d_au_1_5, 0.05,gamma_4f_6d_au_1_5, 3)
conv_au_1_7=flp.ConvolvingFunc(0, E_vals, dE_4f_6d_au_1_7, gf_4f_6d_au_1_7, 0.05,gamma_4f_6d_au_1_7, 3)
conv_au_1_8=flp.ConvolvingFunc(0, E_vals, dE_4f_6d_au_1_8, gf_4f_6d_au_1_8, 0.05,gamma_4f_6d_au_1_8, 3)
conv_au_1_9=flp.ConvolvingFunc(0, E_vals, dE_4f_6d_au_1_9, gf_4f_6d_au_1_9, 0.05,gamma_4f_6d_au_1_9, 3)
conv_au_2_0=flp.ConvolvingFunc(0, E_vals, dE_4f_6d_au_2_0, gf_4f_6d_au_2_0, 0.05,gamma_4f_6d_au_2_0, 3)
conv_au1_1_5=flp.ConvolvingFunc(0, E_vals, dE_4f_6d_au1_1_5, gf_4f_6d_au1_1_5, 0.05,gamma_4f_6d_au1_1_5, 3)
#%%
Eric_data_400ns=np.loadtxt('C:/Users/David McKeagney/Downloads/Eric_data_400ns.txt',dtype=float).T
Eric_data_450ns=np.loadtxt('C:/Users/David McKeagney/Downloads/Eric_data_450ns.txt',dtype=float).T
Eric_data_500ns=np.loadtxt('C:/Users/David McKeagney/Downloads/Eric_data_500ns.txt',dtype=float).T
Eric_data_350ns=np.loadtxt('C:/Users/David McKeagney/Downloads/Eric_data_350ns.txt',dtype=float).T
Eric_data_300ns=np.loadtxt('C:/Users/David McKeagney/Downloads/Eric_data_300ns.txt',dtype=float).T
Eric_data_250ns=np.loadtxt('C:/Users/David McKeagney/Downloads/Eric_data_250ns.txt',dtype=float).T
Intensity_400ns=Eric_data_400ns[1]
Intensity_450ns=Eric_data_450ns[1]
Intensity_500ns=Eric_data_500ns[1]
Intensity_350ns=Eric_data_350ns[1]
Intensity_300ns=Eric_data_300ns[1]
Intensity_250ns=Eric_data_250ns[1]
Energy=Eric_data_400ns[0]
#%%
plt.plot(E_vals+np.repeat(3.2,32000),0.21*(0.95*conv_au_1_5+0.05*conv_au1_1_5) ,label='T=1.5')
#plt.plot(E_vals+np.repeat(3.2,32000),0.19*conv_au_1_6,label='T=1.6')
#plt.plot(E_vals+np.repeat(3.2,32000),0.19*conv_au_1_7,label='T=1.7')
#plt.plot(E_vals+np.repeat(3.2,32000),0.19*conv_au_1_8,label='T=1.8')
#plt.plot(E_vals+np.repeat(3.2,32000),0.19*conv_au_1_9,label='T=1.9')
#plt.plot(E_vals+np.repeat(3.2,32000),0.19*conv_au_2_0,label='T=2.0')
#plt.plot(E_vals+np.repeat(3.2,32000),conv_au1_1_5)
plt.plot(Energy, Intensity_500ns)
plt.plot(Energy,Intensity_450ns)
plt.plot(Energy,Intensity_400ns)
#plt.vlines(95,0,0.273,color='red')
#plt.vlines(98.5,0,0.235,color='blue')
#plt.vlines(97.3,0,0.24,color='green')
#plt.vlines(92,0,0.266,color='blue')
#plt.vlines(93.8,0,0.25,color='red')
plt.legend()
plt.xlim(75,110)


#%%
##Determining the relative intensity 
plt.plot(Energy,Intensity_500ns)
plt.xlim(75,110)
plt.vlines(95,0,0.273,color='red')
plt.vlines(98.5,0,0.235,color='blue')
plt.vlines(97.3,0,0.24,color='green')
plt.vlines(92,0,0.266,color='blue')
plt.vlines(93.8,0,0.25,color='red')


