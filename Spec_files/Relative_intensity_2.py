# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 10:28:22 2025

@author: David McKeagney
"""

import numpy as np 
import function_library_phd as flp
import matplotlib.pyplot as plt
#%%
spec_file_1_5=[]
with open('C:/Users/David McKeagney/Desktop/au.sub.1.5.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            spec_file_1_5.append(lines.split())
spec_file_1_5=np.array(spec_file_1_5)[1:,:]
spec_file_1_6=[]
with open('C:/Users/David McKeagney/Desktop/au.sub.1.6.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            spec_file_1_6.append(lines.split())
spec_file_1_6=np.array(spec_file_1_6)[1:,:]
spec_file_1_7=[]
with open('C:/Users/David McKeagney/Desktop/au.sub.1.7.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            spec_file_1_7.append(lines.split())
spec_file_1_7=np.array(spec_file_1_7)[1:,:]
spec_file_1_8=[]
with open('C:/Users/David McKeagney/Desktop/au.sub.1.8.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            spec_file_1_8.append(lines.split())
spec_file_1_8=np.array(spec_file_1_8)[1:,:]
spec_file_1_9=[]
with open('C:/Users/David McKeagney/Desktop/au.sub.1.9.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            spec_file_1_9.append(lines.split())
spec_file_1_9=np.array(spec_file_1_9)[1:,:]
spec_file_2_0=[]
with open('C:/Users/David McKeagney/Desktop/au.sub.2.0.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            spec_file_2_0.append(lines.split())
spec_file_2_0=np.array(spec_file_2_0)[1:,:]
spec_file_2_2=[]
with open('C:/Users/David McKeagney/Desktop/au.sub.2.2.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            spec_file_2_2.append(lines.split())
spec_file_2_2=np.array(spec_file_2_2)[1:,:]
spec_file_1_1_8=[]
with open('C:/Users/David McKeagney/Desktop/au1.sub.1.8.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            spec_file_1_1_8.append(lines.split())
spec_file_1_1_8=np.array(spec_file_1_1_8)[1:,:]
spec_file_2_9=[]
with open('C:/Users/David McKeagney/Desktop/au.sub.2.9.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            spec_file_2_9.append(lines.split())
spec_file_2_9=np.array(spec_file_2_9)[1:,:]
Eric_data_500ns=np.loadtxt('C:/Users/David McKeagney/Downloads/Eric_data_500ns.txt',dtype=float).T
Intensity_500ns=Eric_data_500ns[1][np.logical_and(Eric_data_500ns[0]>=75,Eric_data_500ns[0]<=100)]
Energy_500ns=Eric_data_500ns[0][np.logical_and(Eric_data_500ns[0]>=75,Eric_data_500ns[0]<=100)]
Eric_data_450ns=np.loadtxt('C:/Users/David McKeagney/Downloads/Eric_data_450ns.txt',dtype=float).T
Intensity_450ns=Eric_data_450ns[1][np.logical_and(Eric_data_450ns[0]>=75,Eric_data_450ns[0]<=100)]
Eric_data_400ns=np.loadtxt('C:/Users/David McKeagney/Downloads/Eric_data_400ns.txt',dtype=float).T
Intensity_400ns=Eric_data_400ns[1][np.logical_and(Eric_data_400ns[0]>=75,Eric_data_400ns[0]<=100)]
Eric_data_350ns=np.loadtxt('C:/Users/David McKeagney/Desktop/Eric_data_350ns.txt',dtype=float).T
Intensity_350ns=Eric_data_350ns[1][np.logical_and(Eric_data_350ns[0]>=75,Eric_data_350ns[0]<=100)]
Eric_data_300ns=np.loadtxt('C:/Users/David McKeagney/Desktop/Eric_data_300ns.txt',dtype=float).T
Intensity_300ns=Eric_data_300ns[1][np.logical_and(Eric_data_300ns[0]>=75,Eric_data_300ns[0]<=100)]
#%%
spec_file_1_2_0=[]
with open('C:/Users/David McKeagney/Desktop/au1.sub.2.0.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            spec_file_1_2_0.append(lines.split())
spec_file_1_2_0=np.array(spec_file_1_2_0)[1:,:]
spec_file_1_2_2=[]
with open('C:/Users/David McKeagney/Desktop/au1.sub.2.2.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            spec_file_1_2_2.append(lines.split())
spec_file_1_2_2=np.array(spec_file_1_2_2)[1:,:]
spec_file_1_1_9=[]
with open('C:/Users/David McKeagney/Desktop/au1.sub.1.9.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            spec_file_1_1_9.append(lines.split())
spec_file_1_1_9=np.array(spec_file_1_1_9)[1:,:]
spec_file_1_2_9=[]
with open('C:/Users/David McKeagney/Desktop/au1.sub.2.9.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            spec_file_1_2_9.append(lines.split())
spec_file_1_2_9=np.array(spec_file_1_2_9)[1:,:]
spec_file_2_2_9=[]
with open('C:/Users/David McKeagney/Desktop/au2.sub.2.9.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            spec_file_2_2_9.append(lines.split())
spec_file_2_2_9=np.array(spec_file_2_2_9)[1:,:]
#%%
auspec_1_1_5=spec_file_1_5[spec_file_1_5[:,8]=='1']
auspec_5_1_5=spec_file_1_5[spec_file_1_5[:,8]=='5']
spec_file_1_5=np.concatenate((auspec_1_1_5,auspec_5_1_5),axis=0)
#%%
auspec_1_1_6=spec_file_1_6[spec_file_1_6[:,8]=='1']
auspec_5_1_6=spec_file_1_6[spec_file_1_6[:,8]=='5']
spec_file_1_6=np.concatenate((auspec_1_1_6,auspec_5_1_6),axis=0)
#%%
auspec_1_1_7=spec_file_1_7[spec_file_1_7[:,8]=='1']
auspec_5_1_7=spec_file_1_7[spec_file_1_7[:,8]=='5']
spec_file_1_7=np.concatenate((auspec_1_1_7,auspec_5_1_7),axis=0)
#%%
auspec_1_1_8=spec_file_1_8[spec_file_1_8[:,8]=='1']
auspec_5_1_8=spec_file_1_8[spec_file_1_8[:,8]=='5']
spec_file_1_8=np.concatenate((auspec_1_1_8,auspec_5_1_8),axis=0)
#%%
auspec_1_1_9=spec_file_1_9[spec_file_1_9[:,8]=='1']
auspec_5_1_9=spec_file_1_9[spec_file_1_9[:,8]=='5']
spec_file_1_9=np.concatenate((auspec_1_1_9,auspec_5_1_9),axis=0)
#%%
auspec_1_2_0=spec_file_2_0[spec_file_2_0[:,8]=='1']
auspec_5_2_0=spec_file_2_0[spec_file_2_0[:,8]=='5']
spec_file_2_0=np.concatenate((auspec_1_2_0,auspec_5_2_0),axis=0)
#%%
auspec_1_2_2=spec_file_2_2[spec_file_2_2[:,8]=='1']
auspec_5_2_2=spec_file_2_2[spec_file_2_2[:,8]=='5']
spec_file_2_2=np.concatenate((auspec_1_2_2,auspec_5_2_2),axis=0)
#%%
auspec_1_2_9=spec_file_2_9[spec_file_2_9[:,8]=='1']
auspec_5_2_9=spec_file_2_9[spec_file_2_9[:,8]=='5']
spec_file_2_9=np.concatenate((auspec_1_2_9,auspec_5_2_9),axis=0)
#%%
auspec_1_1_1_8=spec_file_1_1_8[spec_file_1_1_8[:,8]=='1']
auspec_5_1_1_8=spec_file_1_1_8[spec_file_1_1_8[:,8]=='5']
auspec_9_1_1_8=spec_file_1_1_8[spec_file_1_1_8[:,8]=='9']
auspec_1_1_1_8=np.concatenate((auspec_1_1_1_8,auspec_5_1_1_8),axis=0)
spec_file_1_1_8=np.concatenate((auspec_1_1_1_8,auspec_9_1_1_8),axis=0)
#%%
auspec_1_1_1_9=spec_file_1_1_9[spec_file_1_1_9[:,8]=='1']
auspec_5_1_1_9=spec_file_1_1_9[spec_file_1_1_9[:,8]=='5']
auspec_9_1_1_9=spec_file_1_1_9[spec_file_1_1_9[:,8]=='9']
auspec_1_1_1_9=np.concatenate((auspec_1_1_1_9,auspec_5_1_1_9),axis=0)
spec_file_1_1_9=np.concatenate((auspec_1_1_1_9,auspec_9_1_1_9),axis=0)
#%%
auspec_1_1_2_0=spec_file_1_2_0[spec_file_1_2_0[:,8]=='1']
auspec_5_1_2_0=spec_file_1_2_0[spec_file_1_2_0[:,8]=='5']
auspec_9_1_2_0=spec_file_1_2_0[spec_file_1_2_0[:,8]=='9']
auspec_1_1_2_0=np.concatenate((auspec_1_1_2_0,auspec_5_1_2_0),axis=0)
spec_file_1_2_0=np.concatenate((auspec_1_1_2_0,auspec_9_1_2_0),axis=0) 
#%%
auspec_1_1_2_2=spec_file_1_2_2[spec_file_1_2_2[:,8]=='1']
auspec_5_1_2_2=spec_file_1_2_2[spec_file_1_2_2[:,8]=='5']
auspec_9_1_2_2=spec_file_1_2_2[spec_file_1_2_2[:,8]=='9']
auspec_1_1_2_2=np.concatenate((auspec_1_1_2_2,auspec_5_1_2_2),axis=0)
spec_file_1_2_2=np.concatenate((auspec_1_1_2_2,auspec_9_1_2_2),axis=0)  
#%%
auspec_1_1_2_9=spec_file_1_2_9[spec_file_1_2_9[:,8]=='1']
auspec_5_1_2_9=spec_file_1_2_9[spec_file_1_2_9[:,8]=='5']
auspec_9_1_2_9=spec_file_1_2_9[spec_file_1_2_9[:,8]=='9']
auspec_1_1_2_9=np.concatenate((auspec_1_1_2_9,auspec_5_1_2_9),axis=0)
spec_file_1_2_9=np.concatenate((auspec_1_1_2_9,auspec_9_1_2_9),axis=0)
#%%
auspec_1_2_2_9=spec_file_2_2_9[spec_file_2_2_9[:,8]=='2']
auspec_5_2_2_9=spec_file_2_2_9[spec_file_2_2_9[:,8]=='6']
auspec_9_2_2_9=spec_file_2_2_9[spec_file_2_2_9[:,8]=='10']
auspec_1_2_2_9=np.concatenate((auspec_1_2_2_9,auspec_5_2_2_9),axis=0)
spec_file_2_2_9=np.concatenate((auspec_1_2_2_9,auspec_9_2_2_9),axis=0)       
#%%
upper_levels=list(set(spec_file_1_5[:,6].astype(float)))
gf_vals_1_5=[]
decay_vals_1_5=[]
for ul in upper_levels:
    temp_spec=spec_file_1_5[spec_file_1_5[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_1_5.append(sum(gf_temp))
    decay_vals_1_5.append(sum(decay_temp))
#%%
upper_levels=list(set(spec_file_1_6[:,6].astype(float)))
gf_vals_1_6=[]
decay_vals_1_6=[]
for ul in upper_levels:
    temp_spec=spec_file_1_6[spec_file_1_6[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_1_6.append(sum(gf_temp))
    decay_vals_1_6.append(sum(decay_temp))
#%%
upper_levels=list(set(spec_file_1_7[:,6].astype(float)))
gf_vals_1_7=[]
decay_vals_1_7=[]
for ul in upper_levels:
    temp_spec=spec_file_1_7[spec_file_1_7[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_1_7.append(sum(gf_temp))
    decay_vals_1_7.append(sum(decay_temp))
#%%
upper_levels=list(set(spec_file_1_8[:,6].astype(float)))
gf_vals_1_8=[]
decay_vals_1_8=[]
for ul in upper_levels:
    temp_spec=spec_file_1_8[spec_file_1_8[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_1_8.append(sum(gf_temp))
    decay_vals_1_8.append(sum(decay_temp))
#%%
upper_levels=list(set(spec_file_1_9[:,6].astype(float)))
gf_vals_1_9=[]
decay_vals_1_9=[]
for ul in upper_levels:
    temp_spec=spec_file_1_9[spec_file_1_9[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_1_9.append(sum(gf_temp))
    decay_vals_1_9.append(sum(decay_temp))
#%%
upper_levels=list(set(spec_file_2_0[:,6].astype(float)))
gf_vals_2_0=[]
decay_vals_2_0=[]
for ul in upper_levels:
    temp_spec=spec_file_2_0[spec_file_2_0[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_2_0.append(sum(gf_temp))
    decay_vals_2_0.append(sum(decay_temp))
#%%
upper_levels=list(set(spec_file_2_2[:,6].astype(float)))
gf_vals_2_2=[]
decay_vals_2_2=[]
for ul in upper_levels:
    temp_spec=spec_file_2_2[spec_file_2_2[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_2_2.append(sum(gf_temp))
    decay_vals_2_2.append(sum(decay_temp))
#%%
upper_levels=list(set(spec_file_2_9[:,6].astype(float)))
gf_vals_2_9=[]
decay_vals_2_9=[]
for ul in upper_levels:
    temp_spec=spec_file_2_9[spec_file_2_9[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_2_9.append(sum(gf_temp))
    decay_vals_2_9.append(sum(decay_temp))
#%%
upper_levels_1=list(set(spec_file_1_1_8[:,6].astype(float)))
gf_vals_1_1_8=[]
decay_vals_1_1_8=[]
for ul in upper_levels_1:
    temp_spec=spec_file_1_1_8[spec_file_1_1_8[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_1_1_8.append(sum(gf_temp))
    decay_vals_1_1_8.append(sum(decay_temp))
#%%
upper_levels_1=list(set(spec_file_1_1_9[:,6].astype(float)))
gf_vals_1_1_9=[]
decay_vals_1_1_9=[]
for ul in upper_levels_1:
    temp_spec=spec_file_1_1_9[spec_file_1_1_9[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_1_1_9.append(sum(gf_temp))
    decay_vals_1_1_9.append(sum(decay_temp))
#%%
upper_levels_1=list(set(spec_file_1_2_0[:,6].astype(float)))
gf_vals_1_2_0=[]
decay_vals_1_2_0=[]
for ul in upper_levels_1:
    temp_spec=spec_file_1_2_0[spec_file_1_2_0[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_1_2_0.append(sum(gf_temp))
    decay_vals_1_2_0.append(sum(decay_temp))
#%%
upper_levels_1=list(set(spec_file_1_2_2[:,6].astype(float)))
gf_vals_1_2_2=[]
decay_vals_1_2_2=[]
for ul in upper_levels_1:
    temp_spec=spec_file_1_2_2[spec_file_1_2_2[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_1_2_2.append(sum(gf_temp))
    decay_vals_1_2_2.append(sum(decay_temp))
#%%
upper_levels_1=list(set(spec_file_1_2_9[:,6].astype(float)))
gf_vals_1_2_9=[]
decay_vals_1_2_9=[]
for ul in upper_levels_1:
    temp_spec=spec_file_1_2_9[spec_file_1_2_9[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_1_2_9.append(sum(gf_temp))
    decay_vals_1_2_9.append(sum(decay_temp))
#%%
upper_levels_2=list(set(spec_file_2_2_9[:,6].astype(float)))
gf_vals_2_2_9=[]
decay_vals_2_2_9=[]
for ul in upper_levels_2:
    temp_spec=spec_file_2_2_9[spec_file_2_2_9[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_2_2_9.append(sum(gf_temp))
    decay_vals_2_2_9.append(sum(decay_temp))
#%%
Energy=E_vals=np.arange(88,120,0.001)
conv_vals_1_5=flp.ConvolvingFunc(1, Energy, np.array(upper_levels), np.array(gf_vals_1_5), 0.05, np.array(decay_vals_1_5), 3)
conv_vals_1_6=flp.ConvolvingFunc(1, Energy, np.array(upper_levels), np.array(gf_vals_1_6), 0.05, np.array(decay_vals_1_6), 3)
conv_vals_1_7=flp.ConvolvingFunc(1, Energy, np.array(upper_levels), np.array(gf_vals_1_7), 0.05, np.array(decay_vals_1_7), 3)
conv_vals_1_8=flp.ConvolvingFunc(1, Energy, np.array(upper_levels), np.array(gf_vals_1_8), 0.05, np.array(decay_vals_1_8), 3)
conv_vals_1_9=flp.ConvolvingFunc(1, Energy, np.array(upper_levels), np.array(gf_vals_1_9), 0.05, np.array(decay_vals_1_9), 3)
conv_vals_2_0=flp.ConvolvingFunc(1, Energy, np.array(upper_levels), np.array(gf_vals_2_0), 0.05, np.array(decay_vals_2_0), 3)
conv_vals_2_2=flp.ConvolvingFunc(1, Energy, np.array(upper_levels), np.array(gf_vals_2_2), 0.05, np.array(decay_vals_2_2), 3)
conv_vals_2_9=flp.ConvolvingFunc(1, Energy, np.array(upper_levels), np.array(gf_vals_2_9), 0.05, np.array(decay_vals_2_9), 3)
conv_vals_1_1_8=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_1), np.array(gf_vals_1_1_8), 0.05, np.array(decay_vals_1_1_8), 3)
conv_vals_1_2_0=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_1), np.array(gf_vals_1_2_0), 0.05, np.array(decay_vals_1_2_0), 3)
conv_vals_1_2_2=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_1), np.array(gf_vals_1_2_2), 0.05, np.array(decay_vals_1_2_2), 3)
conv_vals_1_1_9=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_1), np.array(gf_vals_1_1_9), 0.05, np.array(decay_vals_1_1_9), 3)
conv_vals_1_2_9=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_1), np.array(gf_vals_1_2_9), 0.05, np.array(decay_vals_1_2_9), 3)
conv_vals_2_2_9=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_2), np.array(gf_vals_2_2_9), 0.05, np.array(decay_vals_2_2_9), 3)
#%% SINCE the relative intsities when AU2+ is included makes the features in 110eV area more important we can conclued that there is no AU2+ since that isn't seen in the spectra so the temperature is less that 2.6 eV
#plt.plot(Energy+np.repeat(0.3,len(Energy)),0.4*(0.815*conv_vals_1_8+0.185*conv_vals_1_1_8),label='T=1.8')
#plt.plot(Energy+np.repeat(0.3,len(Energy)),0.6*(0.772*conv_vals_1_9+0.228*conv_vals_1_1_9),label='T=1.9')
#plt.plot(Energy+np.repeat(0.3,len(Energy)),0.5*(0.672*conv_vals_2_0+0.328*conv_vals_1_2_0),label='T=2.0')
#plt.plot(Energy+np.repeat(0.3,len(Energy)),0.4*(0.573*conv_vals_2_2+0.427*conv_vals_1_2_2),label='T=2.2')
#plt.plot(Energy+np.repeat(0.3,len(Energy)),0.2*(0.0*conv_vals_2_9+0.54*conv_vals_1_2_9+0.0*conv_vals_2_2_9))
#plt.plot(Energy+np.repeat(0.3,len(Energy)),0.4*conv_vals_1_6,label='T=1.6')
#plt.plot(Energy+np.repeat(0.3,len(Energy)),0.4*conv_vals_1_7,label='T=1.7')
#plt.plot(Energy+np.repeat(0.3,len(Energy)),0.4*conv_vals_1_8,label='T=1.8')
#plt.plot(Energy+np.repeat(0.3,len(Energy)),0.4*conv_vals_1_9,label='T=1.9')
#plt.plot(Energy+np.repeat(0.3,len(Energy)),0.4*conv_vals_2_0,label='T=2.0')
plt.plot(Energy_500ns,Intensity_500ns,label='500ns')
plt.plot(Energy_500ns,Intensity_450ns,label='450ns')
plt.plot(Energy_500ns,Intensity_400ns,label='400ns')
plt.plot(Energy_500ns,Intensity_350ns,label='350ns')
plt.plot(Energy_500ns,Intensity_300ns,label='300ns')
plt.vlines(94.9,0,3,colors='red',linestyles='--')
plt.vlines(98.5,0,3,colors='red',linestyles='--')
plt.vlines(97,0,3,colors='red',linestyles='--')
plt.vlines(91.85,0,3,colors='red',linestyles='--')
plt.vlines(99.7,0,3,colors='blue',linestyles='--')
plt.vlines(96.4,0,3,colors='red',linestyles='--')
plt.grid(True)
plt.xlabel('Energy [eV]')
plt.ylabel('Absorbance [Arb.]')
#plt.xlim(80,90)
plt.ylim(0,1.65)
plt.legend()
#%%
plt.plot(Energy,0.05*conv_vals_2_2_9)
plt.plot(Energy_500ns,Intensity_500ns)
#%%
Energy=E_vals=np.arange(88,120,0.001)
conv_vals_2_9=flp.ConvolvingFunc(1, Energy, np.array(upper_levels), np.array(gf_vals_2_9), 0.05, np.array(decay_vals_2_9), 3)
conv_vals_1_2_9=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_1), np.array(gf_vals_1_2_9), 0.05, np.array(decay_vals_1_2_9), 3)
conv_vals_2_2_9=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_2), np.array(gf_vals_2_2_9), 0.05, np.array(decay_vals_2_2_9), 3)
#%%
plt.plot(Energy,conv_vals_2_9,label='4f-6d Au I')
plt.plot(Energy,conv_vals_1_2_9,label='4f-6d Au II')
plt.plot(Energy,conv_vals_2_2_9,label='4f-6d Au III')
plt.legend()
plt.xlabel('Energy [eV]')
plt.ylabel('Intensity [Arb.]')
#%%
au_spec_I=[]
with open('C:/Users/David McKeagney/Desktop/au.sub.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            au_spec_I.append(lines.split())
au_spec_II=[]
with open('C:/Users/David McKeagney/Desktop/au1.sub.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            au_spec_II.append(lines.split())
au_spec_III=[]
with open('C:/Users/David McKeagney/Desktop/au2.sub.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            au_spec_III.append(lines.split())
au_spec_I=np.array(au_spec_I)[1:,:]
au_spec_II=np.array(au_spec_II)[1:,:]
au_spec_III=np.array(au_spec_III)[1:,:]
#%%
au_spec_1_I=au_spec_I[au_spec_I[:,8]=='1']
au_spec_5_I=au_spec_I[au_spec_I[:,8]=='5']
spec_file_1_I=np.concatenate((au_spec_1_I,au_spec_5_I),axis=0)
#%%
au_spec_1_II=au_spec_II[au_spec_II[:,8]=='1']
au_spec_5_II=au_spec_II[au_spec_II[:,8]=='5']
au_spec_9_II=au_spec_II[au_spec_II[:,8]=='9']
spec_file_1_II=np.concatenate((au_spec_1_II,au_spec_5_II),axis=0)
spec_file_9_II=np.concatenate((spec_file_1_I,au_spec_9_II),axis=0)
#%%
au_spec_2_III=au_spec_III[au_spec_III[:,8]=='2']
au_spec_6_III=au_spec_III[au_spec_III[:,8]=='6']
au_spec_10_III=au_spec_III[au_spec_III[:,8]=='10']
spec_file_2_III=np.concatenate((au_spec_2_III,au_spec_6_III),axis=0)
spec_file_10_III=np.concatenate((spec_file_2_III,au_spec_10_III),axis=0)
#%%
upper_levels_I=list(set(spec_file_1_I[:,6].astype(float)))
gf_vals_I=[]
decay_vals_I=[]
for ul in upper_levels:
    temp_spec=spec_file_1_I[spec_file_1_I[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_I.append(sum(gf_temp))
    decay_vals_I.append(sum(decay_temp))
#%%
upper_levels_1=list(set(spec_file_9_II[:,6].astype(float)))
gf_vals_II=[]
decay_vals_II=[]
for ul in upper_levels_1:
    temp_spec=spec_file_9_II[spec_file_9_II[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_II.append(sum(gf_temp))
    decay_vals_II.append(sum(decay_temp))
#%%
upper_levels_2=list(set(spec_file_10_III[:,6].astype(float)))
gf_vals_III=[]
decay_vals_III=[]
for ul in upper_levels_2:
    temp_spec=spec_file_10_III[spec_file_10_III[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_III.append(sum(gf_temp))
    decay_vals_III.append(sum(decay_temp))
#%%
Energy=E_vals=np.arange(70,130,0.001)
conv_vals_I=flp.ConvolvingFunc(1, Energy, np.array(upper_levels), np.array(gf_vals_I), 0.05, np.array(decay_vals_I), 3)
conv_vals_II=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_1), np.array(gf_vals_II), 0.05, np.array(decay_vals_II), 3)
conv_vals_III=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_2), np.array(gf_vals_III), 0.05, np.array(decay_vals_III), 3)
#%%
plt.plot(Energy,conv_vals_I,label='4f-6d Au I')
plt.plot(Energy,conv_vals_II,label='4f-6d Au II')
plt.plot(Energy,conv_vals_III,label='4f-6d Au III')
plt.legend()
plt.xlabel('Energy [eV]')
plt.ylabel('Intensity [Arb.]')
plt.grid(True)
