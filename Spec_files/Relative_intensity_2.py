# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 10:28:22 2025

@author: David McKeagney
"""

import numpy as np 
import function_library_phd as flp
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})
plt.rcParams["figure.figsize"] = (15,10)
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
Intensity_500ns=Eric_data_500ns[1][np.logical_and(Eric_data_500ns[0]>=78,Eric_data_500ns[0]<=86)]
Energy_500ns=Eric_data_500ns[0][np.logical_and(Eric_data_500ns[0]>=78,Eric_data_500ns[0]<=86)]
Eric_data_450ns=np.loadtxt('C:/Users/David McKeagney/Downloads/Eric_data_450ns.txt',dtype=float).T
Intensity_450ns=Eric_data_450ns[1][np.logical_and(Eric_data_450ns[0]>=78,Eric_data_450ns[0]<=86)]
Eric_data_400ns=np.loadtxt('C:/Users/David McKeagney/Downloads/Eric_data_400ns.txt',dtype=float).T
Intensity_400ns=Eric_data_400ns[1][np.logical_and(Eric_data_400ns[0]>=78,Eric_data_400ns[0]<=86)]
Eric_data_350ns=np.loadtxt('C:/Users/David McKeagney/Desktop/Eric_data_350ns.txt',dtype=float).T
Intensity_350ns=Eric_data_350ns[1][np.logical_and(Eric_data_350ns[0]>=78,Eric_data_350ns[0]<=86)]
Eric_data_300ns=np.loadtxt('C:/Users/David McKeagney/Desktop/Eric_data_300ns.txt',dtype=float).T
Intensity_300ns=Eric_data_300ns[1][np.logical_and(Eric_data_300ns[0]>=78,Eric_data_300ns[0]<=86)]
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
auspec_file_1_1_8=np.concatenate((auspec_1_1_1_8,auspec_9_1_1_8),axis=0)
#%% 4f-5d auII
auspec_4_1_1_8_4f=spec_file_1_1_8[spec_file_1_1_8[:,8]=='4']
auspec_8_1_1_8_4f=spec_file_1_1_8[spec_file_1_1_8[:,8]=='8']
auspec_file_1_1_8_4f=np.concatenate((auspec_4_1_1_8_4f,auspec_8_1_1_8_4f),axis=0)
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
upper_levels_1=list(set(auspec_file_1_1_8[:,6].astype(float)))
gf_vals_1_1_8=[]
decay_vals_1_1_8=[]
for ul in upper_levels_1:
    temp_spec=auspec_file_1_1_8[auspec_file_1_1_8[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_1_1_8.append(sum(gf_temp))
    decay_vals_1_1_8.append(sum(decay_temp))
#%%
upper_levels_1_4f=list(set(auspec_file_1_1_8_4f[:,6].astype(float)))
gf_vals_1_1_8_4f=[]
decay_vals_1_1_8_4f=[]
for ul in upper_levels_1:
    temp_spec=auspec_file_1_1_8_4f[auspec_file_1_1_8_4f[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_1_1_8_4f.append(sum(gf_temp))
    decay_vals_1_1_8_4f.append(sum(decay_temp))
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
#%%
conv_vals_1_1_8_4f=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_1_4f), np.array(gf_vals_1_1_8_4f), 0.05, np.array(decay_vals_1_1_8_4f), 3)
#%% SINCE the relative intsities when AU2+ is included makes the features in 110eV area more important we can conclued that there is no AU2+ since that isn't seen in the spectra so the temperature is less that 2.6 eV
plt.plot(Energy+np.repeat(1.7,len(Energy)),0.31*(conv_vals_1_1_8+conv_vals_1_1_8_4f)+0.69*conv_vals_1_8,label='T=1.8')
#plt.plot(Energy+np.repeat(0.3,len(Energy)),0.6*(0.772*conv_vals_1_9+0.228*conv_vals_1_1_9),label='T=1.9')
#plt.plot(Energy+np.repeat(0.3,len(Energy)),0.5*(0.672*conv_vals_2_0+0.328*conv_vals_1_2_0),label='T=2.0')
#plt.plot(Energy+np.repeat(0.3,len(Energy)),0.4*(0.573*conv_vals_2_2+0.427*conv_vals_1_2_2),label='T=2.2')
#plt.plot(Energy-np.repeat(4.1,len(Energy)),0.27*conv_vals_2_9+0.90*conv_vals_1_2_9+0.03*conv_vals_2_2_9)
#plt.plot(Energy+np.repeat(0.3,len(Energy)),0.4*conv_vals_1_6,label='T=1.6')
#plt.plot(Energy+np.repeat(0.3,len(Energy)),0.4*conv_vals_1_7,label='T=1.7')
#plt.plot(Energy+np.repeat(0.3,len(Energy)),0.4*conv_vals_1_8,label='T=1.8')
#plt.plot(Energy+np.repeat(0.3,len(Energy)),0.4*conv_vals_1_9,label='T=1.9')
#plt.plot(Energy+np.repeat(0.3,len(Energy)),0.4*conv_vals_2_0,label='T=2.0')
plt.plot(Energy_500ns,2.5*Intensity_500ns,label='500ns')
#plt.plot(Energy-np.repeat(4,len(Energy)),conv_vals_1_1_9,label='T=2.2eV')
#plt.plot(Energy-np.repeat(4.2,len(Energy)),conv_vals_1_2_9,label='T=2.9eV')
#plt.plot(Energy_500ns,Intensity_450ns,label='450ns')
#plt.plot(Energy_500ns,Intensity_400ns,label='400ns')
#plt.plot(Energy_500ns,Intensity_350ns,label='350ns')
#plt.plot(Energy_500ns,Intensity_300ns,label='300ns')
#plt.vlines(94.9,0,3,colors='red',linestyles='--')
#plt.vlines(98.5,0,3,colors='red',linestyles='--')
#plt.vlines(97,0,3,colors='red',linestyles='--')
#plt.vlines(91.85,0,3,colors='red',linestyles='--')
#plt.vlines(99.7,0,3,colors='blue',linestyles='--')
#plt.vlines(96.4,0,3,colors='red',linestyles='--')
plt.grid(True)
plt.xlabel('Energy [eV]')
plt.ylabel('Absorbance [Arb.]')
#plt.xlim(79,100)
#plt.ylim(0,1.65)
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
au_spec_II_more_auto=[]
with open('C:/Users/David McKeagney/Desktop/au1_more_auto.sub.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            au_spec_II_more_auto.append(lines.split())
au_spec_III=[]
with open('C:/Users/David McKeagney/Desktop/au2.sub.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            au_spec_III.append(lines.split())
au_spec_I_open_shells=[]
with open('C:/Users/David McKeagney/Downloads/auI_open_shells.sub.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            au_spec_I_open_shells.append(lines.split())
au_spec_II_open_shells=[]
with open('C:/Users/David McKeagney/Downloads/auII_open_shells.sub.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            au_spec_II_open_shells.append(lines.split())

au_spec_I_open_shells=np.array(au_spec_I_open_shells)[1:,:]
au_spec_II_open_shells=np.array(au_spec_II_open_shells)[1:,:]
au_spec_I=np.array(au_spec_I)[1:,:]
au_spec_II=np.array(au_spec_II)[1:,:]
au_spec_more_auto_II=np.array(au_spec_II_more_auto)[1:,:]
au_spec_III=np.array(au_spec_III)[1:,:]
#%%
au_spec_1_I=au_spec_I[np.logical_and(au_spec_I[:,3]=='1',au_spec_I[:,8]=='1')]
au_spec_5_I=au_spec_I[np.logical_and(au_spec_I[:,3]=='2',au_spec_I[:,8]=='5')]
spec_file_1_I=np.concatenate((au_spec_1_I,au_spec_5_I),axis=0)
#%%
au_spec_1_I_open_shells=au_spec_I_open_shells[np.logical_and(au_spec_I_open_shells[:,3]=='1',au_spec_I_open_shells[:,8]=='1')]
au_spec_5_I_open_shells=au_spec_I_open_shells[np.logical_and(au_spec_I_open_shells[:,3]=='2',au_spec_I_open_shells[:,8]=='5')]
spec_file_1_I_open_shells=np.concatenate((au_spec_1_I_open_shells,au_spec_5_I_open_shells),axis=0)
#%%
au_spec_1_II=au_spec_II[np.logical_and(au_spec_II[:,3]=='1',au_spec_II[:,8]=='1')]
au_spec_5_II=au_spec_II[np.logical_and(au_spec_II[:,3]=='2',au_spec_II[:,8]=='5')]
au_spec_9_II=au_spec_II[np.logical_and(au_spec_II[:,3]=='3',au_spec_II[:,8]=='9')]
spec_file_1_II=np.concatenate((au_spec_1_II,au_spec_5_II),axis=0)
spec_file_9_II=np.concatenate((spec_file_1_II,au_spec_9_II),axis=0)
#%%
au_spec_1_II_open_shells=au_spec_II_open_shells[np.logical_and(au_spec_II_open_shells[:,3]=='1',au_spec_II_open_shells[:,8]=='1')]
au_spec_5_II_open_shells=au_spec_II_open_shells[np.logical_and(au_spec_II_open_shells[:,3]=='2',au_spec_II_open_shells[:,8]=='5')]
au_spec_9_II_open_shells=au_spec_II_open_shells[np.logical_and(au_spec_II_open_shells[:,3]=='3',au_spec_II_open_shells[:,8]=='9')]
spec_file_1_II_open_shells=np.concatenate((au_spec_1_II_open_shells,au_spec_5_II_open_shells),axis=0)
spec_file_9_II_open_shells=np.concatenate((spec_file_1_II_open_shells,au_spec_9_II_open_shells),axis=0)
#%%
au_spec_1_II_more_auto=au_spec_more_auto_II[au_spec_more_auto_II[:,8]=='1']
au_spec_5_II_more_auto=au_spec_more_auto_II[au_spec_more_auto_II[:,8]=='5']
au_spec_9_II_more_auto=au_spec_more_auto_II[au_spec_more_auto_II[:,8]=='9']
spec_file_1_II_more_auto=np.concatenate((au_spec_1_II_more_auto,au_spec_5_II_more_auto),axis=0)
spec_file_9_II_more_auto=np.concatenate((spec_file_1_II_more_auto,au_spec_9_II_more_auto),axis=0)
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
for ul in upper_levels_I:
    temp_spec=spec_file_1_I[spec_file_1_I[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_I.append(sum(gf_temp))
    decay_vals_I.append(sum(decay_temp))
#%%
upper_levels_I_open_shells=list(set(spec_file_1_I_open_shells[:,6].astype(float)))
gf_vals_I_open_shells=[]
decay_vals_I_open_shells=[]
for ul in upper_levels_I_open_shells:
    temp_spec=spec_file_1_I_open_shells[spec_file_1_I_open_shells[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_I_open_shells.append(sum(gf_temp))
    decay_vals_I_open_shells.append(sum(decay_temp))
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
upper_levels_1_open_shells=list(set(spec_file_9_II_open_shells[:,6].astype(float)))
gf_vals_II_open_shells=[]
decay_vals_II_open_shells=[]
for ul in upper_levels_1_open_shells:
    temp_spec=spec_file_9_II_open_shells[spec_file_9_II_open_shells[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_II_open_shells.append(sum(gf_temp))
    decay_vals_II_open_shells.append(sum(decay_temp))
#%%
upper_levels_1_more_auto=list(set(spec_file_9_II_more_auto[:,6].astype(float)))
gf_vals_II_more_auto=[]
decay_vals_II_more_auto=[]
for ul in upper_levels_1_more_auto:
    temp_spec_more_auto=spec_file_9_II_more_auto[spec_file_9_II_more_auto[:,6].astype(float)==ul]
    gf_temp_more_auto=np.exp(temp_spec_more_auto[:,15].astype(float))
    decay_temp_more_auto=temp_spec_more_auto[:,16].astype(float)*1e-3
    gf_vals_II_more_auto.append(sum(gf_temp_more_auto))
    decay_vals_II_more_auto.append(sum(decay_temp_more_auto))
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
Energy=E_vals=np.arange(80,100,0.001)
conv_vals_I=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_I), np.array(gf_vals_I), 0.05, np.array(decay_vals_I), 3)
conv_vals_II=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_1), np.array(gf_vals_II), 0.05, np.array(decay_vals_II), 3)
conv_vals_I_open_shells=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_I_open_shells), np.array(gf_vals_I_open_shells), 0.05, np.array(decay_vals_I_open_shells), 3)
conv_vals_II_open_shells=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_1_open_shells), np.array(gf_vals_II_open_shells), 0.05, np.array(decay_vals_II_open_shells), 3)
conv_vals_II_more_auto=flp.ConvolvingFunc(1, Energy-np.repeat(0.5,len(Energy)), np.array(upper_levels_1_more_auto), np.array(gf_vals_II_more_auto), 0.05, np.array(decay_vals_II_more_auto), 3)
#conv_vals_III=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_2), np.array(gf_vals_III), 0.05, np.array(decay_vals_III), 3)
#%%
plt.plot(Energy_500ns,5*Intensity_500ns)
plt.plot(Energy+np.repeat(1.5,len(Energy)),conv_vals_I,label='4f-6d Au I')
#plt.plot(Energy-np.repeat(0.5,len(Energy)),conv_vals_II,label='4f-6d Au II')
#plt.plot(Energy-np.repeat(0.5,len(Energy)),conv_vals_II_more_auto,label='4f-6d Au II more auto')
#plt.plot(Energy,conv_vals_III,label='4f-6d Au III')
plt.xlim(75,120)
plt.legend()
plt.xlabel('Energy [eV]')
plt.ylabel('Intensity [Arb.]')
plt.grid(True)
#%%
scale_factor=(1/5)*(np.sum(Intensity_300ns)+np.sum(Intensity_350ns)+np.sum(Intensity_400ns)+np.sum(Intensity_450ns)+np.sum(Intensity_500ns))
#%%
Norm_intensitiy_300ns=1/np.sum(Intensity_300ns)*Intensity_300ns*scale_factor
Norm_intensitiy_350ns=1/np.sum(Intensity_350ns)*Intensity_350ns*scale_factor
Norm_intensitiy_400ns=1/np.sum(Intensity_400ns)*Intensity_400ns*scale_factor
Norm_intensitiy_450ns=1/np.sum(Intensity_450ns)*Intensity_450ns*scale_factor
Norm_intensitiy_500ns=1/np.sum(Intensity_500ns)*Intensity_500ns*scale_factor
#%%
plt.plot(Energy_500ns,Norm_intensitiy_300ns,label='300ns')
plt.plot(Energy_500ns,Norm_intensitiy_350ns,label='350ns')
plt.plot(Energy_500ns,Norm_intensitiy_400ns,label='400ns')
plt.plot(Energy_500ns,Norm_intensitiy_450ns,label='450ns')
plt.plot(Energy_500ns,Norm_intensitiy_500ns,label='500ns')
plt.xlabel('Energy [eV]')
plt.ylabel('Absorbance (scaled)')
plt.legend()
plt.ylim(0.46,0.94)
plt.xlim(78,86)
#plt.grid(True)
#%% 4f-5d AuII
au_spec_4_II=au_spec_II[np.logical_and(au_spec_II[:,3]=='2',au_spec_II[:,8]=='4')]
au_spec_8_II=au_spec_II[np.logical_and(au_spec_II[:,3]=='3',au_spec_II[:,8]=='8')]
spec_file_4_II=np.concatenate((au_spec_4_II,au_spec_8_II),axis=0)
#%% 4f-5d AuII more auto
au_spec_4_II_more_auto=au_spec_more_auto_II[np.logical_and(au_spec_more_auto_II[:,3]=='2',au_spec_more_auto_II[:,8]=='4')]
au_spec_8_II_more_auto=au_spec_more_auto_II[np.logical_and(au_spec_more_auto_II[:,3]=='3',au_spec_more_auto_II[:,8]=='8')]
spec_file_4_II_more_auto=np.concatenate((au_spec_4_II_more_auto,au_spec_8_II_more_auto),axis=0)
#%% 4f-5d AuII open shells
au_spec_4_II_open_shells=au_spec_II_open_shells[np.logical_and(au_spec_II_open_shells[:,3]=='2',au_spec_II_open_shells[:,8]=='4')]
au_spec_8_II_open_shells=au_spec_II_open_shells[np.logical_and(au_spec_II_open_shells[:,3]=='3',au_spec_II_open_shells[:,8]=='8')]
spec_file_4_II_open_shells=np.concatenate((au_spec_4_II_open_shells,au_spec_8_II_open_shells),axis=0)
#%% 4f-5d AuI
au_spec_4_I=au_spec_I[np.logical_and(au_spec_I[:,3]=='2',au_spec_I[:,8]=='4')]
#%%
upper_levels_I_4f=list(set(au_spec_4_I[:,6].astype(float)))
gf_vals_I_4f=[]
decay_vals_I_4f=[]
for ul in upper_levels_I_4f:
    temp_spec=au_spec_4_I[au_spec_4_I[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_I_4f.append(sum(gf_temp))
    decay_vals_I_4f.append(sum(decay_temp))
#%%
upper_levels_1_4f=list(set(spec_file_4_II[:,6].astype(float)))
gf_vals_II_4f=[]
decay_vals_II_4f=[]
for ul in upper_levels_1_4f:
    temp_spec=spec_file_4_II[spec_file_4_II[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_II_4f.append(sum(gf_temp))
    decay_vals_II_4f.append(sum(decay_temp))
#%%
upper_levels_1_4f_more_auto=list(set(spec_file_4_II_more_auto[:,6].astype(float)))
gf_vals_II_4f_more_auto=[]
decay_vals_II_4f_more_auto=[]
for ul in upper_levels_1_4f_more_auto:
    temp_spec=spec_file_4_II_more_auto[spec_file_4_II_more_auto[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_II_4f_more_auto.append(sum(gf_temp))
    decay_vals_II_4f_more_auto.append(sum(decay_temp))
#%%
upper_levels_1_4f_open_shells=list(set(spec_file_4_II_open_shells[:,6].astype(float)))
gf_vals_II_4f_open_shells=[]
decay_vals_II_4f_open_shells=[]
for ul in upper_levels_1_4f_open_shells:
    temp_spec=spec_file_4_II_open_shells[spec_file_4_II_open_shells[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_II_4f_open_shells.append(sum(gf_temp))
    decay_vals_II_4f_open_shells.append(sum(decay_temp))
#%%
Individual_lines_AuII_4f_dE_3_8=au_spec_8_II[:,11].astype(float)
Individual_lines_AuII_4f_gf_3_8=np.exp(au_spec_8_II[:,15].astype(float))
#%%
Energy=E_vals=np.arange(80,100,0.001)
#conv_vals_I_4f=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_I_4f), np.array(gf_vals_I_4f), 0.05, np.array(decay_vals_I_4f), 3)
conv_vals_II_4f=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_1_4f), np.array(gf_vals_II_4f), 0.05, np.array(decay_vals_II_4f), 3)
conv_vals_II_4f_more_auto=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_1_4f_more_auto), np.array(gf_vals_II_4f_more_auto), 0.05, np.array(decay_vals_II_4f_more_auto), 3)
conv_vals_II_4f_open_shells=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_1_4f_open_shells), np.array(gf_vals_II_4f_open_shells), 0.05, np.array(decay_vals_II_4f_open_shells), 3)
#%%
Individual_lines_AuI_4f_dE=au_spec_4_I[:,11].astype(float)
Individual_lines_AuI_4f_gf=np.exp(au_spec_4_I[:,15].astype(float))

#%%
#plt.plot(Energy,conv_vals_I_4f,label='AuI 4f-5d')
#plt.plot(Energy+np.repeat(3.3,len(Energy)),conv_vals_II_4f,label='Au II 4f-5d')
#plt.plot(Energy+np.repeat(3.3,len(Energy)),conv_vals_II,label='4f-6d Au II')
#plt.plot(Energy+np.repeat(3.3,len(Energy)),conv_vals_I,label='4f-6d Au I')
#plt.plot(Energy+np.repeat(3.3,len(Energy)),1/(np.sum(0.8*conv_vals_I+ 0.2*conv_vals_II_4f))*(0.8*conv_vals_I+0.2*conv_vals_II_4f),label='80% AuI 4f-6d 20% AuII 4f-5d')
#plt.plot(Energy+np.repeat(3.3,len(Energy)),0.87*conv_vals_I+0.13*conv_vals_II_4f,label='87% AuI 4f-6d 13% AuII 4f-5d')
plt.plot(Energy_500ns,Intensity_500ns,label='500ns')
#plt.plot(Energy_500ns,Norm_intensitiy_500ns,label='500ns')
plt.vlines(Individual_lines_AuI_4f_dE+np.repeat(1.5,3),np.zeros(3),Individual_lines_AuI_4f_gf,label='AuI 4f individual trnasitions')
#plt.vlines(Individual_lines_AuII_4f_dE_3_8+np.repeat(1.4,81),np.zeros(81),Individual_lines_AuII_4f_gf_3_8,label='AuII 4f individual transitions',color='red',ls='--')
plt.legend()
#plt.xlim(90,100)
plt.xlabel('Energy [eV]')
plt.ylabel('Intensity')
#%%
trans_6d_I_1=au_spec_I[np.logical_and(au_spec_I[:,3]=='2',au_spec_I[:,8]=='1')]
trans_6d_I_5=au_spec_I[np.logical_and(au_spec_I[:,3]=='1',au_spec_I[:,8]=='5')]
double_promotions=np.concatenate((trans_6d_I_1,trans_6d_I_5),axis=0)
#double_promotions_I=np.logical_not(spec_file_1_I)
#%%
upper_levels_I_dp=list(set(double_promotions[:,6].astype(float)))
gf_vals_I_dp=[]
decay_vals_I_dp=[]
for ul in upper_levels_I_dp:
    temp_spec=double_promotions[double_promotions[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_I_dp.append(sum(gf_temp))
    decay_vals_I_dp.append(sum(decay_temp))
#%%
Energy=E_vals=np.arange(80,100,0.001)
conv_vals_I_dp=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_I_dp), np.array(gf_vals_I_dp), 0.05, np.array(decay_vals_I_dp), 3)
#%%
plt.plot(Energy,conv_vals_I_dp,label='double promotions AuI')
plt.plot(Energy,conv_vals_I,label='4f-6d AuI')
plt.xlabel('Energy [eV]')
plt.ylabel('Intensity')
plt.legend()
#%%
trans_6d_II_1=au_spec_II[au_spec_II[:,8]=='1']
trans_6d_II_1=trans_6d_II_1[np.logical_not(trans_6d_II_1[:,3]=='1')]
trans_6d_II_5=au_spec_II[au_spec_II[:,8]=='5']
trans_6d_II_5=trans_6d_II_5[np.logical_not(trans_6d_II_5[:,3]=='2')]
trans_6d_II_9=au_spec_II[au_spec_II[:,8]=='9']
trans_6d_II_9=trans_6d_II_9[np.logical_not(trans_6d_II_9[:,3]=='3')]
double_promotions_II=np.concatenate((trans_6d_II_1,trans_6d_II_5),axis=0)
double_promotions_II=np.concatenate((double_promotions_II,trans_6d_II_9),axis=0)
#%%
upper_levels_II_dp=list(set(double_promotions_II[:,6].astype(float)))
gf_vals_II_dp=[]
decay_vals_II_dp=[]
for ul in upper_levels_II_dp:
    temp_spec=double_promotions_II[double_promotions_II[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_II_dp.append(sum(gf_temp))
    decay_vals_II_dp.append(sum(decay_temp))
#%%
conv_vals_II_dp=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_II_dp), np.array(gf_vals_II_dp), 0.05, np.array(decay_vals_II_dp), 3)
#%%
#plt.plot(Energy+np.repeat(1.7,len(Energy)),conv_vals_II_dp,label='double promotions AuII')
#plt.plot(Energy+np.repeat(1.7,len(Energy)),conv_vals_II_open_shells,label='AuII 4f-6d')
#plt.plot(Energy+np.repeat(1.7,len(Energy)),conv_vals_I,label='AuI 4f-6d')
#plt.plot(Energy+np.repeat(1.7,len(Energy)),conv_vals_II_4f-conv_vals_II_4f_open_shells,label='AuII 4f-5d: closed shells -open shells')
#plt.plot(Energy+np.repeat(1.7,len(Energy)),conv_vals_II_4f,label='AuII 4f-5d: closed shells')
#plt.plot(Energy+np.repeat(1.7,len(Energy)),conv_vals_II_4f_open_shells,label='AuII 4f-5d')
#plt.plot(Energy+np.repeat(1.7,len(Energy)),conv_vals_II_4f_open_shells-conv_vals_II_4f_more_auto,label='AuII 4f-5d: closed shells')
#plt.plot(Energy+np.repeat(1.7,len(Energy)),0.2*(conv_vals_II+conv_vals_II_4f+conv_vals_I+conv_vals_II_dp))
plt.plot(Energy+np.repeat(1.7,len(Energy)),0.75*conv_vals_I + 0.25*(conv_vals_II_dp+conv_vals_II+conv_vals_II_4f_open_shells),label='87% AuI 13% AuII')
#plt.plot(Energy,conv_vals_I_dp,label='double promotions AuI')
#plt.plot(Energy,conv_vals_I,label='4f-6d AuI')
plt.plot(Energy_500ns,15*Intensity_500ns,label='500ns')
plt.xlim(80,100)
plt.legend()
plt.xlabel('Energy [eV]')
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
moving_avg_500ns=MovingAverage(5, Intensity_500ns)
moving_avg_450ns=MovingAverage(5, Intensity_450ns)
moving_avg_400ns=MovingAverage(5, Intensity_400ns)
moving_avg_energy=MovingAverage(5, Energy_500ns)
#%%
plt.plot(Energy_500ns,Intensity_500ns,label='500ns')
plt.plot(Energy_500ns,Intensity_400ns,label='400ns')
plt.plot(Energy_500ns,Intensity_300ns,label='300ns')
#plt.plot(moving_avg_energy,moving_avg_400ns,label='400ns')
plt.xlabel('Energy [eV]')
plt.ylabel('Absorbance')
#plt.title('5 pt smoothing experimental data')
plt.legend()
plt.xlim(62,99.5)
plt.ylim(0.05,1.7)