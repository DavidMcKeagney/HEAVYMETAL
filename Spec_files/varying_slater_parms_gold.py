# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 11:56:59 2025

@author: David McKeagney
"""

import function_library_phd as flp
import numpy as np 
import matplotlib.pyplot as plt 
#%% Comparing calulcations with scaled spin orbit parameters for the energy levels, au_spin is the varied spin orbit file,au_col is varied colomb integral for non-equiv electrons 
au_spec_II=[]
with open('C:/Users/David McKeagney/Desktop/au1.sub.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            au_spec_II.append(lines.split())
au_spin_spec_II=[]
with open('C:/Users/David McKeagney/Desktop/au1_spin.sub.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            au_spin_spec_II.append(lines.split())
au_col_spec_II=[]
with open('C:/Users/David McKeagney/Desktop/au1_col.sub.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            au_col_spec_II.append(lines.split())
au_eqi_spec_II=[]
with open('C:/Users/David McKeagney/Desktop/au1_eqi.sub.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            au_eqi_spec_II.append(lines.split())
au_gk_spec_II=[]
with open('C:/Users/David McKeagney/Desktop/au1_gk.sub.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            au_gk_spec_II.append(lines.split())
au_spec_II=np.array(au_spec_II)[1:,:]
au_spin_spec_II=np.array(au_spin_spec_II)[1:,:]
au_col_spec_II=np.array(au_col_spec_II)[1:,:]
au_eqi_spec_II=np.array(au_eqi_spec_II)[1:,:]
au_gk_spec_II=np.array(au_gk_spec_II)[1:,:]
#%% fixed spin orbit
au_spec_1_II=au_spec_II[au_spec_II[:,8]=='1']
au_spec_5_II=au_spec_II[au_spec_II[:,8]=='5']
au_spec_9_II=au_spec_II[au_spec_II[:,8]=='9']
spec_file_1_II=np.concatenate((au_spec_1_II,au_spec_5_II),axis=0)
spec_file_9_II=np.concatenate((spec_file_1_II,au_spec_9_II),axis=0)
#%% varied spin orbit
au_spin_spec_1_II=au_spin_spec_II[au_spin_spec_II[:,8]=='1']
au_spin_spec_5_II=au_spin_spec_II[au_spin_spec_II[:,8]=='5']
au_spin_spec_9_II=au_spin_spec_II[au_spin_spec_II[:,8]=='9']
spec_spin_file_1_II=np.concatenate((au_spin_spec_1_II,au_spin_spec_5_II),axis=0)
spec_spin_file_9_II=np.concatenate((spec_spin_file_1_II,au_spin_spec_9_II),axis=0)
#%%varied col parameter
au_col_spec_1_II=au_col_spec_II[au_col_spec_II[:,8]=='1']
au_col_spec_5_II=au_col_spec_II[au_col_spec_II[:,8]=='5']
au_col_spec_9_II=au_col_spec_II[au_col_spec_II[:,8]=='9']
spec_col_file_1_II=np.concatenate((au_col_spec_1_II,au_col_spec_5_II),axis=0)
spec_col_file_9_II=np.concatenate((spec_col_file_1_II,au_col_spec_9_II),axis=0)
#%% varied eqi paremeter
au_eqi_spec_1_II=au_eqi_spec_II[au_eqi_spec_II[:,8]=='1']
au_eqi_spec_5_II=au_eqi_spec_II[au_eqi_spec_II[:,8]=='5']
au_eqi_spec_9_II=au_spin_spec_II[au_eqi_spec_II[:,8]=='9']
spec_eqi_file_1_II=np.concatenate((au_eqi_spec_1_II,au_eqi_spec_5_II),axis=0)
spec_eqi_file_9_II=np.concatenate((spec_eqi_file_1_II,au_eqi_spec_9_II),axis=0)
#%% varied gk paremeter
au_gk_spec_1_II=au_gk_spec_II[au_gk_spec_II[:,8]=='1']
au_gk_spec_5_II=au_gk_spec_II[au_gk_spec_II[:,8]=='5']
au_gk_spec_9_II=au_gk_spec_II[au_gk_spec_II[:,8]=='9']
spec_gk_file_1_II=np.concatenate((au_gk_spec_1_II,au_gk_spec_5_II),axis=0)
spec_gk_file_9_II=np.concatenate((spec_gk_file_1_II,au_gk_spec_9_II),axis=0)

#%% fixed spin orbit
upper_levels_1=list(set(spec_file_9_II[:,6].astype(float)))
gf_vals_II=[]
decay_vals_II=[]
for ul in upper_levels_1:
    temp_spec=spec_file_9_II[spec_file_9_II[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_II.append(sum(gf_temp))
    decay_vals_II.append(sum(decay_temp))
#%% varied spin orbit 
upper_levels_spin_1=list(set(spec_spin_file_9_II[:,6].astype(float)))
gf_vals_spin_II=[]
decay_vals_spin_II=[]
for ul in upper_levels_spin_1:
    temp_spec=spec_spin_file_9_II[spec_spin_file_9_II[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_spin_II.append(sum(gf_temp))
    decay_vals_spin_II.append(sum(decay_temp))
#%%
upper_levels_col_1=list(set(spec_col_file_9_II[:,6].astype(float)))
gf_vals_col_II=[]
decay_vals_col_II=[]
for ul in upper_levels_col_1:
    temp_spec=spec_col_file_9_II[spec_col_file_9_II[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_col_II.append(sum(gf_temp))
    decay_vals_col_II.append(sum(decay_temp))
#%%
upper_levels_eqi_1=list(set(spec_eqi_file_9_II[:,6].astype(float)))
gf_vals_eqi_II=[]
decay_vals_eqi_II=[]
for ul in upper_levels_eqi_1:
    temp_spec=spec_eqi_file_9_II[spec_eqi_file_9_II[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_eqi_II.append(sum(gf_temp))
    decay_vals_eqi_II.append(sum(decay_temp))
#%%
upper_levels_gk_1=list(set(spec_gk_file_9_II[:,6].astype(float)))
gf_vals_gk_II=[]
decay_vals_gk_II=[]
for ul in upper_levels_gk_1:
    temp_spec=spec_gk_file_9_II[spec_gk_file_9_II[:,6].astype(float)==ul]
    gf_temp=np.exp(temp_spec[:,15].astype(float))
    decay_temp=temp_spec[:,16].astype(float)*1e-3
    gf_vals_gk_II.append(sum(gf_temp))
    decay_vals_gk_II.append(sum(decay_temp))
#%%
Energy=E_vals=np.arange(70,130,0.001)
conv_vals_II=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_1), np.array(gf_vals_II), 0.05, np.array(decay_vals_II), 3)
conv_vals_spin_II=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_spin_1), np.array(gf_vals_spin_II), 0.05, np.array(decay_vals_spin_II), 3)
conv_vals_col_II=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_col_1), np.array(gf_vals_col_II), 0.05, np.array(decay_vals_col_II), 3)
conv_vals_eqi_II=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_eqi_1), np.array(gf_vals_eqi_II), 0.05, np.array(decay_vals_eqi_II), 3)
conv_vals_gk_II=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_gk_1), np.array(gf_vals_gk_II), 0.05, np.array(decay_vals_gk_II), 3)
#%%
plt.plot(Energy,conv_vals_II,label='fixed')
plt.plot(Energy,conv_vals_eqi_II,label='eqi')
plt.plot(Energy,conv_vals_gk_II,label='gk')
#plt.plot(Energy,conv_vals_spin_II,label='varied')
plt.legend()
plt.xlabel('Energy [eV]')
plt.ylabel('Intensity [Arb.]')