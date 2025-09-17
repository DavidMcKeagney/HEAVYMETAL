# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 15:01:23 2025

@author: David McKeagney
"""

import numpy as np
import matplotlib.pyplot as plt 
import function_library_phd as flp
import csv 
#%%
gold_levels_I=np.loadtxt('C:/Users/David McKeagney/Downloads/gold_I_levels.txt',delimiter='\t',dtype=str)
gold_levels_II=np.loadtxt('C:/Users/David McKeagney/Downloads/gold_II_levels.txt',delimiter='\t',dtype=str)
gold_levels_I=gold_levels_I[1:4,:5]
gold_levels_II=gold_levels_II[1:7,:5]
#%%
au_I=[]
with open('C:/Users/David McKeagney/Desktop/au.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            au_I.append(lines.split())
au_I=np.array(au_I)[1:,:]
au_II=[]
with open('C:/Users/David McKeagney/Desktop/au1.spec') as file:
    for lines in file:
        if len(lines.split())>17:
            au_II.append(lines.split())
au_II=np.array(au_II)[1:,:]
#%%
au_I_5d106s=np.unique(au_I[au_I[:,3]=='1'][:,1:6],axis=0)
au_I_5d96s2=np.unique(au_I[au_I[:,3]=='2'][:,1:6],axis=0)

au_II_5d10=np.unique(au_II[au_II[:,3]=='1'][:,1:6],axis=0)
au_II_5d96s=np.unique(au_II[au_II[:,3]=='2'][:,1:6],axis=0)
au_II_5d86s2=np.unique(au_II[au_II[:,3]=='3'][:,1:6],axis=0)

gold_levels_I_5d106s=gold_levels_I[0,:]
gold_levels_I_5d96s2=gold_levels_I[1:,:]

gold_levels_II_5d10=gold_levels_II[0:,:]
gold_levels_II_5d96s=gold_levels_II[1:5,:]
gold_levels_II_5d86s2=gold_levels_II[5:,:]
#%%
x_vals=np.arange(0,5,0.01)
plt.plot(x_vals,x_vals)
plt.scatter(au_I_5d106s[0,0].astype(float),au_I_5d106s[0,0].astype(float),label='theory 5d106s')
plt.scatter(au_I_5d96s2[:,0].astype(float),au_I_5d96s2[:,0].astype(float),label='theory 5d96s2')
plt.scatter(gold_levels_I_5d106s[4].astype(float),gold_levels_I_5d106s[4].astype(float),label='exp 5d106s')
plt.scatter(gold_levels_I_5d96s2[:,4].astype(float),gold_levels_I_5d96s2[:,4].astype(float),label='exp 5d96s2')
plt.legend()
#%% theory experimental tuples 
tup_auI= [(au_I_5d106s[0,0],gold_levels_I_5d106s[4]),(au_I_5d96s2[0,0],gold_levels_I_5d96s2[0,4]),(au_I_5d96s2[1,0],gold_levels_I_5d96s2[1,4])]
tup_auII=[(au_II_5d10[0,0],gold_levels_II_5d10[0,4]),(au_II_5d96s[0,0],gold_levels_II_5d10[1,4]),(au_II_5d96s[1,0],gold_levels_II_5d10[2,4]),(au_II_5d96s[2,0],gold_levels_II_5d10[3,4]),(au_II_5d96s[3,0],gold_levels_II_5d10[4,4]),(au_II_5d86s2[1,0],gold_levels_II_5d10[5,4])]
#%%
for lines in tup_auI:
    au_I[au_I[:,1]==lines[0]][:,1]== lines[1]
for lines in tup_auII:
    au_II[au_II[:,1]==lines[0]][:,1]==lines[1]
#%%
for a,lines in enumerate(au_I):
    au_I[a,11]==str(float(lines[6])-float(lines[1]))
    
for b,lines in enumerate(au_II):
    au_II[b,11]==str(float(lines[6])-float(lines[1]))
#%%
au_spec_1_I=au_I[au_I[:,8]=='1']
au_spec_5_I=au_I[au_I[:,8]=='5']
spec_file_1_I=np.concatenate((au_spec_1_I,au_spec_5_I),axis=0)
#%%
au_spec_1_II=au_II[au_II[:,8]=='1']
au_spec_5_II=au_II[au_II[:,8]=='5']
au_spec_9_II=au_II[au_II[:,8]=='9']
spec_file_1_II=np.concatenate((au_spec_1_II,au_spec_5_II),axis=0)
spec_file_9_II=np.concatenate((spec_file_1_I,au_spec_9_II),axis=0)
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
Energy=E_vals=np.arange(70,130,0.001)
conv_vals_I=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_I), np.array(gf_vals_I), 0.05, np.array(decay_vals_I), 3)
conv_vals_II=flp.ConvolvingFunc(1, Energy, np.array(upper_levels_1), np.array(gf_vals_II), 0.05, np.array(decay_vals_II), 3)
#%%
plt.plot(Energy,conv_vals_I,label='4f-6d Au I')
plt.plot(Energy,conv_vals_II,label='4f-6d Au II')
plt.legend()
plt.xlabel('Energy [eV]')
plt.ylabel('Intensity [Arb.]')
plt.grid(True)