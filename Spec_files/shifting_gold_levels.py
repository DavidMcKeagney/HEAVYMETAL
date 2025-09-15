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