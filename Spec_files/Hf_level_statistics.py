# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:27:51 2026

@author: David McKeagney
"""

import numpy as np
import matplotlib.pyplot as plt 
import function_library_phd as flp
#%%
def lvldensity(E,Evals,J):
    num_vals=np.zeros(len(E))
    for a,Energy in enumerate(E):
        vals_ind=np.logical_and(Evals[:,0]<=Energy,Evals[:,2]==J)
        num=len(Evals[vals_ind][:,0])
        num_vals[a]+=num
    return num_vals
def lvldensity_slope(Evals,rho_vals):
    E_diff=np.diff(Evals)
    rho_diff=np.diff(rho_vals)
    return rho_diff/E_diff
    
def levelspacing(Evals,J):
    Vals=Evals[Evals[:,2]==J][:,0][14:]
    spacing=np.diff(Vals)
    return spacing
    
def lvldensityinterpolationdiff(E,Evals,rho):
    N_vals=np.interp(E, Evals, rho)
    N_vals_diff=np.diff(N_vals)
    return N_vals_diff    
def Wigner(s,D):
    return (np.pi*s/(2*D**2))*np.exp(-np.pi*(s/(2*D))**2)     
#%%
Hf_level_info=[]
with open('C:\\Users\David McKeagney\Downloads\Hf_I.sorted') as file:
    for lines in file:
        Hf_level_info.append(lines.split())
Hf_level_info=Hf_level_info[1:]
Hf_level_info=[a[0:3] for a in Hf_level_info]
Hf_level_info=np.array(Hf_level_info).astype(float)
Hf_level_info[:,0]=Hf_level_info[:,0]+np.repeat(np.abs(Hf_level_info[0,0]), len(Hf_level_info[:,0]))
#%%
Hf_positive=Hf_level_info[Hf_level_info[:,1]>0]
Hf_negative=Hf_level_info[Hf_level_info[:,1]<0]
#%%
Energy=np.arange(12,80,0.01)
rho_1_neg=lvldensity(Energy, Hf_negative, 1)
rho_2_neg=lvldensity(Energy, Hf_negative, 2)
rho_3_neg=lvldensity(Energy, Hf_negative, 3)
rho_4_neg=lvldensity(Energy, Hf_negative, 4)
rho_5_neg=lvldensity(Energy, Hf_negative, 5)
#%%
rho_4_neg_slope=lvldensity_slope(Energy, rho_4_neg)
#%%
plt.plot(Energy,rho_1_neg,label='J=1^-')
plt.plot(Energy,rho_2_neg,label='J=2^-')
plt.plot(Energy,rho_3_neg,label='J=3^-')
plt.plot(Energy,rho_4_neg,label='J=4^-')
plt.plot(Energy,rho_5_neg,label='J=5^-')
plt.xlabel('Energy')
plt.ylabel('N(E)')
plt.xlim(12,25)
plt.ylim(0,800)
plt.legend()
#%%
plt.plot(Energy[:-1],rho_4_neg_slope)
#%%
Hf_negative=Hf_negative[np.logical_and(Hf_negative[:,0]>=10,Hf_negative[:,0]<=20)]
#%%
E_2_neg=Hf_negative[Hf_negative[:,2]==2][:,0]
E_3_neg=Hf_negative[Hf_negative[:,2]==3][:,0]
E_4_neg=Hf_negative[Hf_negative[:,2]==4][:,0]
#%%
spacing_2_neg=levelspacing(Hf_negative, 2)
#spacing_2_neg=spacing_2_neg[spacing_2_neg<0.2]
#dN_2_neg=lvldensityinterpolationdiff(E_2_neg, Energy, rho_2_neg)
spacing_3_neg=levelspacing(Hf_negative, 3)
spacing_4_neg=levelspacing(Hf_negative, 4)
#dN_3_neg=lvldensityinterpolationdiff(E_3_neg, Energy, rho_3_neg)
#%%
s=np.arange(0,0.17,0.001)
#%%
plt.hist(spacing_4_neg,bins=20)
plt.plot(s,Wigner(s, 0.0406))
#plt.xlim(0,0.1)