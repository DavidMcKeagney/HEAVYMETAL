# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 16:25:26 2025

@author: David McKeagney
"""

import numpy as np
import matplotlib.pyplot as plt 
from scipy.special import voigt_profile
import function_library_phd as flp
#%%
def Voigt(A,x,x_0,sig,gam):
    Conv_Evals=np.zeros((len(x),len(x_0)))
    i=0

    while i< len(x_0):
        j=0
        conv= A[i]*voigt_profile(x-x_0[i],sig,gam) 
        while j<len(x):
            Conv_Evals[j][i]+=conv[j]
            j+=1
        i+=1
    Conv_Evals=np.sum(Conv_Evals,axis=1)
    return Conv_Evals
#%%
x_vals=np.linspace(-5,15,500)
x_0=np.array([5,5.1,5.2])
gf=np.array([0.01,2,0.5])
V=Voigt(gf,x_vals, x_0, 0.05, 1)
#%%
au2_spec=[]
with open('C:/Users/David McKeagney/Desktop/au2.sub.1.5.spec') as file2:
    for lines in file2:
        if len(lines.split())>17:
            au2_spec.append(lines.split())
au1_spec=[]
with open('C:/Users/David McKeagney/Desktop/au1.sub.1.5.spec') as file1:
    for lines in file1:
        if len(lines.split())>17:
            au1_spec.append(lines.split())
       
au2_spec=np.array(au2_spec)[1:,:]
au1_spec=np.array(au1_spec)[1:,:]

#%%
au2_spec_2=au2_spec[au2_spec[:,8]=='2']
au2_spec_6=au2_spec[au2_spec[:,8]=='6']
au2_spec_10=au2_spec[au2_spec[:,8]=='10']
au2_spec=np.concatenate((au2_spec_2,au2_spec_6),axis=0)
au2_spec=np.concatenate((au2_spec,au2_spec_10),axis=0)
dE_4f_6d_au2=au2_spec[:,11].astype(float)
E_k_4f_d6_au2=au2_spec[:,6].astype(float)-np.repeat(np.min(au2_spec[:,1].astype(float)), len(au2_spec))
gf_4f_6d_au2=np.exp(au2_spec[:,15].astype(float))
gamma_4f_6d_au2=au2_spec[:,16].astype(float)*1e-3
#%%
au2_spec_1=au2_spec[au2_spec[:,8]=='1']
au2_spec_5=au2_spec[au2_spec[:,8]=='5']
au2_spec_9=au2_spec[au2_spec[:,8]=='9']
au2_spec_5d=np.concatenate((au2_spec_1,au2_spec_5),axis=0)
au2_spec_5d=np.concatenate((au2_spec_5d,au2_spec_9),axis=0)
dE_4f_5d_au2=au2_spec_5d[:,11].astype(float)
#E_k_4f_d5_au2=au2_spec_5d[:,6].astype(float)-np.repeat(np.min(au2_spec_5d[:,1].astype(float)), len(au2_spec_5d))
gf_4f_5d_au2=np.exp(au2_spec_5d[:,15].astype(float))
gamma_4f_5d_au2=au2_spec_5d[:,16].astype(float)*1e-3
#%%
au1_spec_8=au1_spec[au1_spec[:,8]=='8']
au1_spec_8=au1_spec_8[au1_spec_8[:,3]=='3']
gf_4f_5d_au1=np.exp(au1_spec_8[:,15].astype(float))
dE_4f_5d_au1=au1_spec_8[:,11].astype(float)
gamma_4f_5d_au1=au1_spec_8[:,16].astype(float)*1e-3
#%%
E_vals=np.arange(70,120,0.01)
conv_au2_6d=flp.ConvolvingFunc(0, E_vals, dE_4f_6d_au2, gf_4f_6d_au2, 0.05, gamma_4f_6d_au2, 3)
conv_au2_5d=flp.ConvolvingFunc(0, E_vals, dE_4f_5d_au2, gf_4f_5d_au2, 0.05, gamma_4f_5d_au2, 3)
conv_au1_5d=flp.ConvolvingFunc(0, E_vals, dE_4f_5d_au1, gf_4f_5d_au1, sig, gam, flag)
#%%
plt.plot(E_vals,conv_au2_6d)
plt.plot(E_vals,conv_au2_5d)