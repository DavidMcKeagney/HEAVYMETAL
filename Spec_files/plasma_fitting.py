# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:56:20 2025

@author: damck
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

plt.rcParams.update({'font.size': 22})
plt.rcParams["figure.figsize"] = (15,10)
#%%
au1_spec=[]
with open('C:/Users/padmin/OneDrive/Desktop/au1.sub.spec') as file1:
    for lines in file1:
        if len(lines.split())>17:
            au1_spec.append(lines.split())
au2_spec=[]
with open('C:/Users/padmin/OneDrive/Desktop/au2.sub.spec') as file2:
    for lines in file2:
        if len(lines.split())>17:
            au2_spec.append(lines.split())
au3_spec=[]
with open('C:/Users/padmin/OneDrive/Desktop/au1.sub.spec') as file3:
    for lines in file3:
        if len(lines.split())>17:
            au3_spec.append(lines.split())
au1_spec=np.array(au1_spec)[1:,:]
au2_spec=np.array(au2_spec)[1:,:]
au3_spec=np.array(au3_spec)[1:,:]
#%% 4f-6d transitions au1+
au1_spec_1=au1_spec[au1_spec[:,8]=='1']
au1_spec_5=au1_spec[au1_spec[:,8]=='5']
au1_spec_9=au1_spec[au1_spec[:,8]=='9']
au1_spec=np.concatenate((au1_spec_1,au1_spec_5),axis=0)
au1_spec=np.concatenate((au1_spec,au1_spec_9),axis=0)
dE_4f_6d_au1=au1_spec[:,11].astype(float)
E_k_4f_d6_au1=au1_spec[:,6].astype(float)-np.repeat(np.min(au1_spec[:,1].astype(float)), len(au1_spec))
gf_4f_6d_au1=np.exp(au1_spec[:,15].astype(float))
gamma_4f_6d_au1=au1_spec[:,16].astype(float)*1e-3
#%% 4f-6d transitions au2+
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
au3_spec_2=au3_spec[au3_spec[:,8]=='2']
au3_spec_6=au3_spec[au3_spec[:,8]=='6']
au3_spec_10=au3_spec[au3_spec[:,8]=='10']
au3_spec=np.concatenate((au3_spec_2,au3_spec_6),axis=0)
au3_spec=np.concatenate((au3_spec,au3_spec_10),axis=0)
dE_4f_6d_au3=au3_spec[:,11].astype(float)
E_k_4f_d6_au3=au3_spec[:,6].astype(float)-np.repeat(np.min(au3_spec[:,1].astype(float)), len(au3_spec))
gf_4f_6d_au3=np.exp(au3_spec[:,15].astype(float))
gamma_4f_6d_au3=au3_spec[:,16].astype(float)*1e-3
#%%
dE_vec=np.concatenate((dE_4f_6d_au1,dE_4f_6d_au2))
dE_vec=np.concatenate((dE_vec,dE_4f_6d_au3))
E_k_4f_6d_vec=np.concatenate((E_k_4f_d6_au1,E_k_4f_d6_au2))
E_k_4f_6d_vec=np.concatenate((E_k_4f_6d_vec,E_k_4f_d6_au3))
gf_4f_6d_vec=np.concatenate((gf_4f_6d_au1,gf_4f_6d_au2))
gf_4f_6d_vec=np.concatenate((gf_4f_6d_vec,gf_4f_6d_au3))
gam_4f_6d_vec=np.concatenate((gamma_4f_6d_au1,gamma_4f_6d_au2))
gam_4f_6d_vec=np.concatenate((gam_4f_6d_vec,gamma_4f_6d_au3))
#%%

dE_vec=dE_vec[gf_4f_6d_vec>0.01]
E_k_4f_6d_vec=E_k_4f_6d_vec[gf_4f_6d_vec>0.01]
gam_4f_6d_vec=gam_4f_6d_vec[gf_4f_6d_vec>0.01]
gf_4f_6d_vec=gf_4f_6d_vec[gf_4f_6d_vec>0.01]
#%%
Eric_data_400ns=np.loadtxt('C:\\Users\padmin\Downloads\Eric_data_400ns.txt',dtype=float).T
Eric_data_450ns=np.loadtxt('C:\\Users\padmin\Downloads\Eric_data_450ns.txt',dtype=float).T
Eric_data_500ns=np.loadtxt('C:\\Users\padmin\Downloads\Eric_data_500ns.txt',dtype=float).T
Intensity_400ns=Eric_data_400ns[1][700:1500]
Intensity_450ns=Eric_data_450ns[1][700:1500]
Intensity_500ns=Eric_data_500ns[1][700:1500]
Energy=Eric_data_400ns[0][700:1500]
#%%
Energy_4f_6d=Energy[np.logical_and(Energy<=100,Energy>=90)]
Intensity_4f_6d=Intensity_400ns[np.logical_and(Energy<=100,Energy>=90)]
E_vals=np.linspace(90,100,156)
#%%
def ConvolvingFunc(E,dE,gf,gamma,e_k,T):
    return 109.7617*(gamma/(2*np.pi))*(np.exp(-(e_k)/T))*(1/((E-dE)**2 + 0.25*gamma**2))
#%%
def Convolve(E,dE,gf,gamma,e_k,T):
    Conv_vals=np.zeros((len(E),len(dE)))
    i=0
    while i<len(dE):
        j=0
        conv=ConvolvingFunc(E, dE[i], gf[i], gamma[i], e_k[i], T)
        while j<len(E):
            Conv_vals[j][i]+=conv[j]
            j+=1
        i+=1
    Conv_vals=np.sum(Conv_vals,axis=1)
    return Conv_vals
#%%
def SynthSpectra(alpha,X):
    E,dE_vec,gf_vec,gam_vec,E_k_vec=X
    func1=alpha[0]*Convolve(E,dE_vec[:len(dE_4f_6d_au1)-1],gf_vec[:len(gf_4f_6d_au1)-1],gam_vec[:len(gamma_4f_6d_au1)-1],E_k_vec[:len(E_k_4f_d6_au1)-1],alpha[3])
    func2=alpha[1]*Convolve(E,dE_vec[len(dE_4f_6d_au1)-1:len(dE_4f_6d_au2)-1],gf_vec[len(gf_4f_6d_au1)-1:len(gf_4f_6d_au2)-1],gam_vec[len(gamma_4f_6d_au1)-1:len(gamma_4f_6d_au2)-1],E_k_vec[len(E_k_4f_d6_au1)-1:len(E_k_4f_d6_au2)-1],alpha[3])
    func3=alpha[2]*Convolve(E,dE_vec[len(dE_4f_6d_au2)-1:],gf_vec[len(gf_4f_6d_au2)-1:],gam_vec[len(gf_4f_6d_au2)-1:],E_k_vec[len(gamma_4f_6d_au2)-1:],alpha[3])
    return func1+func2+func3    
    
def CostFunc(alpha,X,I):
    return SynthSpectra(alpha, X)-I
#%%
alpha_guess=[0.55,0.4,0.05,4.5]
res_lsq=least_squares(CostFunc,alpha_guess,args=((E_vals,dE_vec,gf_4f_6d_vec,gam_4f_6d_vec,E_k_4f_6d_vec),Intensity_4f_6d))
#%%
au1_spec_3_8=au1_spec[np.logical_and(au1_spec[:,8]=='8',au1_spec[:,3]=='3')]
au2_spec_3_9=au2_spec[np.logical_and(au2_spec[:,8]=='9',au2_spec[:,3]=='3')]
dE_1_3_8=au1_spec_3_8[:,11].astype(float)
gf_1_3_8=np.exp(au1_spec_3_8[:,15].astype(float))
dE_1_3_8_cut=dE_1_3_8[gf_1_3_8>0.6]
gf_1_3_8_cut=gf_1_3_8[gf_1_3_8>0.6]
dE_2_3_9=au2_spec_3_9[:,11].astype(float)
dE_2_3_9_cut=dE_2_3_9[dE_2_3_9<=90]
gf_2_3_9=np.exp(au2_spec_3_9[:,15].astype(float))[dE_2_3_9<=90]
gf_2_3_9_cut=gf_2_3_9[gf_2_3_9>0.6]
dE_2_3_9_cut=dE_2_3_9_cut[gf_2_3_9>0.6]
#%%
plt.plot(Energy,Intensity_400ns,color='black',label='400ns')
plt.vlines(dE_2_3_9_cut,ymin=0,ymax=gf_2_3_9_cut,ls='--',lw=2,color='red',label='Au 2+')
plt.vlines(dE_1_3_8_cut+np.repeat(1.8,len(dE_1_3_8_cut)),ymin=0,ymax=gf_1_3_8_cut,ls='--',lw=2,label='Au 1+')
plt.legend()
plt.xlabel('Energy [eV]')
plt.ylabel(' Absorbance [Arb.]')
plt.xlim(75,100)
plt.ylim(0.1,1)
#plt.grid(True)
#%%
unfitted_synth_spec=SynthSpectra(alpha_guess,(E_vals,dE_vec,gf_4f_6d_vec,gam_4f_6d_vec,E_k_4f_6d_vec))

#%%
plt.plot(Energy,Intensity_400ns,color='black',label='400ns')
plt.vlines(95,ymin=0.25,ymax=0.454,ls='--',lw=2,color='blue',label='4f-6d')
plt.vlines(98.5,ymin=0.25,ymax=0.454,ls='--',lw=2,color='blue')
plt.legend()
plt.xlabel('Energy [eV]')
plt.xlim(75,100)
plt.ylim(0.275)
plt.ylabel('Absorbance [Arb.]')
#%%
au1_3_8=au1_spec[np.logical_and(au1_spec[:,3]=='3',au1_spec[:,8]=='8')]
au2_3_9=au2_spec[np.logical_and(au2_spec[:,3]=='3',au2_spec[:,8]=='9')]
au1_3_8=au1_3_8[np.exp(au1_3_8[:,15].astype(float))>0.6]
au2_3_9=au2_3_9[np.exp(au2_3_9[:,15].astype(float))>0.6]

