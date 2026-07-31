# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:12:39 2026

@author: David McKeagney
"""

import numpy as np
import matplotlib.pyplot as plt
import function_library_phd as flp
#%%
Energy=np.arange(40,200.1,0.1)
Hf_I=np.loadtxt('C:\\Users\David McKeagney\Downloads\\total_cross_section_Hf_I.dat',dtype=float)
Hf_II=np.loadtxt('C:\\Users\David McKeagney\Downloads\\total_cross_section_Hf_II.dat',dtype=float)
Hf_III=np.loadtxt('C:\\Users\David McKeagney\Downloads\\total_cross_section_Hf_III.dat',dtype=float)
Hf_IV=np.loadtxt('C:\\Users\David McKeagney\Downloads\\total_cross_section_Hf_IV.dat',dtype=float)
Hf_V=np.loadtxt('C:\\Users\David McKeagney\Downloads\\total_cross_section_Hf_V.dat',dtype=float)
Hf_VI=np.loadtxt('C:\\Users\David McKeagney\Downloads\\total_cross_section_Hf_VI.dat',dtype=float)
#%%
#Abs_55ns=np.loadtxt('C:\\Users\David McKeagney\Desktop\OneDrive_1_24-03-2026\\55ns.csv',dtype=str,delimiter=',')
#Abs_75ns=np.loadtxt('C:\\Users\David McKeagney\Desktop\OneDrive_1_24-03-2026\\75ns.csv',dtype=str,delimiter=',')
#Abs_95ns=np.loadtxt('C:\\Users\David McKeagney\Desktop\OneDrive_1_24-03-2026\\95ns.csv',dtype=str,delimiter=',')
Abs_500ns=np.loadtxt('C:\\Users\David McKeagney\Desktop\OneDrive_1_27-05-2026\\2023_07_13_113.csv',dtype=str,delimiter=',')
Abs_450ns=np.loadtxt('C:\\Users\David McKeagney\Desktop\OneDrive_1_27-05-2026\\2023_07_13_112.csv',dtype=str,delimiter=',')
Abs_400ns=np.loadtxt('C:\\Users\David McKeagney\Desktop\OneDrive_1_27-05-2026\\2023_07_13_111.csv',dtype=str,delimiter=',')
Abs_400ns=Abs_400ns[1:,:].astype(float)
Abs_450ns=Abs_450ns[1:,:].astype(float)
Abs_500ns=Abs_500ns[1:,:].astype(float)
#Abs_55ns=Abs_55ns[1:,:].astype(float)
#Abs_75ns=Abs_75ns[1:,:].astype(float)
#Abs_95ns=Abs_95ns[1:,:].astype(float)
#%%
Hf_I=[]
Hf_II=[]
Hf_III=[]
with open('C:\\Users\David McKeagney\Downloads\Hf_I.spec') as file:
    for lines in file:
        if len(lines.split())>16:
            Hf_I.append(lines.split())
with open('C:\\Users\David McKeagney\Downloads\Hf_II.spec') as file:
    for lines in file:
        if len(lines.split())>16:
            Hf_II.append(lines.split())
with open('C:\\Users\David McKeagney\Downloads\Hf_III.spec') as file:
    for lines in file:
        if len(lines.split())>16:
            Hf_III.append(lines.split())
Hf_I=np.array(Hf_I)
Hf_II=np.array(Hf_II)
Hf_III=np.array(Hf_III)
Hf_I=Hf_I[Hf_I[:,11].astype(float)>=30]
Hf_II=Hf_II[Hf_II[:,11].astype(float)>=30]
Hf_III=Hf_III[Hf_III[:,11].astype(float)>=30]
#%%
gf_HfI=np.exp(Hf_I[:,15].astype(float))
gf_HfII=np.exp(Hf_II[:,15].astype(float))
gf_HfIII=np.exp(Hf_III[:,15].astype(float))

dE_HfI=Hf_I[:,11].astype(float)
dE_HfII=Hf_II[:,11].astype(float)
dE_HfIII=Hf_III[:,11].astype(float)
#%%
Energy_vals=np.arange(0,100,0.01)
sig_HfI=flp.ConvolvingFunc(0, Energy_vals, dE_HfI, gf_HfI, 0.05, 0, 0.01, 0)
#sig_HfII=flp.ConvolvingFunc(0, Energy_vals, dE_HfII, gf_HfII, 0.05, 0, 0.01, 0)
#sig_HfIII=flp.ConvolvingFunc(0, Energy_vals, dE_HfIII, gf_HfIII, 0.05, 0, 0.01, 0)
#%%
#plt.plot(Energy+np.repeat(10.5,len(Energy)),1/400*Hf_III,label='Hf III')
#plt.plot(Energy+np.repeat(11.2,len(Energy)),1/400*Hf_II,label='Hf II')
plt.plot(Energy+np.repeat(9.8,len(Energy)),1/400*Hf_I,label='Hf I')
#plt.plot(Energy+np.repeat(11,len(Energy)),1/400*Hf_II+1/400*Hf_III,label='Hf II + III')
#plt.plot(Abs_55ns[:,0],Abs_55ns[:,1],label='55ns' )
#plt.plot(Abs_55ns[:,0],Abs_75ns[:,1],label='75ns' )
#plt.plot(Abs_55ns[:,0],Abs_95ns[:,1],label='95ns' )
#plt.plot(Energy,Hf_IV,label='Hf IV')
#plt.plot(Energy,Hf_V,label='Hf V')
#plt.plot(Energy,Hf_VI,label='Hf VI')
plt.plot(Energy_vals +np.repeat(17,len(Energy_vals)),1/20*sig_HfI,label='Hf I')
#plt.plot(Energy_vals + np.repeat(15,len(Energy_vals)),1/5*sig_HfII, label='Hf II')
#plt.plot(Energy_vals , sig_HfIII, label= 'Hf III')
plt.plot(Abs_400ns[:,0],Abs_400ns[:,1],label='400ns')
plt.plot(Abs_450ns[:,0],Abs_450ns[:,1],label='450ns')
plt.plot(Abs_500ns[:,0],Abs_500ns[:,1],label='500ns')
plt.xlim(30,80)
plt.ylim(0,1.25)
plt.legend()
plt.xlabel('Energy (eV)')
plt.ylabel('Absorbance')