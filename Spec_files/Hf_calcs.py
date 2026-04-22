# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:12:39 2026

@author: David McKeagney
"""

import numpy as np
import matplotlib.pyplot as plt
#%%
Energy=np.arange(40,200.1,0.1)
Hf_III=np.loadtxt('C:\\Users\David McKeagney\Downloads\\total_cross_section_Hf_III.dat',dtype=float)
Hf_IV=np.loadtxt('C:\\Users\David McKeagney\Downloads\\total_cross_section_Hf_IV.dat',dtype=float)
Hf_V=np.loadtxt('C:\\Users\David McKeagney\Downloads\\total_cross_section_Hf_V.dat',dtype=float)
Hf_VI=np.loadtxt('C:\\Users\David McKeagney\Downloads\\total_cross_section_Hf_VI.dat',dtype=float)
#%%
Abs_55ns=np.loadtxt('C:\\Users\David McKeagney\Desktop\OneDrive_1_24-03-2026\\55ns.csv',dtype=str,delimiter=',')
Abs_75ns=np.loadtxt('C:\\Users\David McKeagney\Desktop\OneDrive_1_24-03-2026\\75ns.csv',dtype=str,delimiter=',')
Abs_95ns=np.loadtxt('C:\\Users\David McKeagney\Desktop\OneDrive_1_24-03-2026\\95ns.csv',dtype=str,delimiter=',')
Abs_55ns=Abs_55ns[1:,:].astype(float)
Abs_75ns=Abs_75ns[1:,:].astype(float)
Abs_95ns=Abs_95ns[1:,:].astype(float)
#%%
#plt.plot(Energy+np.repeat(11,len(Energy)),1/400*Hf_III,label='Hf III')
#plt.plot(Abs_55ns[:,0],Abs_55ns[:,1],label='55ns' )
#plt.plot(Abs_55ns[:,0],Abs_75ns[:,1],label='75ns' )
#plt.plot(Abs_55ns[:,0],Abs_95ns[:,1],label='95ns' )
#plt.plot(Energy,Hf_IV,label='Hf IV')
#plt.plot(Energy,Hf_V,label='Hf V')
plt.plot(Energy,Hf_VI,label='Hf VI')
plt.legend()
plt.xlabel('Energy (eV)')
plt.ylabel('Cross Sections ')