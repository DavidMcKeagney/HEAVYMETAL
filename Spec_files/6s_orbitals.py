# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 16:43:16 2026

@author: David McKeagney
"""

import numpy as np
import matplotlib.pyplot as plt
#%%
Au_6s=np.loadtxt('C:\\Users\David McKeagney\Downloads\\6s_Au', dtype=float)
Pt_6s=np.loadtxt('C:\\Users\David McKeagney\Downloads\\6s_Pt', dtype=float)
Ir_6s=np.loadtxt('C:\\Users\David McKeagney\Downloads\\6s_Ir', dtype=float)
#%%
Au_r=Au_6s[:,0]
Pt_r=Pt_6s[:,0]
Ir_r=Ir_6s[:,0]

Au_P=Au_6s[:,3]
Pt_P=Pt_6s[:,3]
Ir_P=Ir_6s[:,3]
#%%
Au_av_r=np.trapz(Au_r*(Au_P**2),x=Au_r)/(np.trapz((Au_P**2), x=Au_r))
Pt_av_r=np.trapz(Pt_r*(Pt_P**2),x=Pt_r)/(np.trapz((Pt_P**2), x=Pt_r))
Ir_av_r=np.trapz(Ir_r*(Ir_P**2),x=Ir_r)/(np.trapz((Ir_P**2), x=Ir_r))
#%%
plt.plot(Au_r,Au_P**2,label='Au I')
plt.plot(Pt_r,Pt_P**2,label='Pt I')
plt.plot(Ir_r,Ir_P**2,label='Ir I')
plt.vlines(Ir_av_r,0,0.7,color='green',linestyles='dashed')
plt.vlines(Au_av_r,0,0.7,color='blue',linestyles='dashed')
plt.vlines(Pt_av_r,0,0.7,color='orange',linestyles='dashed')
plt.ylabel('6s |P_nl(r)|^2')
plt.xlabel('r (Atomic units)')
plt.legend()
plt.xlim(0,10)

#%%
Au_6s_e=np.loadtxt('C:\\Users\David McKeagney\Downloads\\6s_Au_e', dtype=float)
Pt_6s_e=np.loadtxt('C:\\Users\David McKeagney\Downloads\\6s_Pt_e', dtype=float)
#%%
Au_r_e=Au_6s_e[:,0]
Pt_r_e=Pt_6s_e[:,0]

Au_P_e=Au_6s_e[:,3]
Pt_P_e=Pt_6s_e[:,3]
#%%
Au_av_r_e=np.trapz(Au_r_e*(Au_P_e**2),x=Au_r_e)/(np.trapz((Au_P_e**2), x=Au_r_e))
Pt_av_r_e=np.trapz(Pt_r_e*(Pt_P_e**2),x=Pt_r_e)/(np.trapz((Pt_P_e**2), x=Pt_r_e))
#%%
plt.plot(Au_r_e,Au_P_e**2,label='Au I')
plt.plot(Pt_r_e,Pt_P_e**2,label='Pt I')
plt.plot(Ir_r,Ir_P**2,label='Ir I')
plt.vlines(Ir_av_r,0,0.7,color='green',linestyles='dashed')
plt.vlines(Au_av_r_e,0,0.7,color='blue',linestyles='dashed')
plt.vlines(Pt_av_r_e,0,0.7,color='orange',linestyles='dashed')
plt.ylabel('6s |P_nl(r)|^2')
plt.xlabel('r (Atomic units)')
plt.legend()
plt.xlim(0,10)