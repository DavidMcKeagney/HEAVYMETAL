# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 12:00:54 2026

@author: padmin
"""

import numpy as np
import matplotlib.pyplot as plt
#%%
Hf_5d=np.loadtxt('C:\\Users\padmin\Downloads\\5d_HfIV', dtype=float)[:480]
Hf_5p=np.loadtxt('C:\\Users\padmin\Downloads\\5p_HfIV', dtype=float)

Ta_5d=np.loadtxt('C:\\Users\padmin\Downloads\\5d_TaIV', dtype=float)[:480]
Ta_5p=np.loadtxt('C:\\Users\padmin\Downloads\\5p_TaIV', dtype=float)

W_5d=np.loadtxt('C:\\Users\padmin\Downloads\\5d_WIV', dtype=float)[:480]
W_5p=np.loadtxt('C:\\Users\padmin\Downloads\\5p_WIV', dtype=float)

Re_5d=np.loadtxt('C:\\Users\padmin\Downloads\\5d_ReIV', dtype=float)[:480]
Re_5p=np.loadtxt('C:\\Users\padmin\Downloads\\5p_ReIV', dtype=float)

#%%
Hf_5d_orb=Hf_5d[:,3]
Hf_5d_r=Hf_5d[:,0]

Hf_5p_orb=Hf_5p[:,3]
Hf_5p_r=Hf_5p[:,0]
#One body operators

r_av_1_5d=np.trapz((Hf_5d_orb**2)/Hf_5d_r)/np.trapz(Hf_5d_orb**2)
r_av_2_5d=np.trapz((Hf_5d_orb/Hf_5d_r)**2)/np.trapz(Hf_5d_orb**2)

r_av_1_5p=np.trapz((Hf_5p_orb**2)/Hf_5p_r)/np.trapz(Hf_5p_orb**2)
r_av_2_5p=np.trapz((Hf_5p_orb/Hf_5p_r)**2)/np.trapz(Hf_5p_orb**2)

der_5d=np.gradient(Hf_5d_orb, Hf_5d_r)
der_5d=np.gradient(der_5d, Hf_5d_r)

der_5p=np.gradient(Hf_5p_orb, Hf_5p_r)
der_5p=np.gradient(der_5p, Hf_5p_r)

der_av_5d=np.trapz(Hf_5d_orb*der_5d,Hf_5d_r)/np.trapz(Hf_5d_orb**2)
der_av_5p=np.trapz(Hf_5p_orb*der_5p,Hf_5p_r)/np.trapz(Hf_5p_orb**2)

#%%
E_5d_Hf= -der_av_5d - 2*72*r_av_1_5d + 2*(2+1)*r_av_2_5d
E_5p_Hf= -der_av_5p - 2*72*r_av_1_5p + 1*(1+1)*r_av_2_5p 
#%%
Ta_5d_orb=Ta_5d[:,3]
Ta_5d_r=Ta_5d[:,0]

Ta_5p_orb=Ta_5p[:,3]
Ta_5p_r=Ta_5p[:,0]
#One body operators

r_av_1_5d=np.trapz((Ta_5d_orb**2)/Ta_5d_r)/np.trapz(Ta_5d_orb**2)
r_av_2_5d=np.trapz((Ta_5d_orb/Ta_5d_r)**2)/np.trapz(Ta_5d_orb**2)

r_av_1_5p=np.trapz((Ta_5p_orb**2)/Ta_5p_r)/np.trapz(Ta_5p_orb**2)
r_av_2_5p=np.trapz((Ta_5p_orb/Ta_5p_r)**2)/np.trapz(Ta_5p_orb**2)

der_5d=np.gradient(Ta_5d_orb, Ta_5d_r)
der_5d=np.gradient(der_5d, Ta_5d_r)

der_5p=np.gradient(Ta_5p_orb, Ta_5p_r)
der_5p=np.gradient(der_5p, Ta_5p_r)

der_av_5d=np.trapz(Ta_5d_orb*der_5d,Ta_5d_r)/np.trapz(Ta_5d_orb**2)
der_av_5p=np.trapz(Ta_5p_orb*der_5p,Ta_5p_r)/np.trapz(Ta_5p_orb**2)

#%%
E_5d_Ta= -der_av_5d - 2*73*r_av_1_5d + 2*(2+1)*r_av_2_5d
E_5p_Ta= -der_av_5p - 2*73*r_av_1_5p + 1*(1+1)*r_av_2_5p
#%%
W_5d_orb=W_5d[:,3]
W_5d_r=W_5d[:,0]

W_5p_orb=W_5p[:,3]
W_5p_r=W_5p[:,0]
#One body operators

r_av_1_5d=np.trapz((W_5d_orb**2)/W_5d_r)/np.trapz(W_5d_orb**2)
r_av_2_5d=np.trapz((W_5d_orb/W_5d_r)**2)/np.trapz(W_5d_orb**2)

r_av_1_5p=np.trapz((W_5p_orb**2)/W_5p_r)/np.trapz(W_5p_orb**2)
r_av_2_5p=np.trapz((W_5p_orb/W_5p_r)**2)/np.trapz(W_5p_orb**2)

der_5d=np.gradient(W_5d_orb, W_5d_r)
der_5d=np.gradient(der_5d, W_5d_r)

der_5p=np.gradient(W_5p_orb, W_5p_r)
der_5p=np.gradient(der_5p, W_5p_r)

der_av_5d=np.trapz(W_5d_orb*der_5d,W_5d_r)/np.trapz(W_5d_orb**2)
der_av_5p=np.trapz(W_5p_orb*der_5p,W_5p_r)/np.trapz(W_5p_orb**2)

#%%
E_5d_W= -der_av_5d - 2*74*r_av_1_5d + 2*(2+1)*r_av_2_5d
E_5p_W= -der_av_5p - 2*74*r_av_1_5p + 1*(1+1)*r_av_2_5p
#%%
Re_5d_orb=Re_5d[:,3]
Re_5d_r=Re_5d[:,0]

Re_5p_orb=Re_5p[:,3]
Re_5p_r=Re_5p[:,0]
#One body operators

r_av_1_5d=np.trapz((Re_5d_orb**2)/Re_5d_r)/np.trapz(Re_5d_orb**2)
r_av_2_5d=np.trapz((Re_5d_orb/Re_5d_r)**2)/np.trapz(Re_5d_orb**2)

r_av_1_5p=np.trapz((Re_5p_orb**2)/Re_5p_r)/np.trapz(Re_5p_orb**2)
r_av_2_5p=np.trapz((Re_5p_orb/Re_5p_r)**2)/np.trapz(Re_5p_orb**2)

der_5d=np.gradient(Re_5d_orb, Re_5d_r)
der_5d=np.gradient(der_5d, Re_5d_r)

der_5p=np.gradient(Re_5p_orb, Re_5p_r)
der_5p=np.gradient(der_5p, Re_5p_r)

der_av_5d=np.trapz(Re_5d_orb*der_5d,Re_5d_r)/np.trapz(Re_5d_orb**2)
der_av_5p=np.trapz(Re_5p_orb*der_5p,Re_5p_r)/np.trapz(Re_5p_orb**2)

#%%
E_5d_Re= -der_av_5d - 2*75*r_av_1_5d + 2*(2+1)*r_av_2_5d
E_5p_Re= -der_av_5p - 2*75*r_av_1_5p + 1*(1+1)*r_av_2_5p
#%%
E_5p=[E_5p_Hf,E_5p_Ta,E_5p_W,E_5p_Re]
E_5d=[E_5d_Hf,E_5d_Ta,E_5d_W,E_5d_Re]
Z=[72, 73, 74, 75]
#%%
#plt.plot(Z,E_5p,label='5p')
#plt.plot(Z,E_5d,label='5d')
plt.plot(Z,np.array(E_5d)-np.array(E_5p),label='E_5d - E_5p')
plt.legend()
plt.xlabel('Z')
plt.ylabel('1-body energy (cm^-1)') 
#%%
