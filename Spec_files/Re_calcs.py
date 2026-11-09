# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:25:42 2026

@author: David McKeagney
"""

import numpy as np
import function_library_phd as flp
import matplotlib.pyplot as plt
#%%
Re_40ns=np.loadtxt('C:\\Users\David McKeagney\Downloads\\2023_07_18_122.csv', dtype=str,delimiter=',')
Energy_Re_40ns=Re_40ns[1:,0].astype(float)
Abs_Re_40ns=Re_40ns[1:,1].astype(float)
#%%
Hf_IV=[]
Ta_IV=[]
W_IV=[]
Re_IV=[]


with open('C:\\Users\David McKeagney\Downloads\Re_IV.spec') as file:
    for lines in file:
        if len(lines.split())>16:
            Re_IV.append(lines.split())
            
with open('C:\\Users\David McKeagney\Downloads\Hf_IV.spec') as file:
    for lines in file:
        if len(lines.split())>16:
            Hf_IV.append(lines.split())
            
with open('C:\\Users\David McKeagney\Downloads\Ta_IV.spec') as file:
    for lines in file:
        if len(lines.split())>16:
            Ta_IV.append(lines.split())
            

with open('C:\\Users\David McKeagney\Downloads\W_IV.spec') as file:
    for lines in file:
        if len(lines.split())>16:
            W_IV.append(lines.split())
            
Hf_IV=np.array(Hf_IV)
Ta_IV=np.array(Ta_IV)
W_IV=np.array(W_IV)            
Re_IV=np.array(Re_IV)
#%%
Hf_V=[]
Ta_V=[]
W_V=[]
Re_V=[]


with open('C:\\Users\David McKeagney\Downloads\Re_V.spec') as file:
    for lines in file:
        if len(lines.split())>16:
            Re_V.append(lines.split())
            
with open('C:\\Users\David McKeagney\Downloads\Hf_V.spec') as file:
    for lines in file:
        if len(lines.split())>16:
            Hf_V.append(lines.split())
            
with open('C:\\Users\David McKeagney\Downloads\Ta_V.spec') as file:
    for lines in file:
        if len(lines.split())>16:
            Ta_V.append(lines.split())
            

with open('C:\\Users\David McKeagney\Downloads\W_V.spec') as file:
    for lines in file:
        if len(lines.split())>16:
            W_V.append(lines.split())
            
Hf_V=np.array(Hf_V)
Ta_V=np.array(Ta_V)
W_V=np.array(W_V)            
Re_V=np.array(Re_V)

#%%
Hf_IV_np=Hf_IV[Hf_IV[:,8].astype(float)<=7]
Ta_IV_np=Ta_IV[Ta_IV[:,8].astype(float)<=7]
W_IV_np=W_IV[W_IV[:,8].astype(float)<=9]
Re_IV_np=Re_IV[Re_IV[:,8].astype(float)<=12]
#%%
Hf_V_np=Hf_V[Hf_V[:,8].astype(float)<=6]
Ta_V_np=Ta_V[Ta_V[:,8].astype(float)<=7]
W_V_np=W_V[W_V[:,8].astype(float)<=7]
Re_V_np=Re_V[Re_V[:,8].astype(float)<=9]
#%%
#gf_ReIII_np=np.exp(Re_III_np[:,15].astype(float))
gf_ReIV_np=np.exp(Re_IV_np[:,15].astype(float)) 
gf_HfIV_np=np.exp(Hf_IV_np[:,15].astype(float))
gf_TaIV_np=np.exp(Ta_IV_np[:,15].astype(float))
gf_WIV_np=np.exp(W_IV_np[:,15].astype(float)) 
#%%
gf_ReV_np=np.exp(Re_V_np[:,15].astype(float)) 
gf_HfV_np=np.exp(Hf_V_np[:,15].astype(float))
gf_TaV_np=np.exp(Ta_V_np[:,15].astype(float))
gf_WV_np=np.exp(W_V_np[:,15].astype(float))    
#%%
#dE_ReIII_np=Re_III_np[:,11].astype(float)
dE_ReIV_np=Re_IV_np[:,11].astype(float)
dE_HfIV_np=Hf_IV_np[:,11].astype(float)
dE_TaIV_np=Ta_IV_np[:,11].astype(float)
dE_WIV_np=W_IV_np[:,11].astype(float)
#%%
dE_ReV_np=Re_V_np[:,11].astype(float)
dE_HfV_np=Hf_V_np[:,11].astype(float)
dE_TaV_np=Ta_V_np[:,11].astype(float)
dE_WV_np=W_V_np[:,11].astype(float)
#%%
#plt.stem(dE_ReIII_np,gf_ReIII_np,label='Re III')
plt.stem(dE_HfIV_np,gf_HfIV_np,label='Hf IV',linefmt='blue')
plt.stem(dE_TaIV_np,gf_TaIV_np,label='Ta IV',linefmt='green')
plt.stem(dE_WIV_np,gf_WIV_np,label='W IV',linefmt='orange')
plt.stem(dE_ReIV_np,gf_ReIV_np,label='Re IV',linefmt='red')
plt.legend()
plt.xlabel('Energy (eV)') 
#%%
#plt.stem(dE_ReIII_np,gf_ReIII_np,label='Re III')
plt.stem(dE_HfV_np,gf_HfV_np,label='Hf V',linefmt='blue')
plt.stem(dE_TaV_np,gf_TaV_np,label='Ta V',linefmt='green')
plt.stem(dE_WV_np,gf_WV_np,label='W V',linefmt='orange')
plt.stem(dE_ReV_np,gf_ReV_np,label='Re V',linefmt='red')
plt.legend()
plt.xlabel('Energy (eV)')
#%%
plt.plot(Energy_Re_40ns,Abs_Re_40ns,label='exp 40ns')
plt.stem(dE_ReIV_np-np.repeat(4.5,len(dE_ReIV_np)),gf_ReIV_np/(np.max(gf_ReIV_np)),label='Re IV',linefmt='green')
plt.stem(dE_ReV_np-np.repeat(4.5,len(dE_ReV_np)),gf_ReV_np/np.max(gf_ReV_np),label='Re V',linefmt='orange')
plt.xlim(47,70)
plt.legend()
plt.xlabel('Energy (eV)')
plt.ylabel('Abs')
plt.ylim(0,1.7)
#%%

Hf_IV_orb=np.loadtxt('C:\\Users\David McKeagney\Downloads\\5p_HfIV', dtype=float)
Ta_IV_orb=np.loadtxt('C:\\Users\David McKeagney\Downloads\\5p_TaIV', dtype=float)
W_IV_orb=np.loadtxt('C:\\Users\David McKeagney\Downloads\\5p_WIV', dtype=float)
Re_IV_orb=np.loadtxt('C:\\Users\David McKeagney\Downloads\\5p_ReIV', dtype=float) 

Hf_IV_r=Hf_IV_orb[:,0]
Ta_IV_r=Ta_IV_orb[:,0]
W_IV_r=W_IV_orb[:,0]
Re_IV_r=Re_IV_orb[:,0]

Hf_IV_P=Hf_IV_orb[:,3]
Ta_IV_P=Ta_IV_orb[:,3]
W_IV_P=W_IV_orb[:,3]
Re_IV_P=Re_IV_orb[:,3]
#%%
plt.plot(Hf_IV_r,Hf_IV_P**2,label='Hf IV')
plt.plot(Ta_IV_r,Ta_IV_P**2,label='Ta IV')
plt.plot(W_IV_r,W_IV_P**2,label='W IV')
plt.plot(Re_IV_r,Re_IV_P**2,label='Re IV')
plt.legend()
plt.xlabel('r (a_0)')
plt.ylabel('P^2') 
plt.xlim(0,5)
#%%
Hf_V_orb=np.loadtxt('C:\\Users\David McKeagney\Downloads\\5p_HfV', dtype=float)
Ta_V_orb=np.loadtxt('C:\\Users\David McKeagney\Downloads\\5p_TaV', dtype=float)
W_V_orb=np.loadtxt('C:\\Users\David McKeagney\Downloads\\5p_WV', dtype=float)
Re_V_orb=np.loadtxt('C:\\Users\David McKeagney\Downloads\\5p_ReV', dtype=float) 

Hf_V_r=Hf_V_orb[:,0]
Ta_V_r=Ta_V_orb[:,0]
W_V_r=W_V_orb[:,0]
Re_V_r=Re_V_orb[:,0]

Hf_V_P=Hf_V_orb[:,3]
Ta_V_P=Ta_V_orb[:,3]
W_V_P=W_V_orb[:,3]
Re_V_P=Re_V_orb[:,3]
#%%
plt.plot(Hf_V_r,Hf_V_P**2,label='Hf V')
plt.plot(Ta_V_r,Ta_V_P**2,label='Ta V')
plt.plot(W_V_r,W_V_P**2,label='W V')
plt.plot(Re_V_r,Re_V_P**2,label='Re V')
plt.legend()
plt.xlabel('r (a_0)')
plt.ylabel('P^2') 
plt.xlim(0,5)
#%%
Hf_IV_orb_5d=np.loadtxt('C:\\Users\David McKeagney\Downloads\\5d_HfIV', dtype=float)
Ta_IV_orb_5d=np.loadtxt('C:\\Users\David McKeagney\Downloads\\5d_TaIV', dtype=float)
W_IV_orb_5d=np.loadtxt('C:\\Users\David McKeagney\Downloads\\5d_WIV', dtype=float)
Re_IV_orb_5d=np.loadtxt('C:\\Users\David McKeagney\Downloads\\5d_ReIV', dtype=float) 

Hf_IV_r_5d=Hf_IV_orb_5d[:,0][:480]
Ta_IV_r_5d=Ta_IV_orb_5d[:,0][:480]
W_IV_r_5d=W_IV_orb_5d[:,0][:480]
Re_IV_r_5d=Re_IV_orb_5d[:,0][:480]

Hf_IV_P_5d=Hf_IV_orb_5d[:,3][:480]
Ta_IV_P_5d=Ta_IV_orb_5d[:,3][:480]
W_IV_P_5d=W_IV_orb_5d[:,3][:480]
Re_IV_P_5d=Re_IV_orb_5d[:,3][:480]
#%%
plt.plot(Hf_IV_r_5d,Hf_IV_P_5d**2,label='Hf IV')
plt.plot(Ta_IV_r_5d,Ta_IV_P_5d**2,label='Ta IV')
plt.plot(W_IV_r_5d,W_IV_P_5d**2,label='W IV')
plt.plot(Re_IV_r_5d,Re_IV_P_5d**2,label='Re IV')
plt.legend()
plt.xlabel('r (a_0)')
plt.ylabel('P^2') 
plt.xlim(0,5)
#%%
Z=np.array([72,73,74,75])
r_av_HfIV_5p=np.trapz(Hf_IV_r*Hf_IV_P**2,Hf_IV_r)/np.trapz(Hf_IV_P**2,Hf_IV_r)
r_av_TaIV_5p=np.trapz(Ta_IV_r*Ta_IV_P**2,Ta_IV_r)/np.trapz(Ta_IV_P**2,Ta_IV_r)
r_av_WIV_5p=np.trapz(W_IV_r*W_IV_P**2,W_IV_r)/np.trapz(W_IV_P**2,W_IV_r)
r_av_ReIV_5p=np.trapz(Re_IV_r*Re_IV_P**2,Re_IV_r)/np.trapz(Re_IV_P**2,Re_IV_r)

r_av_arr_5p=np.array([r_av_HfIV_5p,r_av_TaIV_5p,r_av_WIV_5p,r_av_ReIV_5p])
#%%
r_av_HfIV_5d=np.trapz(Hf_IV_r_5d*Hf_IV_P_5d**2,Hf_IV_r_5d)/np.trapz(Hf_IV_P_5d**2,Hf_IV_r_5d)
r_av_TaIV_5d=np.trapz(Ta_IV_r_5d*Ta_IV_P_5d**2,Ta_IV_r_5d)/np.trapz(Ta_IV_P_5d**2,Ta_IV_r_5d)
r_av_WIV_5d=np.trapz(W_IV_r_5d*W_IV_P_5d**2,W_IV_r_5d)/np.trapz(W_IV_P_5d**2,W_IV_r_5d)
r_av_ReIV_5d=np.trapz(Re_IV_r_5d*Re_IV_P_5d**2,Re_IV_r_5d)/np.trapz(Re_IV_P_5d**2,Re_IV_r_5d)

r_av_arr_5d=np.array([r_av_HfIV_5d,r_av_TaIV_5d,r_av_WIV_5d,r_av_ReIV_5d])

#%%
plt.scatter(Z,r_av_arr_5p,label='5p')
plt.scatter(Z,r_av_arr_5d,label='5d')
plt.scatter(Z, r_av_arr_5d-r_av_arr_5p,label='\Delta_r')
plt.xlabel('Z')
plt.ylabel('<r> (a_0)')
plt.legend()                                                                                    
#%%
plt.plot(Hf_IV_P_5d,Hf_IV_P,label='Hf IV')
plt.plot(Ta_IV_P_5d,Ta_IV_P,label='Ta IV')
plt.plot(W_IV_P_5d,W_IV_P,label='W IV')
plt.plot(Re_IV_P_5d,Re_IV_P,label='Re IV')
#%%
rad_overlap_Hf_IV=np.trapz(Hf_IV_P*Hf_IV_r_5d*Hf_IV_P_5d,Hf_IV_r_5d)
rad_overlap_Ta_IV=np.trapz(Ta_IV_P*Hf_IV_r_5d*Ta_IV_P_5d,Ta_IV_r_5d)
rad_overlap_W_IV=np.trapz(W_IV_P*W_IV_r_5d*W_IV_P_5d,W_IV_r_5d)
rad_overlap_Re_IV=np.trapz(Re_IV_P*Re_IV_r_5d*Re_IV_P_5d,Re_IV_r_5d)
#%%
rad_over_lap=np.array([rad_overlap_Hf_IV,rad_overlap_Ta_IV,rad_overlap_W_IV,rad_overlap_Re_IV])
#%%
plt.scatter(Z,np.abs(rad_over_lap)**2,label='5p,5d radial overlap')
plt.legend()
plt.xlabel('Z')
plt.ylabel('|<5p|r|5d>|^2')

