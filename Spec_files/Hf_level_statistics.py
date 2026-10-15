# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:27:51 2026

@author: David McKeagney
"""

import numpy as np
import matplotlib.pyplot as plt 
import function_library_phd as flp
from scipy.optimize import curve_fit
from scipy.special import gamma
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
def N_ave_fit(E,a,rho):
    #Need to rescale the energies so that E=E-E_l
    return (2*rho/(a**2))*(np.exp(a*np.sqrt(E))*(a*np.sqrt(E)-1)+1)
def N_ave_fit2(x,a,rho):
    #Need to rescale the energies so that E=E-E_l
    #Scale E=np.sqrt(E)
    return (2*rho/(a**2))*(np.exp(a*x)*(a*x-1)+1)
def Brody_dist(s,q):
    alpha=gamma(q+2/q+1)**(q+1)
    return alpha*(q+1)*(s**q)*np.exp(-alpha*(s**(q+1))) 
def levelspacing_N(N):
    return np.diff(N)    
def l_space_density(s,s_vals):
    num_vals=np.zeros(len(s))
    for a,b in enumerate(s):
        vals_ind=s_vals<=b
        num=len(s_vals[vals_ind])
        num_vals[a]+=num
    return (1/len(s_vals))*num_vals
#%%
Hf_level_info=[]
Hf_lines=[]
with open('C:\\Users\David McKeagney\Desktop\Hf_data\Hf_I\Hf_I.sorted') as file:
    for lines in file:
        Hf_level_info.append(lines.split())
with open('C:\\Users\David McKeagney\Desktop\Hf_data\Hf_I\Hf_I.spec') as file:
    for lines in file:
        if len(lines.split())>16:
            Hf_lines.append(lines.split())
Hf_level_info=Hf_level_info[1:]
Hf_level_info=[a[0:3] for a in Hf_level_info]
Hf_level_info=np.array(Hf_level_info).astype(float)
Hf_level_info[:,0]=Hf_level_info[:,0]+np.repeat(np.abs(Hf_level_info[0,0]), len(Hf_level_info[:,0]))
Hf_lines=np.array(Hf_lines)
#%%
dE_Hf=Hf_lines[:,11].astype(float)
gf_Hf=np.exp(Hf_lines[:,15].astype(float))
#%%
Hf_positive=Hf_level_info[Hf_level_info[:,1]>0]
Hf_negative=Hf_level_info[Hf_level_info[:,1]<0]
#%%
Energy=np.arange(12,30,0.001)
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
plt.xlim(12,22)
plt.ylim(0,800)
plt.legend()
#%%
plt.plot(Energy[:-1],rho_4_neg_slope)
#%%
Hf_negative=Hf_negative[np.logical_and(Hf_negative[:,0]>=20,Hf_negative[:,0]<=22)]
#%%
E_1_neg=Hf_negative[Hf_negative[:,2]==1][:,0]
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
Hf_negative_4=Hf_negative[Hf_negative[:,2]==4]
avg_space_4=(Hf_negative_4[:,0][-1]-Hf_negative_4[:,0][0])/len(Hf_negative_4)
#%%
s=np.arange(0,0.015,0.0001)
#%%
plt.hist(spacing_4_neg,bins=8)
plt.plot(s,Wigner(s, 0.00408))
#%%
plt.scatter(dE_Hf,gf_Hf/dE_Hf)
#plt.xlim(0,0.1)
#%%
fitting_energy=Energy[np.logical_and(Energy>=12,Energy<=20)]
rho_1_neg_int=rho_1_neg[np.logical_and(Energy>=12,Energy<=20)]
#%%
Min_e_val=np.min(E_1_neg)
fitting_energy=fitting_energy-Min_e_val
#%%
plt.plot(fitting_energy[1476:], rho_1_neg_int[1476:])
#%%
popt,cov=curve_fit(N_ave_fit2, np.sqrt(fitting_energy[1476:]), rho_1_neg_int[1476:])
#%%
plt.plot(fitting_energy[1476:],N_ave_fit2(np.sqrt(fitting_energy[1476:]), popt[0], popt[1]),label='fit')
plt.plot(fitting_energy[1476:],rho_1_neg_int[1476:],label='Cowan')
plt.xlabel('E-E_l')
plt.ylabel('N(E)')
#%%
s_vals=levelspacing_N(N_ave_fit2(np.sqrt(fitting_energy[1476:]), popt[0], popt[1]))
#%%
s_range = np.linspace(0, np.max(s_vals), 6524)
#%%
#Remove the repeated ones
Pi_s=l_space_density(s_range, s_vals)
#Pi_s=Pi_s[:928]
#s_range=s_range[:928]
#%%
plt.scatter(s_range,Pi_s)
#plt.xlim(0,3)
#%%
plt.scatter(np.log(s_range),np.log(np.log((1-Pi_s)**(-1))))
plt.plot()
#plt.xlim(-2.5,1)
#%%
plt.hist(s_vals,bins=15)
