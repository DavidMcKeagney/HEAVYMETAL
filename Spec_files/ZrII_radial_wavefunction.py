# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 14:40:39 2026

@author: padmin
"""

import numpy as np
import matplotlib.pyplot as plt 
plt.rcParams.update({'font.size': 22})
plt.rcParams["figure.figsize"] = (15,10)
#%%
WaveFuncTable_4d_5_2=[]
WaveFuncTable_4d_3_2=[]
WaveFuncTable_5s_1_2=[]
WaveFuncTable_4d_5_2_v2=[]
WaveFuncTable_4d_3_2_v2=[]
WaveFuncTable_5s_1_2_v2=[]
with open('C:\\Users\padmin\OneDrive\Desktop\ZrII.wavefunc') as file:
    for lines in file:
        WaveFuncTable_4d_5_2.append(lines.split())
WaveFuncTable_4d_5_2=np.array(WaveFuncTable_4d_5_2[26:]).astype(float)
with open('C:\\Users\padmin\OneDrive\Desktop\ZrII_4d_3_2.wavefunc') as file:
    for lines in file:
        WaveFuncTable_4d_3_2.append(lines.split())
WaveFuncTable_4d_3_2=np.array(WaveFuncTable_4d_3_2[26:]).astype(float)
with open('C:\\Users\padmin\OneDrive\Desktop\ZrII_5s_1_2.wavefunc') as file:
    for lines in file:
        WaveFuncTable_5s_1_2.append(lines.split())
WaveFuncTable_5s_1_2=np.array(WaveFuncTable_5s_1_2[26:]).astype(float)
with open('C:\\Users\padmin\OneDrive\Desktop\ZrII_4d_5_2_v2.wavefunc') as file:
    for lines in file:
        WaveFuncTable_4d_5_2_v2.append(lines.split())
WaveFuncTable_4d_5_2_v2=np.array(WaveFuncTable_4d_5_2_v2[26:]).astype(float)
with open('C:\\Users\padmin\OneDrive\Desktop\ZrII_4d_3_2_v2.wavefunc') as file:
    for lines in file:
        WaveFuncTable_4d_3_2_v2.append(lines.split())
WaveFuncTable_4d_3_2_v2=np.array(WaveFuncTable_4d_3_2_v2[26:]).astype(float)
with open('C:\\Users\padmin\OneDrive\Desktop\ZrII_5s_1_2_v2.wavefunc') as file:
    for lines in file:
        WaveFuncTable_5s_1_2_v2.append(lines.split())
WaveFuncTable_5s_1_2_v2=np.array(WaveFuncTable_5s_1_2_v2[26:]).astype(float)
#%%
r_vals=WaveFuncTable_4d_5_2[:,1]
r_vals_5s=WaveFuncTable_5s_1_2[:,1]
larg_func_4d_5_2=WaveFuncTable_4d_5_2[:,4]
smol_func_4d_5_2=WaveFuncTable_4d_5_2[:,5]
tot_potential_4d_5_2=(1/r_vals)*(WaveFuncTable_4d_5_2[:,2]+WaveFuncTable_4d_5_2[:,3])
larg_func_4d_3_2=WaveFuncTable_4d_3_2[:,4]
smol_func_4d_3_2=WaveFuncTable_4d_3_2[:,5]
tot_potential_4d_3_2=(1/r_vals)*(WaveFuncTable_4d_3_2[:,2]+WaveFuncTable_4d_3_2[:,3])
larg_func_5s_1_2=WaveFuncTable_5s_1_2[:,4]
smol_func_5s_1_2=WaveFuncTable_5s_1_2[:,5]
tot_potential_5s_1_2=(1/r_vals_5s)*(WaveFuncTable_5s_1_2[:,2]+WaveFuncTable_5s_1_2[:,3])
#%%
r_vals_v2=WaveFuncTable_4d_5_2_v2[:,1]
r_vals_5s_v2=WaveFuncTable_5s_1_2_v2[:,1]
larg_func_4d_5_2_v2=WaveFuncTable_4d_5_2_v2[:,4]
smol_func_4d_5_2_v2=WaveFuncTable_4d_5_2_v2[:,5]
tot_potential_4d_5_2_v2=(1/r_vals_v2)*(WaveFuncTable_4d_5_2_v2[:,2]+WaveFuncTable_4d_5_2_v2[:,3])
larg_func_4d_3_2_v2=WaveFuncTable_4d_3_2_v2[:,4]
smol_func_4d_3_2_v2=WaveFuncTable_4d_3_2_v2[:,5]
tot_potential_4d_3_2_v2=(1/r_vals_v2)*(WaveFuncTable_4d_3_2_v2[:,2]+WaveFuncTable_4d_3_2_v2[:,3])
larg_func_5s_1_2_v2=WaveFuncTable_5s_1_2_v2[:,4]
smol_func_5s_1_2_v2=WaveFuncTable_5s_1_2_v2[:,5]
tot_potential_5s_1_2_v2=(1/r_vals_5s_v2)*(WaveFuncTable_5s_1_2_v2[:,2]+WaveFuncTable_5s_1_2_v2[:,3])
#%%
def PseudoLog(x):
    return np.sign(x)*np.log1p(np.abs(x))
PseudoLog_potential_4d_5_2=PseudoLog(tot_potential_4d_5_2)
PseudoLog_potential_4d_3_2=PseudoLog(tot_potential_4d_3_2)
PseudoLog_potential_5s_1_2=PseudoLog(tot_potential_5s_1_2)
PseudoLog_potential_5s_1_2_v2=PseudoLog(tot_potential_5s_1_2_v2)
#plt.yscale('log')
#%%
fig,axes=plt.subplots(2)
#axes[0].plot(r_vals,(1/r_vals)*larg_func_4d_5_2,label='[P(r)/r]_4d_5/2')
#axes[0].plot(r_vals,(1/r_vals)*smol_func_4d_5_2,label='[Q(r)/r]_4d_5/2')
axes[0].plot(r_vals,((1/r_vals)*(smol_func_4d_5_2))**2+((1/r_vals)*larg_func_4d_5_2)**2,label='rho_4d_5_2')
axes[0].plot(r_vals_v2,((1/r_vals_v2)*(smol_func_4d_5_2_v2))**2+((1/r_vals_v2)*larg_func_4d_5_2_v2)**2,label='rho_4d_5_2_v2')
axes[1].plot(r_vals,PseudoLog_potential_4d_5_2,label='v_1')
axes[1].plot(r_vals_5s_v2,PseudoLog_potential_5s_1_2_v2,label='v_2')
#axes[0].plot(r_vals,(1/r_vals)*larg_func_4d_3_2,label='[P(r)/r]_4d_3/2')
#axes[0].plot(r_vals,(1/r_vals)*smol_func_4d_3_2,label='[Q(r)/r]_4d_3/2')
axes[0].plot(r_vals,((1/r_vals)*(smol_func_4d_3_2))**2+((1/r_vals)*larg_func_4d_3_2)**2,label='rho_4d_3_2')
axes[0].plot(r_vals_v2,((1/r_vals_v2)*(smol_func_4d_3_2_v2))**2+((1/r_vals_v2)*larg_func_4d_3_2_v2)**2,label='rho_4d_3_2_v2')
#axes[1].plot(r_vals,PseudoLog_potential_4d_3_2,label='4d_3/2')
#axes[0].plot(r_vals_5s,(1/r_vals_5s)*larg_func_5s_1_2**2,label='[P(r)/r]_5s_1/2')
#axes[0].plot(r_vals_5s,(1/r_vals_5s)*smol_func_5s_1_2**2,label='[Q(r)/r]_5s_1/2')
axes[0].plot(r_vals_5s,((1/r_vals_5s)*(smol_func_5s_1_2))**2+((1/r_vals_5s)*larg_func_5s_1_2)**2,label='rho_5s_1_2')
axes[0].plot(r_vals_5s_v2,((1/r_vals_5s_v2)*(smol_func_5s_1_2_v2))**2+((1/r_vals_5s_v2)*larg_func_5s_1_2_v2)**2,label='rho_5s_1_2_v2')
#axes[1].plot(r_vals_5s,PseudoLog_potential_5s_1_2,label='5s_1/2')
axes[0].set_xlabel('r [a_0]')
axes[1].set_xlabel('r [a_0]')
axes[0].set_ylabel('Probability density')
axes[1].set_ylabel('sign(V(r))*log(1+|V(r)|)')
axes[0].set_ylim(0,70)
for ax in axes.flat:
    ax.legend()
#%%
#plt.plot(r_vals_5s,larg_func_5s_1_2*tot_potential_5s_1_2)
plt.plot(r_vals_5s,larg_func_5s_1_2)
plt.plot(r_vals_5s,r_vals_5s)
#plt.xlim(0,0.01)    
#%%
plt.plot(r_vals,((1/r_vals)*(smol_func_4d_5_2))**2+((1/r_vals)*larg_func_4d_5_2)**2,label='4d_5_2')
plt.plot(r_vals,((1/r_vals)*(smol_func_4d_3_2))**2+((1/r_vals)*larg_func_4d_3_2)**2,label='4d_3_2')
plt.plot(r_vals_5s,((1/r_vals_5s)*(smol_func_5s_1_2))**2+((1/r_vals_5s)*larg_func_5s_1_2)**2,label='5s_1_2')
plt.xlabel('r (a_0)')
plt.ylabel('Probability Density')
plt.legend()
plt.ylim(0,70)
#%%
plt.plot(r_vals_5s,PseudoLog_potential_5s_1_2,label='4d3')
plt.plot(r_vals_5s_v2,PseudoLog_potential_5s_1_2_v2,label='4d2 5s1, 4d3, 4d2 5p1')
plt.legend()
plt.xlim(-0.02,5)
plt.xlabel('r (a_0)')
plt.ylabel('sign(V(r))*log(1+|V(r)|)')