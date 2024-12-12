# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:14:19 2024

@author: padmin
"""

import numpy as np 
import matplotlib.pyplot as plt
import glob
import csv
import function_library_phd as flp
#%%
csv_NIST=[]
with open('C:/Users/David McKeagney/Downloads/NIST_gold_data.csv') as csvfile:
    file= csv.reader(csvfile,delimiter=',')
    for f in file:
        csv_NIST.append(f[4:14])

#%%
new_path= [p.replace('\\', '/') for p in glob.glob('C:/Users/David*/Desktop/Temp_spec*/*')]
config_even1=flp.configs_even(new_path[1])
config_odd1=flp.configs_odd(new_path[1])
NIST_VALS=flp.NISTformat(csv_NIST, config_even1, config_odd1)
#%%
Energy_eig_vals_low=[flp.findenergysorted(new_path[0], nv[5], nv[6], nv[4]) for nv in NIST_VALS]
Energy_eig_vals_high=[flp.findenergysorted(new_path[0], nv[8], nv[9], nv[7]) for nv in NIST_VALS]
#%%
jj_path=glob.glob('C:/Users/David*/Desktop/au_spin_orbit*')[0].replace('\\','/')
sorted_jj=[]
sorted_energy_jj=[]
with open(jj_path) as jj_file:
    for jj in jj_file:
        if len(jj.split())>7:
            if len(jj.split())==33:
                sorted_jj.append(jj.split())
            else:
                sorted_jj.append(jj.split()+['','','','',''])
sorted_jj=np.array(sorted_jj)
for eh in Energy_eig_vals_high:
    if len(eh)>1:
        boolean_jj=np.logical_or(sorted_jj[:,0]==eh[0,0],sorted_jj[:,0]==eh[1,0])
        sorted_energy_jj.append(sorted_jj[boolean_jj])

            
#%%
spec_info=[]
for el,eh in zip(Energy_eig_vals_low,Energy_eig_vals_high):
    if len(eh)==1:        
        if int(el[0,1])<0:
            spec_info.append(flp.findspec(new_path[1],eh[0,0],el[0,0]))
        else:
            spec_info.append(flp.findspec(new_path[1], el[0,0], eh[0,0]))
    else:
        spec_info.append(flp.findspec(new_path[1], el[0,0], eh[1,0]))
#%%
dE_Cowan=[]
gf_Cowan=[]
for si in spec_info:
    dE_Cowan.append(float(si[0,11]))
    gf_Cowan.append(np.exp(float(si[0,15])))
dE_NIST=NIST_VALS[:,3].astype(float)-NIST_VALS[:,2].astype(float)
gf_NIST=np.exp(NIST_VALS[:,0].astype(float))
#%%
config_13_trans=NIST_VALS[NIST_VALS[:,4]=='13']
E_13_high_vals=[flp.findenergysorted(new_path[0], nv[8], nv[9], nv[7]) for nv in config_13_trans]
E_13_low_vals=[flp.findenergysorted(new_path[0], nv[5], nv[6], nv[4]) for nv in config_13_trans]
#%%
spec_info_13=[]
for el,eh in zip(E_13_low_vals,E_13_high_vals):
    if len(eh)==1:        
        if int(el[0,1])<0:
            spec_info_13.append(flp.findspec(new_path[1],eh[0,0],el[0,0]))
        else:
            spec_info_13.append(flp.findspec(new_path[1], el[0,0], eh[0,0]))
    else:
        spec_info_13.append(flp.findspec(new_path[1], el[0,0], eh[1,0]))
dE_Cowan_13=[]
gf_Cowan_13=[]
for si in spec_info_13:
    dE_Cowan_13.append(float(si[0,11]))
    gf_Cowan_13.append(np.exp(float(si[0,15])))
dE_NIST_13=config_13_trans[:,3].astype(float)-config_13_trans[:,2].astype(float)
gf_NIST_13=np.exp(config_13_trans[:,0].astype(float))
#%%
dE_Cowan_13_2_5=dE_Cowan_13[0:4]
dE_Cowan_13_1_5=dE_Cowan_13[4:]
gf_Cowan_13_2_5=gf_Cowan_13[0:4]
gf_Cowan_13_1_5=gf_Cowan_13[4:]
dE_NIST_13_2_5=dE_NIST_13[0:4]
dE_NIST_13_1_5=dE_NIST_13[4:]
gf_NIST_13_2_5=gf_NIST_13[0:4]
gf_NIST_13_1_5=gf_NIST_13[4:]
#%%
plt.stem(dE_Cowan_13_2_5,gf_Cowan_13_2_5,linefmt='green',label='Cowan')
plt.stem(dE_NIST_13_2_5,gf_NIST_13_2_5,linefmt='blue',label='NIST')
plt.xlabel('dE')
plt.ylabel('gf')
plt.legend()
plt.grid(True)
plt.show()
#%%
plt.stem(dE_Cowan_13_1_5,gf_Cowan_13_1_5,linefmt='green',label='Cowan')
plt.stem(dE_NIST_13_1_5,gf_NIST_13_1_5,linefmt='blue',label='NIST')
plt.xlabel('dE')
plt.ylabel('gf')
plt.legend()
plt.grid(True)
plt.show()
#%%
ddE_13_2_5=np.mean(dE_NIST_13_2_5[1:]-np.array(dE_Cowan_13_2_5)[1:])
dE_Cowan_13_2_5_Shift=[de+ddE_13_2_5 for de in dE_Cowan_13_2_5]
ddE_13_1_5=np.mean(dE_NIST_13_1_5[1:]-np.array(dE_Cowan_13_1_5)[1:])
dE_Cowan_13_1_5_Shift=[de+ddE_13_1_5 for de in dE_Cowan_13_1_5]	
#%%
plt.stem(dE_Cowan_13_2_5_Shift,gf_Cowan_13_2_5,linefmt='green',label='Cowan')
plt.stem(dE_NIST_13_2_5,gf_NIST_13_2_5,linefmt='blue',label='NIST')
plt.xlabel('dE')
plt.ylabel('gf')
plt.legend()
plt.grid(True)
plt.show()
#%%
plt.stem(dE_Cowan_13_1_5_Shift,gf_Cowan_13_1_5,linefmt='green',label='Cowan')
plt.stem(dE_NIST_13_1_5,gf_NIST_13_1_5,linefmt='blue',label='NIST')
plt.xlabel('dE')
plt.ylabel('gf')
plt.legend()
plt.grid(True)
plt.show()