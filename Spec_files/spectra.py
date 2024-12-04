# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:14:19 2024

@author: padmin
"""

import numpy as np 
import matplotlib.pyplot as plt
import glob
import function_library_phd as flphd
#%%
paths= [p.replace('\\','/') for p in glob.glob('C:/Users/padmin/Documents/au_*')]
dele=[]
gf=[]
for p in paths:
    dele.append([])
    gf.append([])
    with open(p) as file:
        for f in file:
            if len(f.split())>16:
                if float(f.split()[11])<6:
                    dele[-1].append(float(f.split()[11]))
                    gf[-1].append(np.exp(float(f.split()[15])))
#%%
NISTDATA=np.loadtxt('C:/Users/padmin/Downloads/NIST_ASD_Output_Lines.txt',delimiter='|',dtype=str)
Booleanindex=NISTDATA[:,4] != NISTDATA[2,4]
SPECIFICVALUES=NISTDATA[Booleanindex]
SPECIFICVALUES2=SPECIFICVALUES
#%%
p_specific=paths[30]
spec=[]
with open(p_specific) as file1:
    for f1 in file1:
        if len(f1.split())>16:
            spec.append(f1.split())
#%%
spec=np.array(spec)   
#%%
bool1=np.logical_and(spec[:,3]=='1',spec[:,8]=='1')
spec_6p=spec[bool1]
bool2=np.logical_and(spec_6p[:,2]=='0.5',spec_6p[:,7]=='0.5')
spec_6p_2=spec_6p[bool2]
#%%
SPECIFICVALUES_spec=[]
for specific in SPECIFICVALUES2:
    bool_config=np.logical_and(spec[:,3]==specific[7],spec[:,8]==specific[10])
    temp_config=spec[bool_config]
    j_bool=np.logical_and(temp_config[:,2]==specific[9],temp_config[:,7]==specific[12])
    temp_j=temp_config[j_bool]
    LS_bool=np.logical_and(temp_j[:,5]==specific[8],temp_j[:,10]==specific[11])
    temp_LS=temp_j[LS_bool]
    SPECIFICVALUES_spec.append(temp_LS)
#%%
spe_val_2=[se[0,:] for se in SPECIFICVALUES_spec]
#%%
dele=[float(s[11]) for s in spe_val_2]
gf=[np.exp(float(s[15])) for s in spe_val_2]
#%%
dele_NIST=[float(f[6]) for f in SPECIFICVALUES2]
gf_NIST=[np.exp(float(f[4])) for f in SPECIFICVALUES2]
#%%
plt.vlines(dele[2:], 0, gf[2:], color='blue', label='Theoretical') 
plt.vlines(dele_NIST[2:], 0, gf_NIST[2:], color='red',label='Experimental') 
plt.vlines(dele[0],0,gf[0],color='green',label='T:5d106s-5d96s6p')
plt.vlines(dele_NIST[0],0,gf_NIST[0],color='black',label='E:5d106s-5d96s6p')
plt.vlines(dele[1],0,gf[1],color='orange',label='T:5d96s2-5d96s6p')
plt.vlines(dele_NIST[1],0,gf_NIST[1],color='deeppink',label='E:5d96s2-5d96s6p')
plt.xlabel('delta_E (eV)')
plt.ylabel('gf')
plt.legend()
plt.grid(True)
              
                    
                    
                    