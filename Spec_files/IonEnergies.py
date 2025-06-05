# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 15:13:30 2025

@author: David McKeagney
"""

import numpy as np
import function_library_phd as flp
#%%
eav='C:/Users/damck/OneDrive/Desktop/au.eav'
config_n_1_6s=['5d86s6de','5d86s7de','5d86s8de']
config_n_1_6s2=['5d76s26de','5d76s27de','5d76s28de']
config_n_6s=['5p65d96s6d','5p65d96s7d','5p65d96s8d']
config_n_6s2=['5d96s26d','5d96s27d','5d96s28d']
config_n_6s2_10=['5p65d106s2']
config_n_1_6s2_10=['5p65d86s2e']
#%%
dE=[]
for i in config_n_1:
    dE.append(flp.IonEnergies(i, config_n, eav))
#%%
def IonEnergies2(c_n_1,c_n,eav):
    eav_file=[]
    with open(eav) as file:
        for lines in file:
            eav_file.append(lines.split())
    dE=[]
    for cn1,cn in zip(c_n_1,c_n):
        for ea in eav_file:
            if ea[-3]==cn1:
                E_1=float(ea[-2])
            elif ea[-3]==cn:
                E_2=float(ea[-2])
        dE.append(abs(E_1-E_2))
    return dE
#%%

dE_6s=IonEnergies2(config_n_1_6s,config_n_6s,eav)
dE_6s2=IonEnergies2(config_n_1_6s2,config_n_6s2,eav)
dE_6s_6s2_mix=IonEnergies2(config_n_1_6s2,config_n_6s,eav)
dE_6s2_10=IonEnergies2(config_n_1_6s2_10,config_n_6s2_10,eav)