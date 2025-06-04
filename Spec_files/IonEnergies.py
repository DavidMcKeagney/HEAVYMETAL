# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 15:13:30 2025

@author: David McKeagney
"""

import numpy as np
import function_library_phd as flp
#%%
eav='C:/Users/David McKeagney/Desktop/au.eav'
config_n_1=['5p65d10e','5p65d96de','5p65d97de','5p65d98de','5p65d96se','5d86s6de','5d86s7de','5d86s8de','5p65d86s2e','5d76s26de','5d76s27de','5d76s28de']
config_n=['5p65d96s6d','5p65d96s7d','5p65d96s8d','5p65d106s2','5d96s26d','5d96s27d','5d96s28d']
#%%
dE=[]
for i in config_n_1:
    dE.append(flp.IonEnergies(i, config_n, eav))

