# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:17:37 2024

@author: G17-2
"""

import numpy as np 
import matplotlib.pyplot as plt
#%%
def spin_value(x):
    =str(x)
    globals()[global_var]=[]
    for i in np.arange(0,5,0.5):
        global_var.append(x*i)
    return global_var
#%%
au_spin_orbit= []
with open("C:/Users/G17-2/Desktop/au_spin_orbit.out36","r") as text_file:
    for row in text_file:
        au_spin_orbit.append(row)
spin_orbit_split=au_spin_orbit[52].split()
spin_orbit_split=spin_orbit_split[2:6]
#%%
