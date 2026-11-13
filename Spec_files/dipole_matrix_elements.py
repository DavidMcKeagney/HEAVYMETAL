# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:38:41 2026

@author: David McKeagney
"""

import numpy as np
import matplotlib.pyplot as plt
#%%
dip_mat_el=np.loadtxt('C:\\Users\David McKeagney\Downloads\dipole_points.csv',dtype=str,delimiter=',')
dip_mat_el=dip_mat_el[1:,:].astype(float)
#%%
first_channel_energy=dip_mat_el[dip_mat_el[:,1]==26][:,4]
first_channel_dip=dip_mat_el[dip_mat_el[:,1]==26][:,6]

#%%
plt.scatter(first_channel_energy,first_channel_energy*first_channel_dip)
