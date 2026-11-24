# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:38:41 2026

@author: David McKeagney
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams.update({'font.size': 22})
plt.rcParams["figure.figsize"] = (15,10) 
#%%
def fitfunc(E,a,b,c):
    return a*np.exp(-b*(E)) + c
def logfunc(x,a,b):
    return a*x + b

#%%
dip_mat_el=np.loadtxt('C:\\Users\David McKeagney\Downloads\dipole_points.csv',dtype=str,delimiter=',')
dip_mat_el=dip_mat_el[1:,:].astype(float)
#%%
first_channel_energy=dip_mat_el[dip_mat_el[:,1]==1][:,4]
first_channel_dip=dip_mat_el[dip_mat_el[:,1]==1][:,6]
second_channel_energy=dip_mat_el[dip_mat_el[:,1]==2][:,4]
second_channel_dip=dip_mat_el[dip_mat_el[:,1]==2][:,6]
                        
#%%
log_points_x=np.log(first_channel_energy)
log_points_y=np.log(first_channel_dip**2)

log_points_y=log_points_y[np.logical_and(log_points_x<=7,log_points_x>=5)]
log_points_x=log_points_x[np.logical_and(log_points_x<=7,log_points_x>=5)]


#%%
popt, cov = curve_fit(fitfunc, first_channel_energy, first_channel_dip**2)
pop_log, cov_log= curve_fit(logfunc, log_points_x, log_points_y)
#%%
plt.scatter(first_channel_energy,(second_channel_dip/first_channel_dip)**2)
#plt.scatter(second_channel_energy,(second_channel_dip)**2)
#plt.plot(log_points_x,logfunc(log_points_x,popt[0],popt[1]))
plt.xlabel('Energy (eV)')
plt.ylabel('Cross section Ratio')
#plt.xscale('log')
#plt.yscale('log')
#plt.ylim(1e2,1.5e2)
#plt.xlim(5,7)
#plt.ylim(-6,-2.5)
#plt.ylim(-1e-4,1e-4)

