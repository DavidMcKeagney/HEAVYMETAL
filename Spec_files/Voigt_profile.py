# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 16:25:26 2025

@author: David McKeagney
"""

import numpy as np
import matplotlib.pylot as plt 
#%%
def integrand(x,y,sig,gam):
    return (gam/np.pi*(np.sqrt(2*np.pi)*sig))*np.exp(-y^2/(2*sig**2))*(gam/(gam**2 + (x-y)**2))

def Voigt(x,y,sig,gam,nx,ny):
    y_vals=np.linspace(-y,y,ny)
    dy=2*y/ny
    Voigt_val=0
    i=0
    while i<len(y_vals)-1:
        Voigt_val+=integrand(x, y_vals[i], sig, gam)*dy
        i+=1
    return Voigt_val
#%%

            