# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 16:25:26 2025

@author: David McKeagney
"""

import numpy as np
import matplotlib.pyplot as plt 
from scipy.special import voigt_profile

#%%
def Voigt(A,x,x_0,sig,gam):
    Conv_Evals=np.zeros((len(x),len(x_0)))
    i=0

    while i< len(x_0):
        j=0
        conv= A[i]*voigt_profile(x-x_0[i],sig,gam) 
        while j<len(x):
            Conv_Evals[j][i]+=conv[j]
            j+=1
        i+=1
    Conv_Evals=np.sum(Conv_Evals,axis=1)
    return Conv_Evals
#%%
x_vals=np.linspace(-5,15,500)
x_0=np.array([5,5.1,5.2])
gf=np.array([0.01,2,0.5])
V=Voigt(gf,x_vals, x_0, 1, 1)
#%%
plt.plot(x_vals,V)