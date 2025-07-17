# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 14:54:47 2025

@author: David McKeagney
"""

import lmfit 
from lmfit import Model
from scipy.interpolate import interp1d
from scipy import signal
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from dynesty import NestedSampler
#%%
def epsilon(x,Er,gamma):
    return (x-Er)*2/gamma
def Fano(x,Er,q,gamma):
     return (q+epsilon(x,Er,gamma))**2/(1+epsilon(x,Er,gamma)**2)
def fitfunc3(x,a,b,c,d,e,f):
     return Fano(x,a,b,c)*d + e*x + f
#%%
params=lmfit.Parameters()
params.add('a',value=82.5,min=82.4, max=82.6) # resonance energy
params.add('b',value=2,min=1e-6,max=10) # q value
params.add('c',value=0.01,min=1e-6,max=10) # linewidth
params.add('d',value=10,min=1e-6,max=15) # intensity of profile 
params.add('e',value=-0.1,min=-10,max=1e-6) # continuum slope
params.add('f',value=-0.3,min=-10,max=1e-6) # continuum constant 
#%%
Eric_data_500ns=np.loadtxt('C:/Users/David McKeagney/Downloads/Eric_data_500ns.txt',dtype=float).T
Intensity_500ns=Eric_data_500ns[1][np.logical_and(Eric_data_500ns[0]>=75,Eric_data_500ns[0]<=110)]
Energy=Eric_data_500ns[0][np.logical_and(Eric_data_500ns[0]>=75,Eric_data_500ns[0]<=110)]
#%% 
def log_likelihood(theta):
    params_dict = dict(zip(params.keys(), theta)) #retrieves the params dictionary as defined above
    params1 = lmfit.Parameters()
    for key, value in params_dict.items():
        params1.add(key, value=value)
        #print(params1)
    model = fitfunc3(Energy, **params1.valuesdict()) #my function is called fit_tot. This is basically the evaluation of your function
    residual = (model - Intensity_500ns) #/ err_ha #calculates the residuals. flux_ha is the "y" that you're fitting and err_ha are its errors
    return -0.5 * np.sum(residual**2) #calculates the log-likelihood of the fit, evaluated at the parameters you gave. 

def prior_transform(unit_cube):
    """Map unit cube [0, 1] to parameter bounds."""
    priors = []
    for i, param in enumerate(params.values()):
        lower = param.min
        upper = param.max
        priors.append(lower + (upper - lower) * unit_cube[i])  # Creats a Uniform prior based on the min and max values of the parameters
        # I believe this is what you'd change to get a gaussian prior. In the flat priors, with dynesty, the attribute "value" of the params
    # as defined in the above cell is meaningless because it samples the priors randomly. 
    return priors
#%%
ndim = len(params)  # Number of parameters
sampler = NestedSampler(loglikelihood=log_likelihood, 
                        prior_transform=prior_transform, 
                        ndim=len(params), 
                        nlive=80*ndim, sample = 'rwalk')

# Run the nested sampling

sampler.run_nested(dlogz=0.001, print_progress=True)

# Extract results
dresults = sampler.results
#%%
ind = np.argmax(dresults.logl)
sols = dresults.samples[ind]
#%% 
plt.plot(Energy,Intensity_500ns)
plt.plot(Energy,fitfunc3(Energy,sols[0],sols[1],sols[2],sols[3],sols[4],sols[5]))