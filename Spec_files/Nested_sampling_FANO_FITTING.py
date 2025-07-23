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
import dynesty
import corner
#%%
def epsilon(x,Er,gamma):
    return (x-Er)*2/gamma
def Fano(x,Er,q,gamma):
     return (q+epsilon(x,Er,gamma))**2/(1+epsilon(x,Er,gamma)**2)
def fitfunc3(x,a,b,c,d,e,f):
     return Fano(x,a,b,c)*d + e*x + f
#%% New parameters
params=lmfit.Parameters()
params.add('a',value=82.5,min=82.4, max=82.6) # resonance energy
params.add('b',value=2,min=-0.2,max=1.2) # q value
params.add('c',value=0.01,min=1e-6,max=0.3) # linewidth
params.add('d',value=10,min=1e-6,max=3) # intensity of profile 
params.add('e',value=-0.1,min=-0.02,max=0.02) # continuum slope
params.add('f',value=-0.3,min=-2.4,max=2.4) # continuum constant 
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
#%%
weights = np.exp(dresults['logwt'] - dresults['logz'][-1])  # Compute normalized weights
samples = dynesty.utils.resample_equal(dresults['samples'], weights)  # Resample based 

parameter_names = list(params.keys())
posterior_samples = {name: samples[:, i] for i, name in enumerate(parameter_names)}

corner.corner(samples, labels=parameter_names, show_titles=True, quantiles=[0.16, 0.5, 0.84])
plt.show()
#%%
#Old parameters that worked 1
params=lmfit.Parameters()
params.add('a',value=82.5,min=82.4, max=82.6) # resonance energy
params.add('b',value=2,min=1e-6,max=10) # q value
params.add('c',value=0.01,min=1e-6,max=10) # linewidth
params.add('d',value=10,min=1e-6,max=15) # intensity of profile 
params.add('e',value=-0.1,min=-10,max=1e-6) # continuum slope
params.add('f',value=-0.3,min=-10,max=1e-6) # continuum constant 
#%%
# Old parameters that worked 2
params=lmfit.Parameters()
params.add('a',value=82.5,min=82.4, max=82.6) # resonance energy
params.add('b',value=2,min=1e-6,max=1.5) # q value
params.add('c',value=0.01,min=1e-6,max=10) # linewidth
params.add('d',value=10,min=1e-6,max=3) # intensity of profile 
params.add('e',value=-0.1,min=-10,max=1e-6) # continuum slope
params.add('f',value=-0.3,min=-3,max=1e-6) # continuum constant 
#%%
# Old parameters that worked 3
params=lmfit.Parameters()
params.add('a',value=82.5,min=82.4, max=82.6) # resonance energy
params.add('b',value=2,min=-0.2,max=1.2) # q value
params.add('c',value=0.01,min=1e-6,max=0.3) # linewidth
params.add('d',value=10,min=1e-6,max=3) # intensity of profile 
params.add('e',value=-0.1,min=-0.02,max=0.02) # continuum slope
params.add('f',value=-0.3,min=-2.4,max=2.4) # continuum constant 
#%%
#Parameters for the second fano profile
params=lmfit.Parameters()
params.add('a',value=84.25,min=84.2, max=84.35) # resonance energy
params.add('b',value=2,min=-3,max=3) # q value
params.add('c',value=0.01,min=1e-6,max=0.3) # linewidth
params.add('d',value=10,min=1e-6,max=3) # intensity of profile 
params.add('e',value=-0.1,min=-0.02,max=0.02) # continuum slope
params.add('f',value=-0.3,min=-2.4,max=2.4) # continuum constant
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
#%%
weights = np.exp(dresults['logwt'] - dresults['logz'][-1])  # Compute normalized weights
samples = dynesty.utils.resample_equal(dresults['samples'], weights)  # Resample based 

parameter_names = list(params.keys())
posterior_samples = {name: samples[:, i] for i, name in enumerate(parameter_names)}

corner.corner(samples, labels=parameter_names, show_titles=True, quantiles=[0.16, 0.5, 0.84])
plt.show()
#%%
# Old Parameters for the second fano profile 1
params=lmfit.Parameters()
params.add('a',value=84.25,min=84.2, max=84.35) # resonance energy
params.add('b',value=2,min=-1,max=2) # q value
params.add('c',value=0.01,min=1e-6,max=0.3) # linewidth
params.add('d',value=10,min=1e-6,max=3) # intensity of profile 
params.add('e',value=-0.1,min=-0.02,max=0.02) # continuum slope
params.add('f',value=-0.3,min=-2.4,max=2.4) # continuum constant
#%%
# Old parameters for the second fano proflie 2 
params=lmfit.Parameters()
params.add('a',value=84.25,min=84.2, max=84.35) # resonance energy
params.add('b',value=2,min=-1,max=1.8) # q value
params.add('c',value=0.01,min=1e-6,max=0.02) # linewidth
params.add('d',value=10,min=1e-6,max=3) # intensity of profile 
params.add('e',value=-0.1,min=-0.02,max=0.02) # continuum slope
params.add('f',value=-0.3,min=-2.4,max=2.4) # continuum constant
