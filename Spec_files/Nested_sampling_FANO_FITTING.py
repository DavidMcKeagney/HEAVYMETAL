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
params.add('b',value=1,min=1e-6,max=1.1) # q value
params.add('c',value=0.01,min=1e-6,max=0.01) # linewidth
params.add('d',value=0.06,min=1e-6,max=0.06) # intensity of profile 
params.add('e',value=-0.01,min=-0.02,max=0.02) # continuum slope
params.add('f',value=-0.3,min=-2.4,max=2.4) # continuum constant
#%%
Eric_data_500ns=np.loadtxt('C:/Users/David McKeagney/Downloads/Eric_data_500ns.txt',dtype=float).T
Intensity_500ns=Eric_data_500ns[1][np.logical_and(Eric_data_500ns[0]>=78,Eric_data_500ns[0]<=100)]
Energy=Eric_data_500ns[0][np.logical_and(Eric_data_500ns[0]>=78,Eric_data_500ns[0]<=100)]
#%%
# Computes moving average
def MovingAverage(window_size,array):
    ws=window_size

    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []

    # Loop through the array to consider
    # every window of size 3
    while i < len(array) - ws + 1:
      
        # Store elements from i to i+window_size
        # in list to get the current window
        window = array[i : i + ws]

        # Calculate the average of current window
        window_average = sum(window) / window_size
        
        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)
        
        # Shift window to right by one position
        i += 1
    return moving_averages 
#%%
moving_avg_500ns=np.array(MovingAverage(5, Intensity_500ns))
moving_avg_energy=np.array(MovingAverage(5, Energy))
#%% 
def log_likelihood(theta):
    params_dict = dict(zip(params.keys(), theta)) #retrieves the params dictionary as defined above
    params1 = lmfit.Parameters()
    for key, value in params_dict.items():
        params1.add(key, value=value)
        #print(params1)
    model = fitfunc3(moving_avg_energy, **params1.valuesdict()) #my function is called fit_tot. This is basically the evaluation of your function
    residual = (model - moving_avg_500ns) #/ err_ha #calculates the residuals. flux_ha is the "y" that you're fitting and err_ha are its errors
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
                        nlive=500*ndim, sample = 'rwalk')

# Run the nested sampling

sampler.run_nested(dlogz=0.001, print_progress=True)

# Extract results
dresults = sampler.results

#%%
ind = np.argmax(dresults.logl)
sols = dresults.samples[ind]
#%% 
plt.plot(moving_avg_energy,moving_avg_500ns)
plt.plot(Energy,fitfunc3(Energy,sols[0],sols[1],sols[2],sols[3],sols[4],sols[5]))
#plt.plot(Energy,fitfunc3(Energy,84.31,1.04,0.032,0.065,-0.005,0.66))
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
ind1 = np.argmax(dresults.logl)
sols1 = dresults.samples[ind1]
#%%
plt.plot(Energy,Intensity_500ns)
plt.plot(Energy,fitfunc3(Energy,84.31,1.04,0.03,0.065,sols1[4],0.66))
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
#%% 
# Old parameters for te second fano profile 3
params=lmfit.Parameters()
params.add('a',value=84.25,min=84.2, max=84.35) # resonance energy
params.add('b',value=2,min=-3,max=3) # q value
params.add('c',value=0.01,min=1e-6,max=0.3) # linewidth
params.add('d',value=10,min=1e-6,max=3) # intensity of profile 
params.add('e',value=-0.1,min=-0.02,max=0.02) # continuum slope
params.add('f',value=-0.3,min=-2.4,max=2.4) # continuum constant
#%%
#THE GIGA FIT
def fitfunc4(x,E1,E2,E3,q1,q2,q3,gam1,gam2,gam3,d1,d2,d3,e,f):
    return Fano(x, E1, q1, gam1)*d1+Fano(x, E2, q2, gam2)*d2+Fano(x, E3, q3, gam3)*d3 +e*x+f
#%%
params=lmfit.Parameters()
params.add('E1',value=82.5,min=82.4, max=82.6) # resonance energy1
params.add('q1',value=0.89,min=-2,max=2) # q value1
params.add('gam1',value=0.01,min=1e-6,max=0.7) # linewidth1
params.add('E2',value=84.25,min=84.2, max=84.35) # resonance energy2
params.add('q2',value=1.04,min=-3,max=3) # q value2
params.add('gam2',value=0.01,min=1e-6,max=0.3) # linewidth2
params.add('d1',value=0.1,min=1e-6,max=3) # intensity of profile1
params.add('d2',value=0.1,min=1e-6,max=3) # intensity of profile2
params.add('E3',value=80.1,min=80,max=80.2) #resonance energy3
params.add('q3',value=3,min=-5,max=5) # q value3
params.add('gam3',value=0.01,min=1e-6,max=1) # linewidth3
params.add('d3',value=1,min=1e-6,max=5)# Intesity of profile3 
params.add('e',value=-0.1,min=-0.02,max=0.02) # continuum slope
params.add('f',value=-0.3,min=-2.4,max=2.4) # continuum constant
#%% 
def log_likelihood(theta):
    params_dict = dict(zip(params.keys(), theta)) #retrieves the params dictionary as defined above
    params1 = lmfit.Parameters()
    for key, value in params_dict.items():
        params1.add(key, value=value)
        #print(params1)
    model = fitfunc4(Energy, **params1.valuesdict()) #my function is called fit_tot. This is basically the evaluation of your function
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
plt.plot(Energy,fitfunc4(Energy,sols[0],sols[3],sols[8],sols[1],sols[4],sols[9],sols[2],sols[5],sols[10],sols[6],sols[7],sols[11],sols[12],sols[13]))
#%%
weights = np.exp(dresults['logwt'] - dresults['logz'][-1])  # Compute normalized weights
samples = dynesty.utils.resample_equal(dresults['samples'], weights)  # Resample based 

parameter_names = list(params.keys())
posterior_samples = {name: samples[:, i] for i, name in enumerate(parameter_names)}

corner.corner(samples, labels=parameter_names, show_titles=True, quantiles=[0.16, 0.5, 0.84])
plt.show()
#%%
def fitfunc5(x,q1,q2,gam1,gam2,d1,d2,e,f):
    return Fano(x, 82.49, q1, gam1)*d1+Fano(x, 84.31, q2, gam2)*d2+ +e*x+f
params=lmfit.Parameters()
#params.add('E1',value=82.5,min=82.4, max=82.6) # resonance energy1
params.add('q1',value=0.89,min=0.5,max=1.5) # q value1
params.add('gam1',value=0.01,min=1e-6,max=1) # linewidth1
#params.add('E2',value=84.25,min=84.2, max=84.35) # resonance energy2
params.add('q2',value=1.1,min=1e-6,max=1.5) # q value2
params.add('gam2',value=0.01,min=1e-6,max=1) # linewidth2
params.add('d1',value=0.1,min=1e-6,max=3) # intensity of profile1
params.add('d2',value=0.1,min=1e-6,max=3) # intensity of profile2
params.add('e',value=-0.1,min=-0.02,max=0.02) # continuum slope
params.add('f',value=-0.3,min=-2.4,max=2.4) # continuum constant
#%%
def log_likelihood(theta):
    params_dict = dict(zip(params.keys(), theta)) #retrieves the params dictionary as defined above
    params1 = lmfit.Parameters()
    for key, value in params_dict.items():
        params1.add(key, value=value)
        #print(params1)
    model = fitfunc5(Energy, **params1.valuesdict()) #my function is called fit_tot. This is basically the evaluation of your function
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

sampler.run_nested(dlogz=1e-6, print_progress=True)

# Extract results
dresults = sampler.results
ind = np.argmax(dresults.logl)
sols = dresults.samples[ind]
#%%
plt.plot(Energy,Intensity_500ns,marker='x')
plt.plot(Energy,fitfunc5(Energy,sols[0],sols[2],sols[1],sols[3],sols[4],sols[5],sols[6],sols[7]))
#%%
weights = np.exp(dresults['logwt'] - dresults['logz'][-1])  # Compute normalized weights
samples = dynesty.utils.resample_equal(dresults['samples'], weights)  # Resample based 

parameter_names = list(params.keys())
posterior_samples = {name: samples[:, i] for i, name in enumerate(parameter_names)}

corner.corner(samples, labels=parameter_names, show_titles=True, quantiles=[0.16, 0.5, 0.84])
plt.show()
#%%
weights = np.exp(dresults['logwt'] - dresults['logz'][-1])
samples = dynesty.utils.resample_equal(dresults['samples'], weights)

median_q1 = np.percentile(samples[:,0],50)
median_gam1=np.percentile(samples[:,1], 50)
median_q2=np.percentile(samples[:,2], 50)
median_gam2=np.percentile(samples[:,3], 50)
median_d1=np.percentile(samples[:,4], 50)
median_d2=np.percentile(samples[:,5], 50)
median_e=np.percentile(samples[:,6], 50)
median_f=np.percentile(samples[:,7], 50) # returns the median of the i-th parameter. Can use 16th and 84th precentiles 
# for 1 sigma uncertainty range
#%%
plt.plot(Energy,Intensity_500ns)
plt.plot(Energy,fitfunc5(Energy,median_q1,median_q2,median_gam1,median_gam2,median_d1,median_d2,median_e,median_f))
#%%
def fitfunc6(x,q1,q2,gam1,gam2,d1,d2):
    return Fano(x, 82.49, q1, gam1)*d1+Fano(x,84.25,q2,gam2)*d2#+f
params=lmfit.Parameters()
params.add('q1',value=0.89,min=0.5,max=2) # q value1
params.add('gam1',value=0.01,min=1e-6,max=1) # linewidth1
params.add('q2',value=1.1,min=0.5,max=1.5) # q value2
params.add('gam2',value=0.01,min=1e-6,max=1) # linewidth2
params.add('d1',value=0.1,min=1e-6,max=3) # intensity of profile1
params.add('d2',value=0.1,min=1e-6,max=3) # intensity of profile2
#params.add('f',value=-0.3,min=-2.4,max=2.4) # continuum constant
#%%
def log_likelihood(theta):
    params_dict = dict(zip(params.keys(), theta)) #retrieves the params dictionary as defined above
    params1 = lmfit.Parameters()
    for key, value in params_dict.items():
        params1.add(key, value=value)
        #print(params1)
    model = fitfunc6(Energy, **params1.valuesdict()) #my function is called fit_tot. This is basically the evaluation of your function
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

sampler.run_nested(dlogz=1e-6, print_progress=True)

# Extract results
dresults = sampler.results
ind = np.argmax(dresults.logl)
sols = dresults.samples[ind]
#%%
plt.plot(Energy,Intensity_500ns,marker='x')
plt.plot(Energy,fitfunc6(Energy,sols[0],sols[2],sols[1],sols[3],sols[4],sols[5]))
#%%
weights = np.exp(dresults['logwt'] - dresults['logz'][-1])  # Compute normalized weights
samples = dynesty.utils.resample_equal(dresults['samples'], weights)  # Resample based 

parameter_names = list(params.keys())
posterior_samples = {name: samples[:, i] for i, name in enumerate(parameter_names)}

corner.corner(samples, labels=parameter_names, show_titles=True, quantiles=[0.16, 0.5, 0.84])
plt.show()
#%%
def fitfunc7(x):
    return Fano(x, 82.49, 0.89, 0.054)*0.06 + Fano(x, 84.31, 1.04, 0.032)*0.065 -0.005*x + 0.57

plt.plot(Energy,fitfunc7(Energy))
plt.plot(Energy,Intensity_500ns,label='500ns')
plt.xlabel('Energy [eV]')
plt.ylabel('Absorbance [Arb.]')
plt.grid(True)
plt.legend()
#%%

params=lmfit.Parameters()
params.add('a',value=80.2,min=79.7, max=81.2) # resonance energy
params.add('b',value=2,min=-2.4,max=2.4) # q value
params.add('c',value=0.01,min=1e-6,max=3) # linewidth
params.add('d',value=0.06,min=-3,max=3) # intensity of profile 
params.add('e',value=-0.1,min=-0.02,max=0.02) # continuum slope
params.add('f',value=-0.3,min=-2.4,max=2.4) # continuum constant
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
                        nlive=500*ndim, sample = 'rwalk')

# Run the nested sampling

sampler.run_nested(dlogz=0.001, print_progress=True)

# Extract results
dresults = sampler.results
#%%

ind2 = np.argmax(dresults.logl)
sols2 = dresults.samples[ind2]
#%%
plt.plot(Energy,Intensity_500ns)
plt.plot(Energy,fitfunc3(Energy,sols2[0],sols2[1],sols2[2],sols2[3],sols2[4],sols2[5]))
#%%
weights = np.exp(dresults['logwt'] - dresults['logz'][-1])  # Compute normalized weights
samples = dynesty.utils.resample_equal(dresults['samples'], weights)  # Resample based 

parameter_names = list(params.keys())
posterior_samples = {name: samples[:, i] for i, name in enumerate(parameter_names)}

corner.corner(samples, labels=parameter_names, show_titles=True, quantiles=[0.16, 0.5, 0.84],bins=50)
plt.show()
#%%
def fitfunc8(x):
    return Fano(x,79.81,1.18,0.476)*0.067+Fano(x, 82.49, 0.89, 0.054)*0.055 + Fano(x, 84.31, 1.04, 0.032)*0.06 -0.0055*x + 0.55

plt.plot(Energy,fitfunc8(Energy))
plt.plot(Energy,Intensity_500ns,label='500ns')
plt.xlabel('Energy [eV]')
plt.ylabel('Absorbance [Arb.]')
plt.grid(True)
plt.legend()
#%%

def fitfunc9(x):
    return Fano(x,80,1.9,0.45)*0.03+Fano(x, 82.49, 1.9, 0.01)*0.05 + Fano(x, 84.31, 1.8, 0.05)*0.02 -0.0055*x + 0.64

plt.plot(Energy,fitfunc9(Energy))
plt.plot(Energy,Intensity_500ns,label='500ns')
plt.xlabel('Energy [eV]')
plt.ylabel('Absorbance [Arb.]')
plt.grid(True)
plt.legend()