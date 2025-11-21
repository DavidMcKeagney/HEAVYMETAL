# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 11:00:31 2025

@author: David McKeagney
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import function_library_phd as flp
from scipy.optimize import curve_fit
import random

plt.rcParams.update({'font.size': 22})
plt.rcParams["figure.figsize"] = (15,10)
#%%
au_1_5_spec=[]
with open('C:/Users/David McKeagney/Desktop/au.sub.1.5.spec') as file4:
    for lines in file4:
        if len(lines.split())>17:
            au_1_5_spec.append(lines.split())
au_1_5_spec=np.array(au_1_5_spec)[1:,:]
Eric_data_500ns=np.loadtxt('C:/Users/David McKeagney/Downloads/Eric_data_500ns.txt',dtype=float).T
Intensity_500ns=Eric_data_500ns[1][np.logical_and(Eric_data_500ns[0]>=75,Eric_data_500ns[0]<=110)]
Energy=Eric_data_500ns[0][np.logical_and(Eric_data_500ns[0]>=75,Eric_data_500ns[0]<=110)]
#%%
au_1_5_spec_2_4=au_1_5_spec[np.logical_and(au_1_5_spec[:,3]=='2',au_1_5_spec[:,8]=='4')]
gf_2_4=np.exp(au_1_5_spec_2_4[:,15].astype(float))
dE_2_4=au_1_5_spec_2_4[:,11].astype(float)
gamma_2_4_au_1_5=au_1_5_spec_2_4[:,16].astype(float)*1e-3
#%%
def epsilon(x,Er,gamma):
    return (x-Er)*2/gamma
def Fano(x,Er,q,gamma):
     return (q+epsilon(x,Er,gamma))**2/(1+epsilon(x,Er,gamma)**2)
def fitfunc(x,a,b,c,d,e,f,g,h):
     return  Fano(x,a,c,d)*e+Fano(x,b,c,d)*f+g*x+h  
#%%
guess2=[82.5,84.3,17,0.00098,10,0.001,0.1,-0.3]
bounds=([82.4,84.25,16.9,9.74e-4,1e-6,1e-6,-np.inf,-np.inf],[82.6,84.35,17.3,1e-3,np.inf,np.inf,np.inf,np.inf])
#guess2_og=[82.5,84.3,7,0.0019,10,0.001,0.1,-0.3]
#bounds_og=([82.4,84.25,1e-6,1e-6,1e-6,1e-6,-np.inf,-np.inf],[82.6,84.35,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])
popt, pcov=curve_fit(fitfunc, Energy, Intensity_500ns, p0=guess2,bounds=bounds)
best=fitfunc(Energy,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7])
sum=0
for i in range(0, (len(Intensity_500ns)-1)):
    sum=sum+(Intensity_500ns[i]-best[i])**2
plt.plot(Energy,best,color='red')
plt.plot(Energy,Intensity_500ns)
#%%
guess2_og=[82.5,84.3,7,0.0019,10,0.001,0.1,-0.3]
bounds_og=([82.4,84.25,1e-6,1e-6,1e-6,1e-6,-np.inf,-np.inf],[82.6,84.35,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])
plt.plot(Energy,Intensity_500ns,marker='x')
#%%
# It looks like I have convergence on the that the q is approximately 17 what I want to go now is to fit the linewidths independently by fixing the q parameter to 17.14 and making a new fiting parameter of the linewidth 

def fitfunc2(x,a,b,c,d,e,f,g,h):
    return Fano(x,a,17.14,c)*e + Fano(x,b,17.14,d)*f + g*x +h 

guess2=[82.5,84.3,0.00097,0.00098,10,0.001,0.1,-0.3]
bounds=([82.4,84.25,1e-6,1e-6,1e-6,1e-6,-np.inf,-np.inf],[82.6,84.35,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])
#guess2_og=[82.5,84.3,7,0.0019,10,0.001,0.1,-0.3]
#bounds_og=([82.4,84.25,1e-6,1e-6,1e-6,1e-6,-np.inf,-np.inf],[82.6,84.35,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])
popt, pcov=curve_fit(fitfunc, Energy, Intensity_500ns, p0=guess2,bounds=bounds)
best=fitfunc2(Energy,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7])
sum=0
for i in range(0, (len(Intensity_500ns)-1)):
    sum=sum+(Intensity_500ns[i]-best[i])**2
plt.plot(Energy,best,color='red')
plt.plot(Energy,Intensity_500ns)
#%%
def fitfunc3(x,a,b,c,d,e,f):
    return Fano(x,a,b,c)*d + e*x + f

guess3=[82.5,2,0.01,10,-0.1,-0.3]
bounds3=([82.3,1e-6,1e-6,1e-6,-np.inf,-np.inf],[82.6,np.inf,np.inf,np.inf,1e-6,1e-6])

popt, pcov=curve_fit(fitfunc3, Energy, Intensity_500ns, p0=guess3,bounds=bounds3)
best=fitfunc3(Energy,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])
sum=0
for i in range(0, (len(Intensity_500ns)-1)):
    sum=sum+(Intensity_500ns[i]-best[i])**2
plt.plot(Energy,best,color='red')
plt.plot(Energy,Intensity_500ns)
#%%
guess4=[84.3,0.1,0.01,10,-0.1,-0.3]
bounds4=([84.25,1e-6,2e-3,1,-np.inf,-10],[84.35,2.3,1e-2,np.inf,1e-6,1e-6])

popt, pcov=curve_fit(fitfunc3, Energy, Intensity_500ns, p0=guess4,bounds=bounds4)
best=fitfunc3(Energy,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])
sum=0
for i in range(0, (len(Intensity_500ns)-1)):
    sum=sum+(Intensity_500ns[i]-best[i])**2
plt.plot(Energy,best,color='red')
plt.plot(Energy,Intensity_500ns)
#%%
guess5=[79.5,2.0,0.2,2,-0.01,-0.3]
bounds5=([79,1.8,1e-6,-3,-0.02,-2.4],[81.2,2.2,3,3,0.02,2.4])


popt, pcov=curve_fit(fitfunc3, Energy, Intensity_500ns, p0=guess5,bounds=bounds5)
best=fitfunc3(Energy,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])
sum=0
for i in range(0, (len(Intensity_500ns)-1)):
    sum=sum+(Intensity_500ns[i]-best[i])**2
plt.plot(Energy,best,color='red')
plt.plot(Energy,Intensity_500ns)
#%%
Fano_plot1=Fano(Energy, 82.8314+1.4, 2.9, 0.26989)*0.008-0.005*Energy+0.66
Fano_plot2=Fano(Energy, 79.1645+1.4, 2.5, 0.28415)*0.011-0.005*Energy+0.66
Fano_plot3=Fano(Energy, 81.2532+1.4, 2.73, 0.26989)*0.01-0.005*Energy+0.66
#Fano_plot4=Fano(Energy, 81.2532+1.4, 2.73, 0.26989)*0.01+Fano(Energy, 79.1645+1.45, 2.5, 0.28415)*0.011+Fano(Energy, 82.8314+1.45, 2.9, 0.26989)*0.007-0.005*Energy+0.64
#plt.plot(Energy,Fano_plot1)
#plt.plot(Energy,Fano_plot2)
#plt.plot(Energy,Fano_plot3)
#plt.plot(Energy,Fano_plot4)
plt.plot(Energy,Intensity_500ns)
#%%
sigma_file=[]
with open('C:\\Users\David McKeagney\Downloads\Au.I.f.J=2.5-2.5..sigma') as file:
    for lines in file:
        sigma_file.append(lines.split())
sigma_file=sigma_file[4:]
sigma_file=np.array(sigma_file).astype(float)
#%%
plt.plot(sigma_file[:,0]+np.repeat(1.7,len(sigma_file[:,1])),sigma_file[:,1])
plt.plot(Energy,7*Intensity_500ns)
#%%
def epsilon(x,gamma):
    return (x-82.5749)*2/gamma
def Fano(x,q,gamma):
     return (q+epsilon(x,gamma))**2/(1+epsilon(x,gamma)**2)
def fitfunc(x,a,b,c,d):
     return  Fano(x,a,b)*c+d
#%%
#Eric_data_500ns=np.loadtxt('C:/Users/Padmin/Downloads/Eric_data_500ns.txt',dtype=float).T
Eric_data_500ns=np.loadtxt('C:/Users/David McKeagney/Downloads/Eric_data_500ns.txt',dtype=float).T
#Intensity_500ns=Eric_data_500ns[1][np.logical_and(Eric_data_500ns[0]>=82.2,Eric_data_500ns[0]<=83)]
#Energy=Eric_data_500ns[0][np.logical_and(Eric_data_500ns[0]>=82.2,Eric_data_500ns[0]<=83)]
Intensity_500ns=Eric_data_500ns[1][np.logical_and(Eric_data_500ns[0]>=84,Eric_data_500ns[0]<=85.5)]
Energy=Eric_data_500ns[0][np.logical_and(Eric_data_500ns[0]>=84,Eric_data_500ns[0]<=85.5)]
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
Avg_Intensity=MovingAverage(5, Intensity_500ns)
Avg_Energy=MovingAverage(5, Energy)
#%%
Guess_fano=[1.5,0.27,0.005,0.01]
Bounds_fano=([1,0.1,1e-6,1e-6],[4,0.3,0.7,0.8])
popt, pcov=curve_fit(fitfunc, Avg_Energy, Avg_Intensity, p0=Guess_fano,bounds=Bounds_fano)
#%%
plt.plot(Avg_Energy,fitfunc(np.array(Avg_Energy), popt[0], popt[1], popt[2],popt[3]),label='fit')
plt.scatter(Avg_Energy,Avg_Intensity,label='smoothed data')
#Fano_plot1=Fano(Energy, 82.8314+1.4, 2.9, 0.26989)*0.004-0.005*Energy+0.68
#Fano_plot2=Fano(Energy, 79.1645+1.4, 2.5, 0.28415)*0.011-0.005*Energy+0.66
#Fano_plot3=Fano(Energy, 81.2532+1.4, 2.73, 0.26989)*0.005-0.005*Energy+0.685
#plt.plot(Energy,Fano_plot3)
#plt.plot(Energy,Fano_plot1)
plt.legend()
plt.xlabel('Energy [eV]')
plt.ylabel('Intensity')
#%%
def ConfidenceEllipses(popt_vals,var,mean):
    i=0
    for ind,j in enumerate(popt_vals):
        i+=(1/2.111713)*((j-mean[ind])/var[ind])**2
    return i

def Normal(x,mu):
    return (1/(np.sqrt(2*np.pi)*0.05))*np.exp(-((x-mu)/0.05)**2)

def PerturbingData(Energy_vals,Intensity):
    Perturbed_intensity_vals=[]
    Energy_range=np.arange(np.min(Energy),np.max(Energy),0.01)
    for i in Intensity:
        Perturbed_intensity_vals.append(random.gauss(i,0.1))
    Perturbed_intensity_vals=np.array(Perturbed_intensity_vals)
    popt_plus,pcov_plus=curve_fit(fitfunc, Energy_vals, Perturbed_intensity_vals, p0=Guess_fano,bounds=Bounds_fano)
    a=np.array(popt_plus)
    
    return a
def maxparm(var,mean):
    return 2.111713*(var**2)+mean
#%%
ellipse_vals=[]
for i in Intensity_500ns:
    ellipse_vals.append(PerturbingData(Energy, Intensity_500ns))
ellipse_vals=np.array(ellipse_vals)
#%%
var_Fano=np.zeros(5)
mean_Fano=np.zeros(5)
for i in range(0,5):
    var_Fano[i]=np.std(ellipse_vals[:,i])**2
    mean_Fano[i]=np.mean(ellipse_vals[:,i])
#%%
ellipse_size=[]
for i in ellipse_vals:
    ellipse_size.append(ConfidenceEllipses(i, var_Fano,mean_Fano))
#%%
maxval=maxparm(var_Fano,mean_Fano)
#%%
def epsilon2(x,gamma):
    return (x-84.3945)*2/gamma
def Fano2(x,q,gamma):
     return (q+epsilon2(x,gamma))**2/(1+epsilon2(x,gamma)**2)
def fitfunc2(x,a,b,c,d):
    return  Fano2(x,a,b)*c+d

#%%
Guess_fano=[1.5,0.27,0.005,0.01]
Bounds_fano=([1,0.1,1e-6,1e-6],[4,0.3,0.7,0.8])
popt, pcov=curve_fit(fitfunc2, Avg_Energy, Avg_Intensity, p0=Guess_fano,bounds=Bounds_fano)

#%%
plt.plot(Avg_Energy,fitfunc2(np.array(Avg_Energy), popt[0], popt[1], popt[2],popt[3]),label='fit')
plt.scatter(Avg_Energy,Avg_Intensity,label='smoothed data')
#Fano_plot1=Fano(Energy, 82.8314+1.4, 2.9, 0.26989)*0.004-0.005*Energy+0.68
#Fano_plot2=Fano(Energy, 79.1645+1.4, 2.5, 0.28415)*0.011-0.005*Energy+0.66
#Fano_plot3=Fano(Energy, 81.2532+1.4, 2.73, 0.26989)*0.005-0.005*Energy+0.685
#plt.plot(Energy,Fano_plot3)
#plt.plot(Energy,Fano_plot1)
plt.legend()
plt.xlabel('Energy [eV]')
plt.ylabel('Intensity')

