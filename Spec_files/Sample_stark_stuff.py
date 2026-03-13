# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 15:41:31 2026

@author: David McKeagney
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 22})
plt.rcParams["figure.figsize"] = (15,10)
#%%
def Lortentzian(A,x,Gamma):
    return (A/(2*np.pi))*(1/(x**2+(Gamma/(2*np.pi))**2))
def Lortentzian_shifted(A,x,Gamma):
    return (A/(2*np.pi))*(1/((x-0.1)**2+(Gamma/(2*np.pi))**2))
def Gaussian(x,FWHM,mu):
    Sigma=FWHM/(2*np.sqrt(2*np.log(2)))
    return 1/(Sigma*np.sqrt(2*np.pi))*(np.exp(-(x-mu)**2/(2*(Sigma)**2)))
#%%
A=1
A_peturb=1.1
Gamma=1
Gamma_peturb=1.1

xrange=np.arange(-4,4,0.01)
#%%
plt.plot(xrange,Lortentzian(A, xrange, Gamma),label='Unbroadened')
plt.plot(xrange,Lortentzian(A_peturb, xrange, Gamma_peturb),label='Broadened')
plt.legend()
plt.xlabel('Energy (arb. units)')
plt.ylabel('Intensity (arb. units)')
plt.show()
#%%
def Fano(x,Gamma,Er,q):
    return (0.5*q*Gamma + x-Er)**2/((0.5*Gamma)**2 + (x-Er)**2)
#%%
plt.plot(xrange,Fano(xrange,0.5*Gamma,0,0.5),label='q=0.5')
plt.plot(xrange,Fano(xrange,0.5*Gamma,0,1),label='q=1')
plt.plot(xrange,Fano(xrange,0.5*Gamma,0,2),label='q=2')
plt.legend()
plt.xlabel('Energy (arb. units)')
plt.ylabel('Cross section (arb. units)')
plt.show()
#%%
Sample_Fano=Fano(xrange,0.5*Gamma,0,2)
Sample_Gauss=Gaussian(xrange,0.15,0)*(1/(np.sum(Gaussian(xrange,0.15,0))*0.01))
L=len(Sample_Gauss)+len(Sample_Fano)-1
fft_Fano=np.fft.fft(Sample_Fano,L)
fft_Gauss=np.fft.fft(Sample_Gauss,L)
prod_Fano_Gauss=fft_Fano*fft_Gauss
conv_Fano_Gauss=np.fft.ifft(prod_Fano_Gauss).real*0.01
x_conv_min = xrange[0] + xrange[0]
x_conv_max = xrange[-1] + xrange[-1]
x_conv = np.linspace(x_conv_min, x_conv_max, L)
#%%
plt.plot(x_conv,conv_Fano_Gauss)
plt.plot(xrange,Sample_Fano)
plt.xlim(-3.9,3.9)