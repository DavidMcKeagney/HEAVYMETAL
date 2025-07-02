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

guess2=[dE_2_4[0]+3.2,dE_2_4[1]+3.2,7,0.0019,10,10,.2,-0.3]
popt, pcov=curve_fit(fitfunc, Energy, Intensity_500ns, p0=guess2,)
best=fitfunc(Energy,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7])
sum=0
for i in range(0, (len(Intensity_500ns)-1)):
    sum=sum+(Intensity_500ns[i]-best[i])**2
plt.plot(Energy,best,color='red')
plt.plot(Energy,Intensity_500ns)
#%%
plt.plot(Energy,Intensity_500ns,marker='x')