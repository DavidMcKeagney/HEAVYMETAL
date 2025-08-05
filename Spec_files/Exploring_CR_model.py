# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 06:53:46 2025

@author: padmin
"""

from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt 
#%%

def N_0(gamma,D,I,N_0J):
    return D*gamma*N_0J/(1+gamma*I)
def f(t,y,D,I):
    return [-D*y[0]+I*y[1],D*y[0]-I*y[1]]

t1=10 #the maximum time value the integrator calculates
dt=0.1 #step size 

y0,t0=[N_0(0.2,1,1,1),1],0

r=ode(f,jac=None).set_integrator('lsoda',method='adams')
r.set_initial_value(y0,t0).set_f_params(1,1)


#%%
Ni=[]
Nj=[]
t=[]
while r.successful() and r.t<t1:
    Ni.append(r.integrate(r.t+dt)[0])
    Nj.append(r.integrate(r.t+dt)[1])
    t.append(r.t+dt)
#%%
plt.plot(t,Ni)
plt.plot(t,Nj)
