# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 19:13:50 2025

@author: padmin
"""

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import odeint 
#%%
def pressure(const,rho,gamma):
    return const*rho**(gamma)
def velocity_diff_eq(rho,y,const,gamma):
    dydt=np.sqrt(gamma*const)*(rho**(0.5*(gamma-3)))
    return dydt
def epsilon(u,rho,const,gamma):
    dudp=velocity_diff_eq(rho, const, gamma)
    return u+rho*dudp
y0=0.0
const=1
gamma=4
rho=np.linspace(0.0, 10, 101)
sol=odeint(velocity_diff_eq, y0, rho,args=(const,gamma))
#%%
plt.scatter(rho,sol[:,0],linestyle='dashed',label='Numerical')
plt.plot(rho,(4/3)*np.sqrt((rho**3)),label='analytical',color='red')
plt.xlabel('Density')
plt.ylabel('u(rho)')
plt.legend()
#%%
epsilon_vals=epsilon(sol[:,0], rho, const,gamma)
