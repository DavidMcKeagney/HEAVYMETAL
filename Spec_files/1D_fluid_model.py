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
def velocity_diff_eq(y,rho,const,gamma):
    dydt=np.sqrt(gamma*const)*(rho**(0.5*(gamma-3)))
    return dydt
def epsilon(y,u,rho,const,gamma):
    dudp=velocity_diff_eq(y,rho, const, gamma)
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
epsilon_vals=epsilon(y0,sol[:,0], rho, const,gamma)
plt.scatter(epsilon_vals,sol[:,0],label='numerical')
plt.plot(epsilon_vals,0.4*epsilon_vals,label='analytical',color='red')
plt.xlabel('epsilon')
plt.ylabel('velocity')
plt.legend()
#%%
x_vals_t_2=2*epsilon_vals
x_vals_t_10=10*epsilon_vals
plt.plot(x_vals_t_2,sol[:,0],label='t=2')
plt.plot(x_vals_t_10,sol[:,0],label='t=10')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
