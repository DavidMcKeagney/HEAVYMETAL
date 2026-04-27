# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 18:07:01 2026

@author: padmin
"""

import numpy as np
import matplotlib.pyplot as plt 
#%%
def rho(rho_0,c_0,x,t):
    return rho_0*((3/4 - x/(4*c_0*t))**3)

def T(T_0,rho_0,x,t):
    return  T_0*((rho(x,t)/rho_0)**(2/3))

def saha_factor(g_ii,g_i,x,t,X_z):
    h=6.62607015e-34
    m_e=9.109e-31
    kb=1,380649e-23
    return (2/(np.sqrt(h/(2*np.pi*m_e*kb*T(x,t)))**3))*(g_ii/g_i)*np.exp(-X_z/(kb*T(x,t)))

def Z_bar(x,t,m_ion,g_ii,g_i,X_z):
    rho_bar=rho(x,t)/m_ion
    alpha_z=saha_factor(g_ii, g_i, x, t, X_z)
    return ((-alpha_z+np.sqrt(alpha_z**2 +4*rho_bar*alpha_z))/(2*rho_bar))

#%%
    