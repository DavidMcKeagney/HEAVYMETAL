# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 18:07:01 2026

@author: padmin
"""

import numpy as np
import matplotlib.pyplot as plt 
#%%
def rho(rho_0, c_0, X, T):
    
    # initialise density array
    R = np.zeros_like(X)

    # admissible region
    mask = (-c_0*T < X) & (X < 3*c_0*T)

    # compute only where valid
    R[mask] = rho_0 * ((3/4 - X[mask]/(4*c_0*T[mask]))**3)

    return R


def T(T_0,rho,rho_0):
    return  T_0*((rho/rho_0)**(2/3)) + 273.15

def saha_factor(g_ii,g_i,T,X_z):
    h = 6.62607015e-34       # J·s
    m_e = 9.109e-31          # kg
    kb = 8.617333262e-5      # eV/K

    # convert k_B*T to Joules for the prefactor
    kb_SI = 1.380649e-23     # J/K

    prefactor = (2*np.pi*m_e*kb_SI*T / h**2)**(3/2)

    return 2 * prefactor * (g_ii/g_i) * np.exp(-X_z/(kb*T))

def Z_bar_lte(T,rho,m_ion,g_ii,g_i,X_z):
    rho_bar=rho/m_ion
    alpha_z=saha_factor(g_ii, g_i,T, X_z)
    y=alpha_z/rho_bar
    return 2*y/(np.sqrt(y**2 +4*y)+y)

def Z_bar_CR(T,rho,S_z,alpha_r,alpha_3b,m_ion):
    rho_bar=rho/m_ion
    B=S_z(T)+alpha_r(T)
    C=4*rho_bar*alpha_3b
    return (-B+np.sqrt(B**2 + C))/(2*rho_bar)

#%%
t=np.arange(0.1,3.1,0.1)
x=np.arange(0.1,3.1,0.1)

X,T_grid=np.meshgrid(x,t)

rho_vals=rho(1.674e-1,1,X,T_grid)
T_vals=T(1,rho_vals,1.674e-1)

X_Au=9.22554*(1.6022e-19)
g_ii_Au=1
g_i_Au=2

X_Pt=8.95883*(1.6022e-19)
g_ii_Pt=6
g_i_Pt=7

X_Ir=8.96702*(1.6022e-19)
g_ii_Ir=11
g_i_Ir=10

alpha_Au=saha_factor(g_ii_Au, g_i_Au, T_vals, X_Au)
alpha_Pt=saha_factor(g_ii_Pt, g_i_Pt, T_vals, X_Pt)
alpha_Ir=saha_factor(g_ii_Ir, g_i_Ir, T_vals, X_Ir)
#%%
Z_bar_Au=Z_bar_lte(T_vals, rho_vals, 1.674e-1, g_ii_Au, g_i_Au, X_Au)
Z_bar_Pt=Z_bar_lte(T_vals, rho_vals, 1.674e-1, g_ii_Pt, g_i_Pt, X_Pt)
Z_bar_Ir=Z_bar_lte(T_vals, rho_vals, 1.674e-1, g_ii_Ir, g_i_Ir, X_Ir)


