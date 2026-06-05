# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:47:37 2026

@author: David McKeagney
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
#%%
def numerov(x_0,x_max,V,dx,l,Z):
    #Need to define V in another function
    #Exterior points excluded to account for boundary conditions
    #We add the numerov corrections to the Hamiltonian manually by adding an if statement when constructing the Hamiltonian
    x_points=np.arange(x_0,x_max+dx,dx)
    x_in=x_points[1:-1]
    n=len(x_in)
    W=V(x_in,l,Z)
    Wmat=np.diag(W)
    
    H=np.zeros((n,n))
    B=np.zeros((n,n))
    
    for i in range(n):
        H[i,i]=-2
        B[i,i]=10
        if i > 0:
            H[i, i-1] = 1
            B[i, i-1] = 1
        if i < n-1:
            H[i, i+1] = 1
            B[i, i+1] = 1
        if i==1:
            H[i,i] += -(1/12)*(Z/dx)
       
    B=1/12*B
    H=-(1/(2*(dx**2)))*H
    A=H+np.matmul(B,Wmat)
    #H_new=np.matmul(LA.inv(B),A)
    eigvalues,eigvectors=LA.eigh((A , B))
    return x_points,x_in,eigvalues,(1/np.sqrt(dx))*eigvectors
def V(x,l,Z):
    return -Z/x + l*(l+1)/(x**2)
#%%
sol=numerov( 0, 5, V, 0.001, 0, 0)
#%% TESTING NO POTENTIAL SOLUTION: IT WORKS
x_vals=np.arange(0,5+0.001,0.001)[1:-1]
analytical=np.sin(x_vals*(np.pi/5))
plt.plot(sol[1],sol[3][0][:,0],label='numerical')
plt.plot(x_vals,np.sqrt(1/np.trapz(analytical**2,x_vals,dx=0.001))*analytical,label='analytical')
plt.legend()
#%%
s_orbit_n_1=numerov(0, 10, V, 0.001, 0, 1)
s_n_1_analytical=np.exp(-s_orbit_n_1[1])
#%%
plt.plot(s_orbit_n_1[1],-s_orbit_n_1[3][0][:,0],label='numerical')
plt.plot(s_orbit_n_1[1],2*s_orbit_n_1[1]*s_n_1_analytical,label='analytical')
plt.legend()
