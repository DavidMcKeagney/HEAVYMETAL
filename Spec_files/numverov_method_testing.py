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
       
    B=1/12*B
    H=-(1/(2*(dx**2)))*H
    A=H+np.matmul(B,Wmat)
    H_new=np.matmul(LA.inv(B),A)
    eigvalues,eigvectors=LA.eig(H_new)
    return x_points,x_in,eigvalues,eigvectors
def V(x,l,Z):
    return Z/x + l*(l+1)/(x**2)
#%%
sol=numerov( 0, 5, V, 0.001, 0, 0)
#%%
x_vals=np.arange(0,5+0.001,0.001)[1:-1]
analytical=np.sin(x_vals*(np.pi/5))
plt.plot(sol[1],sol[3][:,4881],label='numerical')
plt.plot(x_vals,np.sqrt(1/np.trapz(analytical**2,x=x_vals))*analytical,label='analytical')
plt.legend()
