# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 06:53:46 2025

@author: padmin
"""

from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt 
import random
from numpy.linalg import inv
#%%
e=1.602176634e-19
m_e=9.1093837015e-31
eps_0=8.8541878188e-12
#%%
#Absorption Coefficient B_ij
#Stimulated emmsion coefficient B_ji
#Collisional de-excitation coefficient D_ji where T is in units of eV
#E_ij are collisonal excitation rates assumed to be feed in as an array
#We want these to be 2D arrays 
def B_ij(dE,gf,J_i):
    return ((e**2)/(4*np.pi*eps_0*m_e*dE))*(gf/(2*J_i+1))
def B_ji(B_ij,J_i,J_j):
    return ((2*J_i+1)/(2*J_j+1))*B_ij
def D_ji(E_ij,dE,T,J_i,J_j):
    return ((2*J_i+1)/(2*J_j+1))*np.exp(dE/T)*E_ij
#%%
# rates is a 1D array whose elements are the 2D arrays of the rate coefficients
# computes the matrix inverse of C_ji and also returns F_jsig matrix as a tuple
# need to make sure the the rectangular matrices are oriented the right way for matrix multiplication
# Might need to preprose A_jsin,D_jsig,B_jsig, E_sig_i,B_sig_i i.e transpose those matrices 
def C_ji_inv(rates):
    E_ij=rates[0]
    D_ji=rates[1]
    E_jl=rates[2]
    A_jsig=rates[3]
    D_jsig=rates[4]
    B_jsig=rates[5]
    F_jsig=A_jsig+D_jsig+B_jsig
    C_ji=np.zeros((len(E_ij),len(E_ij))) #this array might be one dimension too small since E_ii coefficients do not exist if this is the case increase len size by 1
    i=0  
    while i<len(C_ji[0]):
        j=0 
        while j<len(C_ji[0]):
            if i<j:
                C_ji[j][i]+=E_ij[i][j]
            elif i==j:
                C_ji[j][i]-=np.sum(D_ji[j])+np.sum(E_jl[j])+np.sum(F_jsig[j])
            j+=1
        i+=1
    return np.linalg.inv(C_ji),F_jsig

#Compute C_sig_nu and C_j_sig 
def C_sig_nu(rates):
    E_nu_sig=rates[0]
    D_sig_pho=rates[1]
    E_sig_alpha=rates[2]
    B_sig_j=rates[3]
    E_sig_j=rates[4]
    C_j_sig=B_sig_j+E_sig_j
    C_sig_nu=np.zeros((len(E_nu_sig),len(E_nu_sig))) #potential same issue as C_ji
    nu=0 
    while nu<len(C_sig_nu[0]):
        sig=0 
        while sig<len(C_sig_nu[0]):
            if nu<sig:
                C_sig_nu[sig][nu]+=E_nu_sig[nu][sig]
            elif nu==sig:
                C_sig_nu[sig][nu]-=np.sum(E_sig_alpha[sig])+np.sum(D_sig_pho[sig])+np.sum(C_j_sig[sig])
            sig+=1 
        nu+=1 
    return C_sig_nu,C_j_sig

def RateMatrix(C_sig_nu,C_j_sig,C_ji_inv,F_jsig):
    return C_sig_nu-np.matmul(F_jsig,np.matmul(C_ji_inv,C_j_sig))

def dydt(A,y):
    f=[]
        

def ForwardEuler(A,N_0,dt,t_fin):
    t_steps=np.arange(0,t_fin+dt,dt)
    N_fin=np.zeros((len(N_0),len(t_steps)))
    for a,i in enumerate(t_steps):
        if i==0:
            N_fin[:,a]=N_0.T
        else:
            N_0=N_0+ dt*np.matmul(A,N_0)
            N_fin[:,a]=N_0.T
        
    return N_fin,t_steps

def BackwardEuler(A,N_0,dt,t_fin):
    t_steps=np.arange(0,t_fin+dt,dt)
    N_fin=np.zeros((len(N_0),len(t_steps)))
    for a,i in enumerate(t_steps):
        if i==0:
            N_fin[:,a]=N_0.T
        else:
            N_0=np.matmul(inv(np.identity(len(N_0))-dt*A),N_0)
            N_fin[:,a]=N_0.T
                        
    return N_fin,t_steps
    
#%%
def N_0(gamma,D,I,N_0J):
    return D*gamma*N_0J/(1+gamma*I)
def f(t,y,D,I):
    return [-D*y[0]+I*y[1],D*y[0]-I*y[1]]

t1=10 #the maximum time value the integrator calculates
dt=0.01 #step size 

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
#%%
Ni=np.array(Ni)
t=np.array(t)
av_dN=Ni/t
#%%
plt.plot(t,av_dN)
#%% #Testing forward Euler method 

N_0=np.array([[1],[1]]).astype(float)
A=np.array([[1,1],[1,1]]).astype(float)
#%%
FE_results=ForwardEuler(A, N_0, 0.01, 5)
BE_results=BackwardEuler(A, N_0, 0.001, 5)
#%%
plt.plot(BE_results[1],BE_results[0][0],label='Backward')
plt.plot(FE_results[1],FE_results[0][0],label='Forward')
plt.plot(BE_results[1],np.exp(2*BE_results[1]),label='analytical')
plt.legend()
#%%