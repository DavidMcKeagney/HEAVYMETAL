# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 13:17:04 2026

@author: padmin
"""

import numpy as np 
import matplotlib.pyplot as plt
#%%
#Need to do the ion weights before shifting
def shifting_cross_sections(X_z,X_y,shift_val,tol):
    CS=X_z[:,1].copy()
    for a,E in enumerate(X_z[:,0]):
        i=np.isclose(X_y[:,0],E+shift_val,atol=tol)
        sub_X_y=X_y[i]
        if len(sub_X_y)!=0:
            CS[a]+=sub_X_y[0,1]
    CS=np.column_stack((X_z[:,0],CS))
    return CS
def Gaussian(x,FWHM,mu):
    Sigma=FWHM/(2*np.sqrt(2*np.log(2)))
    return 1/(Sigma*np.sqrt(2*np.pi))*(np.exp(-(x-mu)**2/(2*(Sigma)**2)))
def linear_bck(x,b,c):
    return b*x+c
#%%
Exp_data=np.loadtxt('C:\\Users\padmin\Downloads\\2023_07_19_absorption_shift0.8_eV.csv',dtype=str,delimiter=',')
data_500ns=Exp_data[1:,44].astype(float)
data_400ns=Exp_data[1:,42].astype(float)
data_300ns=Exp_data[1:,40].astype(float)
data_250ns=Exp_data[1:,39].astype(float)
data_200ns=Exp_data[1:,38].astype(float)
Energy_exp=Exp_data[1:,0].astype(float)
#%%
# Au I Cross Sections

Au_I_J_2_5_3_5=[]
Au_I_J_2_5_2_5=[]
Au_I_J_1_5_2_5=[]
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.I.J=2.5-3.5.sigma') as file:
    for lines in file:
        Au_I_J_2_5_3_5.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.I.J=2.5-2.5.sigma') as file:
    for lines in file:
        Au_I_J_2_5_2_5.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.I.J=1.5-2.5.sigma') as file:
    for lines in file:
        Au_I_J_1_5_2_5.append(lines.split())

Au_I_J_1_5_2_5=np.array(Au_I_J_1_5_2_5[4:]).astype(float)
Au_I_J_2_5_2_5=np.array(Au_I_J_2_5_2_5[4:]).astype(float)
Au_I_J_2_5_3_5=np.array(Au_I_J_2_5_3_5[4:]).astype(float)

Energy_AuI=Au_I_J_1_5_2_5[:,0]

Au_I_J_2_5_3_5=(6/2)*np.exp(-1.3931/3.5)*Au_I_J_2_5_3_5[:,1]
Au_I_J_2_5_2_5=(6/2)*np.exp(-1.3931/3.5)*Au_I_J_2_5_2_5[:,1]
Au_I_J_1_5_2_5=(4/2)*np.exp(-2.97397/3.5)*Au_I_J_1_5_2_5[:,1]

X_AuI=Au_I_J_1_5_2_5+Au_I_J_2_5_2_5+Au_I_J_2_5_3_5
X_AuI=0.11*X_AuI
X_AuI=np.column_stack((Energy_AuI,X_AuI))
#%%
# Au II Cross Sections

Au_II_J_1_2=[]
Au_II_J_2_2=[]
Au_II_J_2_3=[]
Au_II_J_3_3=[]
Au_II_J_3_4=[]
Au_II_J_4_4=[]
Au_II_J_4_5=[]
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=1.0-2.0.sigma') as file:
    for lines in file:
        Au_II_J_1_2.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=2.0-2.0.sigma') as file:
    for lines in file:
        Au_II_J_2_2.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=2.0-3.0.sigma') as file:
    for lines in file:
        Au_II_J_2_3.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=3.0-3.0.sigma') as file:
    for lines in file:
        Au_II_J_3_3.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=3.0-4.0.sigma') as file:
    for lines in file:
        Au_II_J_3_4.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=4.0-4.0.sigma') as file:
    for lines in file:
        Au_II_J_4_4.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=4.0-5.0.sigma') as file:
    for lines in file:
        Au_II_J_4_5.append(lines.split())


Au_II_J_1_2=np.array(Au_II_J_1_2[4:]).astype(float)
Au_II_J_2_2=np.array(Au_II_J_2_2[4:]).astype(float)
Au_II_J_2_3=np.array(Au_II_J_2_3[4:]).astype(float)
Au_II_J_3_3=np.array(Au_II_J_3_3[4:]).astype(float)
Au_II_J_3_4=np.array(Au_II_J_3_4[4:]).astype(float)
Au_II_J_4_4=np.array(Au_II_J_4_4[4:]).astype(float)
Au_II_J_4_5=np.array(Au_II_J_4_5[4:]).astype(float)

Energy_AuII= Au_II_J_1_2[:,0]

Au_II_J_1_2=3*np.exp(-(3.4011/3.5))*Au_II_J_1_2[:,1]
Au_II_J_2_2=5*np.exp(-(2.1513/3.5))*Au_II_J_2_2[:,1]
Au_II_J_2_3=5*np.exp(-(2.1513/3.5))*Au_II_J_2_3[:,1]
Au_II_J_3_3=7*np.exp(-(1.7873/3.5))*Au_II_J_3_3[:,1]
Au_II_J_3_4=7*np.exp(-(1.7873/3.5))*Au_II_J_3_4[:,1]
Au_II_J_4_4=9*np.exp(-(5.2425/3.5))*Au_II_J_4_4[:,1]
Au_II_J_4_5=9*np.exp(-(5.2425/3.5))*Au_II_J_4_5[:,1]

X_AuII=Au_II_J_1_2+Au_II_J_2_2+Au_II_J_2_3+Au_II_J_3_3+Au_II_J_3_4+Au_II_J_4_4+Au_II_J_4_5
X_AuII=0.78*X_AuII
X_AuII=np.column_stack((Energy_AuII,X_AuII))
#%%
# Au III Cross Sections
Au_III_J_0_5_0_5=[]
Au_III_J_0_5_1_5=[]
Au_III_J_1_5_1_5=[]
Au_III_J_1_5_2_5=[]
Au_III_J_2_5_2_5=[]
Au_III_J_2_5_3_5=[]
Au_III_J_3_5_3_5=[]
Au_III_J_3_5_4_5=[]
Au_III_J_4_5_4_5=[]
Au_III_J_4_5_5_5=[]
Au_III_J_5_5_5_5=[]
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.III.J=0.5-0.5.sigma') as file:
    for lines in file:
        Au_III_J_0_5_0_5.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.III.J=0.5-1.5.sigma') as file:
    for lines in file:
        Au_III_J_0_5_1_5.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.III.J=1.5-1.5.sigma') as file:
    for lines in file:
        Au_III_J_1_5_1_5.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.III.J=1.5-2.5.sigma') as file:
    for lines in file:
        Au_III_J_1_5_2_5.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.III.J=2.5-2.5.sigma') as file:
    for lines in file:
        Au_III_J_2_5_2_5.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.III.J=2.5-3.5.sigma') as file:
    for lines in file:
        Au_III_J_2_5_3_5.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.III.J=3.5-3.5.sigma') as file:
    for lines in file:
        Au_III_J_3_5_3_5.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.III.J=3.5-4.5.sigma') as file:
    for lines in file:
        Au_III_J_3_5_4_5.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.III.J=4.5-4.5.sigma') as file:
    for lines in file:
        Au_III_J_4_5_4_5.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.III.J=4.5-5.5.sigma') as file:
    for lines in file:
        Au_III_J_4_5_5_5.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.III.J=5.5-5.5.sigma') as file:
    for lines in file:
        Au_III_J_5_5_5_5.append(lines.split())
        
Au_III_J_0_5_0_5=np.array(Au_III_J_0_5_0_5[4:]).astype(float)
Au_III_J_0_5_1_5=np.array(Au_III_J_0_5_1_5[4:]).astype(float)
Au_III_J_1_5_1_5=np.array(Au_III_J_1_5_1_5[4:]).astype(float)
Au_III_J_1_5_2_5=np.array(Au_III_J_1_5_2_5[4:]).astype(float)
Au_III_J_2_5_2_5=np.array(Au_III_J_2_5_2_5[4:]).astype(float)
Au_III_J_2_5_3_5=np.array(Au_III_J_2_5_3_5[4:]).astype(float)
Au_III_J_3_5_3_5=np.array(Au_III_J_3_5_3_5[4:]).astype(float)
Au_III_J_3_5_4_5=np.array(Au_III_J_3_5_4_5[4:]).astype(float)
Au_III_J_4_5_4_5=np.array(Au_III_J_4_5_4_5[4:]).astype(float)
Au_III_J_4_5_5_5=np.array(Au_III_J_4_5_5_5[4:]).astype(float)
Au_III_J_5_5_5_5=np.array(Au_III_J_5_5_5_5[4:]).astype(float)
Energy_AuIII=Au_III_J_0_5_0_5[:,0]

Au_III_J_0_5_0_5=(2/6)*np.exp(-(5.8297+0.8390)/3.5)*Au_III_J_0_5_0_5[:,1]
Au_III_J_0_5_1_5=(2/6)*np.exp(-(5.8297+0.8390)/3.5)*Au_III_J_0_5_1_5[:,1]
Au_III_J_1_5_1_5=(4/6)*np.exp(-(0.7801+0.8390)/3.5)*Au_III_J_1_5_1_5[:,1]
Au_III_J_1_5_2_5=(4/6)*np.exp(-(0.7801+0.8390)/3.5)*Au_III_J_1_5_2_5[:,1]
Au_III_J_3_5_3_5=(8/6)*np.exp(-(3.5370+0.8390)/3.5)*Au_III_J_3_5_3_5[:,1]
Au_III_J_3_5_4_5=(8/6)*np.exp(-(3.5370+0.8390)/3.5)*Au_III_J_3_5_4_5[:,1]
Au_III_J_4_5_4_5=(10/6)*np.exp(-(2.8096+0.8390)/3.5)*Au_III_J_4_5_4_5[:,1]
Au_III_J_4_5_5_5=(10/6)*np.exp(-(2.8096+0.8390)/3.5)*Au_III_J_4_5_5_5[:,1]
Au_III_J_5_5_5_5=(12/6)*np.exp(-(13.2294+0.8390)/3.5)*Au_III_J_5_5_5_5[:,1]

X_AuIII=Au_III_J_0_5_0_5+Au_III_J_0_5_1_5+Au_III_J_1_5_1_5+Au_III_J_1_5_2_5+Au_III_J_2_5_2_5[:,1]+Au_III_J_2_5_3_5[:,1]+Au_III_J_3_5_3_5+Au_III_J_3_5_4_5+Au_III_J_4_5_4_5+Au_III_J_4_5_5_5+Au_III_J_5_5_5_5
X_AuIII=0.11*X_AuIII
X_AuIII=np.column_stack((Energy_AuIII,X_AuIII))
#%%
CS=shifting_cross_sections(X_AuII, X_AuI, -0.0,1e-8)
CS=shifting_cross_sections(CS, X_AuIII, 0, 1e-8)
#%%
linear_500ns=linear_bck(X_AuI[:,0], -0.0045, 0.67)
linear_300ns=linear_bck(X_AuI[:,0], -0.0105, 2.1)
#linear_300ns=linear_bck(X_AuI[:,0], -0.015, 1.93)
#%%

plt.plot(X_AuI[:,0]+np.repeat(0.73,len(X_AuI[:,0])),1/75*CS[:,1]+linear_300ns,label='Theoretical')
#plt.plot(Energy_exp,data_500ns,label='500ns')
#plt.plot(Energy_exp,data_300ns,label='300ns')
#plt.plot(Energy_exp,data_250ns,label='250ns')
plt.plot(Energy_exp,data_200ns,label='200ns')
#plt.plot(Energy_exp,data_400ns,label='400ns')
plt.legend()
plt.xlabel('Energy (eV)')
plt.ylabel('Absorbance')
plt.xlim(78,86)
plt.ylim(1.2,2.8)
#%%
shift_vals=np.arange(-0.5,0.5,0.1)
Gaussian_kernal=Gaussian(shift_vals, 0.4, 0)/(np.sum(Gaussian(shift_vals, 0.4, 0))*0.1)
L=len(Gaussian_kernal)+len(X_AuII)-1
conv_Au_II=np.fft.ifft(np.fft.fft(Gaussian_kernal,L)*np.fft.fft(X_AuII,L)).real*0.1

start = (len(Gaussian_kernal) - 1) // 2
conv_same = conv_Au_II[start:start + len(X_AuII)]

plt.plot(X_AuI[:,0],conv_same,label='convolved')
plt.plot(X_AuI[:,0],X_AuII,label='original')
plt.legend()