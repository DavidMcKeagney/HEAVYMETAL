# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:42:55 2026

@author: padmin
"""

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.rcParams.update({'font.size': 22})
plt.rcParams["figure.figsize"] = (15,10) 
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
data_450ns=Exp_data[1:,43].astype(float)
data_400ns=Exp_data[1:,42].astype(float)
data_350ns=Exp_data[1:,41].astype(float)
data_300ns=Exp_data[1:,40].astype(float)
data_250ns=Exp_data[1:,39].astype(float)
data_200ns=Exp_data[1:,38].astype(float)
Energy_exp=Exp_data[1:,0].astype(float)
#%%
# Au I Cross Sections

Au_I_J_2_5_3_5=[]
Au_I_J_2_5_2_5=[]
Au_I_J_1_5_2_5=[]
Au_I_J_2_5_1_5=[]
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.I.J=2.5-3.5.sigma') as file:
    for lines in file:
        Au_I_J_2_5_3_5.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.I.J=2.5-2.5.sigma') as file:
    for lines in file:
        Au_I_J_2_5_2_5.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.I.J=2.5-1.5.sigma') as file:
    for lines in file:
        Au_I_J_2_5_1_5.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.I.J=1.5-2.5.sigma') as file:
    for lines in file:
        Au_I_J_1_5_2_5.append(lines.split())


Au_I_J_1_5_2_5=np.array(Au_I_J_1_5_2_5[4:]).astype(float)
Au_I_J_2_5_2_5=np.array(Au_I_J_2_5_2_5[4:]).astype(float)
Au_I_J_2_5_1_5=np.array(Au_I_J_2_5_1_5[4:]).astype(float)
Au_I_J_2_5_3_5=np.array(Au_I_J_2_5_3_5[4:]).astype(float)

Energy_AuI=Au_I_J_1_5_2_5[:,0]

Au_I_J_2_5_3_5=np.exp(-1.3931/2.78)*Au_I_J_2_5_3_5[:,1]
Au_I_J_2_5_2_5=np.exp(-1.3931/2.78)*Au_I_J_2_5_2_5[:,1]
Au_I_J_2_5_1_5=np.exp(-1.3931/2.78)*Au_I_J_2_5_1_5[:,1]
Au_I_J_1_5_2_5=np.exp(-2.97397/2.78)*Au_I_J_1_5_2_5[:,1]

X_AuI=Au_I_J_1_5_2_5+Au_I_J_2_5_2_5+Au_I_J_2_5_3_5+Au_I_J_2_5_1_5
X_AuI=0.3*X_AuI
X_AuI=np.column_stack((Energy_AuI,X_AuI))
#%%
# Au II Cross Sections
Au_II_J_1_0=[]
Au_II_J_1_1=[]
Au_II_J_1_2=[]
Au_II_J_2_2=[]
Au_II_J_2_1=[]
Au_II_J_2_3=[]
Au_II_J_3_3=[]
Au_II_J_3_2=[]
Au_II_J_3_4=[]
Au_II_J_4_4=[]
Au_II_J_4_3=[]
Au_II_J_4_5=[]
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=1.0-0.0.sigma') as file:
    for lines in file:
        Au_II_J_1_0.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=1.0-1.0.sigma') as file:
    for lines in file:
        Au_II_J_1_1.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=1.0-2.0.sigma') as file:
    for lines in file:
        Au_II_J_1_2.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=2.0-2.0.sigma') as file:
    for lines in file:
        Au_II_J_2_2.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=2.0-1.0.sigma') as file:
    for lines in file:
        Au_II_J_2_1.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=2.0-3.0.sigma') as file:
    for lines in file:
        Au_II_J_2_3.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=3.0-3.0.sigma') as file:
    for lines in file:
        Au_II_J_3_3.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=3.0-2.0.sigma') as file:
    for lines in file:
        Au_II_J_3_2.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=3.0-4.0.sigma') as file:
    for lines in file:
        Au_II_J_3_4.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=4.0-4.0.sigma') as file:
    for lines in file:
        Au_II_J_4_4.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=4.0-3.0.sigma') as file:
    for lines in file:
        Au_II_J_4_3.append(lines.split())
with open('C:\\Users\padmin\OneDrive\Documents\Github\HEAVYMETAL\Michael_Martins_cowan\Au.II.J=4.0-5.0.sigma') as file:
    for lines in file:
        Au_II_J_4_5.append(lines.split())

Au_II_J_1_0=np.array(Au_II_J_1_0[4:]).astype(float)
Au_II_J_1_1=np.array(Au_II_J_1_1[4:]).astype(float)
Au_II_J_1_2=np.array(Au_II_J_1_2[4:]).astype(float)
Au_II_J_2_2=np.array(Au_II_J_2_2[4:]).astype(float)
Au_II_J_2_1=np.array(Au_II_J_2_1[4:]).astype(float)
Au_II_J_2_3=np.array(Au_II_J_2_3[4:]).astype(float)
Au_II_J_3_3=np.array(Au_II_J_3_3[4:]).astype(float)
Au_II_J_3_2=np.array(Au_II_J_3_2[4:]).astype(float)
Au_II_J_3_4=np.array(Au_II_J_3_4[4:]).astype(float)
Au_II_J_4_4=np.array(Au_II_J_4_4[4:]).astype(float)
Au_II_J_4_3=np.array(Au_II_J_4_3[4:]).astype(float)
Au_II_J_4_5=np.array(Au_II_J_4_5[4:]).astype(float)

Energy_AuII= Au_II_J_1_2[:,0]

Au_II_J_1_0=np.exp(-(3.4011/2.78))*Au_II_J_1_0[:,1]
Au_II_J_1_1=np.exp(-(3.4011/2.78))*Au_II_J_1_1[:,1]
Au_II_J_1_2=np.exp(-(3.4011/2.78))*Au_II_J_1_2[:,1]
Au_II_J_2_2=np.exp(-(2.1513/2.78))*Au_II_J_2_2[:,1]
Au_II_J_2_1=np.exp(-(2.1513/2.78))*Au_II_J_2_1[:,1]
Au_II_J_2_3=np.exp(-(2.1513/2.78))*Au_II_J_2_3[:,1]
Au_II_J_3_3=np.exp(-(1.7873/2.78))*Au_II_J_3_3[:,1]
Au_II_J_3_2=np.exp(-(1.7873/2.78))*Au_II_J_3_2[:,1]
Au_II_J_3_4=np.exp(-(1.7873/2.78))*Au_II_J_3_4[:,1]
Au_II_J_4_4=np.exp(-(5.2425/2.78))*Au_II_J_4_4[:,1]
Au_II_J_4_3=np.exp(-(5.2425/2.78))*Au_II_J_4_3[:,1]
Au_II_J_4_5=np.exp(-(5.2425/2.78))*Au_II_J_4_5[:,1]

X_AuII=Au_II_J_1_2+Au_II_J_2_2+Au_II_J_2_3+Au_II_J_3_3+Au_II_J_3_4+Au_II_J_4_4+Au_II_J_4_5+Au_II_J_2_1+Au_II_J_3_2+Au_II_J_4_3+Au_II_J_1_1+Au_II_J_1_0
X_AuII=0.70*X_AuII
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

Au_III_J_0_5_0_5=(2/6)*np.exp(-(5.8297+0.8390)/2.78)*Au_III_J_0_5_0_5[:,1]
Au_III_J_0_5_1_5=(2/6)*np.exp(-(5.8297+0.8390)/2.78)*Au_III_J_0_5_1_5[:,1]
Au_III_J_1_5_1_5=(4/6)*np.exp(-(0.7801+0.8390)/2.78)*Au_III_J_1_5_1_5[:,1]
Au_III_J_1_5_2_5=(4/6)*np.exp(-(0.7801+0.8390)/2.78)*Au_III_J_1_5_2_5[:,1]
Au_III_J_3_5_3_5=(8/6)*np.exp(-(3.5370+0.8390)/2.78)*Au_III_J_3_5_3_5[:,1]
Au_III_J_3_5_4_5=(8/6)*np.exp(-(3.5370+0.8390)/2.78)*Au_III_J_3_5_4_5[:,1]
Au_III_J_4_5_4_5=(10/6)*np.exp(-(2.8096+0.8390)/2.78)*Au_III_J_4_5_4_5[:,1]
Au_III_J_4_5_5_5=(10/6)*np.exp(-(2.8096+0.8390)/2.78)*Au_III_J_4_5_5_5[:,1]
Au_III_J_5_5_5_5=(12/6)*np.exp(-(13.2294+0.8390)/2.78)*Au_III_J_5_5_5_5[:,1]

X_AuIII=Au_III_J_0_5_0_5+Au_III_J_0_5_1_5+Au_III_J_1_5_1_5+Au_III_J_1_5_2_5+Au_III_J_2_5_2_5[:,1]+Au_III_J_2_5_3_5[:,1]+Au_III_J_3_5_3_5+Au_III_J_3_5_4_5+Au_III_J_4_5_4_5+Au_III_J_4_5_5_5+Au_III_J_5_5_5_5
X_AuIII=0.0*X_AuIII
X_AuIII=np.column_stack((Energy_AuIII,X_AuIII))
#%%
CS_300ns=shifting_cross_sections(X_AuII, X_AuI, -0.0,1e-8)
CS_300ns=shifting_cross_sections(CS_300ns, X_AuIII, 0, 1e-8)

X_AuI_300ns=X_AuI[:,1]
X_AuII_300ns=X_AuII[:,1]
X_AuIII_300ns=X_AuIII[:,1]
#%%
CS_350ns=shifting_cross_sections(X_AuII, X_AuI, -0.0,1e-8)
#CS_350ns=shifting_cross_sections(CS_350ns, X_AuIII, 0, 1e-8)

X_AuI_350ns=X_AuI[:,1]
X_AuII_350ns=X_AuII[:,1]
#X_AuIII_350ns=X_AuIII[:,1]
#%%
CS_400ns=shifting_cross_sections(X_AuII, X_AuI, -0.0,1e-8)
X_AuI_400ns=X_AuI[:,1]
X_AuII_400ns=X_AuII[:,1]
#%%
CS_450ns=shifting_cross_sections(X_AuII, X_AuI, -0.0,1e-8)
X_AuI_450ns=X_AuI[:,1]
X_AuII_450ns=X_AuII[:,1]
#%%
CS_500ns=shifting_cross_sections(X_AuII, X_AuI, -0.0,1e-8)
X_AuI_500ns=X_AuI[:,1]
X_AuII_500ns=X_AuII[:,1]
#CS=shifting_cross_sections(CS, X_AuIII, 0, 1e-8)
#%%
linear_500ns=linear_bck(X_AuI[:,0], -0.00, 0.12)
linear_450ns=linear_bck(X_AuI[:,0], -0.00, 0.185)
linear_400ns=linear_bck(X_AuI[:,0], -0.00, 0.22)
linear_300ns=linear_bck(X_AuI[:,0], 0, 0.5)
linear_350ns=linear_bck(X_AuI[:,0], 0, 0.31)

#linear_300ns=linear_bck(X_AuI[:,0], -0.015, 1.93)
#%%

plt.plot(X_AuI[:,0]+np.repeat(0.74,len(X_AuI[:,0])),1/40*CS_400ns[:,1]+linear_400ns,label='Theoretical')
#plt.plot(X_AuI[:,0]+np.repeat(0.74,len(X_AuI[:,0])),1/100*X_AuI[:,1]+linear_300ns-np.repeat(0.05,len(X_AuI[:,1])),linestyle='dashed',label='Au I')
#plt.plot(X_AuI[:,0]+np.repeat(0.74,len(X_AuI[:,0])),1/100*X_AuII[:,1]+linear_300ns-np.repeat(0.05,len(X_AuII[:,1])),linestyle='dashdot',label='Au II')
plt.plot(Energy_exp,data_400ns,color='black',label='400ns')
#plt.plot(Energy_exp,data_300ns,label='300ns')
#plt.plot(Energy_exp,data_250ns,label='250ns')
#plt.plot(Energy_exp,data_200ns,label='200ns')
#plt.plot(Energy_exp,data_400ns,label='400ns')
plt.legend()
plt.xlabel('Energy (eV)')
plt.ylabel('Absorbance')
plt.xlim(78,95)
plt.ylim(0.3,0.7)
#%%
fig,axes=plt.subplots(2,3,figsize=(9, 10))
axes[0,0].plot(X_AuI[:,0]+np.repeat(0.74,len(X_AuI[:,0])),1/290*CS_500ns[:,1]+linear_500ns)
axes[0,0].plot(X_AuI[:,0]+np.repeat(0.74,len(X_AuI[:,0])),1/290*X_AuI_500ns+linear_500ns+np.repeat(0.01,len(X_AuI[:,1])),linestyle='dashed')
axes[0,0].plot(X_AuI[:,0]+np.repeat(0.74,len(X_AuI[:,0])),1/290*X_AuII_500ns+linear_500ns-np.repeat(0.01,len(X_AuII[:,1])),linestyle='dashdot')
axes[0,0].plot(Energy_exp,data_500ns,color='black')
axes[0,0].set_title('500ns')
axes[0,2].plot(X_AuI[:,0]+np.repeat(0.74,len(X_AuI[:,0])),1/180*CS_400ns[:,1]+linear_400ns)
axes[0,2].plot(X_AuI[:,0]+np.repeat(0.74,len(X_AuI[:,0])),1/180*X_AuI_400ns+linear_400ns+np.repeat(0.07,len(X_AuI[:,1])),linestyle='dashed')
axes[0,2].plot(X_AuI[:,0]+np.repeat(0.74,len(X_AuI[:,0])),1/180*X_AuII_400ns+linear_400ns-np.repeat(0.03,len(X_AuII[:,1])),linestyle='dashdot')
axes[0,2].plot(Energy_exp,data_400ns,color='black')
axes[0,2].set_title('400ns')
line1, =axes[1,1].plot(X_AuI[:,0]+np.repeat(0.74,len(X_AuI[:,0])),1/75*CS_300ns[:,1]+linear_300ns,label='Summed')
line2, =axes[1,1].plot(X_AuI[:,0]+np.repeat(0.74,len(X_AuI[:,0])),1/75*X_AuI_300ns+linear_300ns+np.repeat(0.14,len(X_AuII[:,1])),linestyle='dashed',label='Au I (Offset)')
line3, =axes[1,1].plot(X_AuI[:,0]+np.repeat(0.74,len(X_AuI[:,0])),1/75*X_AuII_300ns+linear_300ns-np.repeat(0.15,len(X_AuII[:,1])),linestyle='dashdot',label='Au II (Offset)')
#line4, =axes[1,1].plot(X_AuI[:,0]+np.repeat(0.74,len(X_AuI[:,0])),1/100*X_AuIII_300ns+linear_300ns+np.repeat(0.1,len(X_AuII[:,1])),linestyle='dotted',label='Au III (Offset)')
line5, =axes[1,1].plot(Energy_exp,data_300ns,color='black',label='Experiment')
axes[1,1].set_title('300ns')
axes[1,0].plot(X_AuI[:,0]+np.repeat(0.74,len(X_AuI[:,0])),1/120*CS_350ns[:,1]+linear_350ns)
axes[1,0].plot(X_AuI[:,0]+np.repeat(0.74,len(X_AuI[:,0])),1/120*X_AuI_350ns+linear_350ns+np.repeat(0.08,len(X_AuII[:,1])),linestyle='dashed')
axes[1,0].plot(X_AuI[:,0]+np.repeat(0.74,len(X_AuI[:,0])),1/120*X_AuII_350ns+linear_350ns-np.repeat(0.1,len(X_AuII[:,1])),linestyle='dashdot')
#axes[1,0].plot(X_AuI[:,0]+np.repeat(0.74,len(X_AuI[:,0])),1/153*X_AuIII_350ns+linear_350ns+np.repeat(0.1,len(X_AuII[:,1])),linestyle='dotted')
axes[1,0].plot(Energy_exp,data_350ns,color='black')
axes[1,0].set_title('350ns')
axes[0,1].plot(X_AuI[:,0]+np.repeat(0.74,len(X_AuI[:,0])),1/225*CS_450ns[:,1]+linear_450ns)
axes[0,1].plot(X_AuI[:,0]+np.repeat(0.74,len(X_AuI[:,0])),1/225*X_AuI_450ns+linear_450ns+np.repeat(0.01,len(X_AuI[:,1])),linestyle='dashed')
axes[0,1].plot(X_AuI[:,0]+np.repeat(0.74,len(X_AuI[:,0])),1/225*X_AuII_450ns+linear_450ns-np.repeat(0.04,len(X_AuII[:,1])),linestyle='dashdot')
axes[0,1].plot(Energy_exp,data_450ns,color='black')
axes[0,1].set_title('450ns')


axes[1,2].axis('off')
handles=[line1,line2,line3,line5]
labels=['Summed','Au I (Offset)', 'Au II (Offset)', 'Experiment']
axes[1,2].legend(handles,labels,loc='center')
#axes[0].vlines(81.9,0.15,0.26,linestyle='dashed',color='green')
#axes[0].vlines(82.3,0.15,0.26,linestyle='dashed',color='green')
#axes[1].vlines(81.9,0.3,0.49,linestyle='dashed',color='green')
#axes[1].vlines(82.3,0.3,0.5,linestyle='dashed',color='green')
#axes[2].vlines(81.9,0.3,1.24,linestyle='dashed',color='green')
#axes[2].vlines(82.3,0.3,1.24,linestyle='dashed',color='green')
fig.supxlabel('Energy [eV]',fontsize='x-large')
fig.supylabel('Absorbance',fontsize='x-large')
#axes[1,0].set_ylabel('Absorbance',fontsize='x-large')
#axes[1,1].set_xlabel('Energy [eV]', fontsize='x-large')

axes[0,0].set_ylim(0.15,0.36)
axes[0,0].set_yticks([0.15,0.22,0.29,0.37])
axes[0,1].set_ylim(0.2,0.5)
axes[0,2].set_ylim(0.3,0.66)
axes[0,2].set_yticks([0.3,0.43,0.55,0.67])
axes[1,0].set_ylim(0.4,1)
axes[1,1].set_ylim(0.7,1.62)
axes[1,1].set_yticks([0.70,1.01,1.32,1.64])
#for ax in axes.flat:
#    ax.label_outer()
for ax in axes.flat:
    ax.set_xlim(78,84)
for ax in axes.flat:
    ax.set_xticks([79,81,83])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#for ax in axes.flat:
#    ax.legend(fontsize=10)
#axes[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

#labels=['a)','b)','c)']
    
#for ax, label in zip(axes, labels):
    # Place label in upper-left corner of each subplot
#    ax.text(
#        0.02, 0.95, label, 
#        transform=ax.transAxes,  # Coordinates relative to axes
#        fontsize=14, fontweight='bold',
#        va='top', ha='left'
#    )
plt.tight_layout(pad=0.2,h_pad=0.2,w_pad=0.2)
plt.show()
#%%
plt.plot(Energy_exp,data_300ns,color='black',label='300ns')
#plt.plot(Energy_exp,data_400ns,label='400ns')
#plt.plot(Energy_exp,data_500ns,label='500ns')
plt.xlabel('Energy [eV]')
plt.ylabel('Absorbance')
plt.legend()
plt.xlim(78,86)
plt.ylim(0.8,1.6)
#%%
fig,axes=plt.subplots(1,3,figsize=(9, 10))
axes[2].plot(Energy_exp,data_300ns,color='black',label='Exp. 300ns')
axes[1].plot(Energy_exp,data_400ns,color='black',label='Exp. 400ns')
axes[0].plot(Energy_exp,data_500ns,color='black',label='Exp. 500ns')
#axes[0].set_ylabel('Absorbance',fontsize='x-large')
#axes[0].set_xlabel('Energy [eV]', fontsize='x-large')
axes[0].set_ylim(0.19,0.34)
axes[1].set_ylim(0.35,0.66)
axes[2].set_ylim(0.85,1.62)
#axes[0].set_yticks([0.15,0.22,0.29,0.37])
#axes[1].set_yticks([0.3,0.43,0.55,0.67])
#axes[2].set_yticks([0.70,1.01,1.32,1.64])

fig.supxlabel('Energy [eV]',fontsize='x-large')
fig.supylabel('Absorbance',fontsize='x-large')
for ax in axes.flat:
    ax.legend(fontsize=12)
for ax in axes.flat:
    ax.set_xlim(78,84)
for ax in axes.flat:
    ax.set_xticks([79,81,83])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#for ax in axes.flat:
#    ax.legend(fontsize=10)
#axes[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

#labels=['a)','b)','c)']
    
#for ax, label in zip(axes, labels):
    # Place label in upper-left corner of each subplot
#    ax.text(
#        0.02, 0.95, label, 
#        transform=ax.transAxes,  # Coordinates relative to axes
#        fontsize=14, fontweight='bold',
#        va='top', ha='left'
#    )
plt.tight_layout(pad=0.2,h_pad=0.2,w_pad=0.2)
plt.show()

#%%

plt.plot(Energy_exp,data_300ns,label='Exp. 300ns')
plt.plot(Energy_exp,data_400ns+np.repeat(0.5,len(Energy_exp)),label='Exp. 400ns (Offset)')
plt.plot(Energy_exp,data_500ns+np.repeat(0.6,len(Energy_exp)),label='Exp. 500ns (Offset)')
plt.legend()
plt.xlabel('Energy [eV]')
plt.ylabel('Absorbance')
plt.xlim(78,84)
plt.ylim(0.77,1.6)