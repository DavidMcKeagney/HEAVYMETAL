# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 11:23:20 2025

@author: padmin
"""

import numpy as np
import matplotlib.pyplot as plt
#%%
Au_I_J_2_5_3_5=[]
Au_I_J_2_5_2_5=[]
Au_I_J_1_5_2_5=[]
with open('C:\\Users\padmin\Downloads\Au.I.J=2.5-3.5.sigma') as file:
    for lines in file:
        Au_I_J_2_5_3_5.append(lines.split())
with open('C:\\Users\padmin\Downloads\Au.I.J=2.5-2.5.sigma') as file:
    for lines in file:
        Au_I_J_2_5_2_5.append(lines.split())
with open('C:\\Users\padmin\Downloads\Au.I.J=1.5-2.5.sigma') as file:
    for lines in file:
        Au_I_J_1_5_2_5.append(lines.split())
#%%
Au_II_J_1_2=[]
Au_II_J_2_2=[]
Au_II_J_2_3=[]
Au_II_J_3_3=[]
Au_II_J_3_4=[]
Au_II_J_4_4=[]
Au_II_J_4_5=[]
with open('C:\\Users\padmin\Downloads\Au.II.J=1.0-2.0.sigma') as file:
    for lines in file:
        Au_II_J_1_2.append(lines.split())
with open('C:\\Users\padmin\Downloads\Au.II.J=2.0-2.0.sigma') as file:
    for lines in file:
        Au_II_J_2_2.append(lines.split())
with open('C:\\Users\padmin\Downloads\Au.II.J=2.0-3.0.sigma') as file:
    for lines in file:
        Au_II_J_2_3.append(lines.split())
with open('C:\\Users\padmin\Downloads\Au.II.J=3.0-3.0.sigma') as file:
    for lines in file:
        Au_II_J_3_3.append(lines.split())
with open('C:\\Users\padmin\Downloads\Au.II.J=3.0-4.0.sigma') as file:
    for lines in file:
        Au_II_J_3_4.append(lines.split())
with open('C:\\Users\padmin\Downloads\Au.II.J=4.0-4.0.sigma') as file:
    for lines in file:
        Au_II_J_4_4.append(lines.split())
with open('C:\\Users\padmin\Downloads\Au.II.J=4.0-5.0.sigma') as file:
    for lines in file:
        Au_II_J_4_5.append(lines.split())

#%%
Au_I_J_1_5_2_5=np.array(Au_I_J_1_5_2_5[4:]).astype(float)
Au_I_J_2_5_2_5=np.array(Au_I_J_2_5_2_5[4:]).astype(float)
Au_I_J_2_5_3_5=np.array(Au_I_J_2_5_3_5[4:]).astype(float)
#%%
Au_II_J_1_2=np.array(Au_II_J_1_2[4:]).astype(float)
Au_II_J_2_2=np.array(Au_II_J_2_2[4:]).astype(float)
Au_II_J_2_3=np.array(Au_II_J_2_3[4:]).astype(float)
Au_II_J_3_3=np.array(Au_II_J_3_3[4:]).astype(float)
Au_II_J_3_4=np.array(Au_II_J_3_4[4:]).astype(float)
Au_II_J_4_4=np.array(Au_II_J_4_4[4:]).astype(float)
Au_II_J_4_5=np.array(Au_II_J_4_5[4:]).astype(float)
#%%
total_cross_sections_AuI=Au_I_J_1_5_2_5[:,1]+Au_I_J_2_5_2_5[:,1]+Au_I_J_2_5_3_5[:,1]
Energy=Au_I_J_1_5_2_5[:,0]
#%%
total_cross_sections_AuII=Au_II_J_1_2[:,1]+Au_II_J_2_2[:,1]+Au_II_J_2_3[:,1]+Au_II_J_3_3[:,1]+Au_II_J_3_4[:,1]+Au_II_J_4_4[:,1]+Au_II_J_4_5[:,1]

#%%
plt.plot(Energy,Au_I_J_1_5_2_5[:,1],label='J:1.5-2.5')
plt.plot(Energy,Au_I_J_2_5_2_5[:,1],label='J:2.5-2.5')
plt.plot(Energy,Au_I_J_2_5_3_5[:,1],label='J:2.5-3.5')
#plt.plot(Energy,total_cross_sections_AuII)
#plt.plot(Energy,Au_II_J_1_2[:,1],label='J:1-2')
#plt.plot(Energy,Au_II_J_2_2[:,1],label='J:2-2')
#plt.plot(Energy,Au_II_J_2_3[:,1],label='J:2-3')
#plt.plot(Energy,Au_II_J_3_3[:,1],label='J:3-3')
#plt.plot(Energy,Au_II_J_3_4[:,1],label='J:3-4')
#plt.plot(Energy,Au_II_J_4_4[:,1],label='J:4-4')
#plt.plot(Energy,Au_II_J_4_5[:,1],label='J:4-5')
plt.legend()
plt.xlabel('Energy [eV]')
plt.ylabel('Cross section [mb]')