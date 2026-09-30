# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:32:24 2026

@author: David McKeagney
"""

import numpy as np
import matplotlib.pyplot as plt
#%%
Hf_I_5p=np.loadtxt('C:\\Users\David McKeagney\Downloads\\5p_Hf_I', dtype=float)
Hf_I_4f=np.loadtxt('C:\\Users\David McKeagney\Downloads\\4f_Hf_I', dtype=float)

Hf_II_5p=np.loadtxt('C:\\Users\David McKeagney\Downloads\\5p_Hf_II', dtype=float)
Hf_II_4f=np.loadtxt('C:\\Users\David McKeagney\Downloads\\4f_Hf_II', dtype=float)

Hf_III_5p=np.loadtxt('C:\\Users\David McKeagney\Downloads\\5p_Hf_III', dtype=float)
Hf_III_4f=np.loadtxt('C:\\Users\David McKeagney\Downloads\\4f_Hf_III', dtype=float)

Hf_I_5p_r=Hf_I_5p[:,0]
Hf_I_4f_r=Hf_I_4f[:,0]

Hf_II_5p_r=Hf_II_5p[:,0]
Hf_II_4f_r=Hf_II_4f[:,0]

Hf_III_5p_r=Hf_III_5p[:,0]
Hf_III_4f_r=Hf_III_4f[:,0]

Hf_I_5p_P=Hf_I_5p[:,3]
Hf_I_4f_P=Hf_I_4f[:,3]

Hf_II_5p_P=Hf_II_5p[:,3]
Hf_II_4f_P=Hf_II_4f[:,3]

Hf_III_5p_P=Hf_III_5p[:,3]
Hf_III_4f_P=Hf_III_4f[:,3]
#%%
plt.plot(Hf_I_4f_r,Hf_I_4f_P**2,label='4f')
plt.plot(Hf_I_5p_r,Hf_I_5p_P**2,label='5p')
plt.xlabel('r (a_0)')
plt.ylabel('P^2')
plt.title('Hf I')
plt.legend()
plt.xlim(0,5)
#%%
plt.plot(Hf_II_4f_r,Hf_II_4f_P**2,label='4f')
plt.plot(Hf_II_5p_r,Hf_II_5p_P**2,label='5p')
plt.xlabel('r (a_0)')
plt.ylabel('P^2')
plt.title('Hf II')
plt.legend()
plt.xlim(0,5)
#%%
plt.plot(Hf_III_4f_r,Hf_III_4f_P**2,label='4f')
plt.plot(Hf_III_5p_r,Hf_III_5p_P**2,label='5p')
plt.xlabel('r (a_0)')
plt.ylabel('P^2')
plt.title('Hf III')
plt.legend()
plt.xlim(0,5)
#%%
plt.plot(Hf_I_4f_r,Hf_I_4f_P**2,label='Hf I')
plt.plot(Hf_II_4f_r,Hf_II_4f_P**2,label='Hf II')
plt.plot(Hf_III_4f_r,Hf_III_4f_P**2,label='Hf III')
plt.xlabel('r (a_0)')
plt.ylabel('P^2')
plt.title('4f')
plt.legend()
plt.xlim(0,5)
#%%
plt.plot(Hf_I_5p_r,Hf_I_5p_P**2,label='Hf I')
plt.plot(Hf_II_5p_r,Hf_II_5p_P**2,label='Hf II')
plt.plot(Hf_III_5p_r,Hf_III_5p_P**2,label='Hf III')
plt.xlabel('r (a_0)')
plt.ylabel('P^2')
plt.title('5p')
plt.legend()
plt.xlim(0,5)