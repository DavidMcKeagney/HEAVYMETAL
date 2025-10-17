# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 18:31:10 2025

@author: padmin
"""

import numpy as np
import matplotlib.pyplot as plt 
import function_library_phd
import glob as glob 
import re
#%%
partial_cross_sections=glob.glob('C:\\Users\padmin\OneDrive\Desktop\partial_cross_sections\*')
partial_cross_sections_new_scalings=glob.glob('C:\\Users\padmin\OneDrive\Desktop\partial_cross_sections_new_scalings\*')
partial_cross_sections_new_scalings2=glob.glob('C:\\Users\padmin\OneDrive\Desktop\partial_cross_sections_new_scalings2\*')
def pcs(filepath):
    partial=[]
    with open(filepath) as file:
        for lines in file:
            partial.append(lines.split())
    partial=np.array(partial,dtype=float)
    return partial
#%% 5D3_2 all configs
partial_AuII_5D3_2=pcs(partial_cross_sections[0])
partial_AuII_5d86s2_1_5_5D3_2=pcs(partial_cross_sections[2])
partial_AuII_5d86s2_2_5_1_5_5D3_2=pcs(partial_cross_sections[6])
partial_AuII_5d86s2_2_5_5D3_2=pcs(partial_cross_sections[10])
partial_AuII_5d96s_1_5_5D3_2=pcs(partial_cross_sections[14])
partial_AuII_5d96s_2_5_5D3_2=pcs(partial_cross_sections[17])
partial_AuI_5d96s2_1_5_5D3_2=pcs(partial_cross_sections[24])
partial_AuI_5d96s2_2_5_5D3_2=pcs(partial_cross_sections[28])
#%% 5D5_2 all configs 
partial_AuII_5D5_2=pcs(partial_cross_sections[1])
partial_AuII_5d86s2_1_5_5D5_2=pcs(partial_cross_sections[3])
partial_AuII_5d86s2_2_5_1_5_5D5_2=pcs(partial_cross_sections[7])
partial_AuII_5d86s2_2_5_5D5_2=pcs(partial_cross_sections[11])
partial_AuII_5d96s_1_5_5D5_2=pcs(partial_cross_sections[15])
partial_AuII_5d96s_2_5_5D5_2=pcs(partial_cross_sections[18])
partial_AuI_5d96s2_1_5_5D5_2=pcs(partial_cross_sections[25])
partial_AuI_5d96s2_2_5_5D5_2=pcs(partial_cross_sections[29])
#%% 5P3_2 all configs
partial_AuII_5d86s2_1_5_5P3_2=pcs(partial_cross_sections[4])
partial_AuII_5d86s2_2_5_1_5_5P3_2=pcs(partial_cross_sections[8])
partial_AuII_5d86s2_2_5_5P3_2=pcs(partial_cross_sections[12])
partial_AuII_5d96s_1_5_5P3_2=pcs(partial_cross_sections[16])
partial_AuII_5d96s_2_5_5P3_2=pcs(partial_cross_sections[20])
partial_AuI_5d96s2_1_5_5P3_2=pcs(partial_cross_sections[27])
partial_AuI_5d96s2_2_5_5P3_2=pcs(partial_cross_sections[31])
#%% 6S1_2 all configs
partial_AuII_5d86s2_1_5_6S1_2=pcs(partial_cross_sections[5])
partial_AuII_5d86s2_2_5_1_5_6S1_2=pcs(partial_cross_sections[9])
partial_AuII_5d86s2_2_5_6S1_2=pcs(partial_cross_sections[13])
partial_AuII_5d96s_2_5_6S1_2=pcs(partial_cross_sections[21])
#%% 5P1_2 all configs 
partial_AuII_5d96s_2_5_5P1_2=pcs(partial_cross_sections[19])
partial_AuII_5P1_2=pcs(partial_cross_sections[22])
partial_AuI_5d96s2_1_5_5P1_2=pcs(partial_cross_sections[26])
partial_AuI_5d96s2_2_5_5P1_2=pcs(partial_cross_sections[30])
#%% 5P1_2 plots
plt.plot(partial_AuII_5d96s_2_5_5P1_2[:,0],partial_AuII_5d96s_2_5_5P1_2[:,1],label='AuII: 5d96s, j_5d=5/2')
plt.plot(partial_AuII_5P1_2[:,0],partial_AuII_5P1_2[:,1],label='AuII: 5d10')
plt.plot(partial_AuI_5d96s2_1_5_5P1_2[:,0],partial_AuI_5d96s2_1_5_5P1_2[:,1],label='AuI: 5d96s2, j_5d=3/2')
plt.plot(partial_AuI_5d96s2_2_5_5P1_2[:,0],partial_AuI_5d96s2_2_5_5P1_2[:,1],label='AuI: 5d96s2, j_5d=5/2')
plt.xlabel('Energy')
plt.ylabel('Intensity')
plt.title('5P1_2')
plt.legend()
#%% 6S1_2 plots
plt.plot(partial_AuII_5d86s2_1_5_6S1_2[:,0],partial_AuII_5d86s2_1_5_6S1_2[:,1],label='AuII: 5d86s2 2 x j_5d=3/2')
plt.plot(partial_AuII_5d86s2_2_5_1_5_6S1_2[:,0],partial_AuII_5d86s2_2_5_1_5_6S1_2[:,1],label='AuII: 5d86s2 j_5d=3/2, j_6s=5/2')
plt.plot(partial_AuII_5d86s2_2_5_6S1_2[:,0],partial_AuII_5d86s2_2_5_6S1_2[:,1],label='AuII: 5d86s2 2 x j_5d=5/2')
plt.plot(partial_AuII_5d96s_2_5_6S1_2[:,0],partial_AuII_5d96s_2_5_6S1_2[:,1],label='AuII: 5d96s j_5d= 5/2')
plt.xlabel('Energy')
plt.ylabel('Intensity')
plt.title('6S1_2')
plt.legend()
#%% 5P3_2 plots
plt.plot(partial_AuII_5d86s2_1_5_5P3_2[:,0],partial_AuII_5d86s2_1_5_5P3_2[:,1],label='AuII: 5d86s2 j_5d=3/2')
plt.plot(partial_AuII_5d86s2_2_5_1_5_5P3_2[:,0],partial_AuII_5d86s2_2_5_1_5_5P3_2[:,1],label='AuII: 5d86s2 j_5d=3/2,j_5d=5/2')
plt.plot(partial_AuII_5d86s2_2_5_5P3_2[:,0],partial_AuII_5d86s2_2_5_5P3_2[:,1],label='AuII: 5d86s2 j_5d=5/2')
plt.plot(partial_AuII_5d96s_1_5_5P3_2[:,0],partial_AuII_5d96s_1_5_5P3_2[:,1],label='AuII: 5d96s j_5d=3/2')
plt.plot(partial_AuII_5d96s_2_5_5P3_2[:,0],partial_AuII_5d96s_2_5_5P3_2[:,1],label='AuII: 5d96s j_5d=5/2')
plt.plot(partial_AuI_5d96s2_1_5_5P3_2[:,0],partial_AuI_5d96s2_1_5_5P3_2[:,1],label='AuI: 5d96s2 j_5d=3/2')
plt.plot(partial_AuI_5d96s2_2_5_5P3_2[:,0],partial_AuI_5d96s2_2_5_5P3_2[:,1],label='AuI: 5d96s2 j_5d=5/2')
plt.xlabel('Energy')
plt.ylabel('Intensity')
plt.title('5P3_2')
plt.legend()
#%% 5D5_2 plots
plt.plot(partial_AuII_5D5_2[:,0],partial_AuII_5D5_2[:,1],label='AuII: 5d10')
plt.plot(partial_AuII_5d86s2_1_5_5D5_2[:,0],partial_AuII_5d86s2_1_5_5D5_2[:,1],label='AuII: 5d86s2 2 x j_5d=3/2')
plt.plot(partial_AuII_5d86s2_2_5_1_5_5D5_2[:,0],partial_AuII_5d86s2_2_5_1_5_5D5_2[:,1],label='AuII: 5d86s2 j_5d=3/2, j_5d=5/2')
plt.plot(partial_AuII_5d86s2_2_5_5D5_2[:,0],partial_AuII_5d86s2_2_5_5D5_2[:,1],label='AuII: 5d86s2 2 x j_5d=5/2')
plt.plot(partial_AuII_5d96s_1_5_5D5_2[:,0],partial_AuII_5d96s_1_5_5D5_2[:,1],label='AuII: 5d96s j_5d=3/2')
plt.plot(partial_AuII_5d96s_2_5_5D5_2[:,0],partial_AuII_5d96s_2_5_5D5_2[:,1],label='AuII: 5d96s j_5d=5/2')
plt.plot(partial_AuI_5d96s2_1_5_5D5_2[:,0],partial_AuI_5d96s2_1_5_5D5_2[:,1],label='AuI: 5d96s2 j_5d=3/2 ')
plt.plot(partial_AuI_5d96s2_2_5_5D5_2[:,0],partial_AuI_5d96s2_2_5_5D5_2[:,1],label='AuI: 5d96s2 j_5d=5/2')
plt.xlabel('Energy')
plt.ylabel('Intensity')
plt.title('5D5_2')
plt.legend()
#%% 5D3_2 plots 
plt.plot(partial_AuII_5D3_2[:,0],partial_AuII_5D3_2[:,1],label='AuII: 5d10')
plt.plot(partial_AuII_5d86s2_1_5_5D3_2[:,0],partial_AuII_5d86s2_1_5_5D3_2[:,1],label='AuII: 5d86s2 2 x j_5d=3/2')
plt.plot(partial_AuII_5d86s2_2_5_1_5_5D3_2[:,0],partial_AuII_5d86s2_2_5_1_5_5D3_2[:,1],label='AuII: 5d86s2 j_5d=3/2, j_5d=5/2')
plt.plot(partial_AuII_5d86s2_2_5_5D3_2[:,0],partial_AuII_5d86s2_2_5_5D3_2[:,1],label='AuII: 5d86s2 2 x j_5d=5/2')
plt.plot(partial_AuII_5d96s_1_5_5D3_2[:,0],partial_AuII_5d96s_1_5_5D3_2[:,1],label='AuII: 5d96s j_5d=3/2')
plt.plot(partial_AuII_5d96s_2_5_5D3_2[:,0],partial_AuII_5d96s_2_5_5D3_2[:,1],label='AuII: 5d96s j_5d=5/2')
plt.plot(partial_AuI_5d96s2_1_5_5D3_2[:,0],partial_AuI_5d96s2_1_5_5D3_2[:,1],label='AuI: 5d96s2 j_5d=3/2 ')
plt.plot(partial_AuI_5d96s2_2_5_5D3_2[:,0],partial_AuI_5d96s2_2_5_5D3_2[:,1],label='AuI: 5d96s2 j_5d=5/2')
plt.xlabel('Energy')
plt.ylabel('Intensity')
plt.title('5D3_2')
plt.legend()
#%% 5D3_2 80% scaling
partial_AuII_5d96s_1_5_5D3_2_new_scaling=pcs(partial_cross_sections_new_scalings[0])
partial_AuII_5d96s_2_5_5D3_2_new_scaling=pcs(partial_cross_sections_new_scalings[3])
#%% 5D5_2 80% scaling
partial_AuII_5d96s_1_5_5D5_2_new_scaling=pcs(partial_cross_sections_new_scalings[1])
partial_AuII_5d96s_2_5_5D5_2_new_scaling=pcs(partial_cross_sections_new_scalings[4])
#%% 5P3_2 80% scaling
partial_AuII_5d96s_1_5_5P3_2_new_scaling=pcs(partial_cross_sections_new_scalings[2])
partial_AuII_5d96s_2_5_5P3_2_new_scaling=pcs(partial_cross_sections_new_scalings[5])
#%% plots 5D3_2 new vs old scaling
plt.plot(partial_AuII_5d96s_1_5_5D3_2_new_scaling[:,0],partial_AuII_5d96s_1_5_5D3_2_new_scaling[:,1],label='AuII: 5d96s j_5d=3/2 new scaling')
plt.plot(partial_AuII_5d96s_2_5_5D3_2_new_scaling[:,0],partial_AuII_5d96s_2_5_5D3_2_new_scaling[:,1],label='AuII: 5d96s j_5d=5/2 new scaling')
plt.plot(partial_AuII_5d96s_1_5_5D3_2[:,0],partial_AuII_5d96s_1_5_5D3_2[:,1],label='AuII: 5d96s j_5d=3/2 old scaling')
plt.plot(partial_AuII_5d96s_2_5_5D3_2[:,0],partial_AuII_5d96s_2_5_5D3_2[:,1],label='AuII: 5d96s j_5d=5/2 old scaling')
plt.xlabel('Energy')
plt.ylabel('Intensity')
plt.title('5D3_2')
plt.legend()
#%% plots of 5D5_2 new vs old scaling 
plt.plot(partial_AuII_5d96s_1_5_5D5_2_new_scaling[:,0],partial_AuII_5d96s_1_5_5D5_2_new_scaling[:,1],label='AuII: 5d96s j_5d=3/2 new scaling')
plt.plot(partial_AuII_5d96s_2_5_5D5_2_new_scaling[:,0],partial_AuII_5d96s_2_5_5D5_2_new_scaling[:,1],label='AuII: 5d96s j_5d=5/2 new scaling')
plt.plot(partial_AuII_5d96s_1_5_5D5_2[:,0],partial_AuII_5d96s_1_5_5D5_2[:,1],label='AuII: 5d96s j_5d=3/2 old scaling')
plt.plot(partial_AuII_5d96s_2_5_5D5_2[:,0],partial_AuII_5d96s_2_5_5D5_2[:,1],label='AuII: 5d96s j_5d=5/2 old scaling')
plt.xlabel('Energy')
plt.ylabel('Intensity')
plt.title('5D5_2')
plt.legend()
#%% plots of 5P3_2 new vs old scaling
plt.plot(partial_AuII_5d96s_1_5_5P3_2_new_scaling[:,0],partial_AuII_5d96s_1_5_5P3_2_new_scaling[:,1],label='AuII: 5d96s j_5d=3/2 new scaling')
plt.plot(partial_AuII_5d96s_2_5_5P3_2_new_scaling[:,0],partial_AuII_5d96s_2_5_5P3_2_new_scaling[:,1],label='AuII: 5d96s j_5d=5/2 new scaling')
plt.plot(partial_AuII_5d96s_1_5_5P3_2[:,0],partial_AuII_5d96s_1_5_5P3_2[:,1],label='AuII: 5d96s j_5d=3/2 old scaling')
plt.plot(partial_AuII_5d96s_2_5_5P3_2[:,0],partial_AuII_5d96s_2_5_5P3_2[:,1],label='AuII: 5d96s j_5d=5/2 old scaling')
plt.xlabel('Energy')
plt.ylabel('Intensity')
plt.title('5P3_2')
plt.legend()
#%% 5D3_2 90% scaling
partial_AuII_5d96s_1_5_5D3_2_new_scaling2=pcs(partial_cross_sections_new_scalings2[0])
partial_AuII_5d96s_2_5_5D3_2_new_scaling2=pcs(partial_cross_sections_new_scalings2[3])
#%% 5D5_2 90% scaling
partial_AuII_5d96s_1_5_5D5_2_new_scaling2=pcs(partial_cross_sections_new_scalings2[1])
partial_AuII_5d96s_2_5_5D5_2_new_scaling2=pcs(partial_cross_sections_new_scalings2[4])
#%% 5P3_2 90% scaling
partial_AuII_5d96s_1_5_5P3_2_new_scaling2=pcs(partial_cross_sections_new_scalings2[2])
partial_AuII_5d96s_2_5_5P3_2_new_scaling2=pcs(partial_cross_sections_new_scalings2[5])
#%% plots 5D3_2 new vs old scaling
plt.plot(partial_AuII_5d96s_1_5_5D3_2_new_scaling2[:,0],partial_AuII_5d96s_1_5_5D3_2_new_scaling2[:,1],label='AuII: 5d96s j_5d=3/2 new scaling')
plt.plot(partial_AuII_5d96s_2_5_5D3_2_new_scaling2[:,0],partial_AuII_5d96s_2_5_5D3_2_new_scaling2[:,1],label='AuII: 5d96s j_5d=5/2 new scaling')
plt.plot(partial_AuII_5d96s_1_5_5D3_2[:,0],partial_AuII_5d96s_1_5_5D3_2[:,1],label='AuII: 5d96s j_5d=3/2 old scaling')
plt.plot(partial_AuII_5d96s_2_5_5D3_2[:,0],partial_AuII_5d96s_2_5_5D3_2[:,1],label='AuII: 5d96s j_5d=5/2 old scaling')
plt.xlabel('Energy')
plt.ylabel('Intensity')
plt.title('5D3_2')
plt.legend()
#%% plots of 5D5_2 new vs old scaling 
plt.plot(partial_AuII_5d96s_1_5_5D5_2_new_scaling2[:,0],partial_AuII_5d96s_1_5_5D5_2_new_scaling2[:,1],label='AuII: 5d96s j_5d=3/2 new scaling')
plt.plot(partial_AuII_5d96s_2_5_5D5_2_new_scaling2[:,0],partial_AuII_5d96s_2_5_5D5_2_new_scaling2[:,1],label='AuII: 5d96s j_5d=5/2 new scaling')
plt.plot(partial_AuII_5d96s_1_5_5D5_2[:,0],partial_AuII_5d96s_1_5_5D5_2[:,1],label='AuII: 5d96s j_5d=3/2 old scaling')
plt.plot(partial_AuII_5d96s_2_5_5D5_2[:,0],partial_AuII_5d96s_2_5_5D5_2[:,1],label='AuII: 5d96s j_5d=5/2 old scaling')
plt.xlabel('Energy')
plt.ylabel('Intensity')
plt.title('5D5_2')
plt.legend()
#%% plots of 5P3_2 new vs old scaling
plt.plot(partial_AuII_5d96s_1_5_5P3_2_new_scaling2[:,0],partial_AuII_5d96s_1_5_5P3_2_new_scaling2[:,1],label='AuII: 5d96s j_5d=3/2 new scaling')
plt.plot(partial_AuII_5d96s_2_5_5P3_2_new_scaling2[:,0],partial_AuII_5d96s_2_5_5P3_2_new_scaling2[:,1],label='AuII: 5d96s j_5d=5/2 new scaling')
plt.plot(partial_AuII_5d96s_1_5_5P3_2[:,0],partial_AuII_5d96s_1_5_5P3_2[:,1],label='AuII: 5d96s j_5d=3/2 old scaling')
plt.plot(partial_AuII_5d96s_2_5_5P3_2[:,0],partial_AuII_5d96s_2_5_5P3_2[:,1],label='AuII: 5d96s j_5d=5/2 old scaling')
plt.xlabel('Energy')
plt.ylabel('Intensity')
plt.title('5P3_2')
plt.legend()