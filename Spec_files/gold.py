# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:02:02 2024

@author: G17-2
"""

import numpy as np
import matplotlib.pyplot as plt 
import csv
#%%
spec_data=np.loadtxt('C:/Users/G17-2/Desktop/au.spec',skiprows=3,dtype=str)
gold_NIST=[]
with open('C:/Users/G17-2/Desktop/lines1.csv') as csv_file:
    csv_read= csv.reader(csv_file, delimiter='"')
    for rows in csv_read:
        gold_NIST.append(rows)
#%%
i=1 
wavelength_NIST=[]
while i<len(gold_NIST):
    if gold_NIST[i][0]==gold_NIST[0][0]:
        i+=1
    else:
        wavelength_NIST.append(gold_NIST[i][0][2:9])
        i+=1
wavelength_NIST=np.array(wavelength_NIST,dtype=float)