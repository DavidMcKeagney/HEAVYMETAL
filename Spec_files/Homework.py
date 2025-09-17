# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 20:40:34 2025

@author: padmin
"""

import numpy as np 
#%%
sorted_file=[]
with open('C:/Users/padmin/OneDrive/Desktop/B.sorted') as file:
    for lines in file:
        if len(lines.split())>10:
            sorted_file.append(lines.split()[:27])
sorted_file=np.array(sorted_file)
#%%
energies=sorted_file[:,0].astype(float)
unique_energies=list(set(energies))
num_levels=len(list(set(energies)))
#%%
possible_terms=['2P','2D','2F','2G','2H','4P','4D','4F','4G','4H']
def checknumlevels(LSterm,sortedfile):
    numofeachlevel=[]
    for term in LSterm:
        temparray=sortedfile[sortedfile[:,6]==term]
        temparray=temparray[:,0]
        numofeachlevel.append(len(list(set(temparray))))
    return numofeachlevel
eachlevel=checknumlevels(possible_terms,sorted_file)

    
    