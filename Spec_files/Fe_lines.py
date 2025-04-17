# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 18:54:44 2025

@author: David McKeagney
"""

import numpy as np
import re
import csv
#%%
def NISTformat(data):
    J_dict={'1/2':'0.5','3/2':'1.5','5/2':'2.5','7/2':'3.5','9/2':'4.5'}
    for a,gn in enumerate(data):
        data[a][12]=re.sub('[.]','',gn[12])
        data[a][15]=re.sub('[.]','',gn[15])
        if a !=0:
            data[a][6]=re.sub('[a-zA-Z]','',gn[6])
        for b,ggn in enumerate(gn):
            data[a][b]=re.sub('[=*"]','',ggn)
            #for loops remove funky symbols from the nist csv file except the bracket terms which I haven't figured out yet    
    for a,gn in enumerate(data):
        for b,ggn in enumerate(gn):
            data[a][b]=re.sub('<.*?>','',ggn)
    for a,gn in enumerate(data):
        for b,ggn in enumerate(gn):
            if re.search('[()]', data[a][b]) != None:
                data[a][b]=re.sub(r'\(.*?\)', '', data[a][b])
    data=np.array(data) 
    return data
#%%
NIST_values=[]
with open('C:/Users/David McKeagney/Downloads/Fe_lines.csv') as file:
    csv_reader=csv.reader(file,delimiter=',')
    for lines in csv_reader:
        NIST_values.append(lines)
        
#%%
data=NISTformat(NIST_values)[:, :19]  
#%%


#data[a][b]=data[a][b].replace('(','').replace(')','')
#%%
