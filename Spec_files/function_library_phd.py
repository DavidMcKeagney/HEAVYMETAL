# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 16:55:09 2024

@author: padmin
"""

import numpy as np
import matplotlib.pyplot as plt
import glob 
import re 
import csv
import pandas as pd
#%% requires spec file
def configs(spec_file):
    config_even={}
    config_odd={}
    all_config=[]
    with open(spec_file) as textfile:
        for lines in textfile:
            if len(lines.split())<7:
                all_config.append(lines.split())
    for spec_config in all_config:
        if spec_config[2]=='ground':
            config_even.update({'5d106s':'1'})
            config_odd.update({spec_config[5]:'1'})
        else:
            config_even.update({spec_config[2]:spec_config[0]})
            config_odd.update({spec_config[5]:spec_config[0]})
    return config_even,config_odd
        
#%% Extracts LS,J and config of even and odd terms in radiative transitions, requires Nist file preformatted and spec file
def NISTformat(spec_file,data,even_odd_split):
    for a,gn in enumerate(data):
        for b,ggn in enumerate(gn):
            data[a][b]=re.sub('[.,*"]','',ggn)
    for a,gn in enumerate(data):
        for b,ggn in enumerate(gn):
            data[a][b]=re.sub('<.*?>','',ggn)
    for a in data:
        if a[0]=='':
            data.remove(a)
    for d,a in enumerate(data):    
        if configs[0][a[4]]==None:
            c1=a[4]
            c2=a[5]
            data[d][4]=configs(spec_file)[0][c2]
            data[d][5]=configs(spec_file)[1][c1]
        else:
            data[d][4]=configs(spec_file)[0][data[d][4]]
            data[d][5]=configs(spec_file)[1][data[d][5]]
    if even_odd_split==1:
        data_even=[]
        data_odd=[]
        for d in data:
            data_even.append([d[0],d[2],d[4]])
            data_odd.append(d[1],d[3],d[5])
        return data_even,data_odd
    else:
        return data
    
    
#%% requires the sorted file from cowan gives the energy level and its configuration could add J later
def findenergysorted(data,T,J,c):
    df=pd.DataFrame(data,columns=['E','conf','J','frac','num','par','LS'])
    sub_df=df.loc[(df.LS==T)&(df.J==J)&(df.conf==c)]
    if sub_df.size>0:
        return sub_df.iloc[0]['E'],sub_df.iloc[0]['c']