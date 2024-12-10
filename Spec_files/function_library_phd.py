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
# for ground state config put in the config labellng and not the ground state
def configs_even(spec_file):
    config_even={}
    all_config=[]
    with open(spec_file) as textfile:
        for lines in textfile:
            if len(lines.split())<7:
                all_config.append(lines.split())
    
    for spec_config in all_config:
        if spec_config[2]=='ground':
            config_even.update({'5d106s':'1'})
        elif spec_config[2]=='au':
            break 
        else:
            config_even.update({spec_config[2]:spec_config[0]})
    return config_even

def configs_odd(spec_file):
    config_odd={}
    all_config_odd=[]
    with open(spec_file) as textfile:
        for lines in textfile:
            if len(lines.split())<7:
                all_config_odd.append(lines.split())
                
    
    for spec_config_odd in all_config_odd:
        a='-'+spec_config_odd[0]
        if spec_config_odd[2] != 'au':
            config_odd.update({spec_config_odd[5]:a})
        else:
            config_odd.update({spec_config_odd[4]:a})
    return config_odd
        
#%% Extracts LS,J and config of even and odd terms in radiative transitions, requires Nist file preformatted and spec file
#Requires Nist file to be csv file with ',' delimited
def NISTformat(data):
    for a,gn in enumerate(data):
        data[a][4]=re.sub('[.]','',gn[4])
        data[a][7]=re.sub('[.]','',gn[7])
        for b,ggn in enumerate(gn):
            data[a][b]=re.sub('[=*"]','',ggn)
            #for loops remove funky symbols from the nist csv file except the bracket terms which I haven't figured out yet    
    for a,gn in enumerate(data):
        for b,ggn in enumerate(gn):
            data[a][b]=re.sub('<.*?>','',ggn)
    for a,gn in enumerate(data):
        for b,ggn in enumerate(gn):
            if re.search('[()]', data[a][b]) != None:
                data[a][b]=data[a][b].replace('(','').replace(')','')
    data=np.array(data) 
    boolean_index=data[:,0]!='' #this only extracts nist values with known oscillator strenghts this is the zero index
    sliced_data=data[boolean_index] 
    # TODO The next chunk of code replaces config with config serial number from cowan assumes dictionary of configs 
    # TODO Find a way to swap odd parity configs appearing in even parity coolumn 
    #for a,gn in enumerate(sliced_data):
    #    if gn[4] in config_even.keys() == False:
    #        temp_odd=sliced_data[a][4]
    #        temp_even=sliced_data[a][5]
    #        sliced_data[a][4]=config_even[temp_even]
    #        sliced_data[a][5]=config_odd[temp_odd]
    #    else:
    #        sliced_data[a][4]=config_even[gn[4]]
    #        sliced_data[a][5]=config_odd[gn[5]]
    return sliced_data


#%% requires the sorted file from cowan gives the energy level and its configuration could add J later using boolean logic 
# requires the data to be in a numpy array
#Need to ensure that the second parity configurations are the absolute values so the minus sign isn't there
def findenergysorted(data,T,J,c):
    boolean_index= np.logical_and(data[:,1]==c,data[:,2]==J)
    data_T_c=data[boolean_index]
    boolean_T=data_T_c==data[:,6]
    data_final=data_T_c[boolean_T]
    return data_final
#%% Use the energies from the previous funtions
#Make spec file into numpy array
def findspec(data,El,Eu):
    boolean=np.logical_and(data[:,0]==El,data[0:,5]==Eu)
    spec_data=data[boolean]
    return spec_data
#%%
samp_conf=[]
paths= [p.replace('\\','/') for p in glob.glob('C:/Users/David*/Desktop/AU_*/*')]
with open(paths[0]) as file:
    for f in file:
        if len(f.split())<7:
            samp_conf.append(f.split())
    
    
    