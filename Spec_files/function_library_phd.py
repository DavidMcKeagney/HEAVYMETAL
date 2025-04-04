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
# TODO the if conditions involving spec_config[2]=='au' will need to be changed to whatever element is being used in the calculation 
# TODO the spec_config_odd[2] != 'au' will need to be changed according to 
def configs_even(spec_file):
    config_even={}
    all_config=[]
    with open(spec_file) as textfile:
        for lines in textfile:
            if len(lines.split())<7:
                all_config.append(lines.split())
    element=all_config[0][1]
    for spec_config in all_config:
        if spec_config[2]=='ground':
            config_even.update({'5d106s':'1'})
        elif spec_config[2]==element:
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
                
    element=all_config_odd[0][1]
    for spec_config_odd in all_config_odd:
        a='-'+spec_config_odd[0]
        if spec_config_odd[2] != element:
            config_odd.update({spec_config_odd[5]:a})
        else:
            config_odd.update({spec_config_odd[3]:a})
    return config_odd

        
#Extracts LS,J and config of even and odd terms in radiative transitions, requires Nist file preformatted and spec file
#Requires Nist file to be csv file with ',' delimited
#Requires that log_gf is a selected whaen saving the NIST page
#Requires that the config serial names are formatted such that they are the same as the nist format once regex has altered the Nist versions 
def NISTformat(data,config_even_dict,config_odd_dict):
    J_dict={'1/2':'0.5','3/2':'1.5','5/2':'2.5','7/2':'3.5','9/2':'4.5'}
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
    #The chunk of code below can be commented out if you are just looking for a more formatted version of the NIST page
    #The next chunk of code changes the NIST cofiguration names to the config serial number from Cowan
    boolean_index=data[:,0]!='' #this only extracts nist values with known oscillator strenghts, this implies nist data is preformatted so that log_gf is the first column
    sliced_data=data[boolean_index][1:] 
    # TODO The next chunk of code replaces config with config serial number from cowan assumes dictionary of configs 
    # TODO Find a way to swap odd parity configs appearing in even parity coolumn 
    for a,gn in enumerate(sliced_data):
        if gn[4] in config_even_dict.keys():
            sliced_data[a][4]=config_even_dict[gn[4]]
            sliced_data[a][7]=config_odd_dict[gn[7]]
        else:
            sliced_data[a][4]=config_odd_dict[gn[4]]
            sliced_data[a][7]=config_even_dict[gn[7]]

    for a,gn in enumerate(sliced_data):
        sliced_data[a,9]=J_dict[sliced_data[a,9]]
        sliced_data[a,6]=J_dict[sliced_data[a,6]]
    return sliced_data


# requires the sorted file from cowan gives the energy level and its configuration could add J later using boolean logic 
# requires the data to be in a numpy array
#Uses sorted file in LS coupling
def findenergysorted(data,T,J,c):
    sorted_file=[]
    with open(data) as sorte:
        for sf in sorte:
            if len(sf.split())>7:
                if len(sf.split())==33:
                    sorted_file.append(sf.split())
                else:
                    sorted_file.append(sf.split() + ['','','','',''])
    
    sorted_file=np.array(sorted_file)
    boolean_index= np.logical_and(sorted_file[:,1]==c,sorted_file[:,2]==J)
    data_T_c=sorted_file[boolean_index]
    boolean_T_c=data_T_c[:,6]==T
    data_final=data_T_c[boolean_T_c]
    return data_final
# Use the energies from the previous funtions
#Make spec file into numpy array
def findspec(data,El,Eu):
    spec_file=[]
    with open(data) as spec:
        for s in spec:
            if len(s.split())>16:
                spec_file.append(s.split())
    spec_file=np.array(spec_file)
    boolean=np.logical_and(spec_file[:,1]==El,spec_file[:,6]==Eu)
    spec_data=spec_file[boolean]
    return spec_data

def ConvolvingFunc(q,x,E,amp,sig,flag):
    if flag==0:
        return amp*(1/(np.sqrt(2*np.pi)*sig))*np.exp(-((x-E)/(sig))**2) #+ (A_vals/(2*np.pi))*(1/((x-E)**2 + (0.5*A_vals)**2)))
    elif flag==1:
        return amp*((sig*0.5*q+E-x)**2/((0.5*sig)**2+(E-x)**2))

    
    
    