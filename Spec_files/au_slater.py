#!/usr/bin/env python3


import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--slater_arg")
parser.add_argument("--file")
parser.add_argument("--output_file")
args=parser.parse_args()
#%%
in2=[]
with open(args.file,'r') as text_file:
    for lines in text_file:
        in2.append(lines)
#%%
in2[0]=in2[0][0:54]+str(args.slater_arg)+str(args.slater_arg)+in2[0][58:]
#%%
with open(args.output_file,'x') as f:
    print(f) 
    for lines in in2:
        f.write(lines)
f.close()

