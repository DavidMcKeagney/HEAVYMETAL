#!/usr/bin/env python3


import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--slater_arg")
parser.add_argument("--file")
parser.add_argument("--output_file")
parser.add_argument("--f1")
parser.add_argument("--f2")
parser.add_argument("--f3")
parser.add_argument("--f4")
parser.add_argument("--f5")
parser.add_argument("--s_val")
args=parser.parse_args()

def slaterscaling(file,f1,f2,f3,f4,f5,slater_val):
    if f1==1:
        file[0]=file[0][0:50]+slater_val + file[0][52:]
    elif f2==1:
        file[0]=file[0][0:52] + slater_val + file[0][54:]
    elif f3==1:
        file[0]=file[0][0:54]+slater_val+file[0][56:]
    elif f4==1:
        file[0]=file[0][0:56]+slater_val+file[0][58:]
    elif f5==1:
        file[0]=file[0][0:58] + slater_val+file[0][60:]
    return file[0]
#%%
in2=[]
with open(args.file,'r') as text_file:
    for lines in text_file:
        in2.append(lines)
slaterscaling(in2, args.f1, args.f2, args.f3, args.f4, args.f5, args.s_val)

#%%
with open(args.output_file,'x') as f:
    print(f) 
    for lines in in2:
        f.write(lines)
f.close()

