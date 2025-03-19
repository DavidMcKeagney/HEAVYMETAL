#!/usr/bin/python3

import numpy as np 

spec_file=[]
with open('au2.spec') as file:
	for lines in file:
		if len(lines.split())>17:
			spec_file.append(lines.split())


#returns list of absorption coefficients
def Absorption(file_array):
	B_vals=[]
	for lines in spec_file:
		if float(lines[1])<float(lines[6]):
			B=(0.004965)*(10**24)*(np.exp(float(lines[15]))/(float(lines[11])*(2*float(lines[7])+1)))
		else:
			B=(0.004965)*(10**24)*(np.exp(float(lines[15]))/(float(lines[11])*(2*float(lines[2])+1)))
	
		B_vals.append(B)
	return np.array(B_vals)

#B_vals=Absorption(spec_file)

gf_vals=[np.exp(float(lines[15])) for lines in spec_file]
dE=np.array([float(i) for i in np.array(spec_file)[:,11]])

#returns array of values for convolved gaussian the mean is the dE value and the amplitude is either absorbption coefficient or absorbption cross section
def ConvolvingFunc(x,E,amp):
	return 40*np.sqrt(np.log(2)/(np.pi))*amp*np.exp(-4*np.log(2)*((x-E)/(0.05))**2)

	
E_vals=np.arange(50,130,0.1)
Conv_total=np.zeros((len(E_vals),len(dE)))
i=0
while i<len(dE):
	j=0
	conv=ConvolvingFunc(E_vals,dE[i],gf_vals[i])
	while j<len(E_vals):
		Conv_total[j][i]+=conv[j]
		j+=1
	i+=1

Conv_gfvals=np.sum(Conv_total,axis=1)

file_data=[' '.join([str(i) for i in Conv_gfvals]),' '.join([str(i) for i in E_vals])]


#text files first line is the convolved values and the second line is E_vals
with open('conv_vals.txt','w') as text_file:
	for data_lines in file_data:
		text_file.write('%s/n' %data_lines)
	text_file.close()
	

	
		 
