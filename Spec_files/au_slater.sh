#!/usr/bin/bash


CURRENT_PATH=$(pwd)
EXECUTABLE=${CURRENT_PATH}/au_slater.py
FILE=${CURRENT_PATH}/test_file.in2
OUTPUT=${CURRENT_PATH}/au_spin_orbit.in2

for ((i=50 ; i<100; i++)); do
        rm au_spin_orbit.in2 
	${EXECUTABLE} --slater_arg ${i} --file ${FILE} --output_file ${OUTPUT} 
	cowan.sh -dxxl au_spin_orbit
	mv au_spin_orbit.spec ${CURRENT_PATH}/slater_spec/au_spin_orbit_${i}.spec
done
