#!/bin/bash

# Read the namelist file line by line
input_file='experiments/AE_2_s24.yml.bk'
output_file='experiments/AE_2_s24.yml'
name_file='data/list_46_re.txt'

# loop by name file. skip the first line.
#tail -n +2 "$name_file" | while IFS= read -r line; do
while IFS= read -r line; do
    # Extract the ID, from line's first 6 chracter
    sid=$(echo $line | cut -c1-6)
    # change sid in input file to SID 
    sed "s/sid/$sid/g" "$input_file" > "$output_file"
    # run the experiment
    pipenv run python main.py "$output_file" -G 1 #--omit_complete
done < $name_file
