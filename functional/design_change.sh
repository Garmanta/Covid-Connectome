#!/bin/bash
#This script is meant to create the feat_design.fsf in the directory 
#Its meant to be run with for_each in the following form:
#for_each sub-* : design_change.sh /home/sphyrna/thesis/functional/fsl_design/design_aroma_treatment.fsf PRE IN/PRE_feat_design.fsf
#for_each sub-* : design_change.sh /home/sphyrna/thesis/functional/fsl_design/design_aroma_control.fsf PRE IN/PRE_feat_design.fsf

# Check if exactly three arguments are given
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 INPUT_PATH CHANGE_STRING OUTPUT_PATH"
    exit 1
fi

# Assign arguments to variables
INPUT_PATH="${fsl_design/feat_design.fsf}"
CHANGE_STRING="$2"
OUTPUT_PATH="$3"

# Check if the input file exists
if [ ! -f "$INPUT_PATH" ]; then
    echo "Error: Input file does not exist."
    exit 2
fi

# Use sed to replace all instances of "sub-01" with CHANGE_STRING
sed "s/sub-01/${CHANGE_STRING}/g" "$INPUT_PATH" > "$OUTPUT_PATH"

echo "Processing complete. Output saved to $OUTPUT_PATH"

