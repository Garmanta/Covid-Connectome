#!/bin/bash
#design_change_fsl.sh v1.0
#This script is meant to create the feat_design.fsf in the BIDS directory 
#After creating the .fsf files, the objective is to run FSL's Feat.
#This is the first step in the preprocessing steps for the fmri images

#Its meant to be run with for_each in the main BIDS directory (where all the sub-XX are located) in the following form:
#for_each sub-* : design_change.sh /PATH/TO/BIDS/FILE/fsl_design/design_aroma_treatment.fsf PRE IN/PRE_feat_design.fsf

#This script must be initialized in the users bin directory in /opt/bin/
#The usual guideline is to create your custom pipeline in FEAT with sub-01. Then import the .fsf file and use it with design_change

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


#--------------------------------------------------------------------------------------
#Version v1.0.
#--------------------------------------------------------------------------------------
#Get the lastest version at:
#--------------------------------------------------------------------------------------
#script by Alejandro Garma Oehmichen
#--------------------------------------------------------------------------------------


