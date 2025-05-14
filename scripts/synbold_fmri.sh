#!/usr/bin/env bash
#synbold_fmri.sh v1.0
#The objective of this script is to run SynBOLD-DisCo to generate synthetic fieldmaps to run a complete Eddy Distortion correction. This process is usually done with other b0 distortion corrections algos with fields maps, but alas, life happns
#The script is a continuation of design_change.sh
#design_change_fsl --> synbold_fmri

#This script does the following steps:
#1: Modifies the design.fsf file so its compatible with each subject
#2: Runs multiple instances of Feat
#3: MOves and copies the required files for SynBOLD
#4: Runs SynBOLD

#The input files are:  synbold_design.fsf, ${sub}_T1w_bet.nii.gz
#The principal output is: BOLD_u.nii.gz



for_each sub-* : design_change.sh /home/sphyrna/storage/subjects/control_processed/synbold_design/synbold_design.fsf PRE IN/func/PRE_synbold_design.fsf
#Faster than FSL's crappy multithreading option. 
for_each -nthreads 10 sub-* : feat IN/func/IN_feat_design.fsf

for i in {1..49}
do
printf -v sub "sub-%02d" $i
echo "-------------------------------------"
echo "|"
echo "| Running : " ${sub}
echo "|"
echo "-------------------------------------"
cd ${sub}/func 

#Prepares directories and files for SynBOLD
mkdir syn_inputs
mkdir syn_outputs

cp func.feat/filtered_func_data.nii.gz syn_inputs/BOLD_d.nii.gz
cp ../anat/${sub}_T1w_bet.nii.gz syn_inputs/T1.nii.gz

#Runs SynBOLD
docker run --rm \
-v $(pwd)/syn_inputs/:/INPUTS/ \
-v $(pwd)/syn_outputs/:/OUTPUTS/ \
-v /home/sphyrna/neuro_software/freesurfer/license.txt:/opt/freesurfer/license.txt \
-v /tmp:/tmp \
--user $(id -u):$(id -g) \
ytzero/synbold-disco:v1.4 --skull_stripped --motion_corrected --no_smoothing


cd ../..
done

#--------------------------------------------------------------------------------------
#Version v1.0.
#--------------------------------------------------------------------------------------
#Get the lastest version at:
#--------------------------------------------------------------------------------------
#script by Alejandro Garma Oehmichen
#--------------------------------------------------------------------------------------

