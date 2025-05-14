#!/usr/bin/env bash
#connectome_dwi.sh v1.0
#The objective of this script is to use the tractography and recon-all to generate a structural connectome in DK atlas space.
#The script is a continuation of registration_dwi.sh
#preprocessing_dwi --> registration_dwi --> processing_dwi --> recon_all --> connectome_dwi

#This script does the following steps:
#1: Use labelconvert to obtain a node image from the subject T1w in DK atlas space
#2: Transform the node image from T1w native space to DWI space
#3: Create connectome from tractography. Applies symmetry, zero diagonal, size normalization

#The input files are:  aparc+aseg.mgz, FreeSurferColorLUT.tx, fs_default.txt, dwi_2_t1 registration
#The principal output is: ${sub}_tracks_connectome_dk.csv 

for i in {1..51}
do
printf -v sub "sub-%02d" $i
echo "-------------------------------------"
echo "|"
echo "| Running : " ${sub}
echo "|"
echo "-------------------------------------"
cd ${sub}/anat/reconall

#Create node image using fs_default. aparc+aseg.mgz is the T1w subject image in DK atlas space.
labelconvert ${sub}/mri/aparc+aseg.mgz ../../../FreeSurferColorLUT.txt ../../../fs_default.txt ${sub}_nodes_dk.nii.gz
mv ${sub}_nodes_dk.nii.gz ../../dwi/${sub}_nodes_dk.nii.gz
cd ../../dwi

#Transform to MNI space
antsApplyTransforms -d 3 -i ${sub}_nodes_dk.nii.gz -r ${sub}_corrected_b0_bet.nii.gz -n NearestNeighbor -t [ants_reg/dwi_2_t1/dwi_2_t10GenericAffine.mat,1] -o ${sub}_nodes_dk_in_dwi.nii.gz
mrconvert ${sub}_nodes_dk_in_mni.nii.gz ${sub}_nodes_dk_in_mni.mif
tck2connectome ${sub}_sift_1mtrack.tck ${sub}_nodes_dk_in_dwi.mif ${sub}_connectome_dk.csv -symmetric -zero_diagonal -scale_invnodevol -out_assignment ${sub}_tracks_connectome_dk.csv 

cd ../..
done

#--------------------------------------------------------------------------------------
#Version v1.0.
#--------------------------------------------------------------------------------------
#Get the lastest version at:
#--------------------------------------------------------------------------------------
#script by Alejandro Garma Oehmichen
#--------------------------------------------------------------------------------------

