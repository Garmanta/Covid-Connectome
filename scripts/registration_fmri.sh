#!/usr/bin/env bash
#registration_fmri.sh v1.0
#The objective of this script is to complete the preprocessing of fMRI images to create a FC connectome. After this script is done, CONN must be used to follow the same steps done in the thesis. Unfortenetly, the individual script to run CONN doesnt exist. Read the methodology in the thesis for guidelines.
#The script is a continuation of registration_dwi.sh
#design_change_fsl --> synbold_fmri --> registration_fmri

#This script does the following steps:
#1: Register the T1w to the MNI's T1w through a 12dof and warp transformation. 
#2: Use affine matrix and deformation field to obtain native T1w in MNI space.
#3: Extracts a single volume of the fMRI to be used in the registration
#6: Transforms the fMRI to native T1w space and then to MNI space through the usage of two deformation fields

#The input files are:  *_T1w_bet.nii.gz,  BOLD_u.nii.gz 
#The principal output is: ${sub}_rest_in_mni_warp.nii.gz 


for i in {1..51}
do
printf -v sub "sub-%02d" $i
echo "-------------------------------------"
echo "|"
echo "| Running : " ${sub}
echo "|"
echo "-------------------------------------"
cd ${sub}/func

#In case dwi registration was done before this script is run.
cp -r ../dwi/ants_reg ants_reg

#Registration T1 --> MNI
mkdir ants_reg
mkdir ants_reg/t1_2_mni2mm
../../antsRegistrationSyN.sh -d 3 -f ../../MNI152_T1_2mm_brain.nii.gz -m ../anat/*_T1w_bet.nii.gz -o ants_reg/t1_2_mni2mm/t1_2_mni2mm -n 12

#Registration fMRI --> T1
mkdir ants_reg/fmri_2_t1
cp syn_outputs/BOLD_u.nii.gz ${sub}_rest_preprocessed.nii.gz

#Extract a single volume to register to
fslroi ${sub}_rest_preprocessed.nii.gz ${sub}_rest_preprocessed_3dvol.nii.gz 250 1
../../antsRegistrationSyN.sh -d 3 -f ../anat/*_T1w_bet.nii.gz -m ${sub}_rest_preprocessed_3dvol.nii.gz -o ants_reg/fmri_2_t1/fmri_2_t1 -n 12 -t a

#Apply transform fMRI --> T1 --> MNI
antsApplyTransforms -d 3 -e 3 -r ../../MNI152_T1_2mm_brain.nii.gz -i ${sub}_rest_preprocessed.nii.gz -t ants_reg/t1_2_mni2mm/t1_2_mni2mm1Warp.nii.gz -t ants_reg/t1_2_mni2mm/t1_2_mni2mm0GenericAffine.mat -t ants_reg/fmri_2_t1/fmri_2_t10GenericAffine.mat -o ${sub}_rest_in_mni_warp.nii.gz --float

cd ../..
done

echo "----------------------------------"
echo "|"
echo "| Finished structrual registration!"
echo "|"
echo "----------------------------------"


#--------------------------------------------------------------------------------------
#Version v1.0.
#--------------------------------------------------------------------------------------
#Get the lastest version at:
#--------------------------------------------------------------------------------------
#script by Alejandro Garma Oehmichen
#--------------------------------------------------------------------------------------

