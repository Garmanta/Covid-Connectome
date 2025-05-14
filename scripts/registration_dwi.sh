#!/usr/bin/env bash
#registration_dwi.sh v1.0
#The objective of this script is to register the DWI image to T1w space and then to MNI space
#The script is a continuation of preprocessing_dwi.sh
#preprocessing_dwi --> registration_dwi

#This script does the following steps:
#1: Register the T1w to the MNI's T1w through a 12dof and warp transformation. 
#2: Use affine matrix and deformation field to obtain native T1w in MNI space.
#3: Extracts the corrected (after eddy distortion correction) b0
#4: Performs BET to the corrected b0
#5: Registers the corrected b0 to native T1w.
#6: Transforms the b0 to native T1w space and then to MNI space through the usage of two deformation fields

#Requires antsRegistrationSyN.sh from ANTs github
#The input files are:  *_T1w_bet.nii.gz, *_unbias.mif, MNI152_T1_1mm_brain.nii.gz
#The principal output is: ${sub}_dwi_in_mni_warp.nii.gz, ${sub}_t1_in_mni_warp.nii.gz


for i in {1..51}
do
printf -v sub "sub-%02d" $i
echo "-------------------------------------"
echo "|"
echo "| Running : " ${sub}
echo "|"
echo "-------------------------------------"
cd ${sub}/dwi


mkdir ants_reg
mkdir ants_reg/dwi_2_t1
mkdir ants_reg/t1_2_mni


#Registration t1 -> mni
../../antsRegistrationSyN.sh -d 3 -f ../../MNI152_T1_1mm_brain.nii.gz -m ../anat/*_T1w_bet.nii.gz -o ants_reg/t1_2_mni/t1_2_mni -n 12

#Applies the Affine matrix and the deformation field to the native T1w
antsApplyTransforms -d 3 -i ../anat/*_T1w_bet.nii.gz -r ../../MNI152_T1_1mm_brain.nii.gz -t ants_reg/t1_2_mni/t1_2_mni1Warp.nii.gz -t ants_reg/t1_2_mni/t1_2_mni0GenericAffine.mat -o ${sub}_t1_in_mni_warp.nii.gz


#Registration dwi -> t1 -> mni 
#Starts by obtaining a b0 image which has been corrected for Eddy distortions
dwiextract *_unbias.mif ${sub}_corrected_b0.nii.gz -bzero
bet ${sub}_corrected_b0.nii.gz ${sub}_corrected_b0_bet.nii.gz -f 0.4 -m 

# dwi -> t1 registration
../../antsRegistrationSyN.sh -d 3 -f ../anat/*_T1w_bet.nii.gz -m ${sub}_corrected_b0_bet.nii.gz -o ants_reg/dwi_2_t1/dwi_2_t1 -n 12 -t r

#Applies dwi -> t1 transform
antsApplyTransforms -d 3 -r ../anat/*_T1w_bet.nii.gz -i ${sub}_corrected_b0_bet.nii.gz -t ants_reg/dwi_2_t1/dwi_2_t10GenericAffine.mat -o ${sub}_dwi_in_t1_rigid.nii.gz

#Applies dwi -> t1 -> mni transform
antsApplyTransforms -d 3 -r ../../MNI152_T1_1mm_brain.nii.gz -i ${sub}_corrected_b0_bet.nii.gz -t ants_reg/t1_2_mni/t1_2_mni1Warp.nii.gz -t ants_reg/t1_2_mni/t1_2_mni0GenericAffine.mat -t ants_reg/dwi_2_t1/dwi_2_t10GenericAffine.mat -o ${sub}_dwi_in_mni_warp.nii.gz

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

