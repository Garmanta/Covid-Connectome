#!/usr/bin/env bash
#processing_dwi.sh v1.0
#The objective of this script is to complete the processing of DWI images and generate a tractography
#The script is a continuation of registration_dwi.sh
#preprocessing_dwi --> registration_dwi --> processing_dwi

#This script does the following steps:
#1: Runs single threaded 5tt, since multithread processing sucks with FSL
#2: Obtains the grey matter and white matter interface
#3: Register the gmwm_interface into DWI space
#4: Generates the FODs
#5: Obtain the average normalized response function for white matter, grey matter and CSF
#6: Performs the tractography followed by SIFT
#7: Calculates DTI metrics

#The input files are:  *_T1w.nii.gz, ${sub}_corrected_b0_bet.nii.gz ${sub}_dwi_unbias.mif
#The principal output is: ${sub}_sift_1mtrack.tck, ${sub}_sift_weights.txt, ${sub}_dti.mif, ${sub}_adc.mif, ${sub}_fa.mif


echo "Running 5tt"
#Performs multiple single thread 5ttgen. Change -nthreads with how many logical cores you want to use. I use 80% of my capacity
for_each -nthreads 12 sub-* : 5ttgen fsl IN/anat/IN_T1w.nii.gz IN/dwi/IN_5tt.mif -mask IN/anat/IN_T1w_bet_mask.nii.gz -nocrop -sgm_amyg_hipp
for_each -nthreads 12 sub-* : 5tt2gmwmi IN/dwi/IN_5tt.mif IN/dwi/IN_gmwm_interface.mif

for i in {1..51}
do
printf -v sub "sub-%02d" $i
echo "-------------------------------------"
echo "|"
echo "| Running : " ${sub}
echo "|"
echo "-------------------------------------"
cd ${sub}/dwi

#To use ANTs the images must be in NIFTI format
mrconvert ${sub}_gmwm_interface.mif ${sub}_gmwm_interface.nii.gz
mrconvert ${sub}_5tt.mif ${sub}_5tt.nii.gz

#The gmwm_interface is in T1 space, must be transformed to DWI space
antsApplyTransforms -d 3 -i ${sub}_gmwm_interface.nii.gz -r ${sub}_corrected_b0_bet.nii.gz -t [ants_reg/dwi_2_t1/dwi_2_t10GenericAffine.mat,1] -o ${sub}_gmwm_interface_in_dwi.nii.gz
antsApplyTransforms -d 3 -e 3 -i ${sub}_5tt.nii.gz -r ${sub}_corrected_b0_bet.nii.gz -t [ants_reg/dwi_2_t1/dwi_2_t10GenericAffine.mat,1] -o ${sub}_5tt_in_dwi.nii.gz

#Back to MIF
mrconvert ${sub}_gmwm_interface_in_dwi.nii.gz ${sub}_gmwm_interface_in_dwi.mif
mrconvert ${sub}_5tt_in_dwi.nii.gz ${sub}_5tt_in_dwi.mif

#Obtains the FODs
dwi2response dhollander -mask ${sub}_corrected_b0_bet_mask.nii.gz ${sub}_dwi_unbias.mif ${sub}_wm.txt ${sub}_gm.txt ${sub}_csf.txt 

dwi2fod msmt_csd ${sub}_dwi_unbias.mif -mask ${sub}_corrected_b0_bet_mask.nii.gz ${sub}_wm.txt ${sub}_wmfod.mif ${sub}_gm.txt ${sub}_gmfod.mif ${sub}_csf.txt ${sub}_csffod.mif

mrconvert -coord 3 0 ${sub}_wmfod.mif - | mrcat ${sub}_csffod.mif ${sub}_gmfod.mif - ${sub}_vf.mif
mtnormalise ${sub}_wmfod.mif ${sub}_wmfod_norm.mif ${sub}_gmfod.mif ${sub}_gmfod_norm.mif ${sub}_csffod.mif ${sub}_csffod_norm.mif -mask ${sub}_corrected_b0_bet_mask.nii.gz

#Generates the tractography. 8 million tracts, down to 1 million through SIFT, and then 250k for visualization purposes.
#SIFT2 is used for its weights on the connectome
tckgen -act ${sub}_5tt_in_dwi.mif -seed_gmwmi ${sub}_gmwm_interface_in_dwi.mif -mask ${sub}_corrected_b0_bet_mask.nii.gz -select 8000000 -backtrack -nthreads 14 ${sub}_wmfod_norm.mif ${sub}_8mtract.tck
tcksift ${sub}_8mtract.tck ${sub}_wmfod_norm.mif ${sub}_sift_1mtrack.tck -term_number 1000000 -nthreads 14
tcksift2 ${sub}_8mtract.tck ${sub}_wmfod_norm.mif ${sub}_sift_weights.txt -act ${sub}_5tt_in_dwi.mif
tckedit ${sub}_sift_1mtrack.tck -number 250000 ${sub}_sift_250ktrack.tck

#Tensor metrics required for some hypothesis testing
dwi2tensor ${sub}_dwi_unbias.mif ${sub}_dti.mif -m ${sub}_corrected_b0_bet_mask.nii.gz
tensor2metric ${sub}_dti.mif -adc ${sub}_adc.mif
tensor2metric ${sub}_dti.mif -fa ${sub}_fa.mif


cd ../..
done
echo "----------------------------------"
echo "|"
echo "| Finished DWI processing!"
echo "|"
echo "----------------------------------"



#--------------------------------------------------------------------------------------
#Version v1.0.
#--------------------------------------------------------------------------------------
#Get the lastest version at:
#--------------------------------------------------------------------------------------
#script by Alejandro Garma Oehmichen
#--------------------------------------------------------------------------------------

