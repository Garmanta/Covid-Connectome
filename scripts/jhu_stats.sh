#!/usr/bin/env bash
#jhu_stats.sh v1.0
#The objective of this script is to obtain DTI metrics from a DTI image according to the JHU white matter atlas. It obtains average metrics from the ROIs voxels and writes in a .txt file. This script obtains FA and ADC metrics
#The script is a continuation of processing_dwi.sh
#preprocessing_dwi --> registration_dwi --> processing_dwi --> jhu_stats

#This script does the following steps:
#1: Transforms fa and adc images to MNI space with Affine and warp transformation
#2: Calculates mean and std from fa and adc image according to JHU atlas ROIs
#3: Outputs the results in tw .txt files. One for fa and another one for adc

#The input files are:  ${sub}_adc.mif ${sub}_fa.mif
#The principal output files are: adc_statistics, fa_statistics


for i in {1..51}
do
printf -v sub "sub-%02d" $i
echo "Running : " ${sub}
cd ${sub}/dwi


mrconvert ${sub}_adc.mif ${sub}_adc.nii.gz
mrconvert ${sub}_fa.mif ${sub}_fa.nii.gz

#Transforms to MNI space
antsApplyTransforms -d 3 -r ../../MNI152_T1_1mm_brain.nii.gz -i ${sub}_adc.nii.gz -t ants_reg/t1_2_mni/t1_2_mni1Warp.nii.gz -t ants_reg/t1_2_mni/t1_2_mni0GenericAffine.mat -t ants_reg/dwi_2_t1/dwi_2_t10GenericAffine.mat -o ${sub}_adc_in_mni_warp.nii.gz

antsApplyTransforms -d 3 -r ../../MNI152_T1_1mm_brain.nii.gz -i ${sub}_fa.nii.gz -t ants_reg/t1_2_mni/t1_2_mni1Warp.nii.gz -t ants_reg/t1_2_mni/t1_2_mni0GenericAffine.mat -t ants_reg/dwi_2_t1/dwi_2_t10GenericAffine.mat -o ${sub}_fa_in_mni_warp.nii.gz

mkdir results

#--------------------------------
#adc statistics
#--------------------------------

adc_statistics="adc_statistics.txt"

# Remove the output file if it already exists
#[ -f "$output_file" ] && rm "$output_file"

echo -e "${sub}" >> "results/$adc_statistics"
echo -e "The following statistics were obtaned using JHU-ICBM-labels-1mm.nii.gz with the ${sub}_adc_in_mni_warp image.nii.gz" >> "results/$adc_statistics"
echo -e "N\tmean\tstd" >> "results/$adc_statistics"

# Run the loop and append data to the adc_statistics.txt file
mean=$(mrstats ${sub}_adc_in_mni_warp.nii.gz -output mean -quiet)
std=$(mrstats ${sub}_adc_in_mni_warp.nii.gz -output std -quiet)
echo -e "g\t$mean\t$std" >> "results/$adc_statistics"
for N in {0..50}; do
  mean=$(mrcalc ../../JHU-ICBM-labels-1mm.nii.gz $N -eq - -quiet| mrstats ${sub}_adc_in_mni_warp.nii.gz -mask - -output mean -quiet)
  std=$(mrcalc ../../JHU-ICBM-labels-1mm.nii.gz $N -eq - -quiet | mrstats ${sub}_adc_in_mni_warp.nii.gz -mask - -output std -quiet)
  echo -e "$N\t$mean\t$std" >> "results/$adc_statistics"
done

#--------------------------------
#fa statistics

fa_statistics="fa_statistics.txt"
# Remove the output file if it already exists
#[ -f "$output_file" ] && rm "$output_file"

echo -e "${sub}" >> "results/$fa_statistics"
echo -e "The following statistics were obtaned using JHU-ICBM-labels-1mm.nii.gz with the ${sub}_fa_in_mni_warp image.nii.gz" >> "results/$fa_statistics"
echo -e "N\tmean\tstd" >> "results/$fa_statistics"


mean=$(mrstats ${sub}_fa_in_mni_warp.nii.gz -output mean -quiet)
std=$(mrstats ${sub}_fa_in_mni_warp.nii.gz -output std -quiet)
echo -e "g\t$mean\t$std" >> "results/$fa_statistics"
# Run the loop and append data to the fa_statistics.txt file
for N in {0..50}; do
  mean=$(mrcalc ../../JHU-ICBM-labels-1mm.nii.gz $N -eq - -quiet| mrstats ${sub}_fa_in_mni_warp.nii.gz -mask - -output mean -quiet)
  std=$(mrcalc ../../JHU-ICBM-labels-1mm.nii.gz $N -eq - -quiet | mrstats ${sub}_fa_in_mni_warp.nii.gz -mask - -output std -quiet)
  echo -e "$N\t$mean\t$std" >> "results/$fa_statistics"
done

cd ../..
done

echo "----------------------------------"
echo "|"
echo "| Finished JHU stats!"
echo "|"
echo "----------------------------------"


#--------------------------------------------------------------------------------------
#Version v1.0.
#--------------------------------------------------------------------------------------
#Get the lastest version at:
#--------------------------------------------------------------------------------------
#script by Alejandro Garma Oehmichen
#--------------------------------------------------------------------------------------


