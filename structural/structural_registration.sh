
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

# t1 -> mni

../../antsRegistrationSyN.sh -d 3 -f ../../MNI152_T1_1mm_brain.nii.gz -m ../anat/*_T1w_bet.nii.gz -o ants_reg/t1_2_mni/t1_2_mni -n 12

antsApplyTransforms -d 3 -i ../anat/*_T1w_bet.nii.gz -r ../../MNI152_T1_1mm_brain.nii.gz -t ants_reg/t1_2_mni/t1_2_mni1Warp.nii.gz -t ants_reg/t1_2_mni/t1_2_mni0GenericAffine.mat -o ${sub}_t1_in_mni_warp.nii.gz

# dwi -> t1

dwiextract *_unbias.mif corrected_b0.nii.gz -bzero
bet corrected_b0.nii.gz corrected_b0_bet.nii.gz -f 0.4 -m 

../../antsRegistrationSyN.sh -d 3 -f ../anat/*_T1w_bet.nii.gz -m corrected_b0_bet.nii.gz -o ants_reg/dwi_2_t1/dwi_2_t1 -n 12 -t r

antsApplyTransforms -d 3 -r ../anat/*_T1w_bet.nii.gz -i corrected_b0_bet.nii.gz -t ants_reg/dwi_2_t1/dwi_2_t10GenericAffine.mat -o ${sub}_dwi_in_t1_rigid.nii.gz

# dwi -> t1 -> mni
antsApplyTransforms -d 3 -r ../../MNI152_T1_1mm_brain.nii.gz -i corrected_b0_bet.nii.gz -t ants_reg/t1_2_mni/t1_2_mni1Warp.nii.gz -t ants_reg/t1_2_mni/t1_2_mni0GenericAffine.mat -t ants_reg/dwi_2_t1/dwi_2_t10GenericAffine.mat -o ${sub}_dwi_in_mni_warp.nii.gz

cd ../..
done

echo "----------------------------------"
echo "|"
echo "| Finished structrual registration!"
echo "|"
echo "----------------------------------"

