for i in {2..3}
do
printf -v sub "sub-%02d" $i
echo "-------------------------------------"
echo "|"
echo "| Running : " ${sub}
echo "|"
echo "-------------------------------------"
cd ${sub}/func

cp -r ../dwi/ants_reg ants_reg

mkdir ants_reg
mkdir ants_reg/t1_2_mni2mm
../../antsRegistrationSyN.sh -d 3 -f ../../MNI152_T1_2mm_brain.nii.gz -m ../anat/*_T1w_bet.nii.gz -o ants_reg/t1_2_mni2mm/t1_2_mni2mm -n 12


mkdir ants_reg/fmri_2_t1
cp syn_outputs/BOLD_u.nii.gz ${sub}_rest_preprocessed.nii.gz

fslroi ${sub}_rest_preprocessed.nii.gz ${sub}_rest_preprocessed_3dvol.nii.gz 250 1
../../antsRegistrationSyN.sh -d 3 -f ../anat/*_T1w_bet.nii.gz -m ${sub}_rest_preprocessed_3dvol.nii.gz -o ants_reg/fmri_2_t1/fmri_2_t1 -n 12 -t a

antsApplyTransforms -d 3 -e 3 -r ../../MNI152_T1_2mm_brain.nii.gz -i ${sub}_rest_preprocessed.nii.gz -t ants_reg/t1_2_mni2mm/t1_2_mni2mm1Warp.nii.gz -t ants_reg/t1_2_mni2mm/t1_2_mni2mm0GenericAffine.mat -t ants_reg/fmri_2_t1/fmri_2_t10GenericAffine.mat -o ${sub}_rest_in_mni_warp.nii.gz --float

cd ../..
done

echo "----------------------------------"
echo "|"
echo "| Finished structrual registration!"
echo "|"
echo "----------------------------------"
