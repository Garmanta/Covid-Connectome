echo "Running 5tt"

for_each -nthreads 12 sub-* : 5ttgen fsl IN/anat/IN_T1w.nii.gz IN/dwi/IN_5tt.mif -mask IN/anat/IN_T1w_bet_mask.nii.gz -nocrop -sgm_amyg_hipp
for_each -nthreads 12 sub-* : 5tt2gmwmi IN/dwi/IN_5tt.mif IN/dwi/IN_gmwm_interface.mif

for i in {1..20}
do
printf -v sub "sub-%02d" $i
echo "-------------------------------------"
echo "|"
echo "| Running : " ${sub}
echo "|"
echo "-------------------------------------"
cd ${sub}/dwi

mrconvert ${sub}_gmwm_interface.mif ${sub}_gmwm_interface.nii.gz
mrconvert ${sub}_5tt.mif ${sub}_5tt.nii.gz

antsApplyTransforms -d 3 -i ${sub}_gmwm_interface.nii.gz -r corrected_b0_bet.nii.gz -t [ants_reg/dwi_2_t1/dwi_2_t10GenericAffine.mat,1] -o ${sub}_gmwm_interface_in_dwi.nii.gz
antsApplyTransforms -d 3 -e 3 -i ${sub}_5tt.nii.gz -r corrected_b0_bet.nii.gz -t [ants_reg/dwi_2_t1/dwi_2_t10GenericAffine.mat,1] -o ${sub}_5tt_in_dwi.nii.gz

mrconvert ${sub}_gmwm_interface_in_dwi.nii.gz ${sub}_gmwm_interface_in_dwi.mif
mrconvert ${sub}_5tt_in_dwi.nii.gz ${sub}_5tt_in_dwi.mif

dwi2response dhollander -mask corrected_b0_bet_mask.nii.gz ${sub}_dwi_unbias.mif ${sub}_wm.txt ${sub}_gm.txt ${sub}_csf.txt 

dwi2fod msmt_csd ${sub}_dwi_unbias.mif -mask corrected_b0_bet_mask.nii.gz ${sub}_wm.txt ${sub}_wmfod.mif ${sub}_gm.txt ${sub}_gmfod.mif ${sub}_csf.txt ${sub}_csffod.mif

mrconvert -coord 3 0 ${sub}_wmfod.mif - | mrcat ${sub}_csffod.mif ${sub}_gmfod.mif - ${sub}_vf.mif

mtnormalise ${sub}_wmfod.mif ${sub}_wmfod_norm.mif ${sub}_gmfod.mif ${sub}_gmfod_norm.mif ${sub}_csffod.mif ${sub}_csffod_norm.mif -mask corrected_b0_bet_mask.nii.gz

tckgen -act ${sub}_5tt_in_dwi.mif -seed_gmwmi ${sub}_gmwm_interface_in_dwi.mif -mask corrected_b0_bet_mask.nii.gz -select 8000000 -backtrack -nthreads 14 ${sub}_wmfod_norm.mif ${sub}_8mtract.tck
tcksift ${sub}_8mtract.tck ${sub}_wmfod_norm.mif ${sub}_sift_1mtrack.tck -term_number 1000000 -nthreads 14
tcksift2 ${sub}_8mtract.tck ${sub}_wmfod_norm.mif ${sub}_sift_weights.txt -act ${sub}_5tt_in_dwi.mif
tckedit ${sub}_sift_1mtrack.tck -number 250000 ${sub}_sift_250ktrack.tck

dwi2tensor ${sub}_dwi_unbias.mif ${sub}_dti.mif -m corrected_b0_bet_mask.nii.gz
tensor2metric ${sub}_dti.mif -adc ${sub}_adc.mif
tensor2metric ${sub}_dti.mif -fa ${sub}_fa.mif


cd ../..
done

#wb_command
