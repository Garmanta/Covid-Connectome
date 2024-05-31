
for_each sub-* : design_change.sh /home/sphyrna/storage/subjects/tests/complete_synbold_pipeline/design/synbold_design.fsf PRE IN/func/PRE_synbold_design.fsf
for_each -nthreads 8 sub-* : feat IN/func/IN_synbold_design.fsf

for i in {2..3}
do
printf -v sub "sub-%02d" $i
echo "-------------------------------------"
echo "|"
echo "| Running : " ${sub}
echo "|"
echo "-------------------------------------"
cd ${sub}/func 

mkdir syn_inputs
mkdir syn_outputs

cp func.feat/filtered_func_data.nii.gz syn_inputs/BOLD_d.nii.gz
cp ../anat/${sub}_T1w_bet.nii.gz syn_inputs/T1.nii.gz

docker run --rm \
-v $(pwd)/syn_inputs/:/INPUTS/ \
-v $(pwd)/syn_outputs/:/OUTPUTS/ \
-v /home/sphyrna/neuro_software/freesurfer/license.txt:/opt/freesurfer/license.txt \
-v /tmp:/tmp \
--user $(id -u):$(id -g) \
ytzero/synbold-disco:v1.4 --skull_stripped --motion_corrected --no_smoothing


cd ../..
done


#python ~/neuro_software/fsl/aroma/ICA_AROMA.py -in $(pwd)/syn_outputs/BOLD_u.nii.gz -out $(pwd)/aroma.preproc -mc $(pwd)/feat_preproc.feat/mc/prefiltered_func_data_mcf.par -affmat -warp -m $(pwd)/feat_preproc.feat/mask_func.nii.gz

#python2.7 ~/neuro_software/fsl/aroma/ICA_AROMA.py -in $(pwd)/${sub}rest_in_mni_warp.nii.gz -out $(pwd)/aroma -mc $(pwd)/feat_preproc.feat/mc/prefiltered_func_data_mcf.par -affmat -warp -m $(pwd)/feat_preproc.feat/mask_func.nii.gz

