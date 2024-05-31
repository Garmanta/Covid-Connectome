#!/usr/bin/env bash
#This pipelines runs the preprocessing for dwi images
#1: General denoising using Mrtrix
#2: Runs Synb0-Disco
#3: Does eddy correction 
#4: Finishes with bias field correction ready for registration
# For this pipeline, the following files must exists: acqparams.txt, slspec.txt and index.txt 
# For the acqparams, search for the 'Total Readout Time' in the json or mrinfo
# For the slspec use the matlab scrip slspec_filecreator.m adjunct in this package
# For the index.txt do:

#indx=""
#for ((i=1; i<=129; i+=1)); do indx="$indx 1"; done
#echo $indx > index.txt


for i in {1..51}
do
printf -v sub "sub-%02d" $i
echo "-------------------------------------"
echo "|"
echo "| Running : " ${sub}
echo "|"
echo "-------------------------------------"
cd ${sub}/dwi

mkdir syn_inputs
mkdir syn_outputs
mkdir topup
mkdir eddy

hd-bet -i ../anat/*_T1w.nii.gz
mrconvert *_dwi.nii.gz ${sub}_dwi.mif -fslgrad *.bvec *.bval -json_import *_dwi.json

dwidenoise ${sub}_dwi.mif ${sub}_dwi_denoise.mif
mrdegibbs  ${sub}_dwi_denoise.mif ${sub}_dwi_unringed.mif

dwiextract ${sub}_dwi_unringed.mif syn_inputs/b0.nii.gz -bzero

cp ../anat/*_T1w_bet.nii.gz syn_inputs/T1.nii.gz
cp ../anat/*_T1w_bet_mask.nii.gz ${sub}_T1_mask.nii.gz
cp ../../acqparams_treatment.txt syn_inputs/acqparams.txt
cp ../../slspec_treatment_dwi.txt slspec.txt
cp ../../index.txt index.txt

docker run --rm -v $(pwd)/syn_inputs/:/INPUTS/ -v $(pwd)/syn_outputs:/OUTPUTS/ -v ~/neuro_software/freesurfer/license.txt:/extra/freesurfer/license.txt --user $(id -u):$(id -g) leonyichencai/synb0-disco:v3.0 --notopup --stripped

fslmerge -t ${sub}_b0_pair.nii.gz syn_inputs/b0.nii.gz syn_outputs/b0_u.nii.gz

cd ../..
done

#Speeds up significantly the script
for_each -info -nthreads 10 sub-* : topup --imain=IN/dwi/IN_b0_pair.nii.gz --datain=IN/dwi/syn_inputs/acqparams.txt --config=b02b0.cnf --out=IN/dwi/topup/topup_results --fout=IN/dwi/topup/field --iout=IN/dwi/topup/unwarped_images

for i in {1..51}
do
printf -v sub "sub-%02d" $i
echo "Running : " ${sub}
cd ${sub}/dwi

dwi2mask ${sub}_dwi_unringed.mif - | maskfilter - dilate - | mrconvert - ${sub}_eddy_mask.nii -datatype float32 -strides -1,+2,+3

mrconvert ${sub}_dwi_unringed.mif ${sub}_dwi_unringed.nii.gz 

eddy --imain=${sub}_dwi_unringed.nii.gz --mask=${sub}_eddy_mask.nii.gz --acqp=syn_inputs/acqparams.txt --index=index.txt --bvecs=${sub}_dir-AP_dwi.bvec --bvals=${sub}_dir-AP_dwi.bval --topup=topup/topup_results --repol --mporder=9 --flm=quadratic --cnr_maps --slspec=slspec.txt --out=eddy/eddy_unwarped_images 

mrconvert eddy/eddy_unwarped_images.nii.gz ${sub}_dwi_eddy.mif -fslgrad *.bvec *.bval -json_import *_dwi.json
dwibiascorrect ants ${sub}_dwi_eddy.mif ${sub}_dwi_unbias.mif -bias ${sub}_bias.mif

cd ../..
done

echo "----------------------------------"
echo "|"
echo "| Finished preprocessing!"
echo "|"
echo "----------------------------------"


