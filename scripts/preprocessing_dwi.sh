#!/usr/bin/env bash
#preprocessing_dwi.sh v1.0
#The objective of this script is to complete the preprocessing of DWI images
#The script is structure to work with a BIDS formart.

#The script performs the following stepts:
#1: Performs a PCA-based denoising
#2: Performs a gibbs unringing algorithm
#3: Creates a synthetic b0 image in the opposite direction (in this case PA)
#4: Corrects eddy distortion 
#5: Finishes with bias field correction ready for registration 

# Requires the following files: acqparams.txt, slspec.txt and index.txt 
# For the acqparams, search for the 'Total Readout Time' in the json or mrinfo
# For the slspec use the matlab scrip slspec_filecreator.m adjunct in this package
# For the index.txt do:

#indx=""
#for ((i=1; i<=129; i+=1)); do indx="$indx 1"; done
#echo $indx > index.txt

#The input files are:  *_T1w.nii.gz, *_dwi.nii.gz, acqparams.txt, slspec.txt, index.txt
#The principal output is: ${sub}_T1w_bet.nii.gz, ${sub}_bias.mif

for i in {1..49}
do
printf -v sub "sub-%02d" $i
echo "-------------------------------------"
echo "|"
echo "| Running : " ${sub}
echo "|"
echo "-------------------------------------"
cd ${sub}/dwi

#Creates basic directories
mkdir syn_inputs
mkdir syn_outputs
mkdir topup
mkdir eddy

#Performs hd-bet
hd-bet -i ../anat/*_T1w.nii.gz

#Transforms files from NIFTI format to MIF format
mrconvert *_dwi.nii.gz ${sub}_dwi.mif -fslgrad *.bvec *.bval -json_import *_dwi.json

#Performs usual denoising
dwidenoise ${sub}_dwi.mif ${sub}_dwi_denoise.mif
mrdegibbs  ${sub}_dwi_denoise.mif ${sub}_dwi_unringed.mif

#Prepares for Synb0-DisCo
dwiextract ${sub}_dwi_unringed.mif syn_inputs/b0.nii.gz -bzero

cp ../anat/*_T1w_bet.nii.gz syn_inputs/T1.nii.gz
cp ../anat/*_T1w_bet_mask.nii.gz ${sub}_T1_mask.nii.gz
cp ../../acqparams_control.txt syn_inputs/acqparams.txt
cp ../../slspec_control_dwi.txt slspec.txt
cp ../../index.txt index.txt

#Runs Synb0, then completes required inputs for FSL's Eddy
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

#Finalizes preprocessing with bias correct.
mrconvert eddy/eddy_unwarped_images.nii.gz ${sub}_dwi_eddy.mif -fslgrad *.bvec *.bval -json_import *_dwi.json
dwibiascorrect ants ${sub}_dwi_eddy.mif ${sub}_dwi_unbias.mif -bias ${sub}_bias.mif

cd ../..
done

echo "----------------------------------"
echo "|"
echo "| Finished preprocessing!"
echo "|"
echo "----------------------------------"

#--------------------------------------------------------------------------------------
#Version v1.0.
#--------------------------------------------------------------------------------------
#Get the lastest version at:
#--------------------------------------------------------------------------------------
#script by Alejandro Garma Oehmichen
#--------------------------------------------------------------------------------------

