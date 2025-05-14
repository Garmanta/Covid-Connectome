#!/usr/bin/env bash
#recon_all.sh v1.0
#The objective of this script is to perform recon_all in all T1w images
#The script is a continuation of procesing_dwi.sh and preprocessing_fmri.sh
#preprocessing_dwi --> registration_dwi --> processing_dwi --> recon_all_dk
#design_change_fsl --> synbold_fmri --> registration_fmri --> recon_all_dk

#This script does the following steps:
#1: Performs recon-all to all subjects 

#The input files are:  *_T1w.nii.gz, 
#The principal output is:  ${sub}/mri/aparc+aseg.mgz


#Similarly to FSL, the program doesnt handle multithreading that well. Just as a fun fact, I tested how fast recon-all became as a function of how many logic cores I inputed in.
#One core run time: 240mins. Two core run time: 180mins. Four core run time: 120mins.
#Since it doesnt scale linearly and it tampers off pretty quickly, its just better to run multiple instances in single core
#I used only 50% of the available cores in my computer. Going to 80% usage causes crashes and you wont know which process is the one that crashed so good luck. 
#The -s option switches FreeSurfer default file system management to BIDS format sub-XX.
for_each sub-* : mkdir IN/anat/reconall
for_each -nthreads 8 sub-* : recon-all -all -s IN -i IN/anat/IN_T1w.nii.gz -sd IN/anat/reconall -qcache


#--------------------------------------------------------------------------------------
#Version v1.0.
#--------------------------------------------------------------------------------------
#Get the lastest version at:
#--------------------------------------------------------------------------------------
#script by Alejandro Garma Oehmichen
#--------------------------------------------------------------------------------------

