import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

"""
The objective of dti_results.py is to
"""

# Function to read and process each file
def process_file(file_path):
    """
    In BIDS format, each patient has a summary of fa and adc statistics obtain from a JHU atlas an a DTI image.
    This function opens the .txt file and stores the values in a df

    ----------
    Inputs: 
        file_path (str): Path to the fa or adc summary statistics
 
    Outputs:
        summary_metrics_df (df): A df series containing summary statistics

    """
    with open(file_path, 'r') as file:
        lines = file.readlines()[3:]  # Skip the first three header lines
        data = [line.strip().split() for line in lines]
        summary_metrics_df = pd.DataFrame(data, columns=['N', 'Mean', 'Std'])
        return summary_metrics_df
    
    
def aggregate_data(subject_dir_path, n_sub):
    """
    This function obtains and aggregates both the fa and adc data from a control and covid cohort
    in a multiple df.

    ----------
    Inputs: 
        subject_dir_path (string): Path to the group directory. In this thesis, there are two directories, contorl and covid
        n_sub(int): Number of total subjects in the directory
 
    Outputs:
        adc_mean (df): Apparent diffusion coeficient mean across the entire subject cohort 
        adc_std (df): Apparent diffusion coeficient standard deviation across the entire subject cohort 
        fa_mean (df): fractional anisotropy mean across the entire subject cohort 
        fa_std (df):  fractional anisotropy standard deviation the entire subject cohort 
 
    """
    
    adc_mean_data = {}
    adc_std_data = {}
    fa_mean_data = {}
    fa_std_data = {}
    
    # Loop through each subject directory
    for i in range(1, n_sub):
        sub_dir = f"sub-{i:02d}"
        adc_file_path = os.path.join(subject_dir_path,sub_dir, 'dwi', 'results', 'adc_statistics.txt')
        fa_file_path = os.path.join(subject_dir_path,sub_dir, 'dwi', 'results', 'fa_statistics.txt')
    
        # Process adc_statistics.txt
        adc_df = process_file(adc_file_path)
        adc_mean_data[sub_dir] = adc_df['Mean'].values
        adc_std_data[sub_dir] = adc_df['Std'].values
    
        # Process fa_statistics.txt
        fa_df = process_file(fa_file_path)
        fa_mean_data[sub_dir] = fa_df['Mean'].values
        fa_std_data[sub_dir] = fa_df['Std'].values
    
    # Create DataFrames for adc mean and std
    adc_mean = pd.DataFrame.from_dict(adc_mean_data, orient='index', columns=adc_df['N'].values).astype(float)
    adc_std = pd.DataFrame.from_dict(adc_std_data, orient='index', columns=adc_df['N'].values).astype(float)
    
    # Create DataFrames for fa mean and std
    fa_mean = pd.DataFrame.from_dict(fa_mean_data, orient='index', columns=fa_df['N'].values).astype(float)
    fa_std = pd.DataFrame.from_dict(fa_std_data, orient='index', columns=fa_df['N'].values).astype(float)
    return adc_mean, adc_std, fa_mean, fa_std


results_data_path = 'results/fa_adc/'

#I would generaly make a single path, but since the program has to recollect data divided in BIDS format, its easier this way
control_mean_adc, control_std_adc, control_mean_fa, control_std_fa = aggregate_data('control_processed',50)
treatment_mean_adc, treatment_std_adc, treatment_mean_fa, treatment_std_fa = aggregate_data('treatment_processed',52)

control_mean_adc.to_csv(results_data_path + 'control_adc_mean.txt')
control_std_adc.to_csv(results_data_path + 'control_adc_std.txt')
control_mean_fa.to_csv(results_data_path + 'control_fa_mean.txt')
control_std_fa.to_csv(results_data_path + 'control_fa_std.txt')

treatment_mean_adc.to_csv(results_data_path + 'treatment_adc_mean.txt')
treatment_std_adc.to_csv(results_data_path + 'treatment_adc_std.txt')
treatment_mean_fa.to_csv(results_data_path + 'treatment_fa_mean.txt')
treatment_std_fa.to_csv(results_data_path + 'treatment_fa_std.txt')



#%%

def multiple_tests_image_metrics(control_mean, treatment_mean):
    """
    Performs independed groups t-test for unequal variance. Then performs false discovery rate with the 
    Benjamini-Hochberg procedure across all ROIs from an specific atlas. In this thesis, JHU white matter atlas
    Finally, the function outputs the proper results

    ----------
    Inputs: 
        control_mean (df): A df matrix containing the mean of any imagenological metric across multiple ROIs for the control group
        treatment_mean (df): A df matrix containing the mean of any imagenological metric across multiple ROIs for the treatment group
 
    Outputs:
        stat_results (df): A matrix df containing the raw t statistics and p values
        bool_results (array): A boolean array containig a True if the corrected p-vaules are smaller than 0.05
        adjusted_p_values (array): An array containig the adjusted p values after fdr correction 

    """

    # Initialize dictionaries to store t-statistics and p-values
    t_statistics = []
    p_values = []
    
    # Iterate over each column and calculate t-statistic and p-value
    for column in control_mean.columns:
        #Assuming unequal variance
        t_stat, p_val = ttest_ind(control_mean[column].astype(float), treatment_mean[column].astype(float), equal_var=False) 
        p_values.append(p_val)
    
    # Create a DataFrame with t-statistics and p-values
    stat_results = pd.DataFrame([t_statistics, p_values], index=['t_statistic', 'p_value'], columns=control_mean.columns)
    bool_results, adjusted_p_values, _, _ = multipletests(stat_results.loc['p_value'], alpha=0.05, method='fdr_bh')
    
    return stat_results, bool_results, adjusted_p_values




_, bool_results_fa, adjusted_p_fa = multiple_tests_image_metrics(control_mean_fa, treatment_mean_fa)
_, bool_results_adc, adjusted_p_adc = multiple_tests_image_metrics(control_mean_adc, treatment_mean_adc)

#%%


def histogram_roi(control_mean, treatment_mean, roi_name):
   """
   A simple plot with two histograms constrasting the control and treatment groups. Inclused a vertical line
   to denote the mean of the groups   

   ----------
   Inputs: 
       control_mean (df): A df matrix containing the mean of any imagenological metric across multiple ROIs for the control group
       treatment_mean (df): A df matrix containing the mean of any imagenological metric across multiple ROIs for the treatment group

   Outputs:
       histogram plot

   """
   # Calculate the mean of the treatment and control data across all subjects
   treatment_mean_group = treatment_mean.mean()
   control_mean_group = control_mean.mean()
    

   plt.figure(figsize=(10, 6))
   plt.hist(treatment_mean, bins=20, alpha=0.5, label='Treatment FA Mean (g)', color='red')
   plt.hist(control_mean, bins=20, alpha=0.5, label='Control FA Mean (g)', color='blue')
    

   plt.axvline(treatment_mean_group, color='red', linestyle='dashed', linewidth=2, label='Treatment Mean')
   plt.axvline(control_mean_group, color='blue', linestyle='dashed', linewidth=2, label='Control Mean')
    

   plt.title('Histogram of FA Mean ' + roi_name + ' for Treatment and Control Groups')
   plt.xlabel('FA Mean (' + roi_name +')')
   plt.ylabel('Frequency')
   plt.grid()
   plt.legend(loc='upper right')



histogram_roi(control_mean_fa['g'], treatment_mean_fa['g'], 'global')
histogram_roi(control_mean_adc['12'], treatment_mean_adc['12'], 'hippocampus')


#%%

def multiple_roi_boxplots(control_mean, treatment_mean, roi_regions, name_regions):
    """
    Create a 2x3 grid of boxplots comparing Control vs. Treatment data,
      
    ----------
    Inputs
        control_mean (df) : Mean data per patients for control. 
        treatment_mean (df): Mean data per patients for treatment, in this case, covid
        roi_regions (list,str): A list of ROIs from the atlas as encoded in the atlas
        name_regions (list,str): List with atlas names to be used in the subplot title
        
    Outputs
        2x3 subplot of multiple rois.
    """
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
    axes = axes.flatten()  # Flatten for easy iteration

    for i, col in enumerate(roi_regions):
        if i >= len(axes):
            break
        
        ax = axes[i]

        # Prepare data
        control_mean = control_mean[col]
        treatment_mean = treatment_mean[col]
        
        # Plot boxplots (patch_artist=True to allow facecolor changes)
        bp = ax.boxplot([control_mean, treatment_mean],
                        labels=["Control", "Treatment"],
                        patch_artist=True)

        # Color the boxes
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('lightcoral')
        bp['boxes'][1].set_alpha(0.6)
        
        # Set subplot title
        ax.set_title(f"{name_regions[i]}", fontsize =14)
        
        # Grid
        ax.grid(True)

        # Enforce 5 y-ticks
        data_min = min(control_mean.min(), treatment_mean.min())
        data_max = max(control_mean.max(), treatment_mean.max())
        if data_min == data_max:
            data_min -= 0.1
            data_max += 0.1
        y_ticks = np.linspace(data_min, data_max, 5)
        ax.set_yticks(y_ticks)
        
    # Hide any remaining unused subplots if fewer than 6
    for j in range(len(roi_regions), 6):
        axes[j].set_visible(False)

    # Add a main figure title
    fig.suptitle("Comparison of multiple ROI in Control vs Covid",
                 fontsize=18)

    # Adjust subplot spacing
    fig.subplots_adjust(wspace=0.4, hspace=0.3)
    # or if you prefer:
    # plt.tight_layout(pad=2.0, w_pad=2.0, h_pad=2.0)

    plt.show()

    
roi_regions =  ['g', '1', '2', '3', '4', '5']
name_regions =  ['global', 'region_1', 'region_2', 'region_3', 'region_4', 'region_5']
multiple_roi_boxplots(control_mean_fa, treatment_mean_fa, roi_regions, name_regions)



#--------------------------------------------------------------------------------------
#Version v1.0.
#--------------------------------------------------------------------------------------
#Get the lastest version at:
#--------------------------------------------------------------------------------------
#script by Alejandro Garma Oehmichen
#--------------------------------------------------------------------------------------
