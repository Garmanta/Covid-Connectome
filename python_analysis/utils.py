import os
import nibabel as nib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter



"""
The objective of utils.py is to accomodate a list of usable functions used on my thesis

"""

def get_snr(n, path_template, mask_path):
    """
    Process fMRI data to generate appropiate SNR and tSNR values,
    also computes group-level statistics.
    (Altough it can easily be modified to also work with other type of images like DWI)
    Just change the path and the name of the file
    
    ---------
    Inputs:
        n (int): Number of subjects.
        path_template (str): Path for subject-specific fMRI images.
        mask_path (str): Path to the brain mask image.

    Outpus:
        np array: SNR and tSNR value arrays
    """
    # Initialize lists to store SNR and tSNR values for each subject
    snr_list = []
    tsnr_list = []

    for i in range(1, n):
        # Define the paths for the subject's fMRI data and mask
        path = path_template.format(subject_id=i)
        print(f"Processing subject {i} with path {path}")

        # Load fMRI data and mask
        fmri_data = nib.load(path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        # Calculate SNR
        brain_region = fmri_data[mask == 1]  # Brain region where mask == 1
        background_region = fmri_data[mask == 0]  # Background where mask == 0

        mean_signal = np.mean(brain_region)
        std_noise = np.std(background_region)
        snr = mean_signal / std_noise
        snr_list.append(snr)

        # Calculate Temporal SNR (tSNR)
        mean_time_series = np.mean(fmri_data, axis=3)  # Mean over time
        std_time_series = np.std(fmri_data, axis=3)    # Standard deviation over time

        # Compute tSNR for brain voxels and average across those voxels
        tsnr_values = mean_time_series[mask == 1] / std_time_series[mask == 1]
        tsnr_list.append(np.mean(tsnr_values))

    # Convert lists to numpy arrays for easy manipulation
    snr_array = np.array(snr_list)
    tsnr_array = np.array(tsnr_list)

    # Group-level statistics
    print("Group-level SNR:")
    print(f"Mean SNR: {snr_array.mean()}, Median SNR: {np.median(snr_array)}")
    print("Group-level tSNR:")
    print(f"Mean tSNR: {tsnr_array.mean()}, Median tSNR: {np.median(tsnr_array)}")
    return snr_array, tsnr_array



def snr_histogram(snr_data, bins=12, type='snr'):
    """
    Plot a histogram with a mean line and default styling.

    ---------
    Parameters:
        data (array-like): Data to plot.
        mean_value (float): Mean value to display as a vertical line.
        bins (int, optional): Number of bins for the histogram. Defaults to 12.
        type (string, optional): For title and color preferences.
    """
    
    if type == 'snr':
        title='SNR Histogram'
        color='blue'
    elif type == 'tsnr':
        title='tSNR Histogram'
        color='green'
        
    mean_snr = np.mean(snr_data)
    plt.figure(figsize=(10, 6))  # Adjust the figure size for a better visual impact
    plt.hist(snr_data, bins=bins, color='skyblue', edgecolor='black', alpha=0.8)  # Set color, edge color, and transparency
    plt.title(title, fontsize=18, fontweight='bold')  # Adjust font size for title
    plt.xlabel('Signal to Noise Ratio', fontsize=12)  # Adjust font size for x-label
    plt.ylabel('Covid subjects', fontsize=12)  # Optionally add a y-label with the same font size
    plt.xticks(fontsize=10)  # Adjust font size for x-axis tick labels
    plt.yticks(fontsize=10)  # Adjust font size for y-axis tick labels
    plt.axvline(mean_snr, color=color, linestyle='dashed', linewidth=1, label=f'Mean: {mean_snr:.2f}')
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.9)
    plt.legend(fontsize=12)  # Display legend for mean and median lines
    plt.show()



# Paths to data
path_template = 'treatment/sub-{subject_id:02d}/func/sub-{subject_id:02d}_rest_in_mni_warp.nii.gz'
mask_path = 'MNI152_T1_2mm_brain_mask.nii.gz'


snr_array, tsnr_array = get_snr(51, path_template, mask_path)

snr_histogram(snr_array)
snr_histogram(tsnr_array, type='tsnr')
#%%


def process_connectome_data(file_path_nbs, file_path_dk):
    """
    Inputs a statistically significant network from NBS. Then outputs the same network with
    columns and indexes of DK ROIs
    
    ---------
    Inputs:
        file_path_nbs (str): Path to the structural connectivity matrix file.
        file_path_dk (str): Path to the DK atlas metadata file.

    Outputs:
        connectome_df: DataFrame of the structural connectivity matrix with region tags.
        region_counts_df: DataFrame of region connection counts.
        dk: DataFrame of DK atlas metadata.
    """
    # Load the structural connectivity matrix
    sc = np.loadtxt(file_path_nbs)

    # Load the DK atlas file into a DataFrame
    dk = pd.read_csv(file_path_dk, delim_whitespace=True, comment='#', header=None, names=['ID', 'TAG', 'NAME', 'R', 'G', 'B', 'A'])

    # Extract relevant columns and set the ID as the index
    dk = dk[['ID', 'TAG', 'NAME']].set_index('ID')
    dk = dk.drop(index=0)  # Drop the row with index 0 if it exists

    # Extract connections from the upper triangular part of the SC matrix then count #repeticion
    connections_list = [(dk.iloc[i].NAME, dk.iloc[j].NAME) for i, j in np.argwhere(np.triu(sc, k=1) == 1)]
    region_counts = Counter([region for connection in connections_list for region in connection])
    region_counts_df = pd.DataFrame(region_counts.items(), columns=['region', 'connection_count'])

    # Create a labeled DataFrame for the structural connectivity matrix
    connectome_df = pd.DataFrame(sc, index=dk['TAG'], columns=dk['TAG'])

    return connectome_df, region_counts_df, dk


def stat_network_nbs_plot(connectome_df):
    """
    Plots in a fancy matter the NBS statistically significant network.

    ----------
    Inputs: 
        connectome_df (df): The matrix containing the NBS connectome
 
    Outputs:
        imshow NBS plot

    """
    
    plt.figure(figsize=(20, 15), dpi=300)
    plt.imshow(sc, cmap='binary')
    
    x, y = np.where(sc == 1)
    plt.scatter(y, x, color='black', s=10, label='Binary 1')  
    
    tick_positions = np.arange(0, 85, 10)  
    plt.xticks(tick_positions, fontsize=12) 
    plt.yticks(tick_positions, fontsize=12)  
    
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.gca().set_xticks(np.arange(-0.5, 84, 1), minor=True) 
    plt.gca().set_yticks(np.arange(-0.5, 84, 1), minor=True)
    plt.grid(which='minor', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tick_params(which='minor', size=0)  
    
    plt.title('Structural Connectome NBS', fontsize=22, pad=15)  
    plt.xlabel('Desikan-Killiany ROI', fontsize=19)  
    plt.ylabel('Desikan-Killiany ROI', fontsize=19) 
    
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    plt.show()


def region_counbts_plot(region_counts_df):
    """
    Generates two plots from the NBS statistically signifiicant network.
    The first plot orders the ROI from highest to lowest degree.
    The second plot focuses entirely on the top 10 highest connected ROIs and contrast them in green color the ROIs
    that were chosen in the hypothesis to correlate with the pathophysiology of Covid

    ----------
    Inputs: 
        region_counts_df (df): A df containing the ROIs and their degrees
 
    Outputs:
        Plot 1: ROIs degree
        Plot 2: TOp 10 region connection counts

    """
   
    #Plot 1: Highest connected ROI to lowest connected ROI
    region_counts_df = region_counts_df.sort_values(by='connection_count', ascending=False)
    
    plt.figure(figsize=(16, 9))
    plt.bar(range(len(region_counts_df)), region_counts_df['connection_count'], alpha=0.7)
    
    plt.ylabel('Degree', fontsize=18)  
    plt.xticks([], [])  
    plt.xlim(-0.5, len(region_counts_df) - 0.5)
    plt.title('Degree on NBS connectome DK ROIs', fontsize=20, pad = 20)  #
    
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.9)
    plt.tight_layout()
    plt.show()


def top_10_regions_plot(region_counts_df):
    top_10_df = region_counts_df.head(10)
    top_10_df['region'] = top_10_df['region'].str.replace('ctx-', '', regex=False)
    
    bar_colors = ['green' if i in [0, 1, 4, 6, 9] else 'blue' for i in range(len(top_10_df))]
    
    plt.figure(figsize=(16, 9))
    plt.bar(top_10_df['region'], top_10_df['connection_count'], color=bar_colors, alpha=0.8)
    
    plt.ylabel('Connection Counts', fontsize=20)  # Adjust y-axis label font size
    plt.xticks(rotation=55, fontsize=12)  # Rotate x-axis labels for readability
    plt.yticks(fontsize=16)
    plt.title('Top 10 Region Connection Counts', fontsize=20, pad=20)  # Add padding to the title
    
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.9)
    plt.tight_layout()
    plt.show()




# Load the data
file_path_nbs = 'results/nbs/sc_dk/weight/scw_nbs_connectome.txt'
sc = np.loadtxt(file_path_nbs)
file_path_dk = 'results/fs_default.txt'

connectome_df, region_counts_df, dk = process_connectome_data(file_path_nbs, file_path_dk)
stat_network_nbs_plot(connectome_df)
region_counbts_plot(region_counts_df)
top_10_regions_plot(region_counts_df)



#--------------------------------------------------------------------------------------
#Version v1.0.
#--------------------------------------------------------------------------------------
#Get the lastest version at:
#--------------------------------------------------------------------------------------
#script by Alejandro Garma Oehmichen
#--------------------------------------------------------------------------------------


