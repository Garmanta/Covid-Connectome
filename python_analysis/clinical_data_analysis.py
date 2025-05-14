import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import scipy.stats as stats

from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_curve, auc

#%% Functions

"""
The objective of clinical_data_analysis.py is to generate critical analysis of the conjunction between clinical
and imaginological data. This include the clinical database generated at the INER and the DWI images at INNN.
DWI images were used to obtain DTI metrics like FA and ADC, and network metrics like degree, betweenes centrality, etc.

"""


def replace_outliers(data, std_out=4):
    """
    Replace outliers in a the input data with the mean of the respective variable.
    An outlier is defined as a value more than four standard deviations from the mean.
    Prints the column name and row index of each outlier.
    
    Inputs:
    data (df): The input DataFrame with variables to be cleaned with its mean.
    
    Outputs:
    cleaned_data (df): A copy of the DataFrame with outliers replaced by mean values.
    """
    # Create a copy of the DataFrame to avoid modifying the original


    # Calculate means and standard deviations
    means = data.mean()
    stds = data.std()

    # Replace outliers with the mean
    for col in data.columns:
        col_mean = means[col]
        col_std = stds[col]

        # Identify outliers
        outliers_mask = (data[col] - col_mean).abs() > std_out * col_std

        # Print information about outliers
        if outliers_mask.any():
            outlier_indices = outliers_mask[outliers_mask].index.tolist()
            print(f"Column '{col}' outliers at rows: {outlier_indices}")

            # Replace outliers with mean
            data.loc[outliers_mask, col] = col_mean

    return data

def covid_control_df(covid_data_path, control_data_path, save_path):
    """
    Creates a Dataframe containing all the relevant data from the Covid and Control database.
    Also creates a copy of the datframes to a .csv in the same folder

    ----------
    Inputs: 
        covid_data_path (string): Path to the covid database
        control_data_path (string): Path to the control database
        save_path (string) : Path to where to save the resulting final databases.
 
    Outputs:
        covid_data (df): Dataframe with all relevant covid subjects data
        control_data (df): Dataframe with all relevant control subjects data

    """
    #Covid data
    xls = pd.ExcelFile(covid_data_path)
    data = {}
    for sheet_name in xls.sheet_names:
        data[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name, header=1)
    
    fisio1 = data['Marcadores fisio 1']
    fisio2 = data['Marcadores fisio 2']
    demog = data['Datos demog']
    neuro = data['Comorb neuro']
    resp = data['Marc resp']
    
    #Filling missing vaLues with mean
    demog.fillna(value = demog.mean(), inplace = True)
    fisio1.fillna(value = fisio1.mean(), inplace = True)
    fisio2.fillna(value = fisio2.mean(), inplace = True)
    neuro.fillna(value = neuro.mean(), inplace = True)
    resp.fillna(value = resp.mean(), inplace = True)
    #Filling wrong values with correct or assumed ones.
    resp.neumonia_ingreso[46] = 1
    
    #Resumed covid data
    covid_data = pd.DataFrame({
        'sexo': demog['sexo'],
        'edad': demog['edad2021'],
        'no_recupera_olfatogusto': 1 - neuro['recupera_olfatogusto'],
        'cefalea_postcovid': neuro['cefalea_postcovid'],
        'persistencia_delirio': neuro['persistencia_delirio'],
        'mov_anormales': neuro['mov_anormales'],
        'moca_pnts': neuro['moca_pnts'],
        'mmse_pnts': neuro['mmse_pnts'],
        'duracion_fatiga': neuro['duracion_fatiga'],
        'spo2': resp['spo2urgencias'],
        'saturacion_ingreso': resp['saturacion_ingreso'],
        'prot_c_react': fisio1['pcr_1'],
        'dimero_d': fisio1['dimd_1'],
        'neutrofilos': fisio1['neutro_1']
    })
        
    #pcr_data = fisio1[["linfo_1", "neutro_1","plaq_1", "ldh_1", "pcr_1", "dimd_1", "fibri_1"]].copy()
    
    #Control data
    xls = pd.ExcelFile(control_data_path)
    data = {}
    for sheet_name in xls.sheet_names:
        data[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name, header=0)
    
    
    control_data = data['Control Filter'].fillna(data['Control Filter'].mean())
    
    control_data = pd.DataFrame({
        'sexo': control_data['sexo'],
        'edad': control_data['edad'],
        'mmse_pnts': control_data['mmse']
    })

    
    
    # covid_data.to_csv(save_path + 'covid_data.csv')
    # #pcr_data.to_csv('pcr_data.csv')
    # control_data.to_csv(save_path + 'control_data.csv')
    
    return covid_data, control_data


#%% Table 1 data

#############################
# Generating the basic COntrol and Covid database
#############################

covid_data_path = 'results/clinical_db/datos_clinicos_covidsp.xlsx'
control_data_path = 'results/clinical_db/datos_clinicos_control.xlsx'
save_path = 'results/clincal_db/'

covid_data, control_data = covid_control_df(covid_data_path, control_data_path, save_path)


#Datos demograficos
print('--Table 1------------------')
print('Covid demographic data')
print("Sexo --------------------------")
covid_data.sexo.mean()
covid_data.sexo.std()
print("Edad --------------------------")
covid_data.edad.mean()
covid_data.edad.std()

print("MMSE --------------------------")
covid_data.mmse_pnts.mean()
covid_data.mmse_pnts.std()
print('Control demographic data')

print("Sexo --------------------------")
control_data.sexo.mean()
control_data.sexo.std()
print("Edad --------------------------")
control_data.edad.mean()
control_data.edad.std()

print("MMSE --------------------------")
control_data.mmse_pnts.mean()
control_data.mmse_pnts.std()



sm.stats.normal_ad(covid_data.edad)
sm.stats.normal_ad(control_data.edad)

stats.mannwhitneyu(covid_data.edad, control_data.edad)
stats.mannwhitneyu(covid_data.mmse_pnts, control_data.mmse_pnts)



covid_corr = covid_data.corr()
control_corr = control_data.corr()

del control_data_path, covid_data_path, save_path



#%% Plotting
#############################
# Functions for plot of clinical variables
#############################

def plot_histogram_with_binary_outcome(continuous_variable, binary_variable, bins, xlabelstring, ylabelstring):
    """
    Plots a histogram of a continuous variable where each bin is divided according to the proportion of a 
    discrete variable.

    ----------
    Inputs: 
        continous_variable (df): df series containing a continous variable
        binary_variable (df): df series containing a discrete, binary variable with either 0 or 1
        bins (int) : number of bins in the plot
        xlabelstring (string): String for the name on the x axis
        ylabelstring (string): String for the name on the y axis
 
    Outputs:
        histogram plot

    """
    # Calculate Histogram Bins with specified number of bins
    bin_edges = np.linspace(min(continuous_variable), max(continuous_variable), bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate bin centers for plotting

    # Initialize counts for each category within each bin
    counts_0 = np.zeros(len(bin_edges)-1)
    counts_1 = np.zeros(len(bin_edges)-1)

    # Calculate counts for each bin
    for i in range(len(bin_edges)-1):
        bin_mask = (continuous_variable >= bin_edges[i]) & (continuous_variable < bin_edges[i+1])
        counts_0[i] = np.sum(binary_variable[bin_mask] == 0)
        counts_1[i] = np.sum(binary_variable[bin_mask] == 1)

    # Plotting
    fig, ax = plt.subplots()
    ax.bar(bin_centers, counts_0, width=np.diff(bin_edges), label='Sin ' + ylabelstring, color='blue', alpha =0.7)
    ax.bar(bin_centers, counts_1, width=np.diff(bin_edges), bottom=counts_0, label='Con ' + ylabelstring, color='red', alpha=0.7)

    # Adjust Aesthetics
    ax.set_xlabel(xlabelstring, fontsize=14)
    ax.set_ylabel(ylabelstring, fontsize=14)
    ax.set_title('Histograma puntuacion Moca con Proporciones', fontsize=15)
    ax.legend()
    ax.grid(True)
    plt.show()
 

def plot_binary_data(binary_data,title="Variabilidad en variables dicotomicas"):
    """
    Plots a collection of columns containing different binary data. This is thought to be used with a threshold
    in a continous variable to ascertain the effect of the continous variable in the binary outcome.

    ----------
    Inputs: 
        binary_data (df) : A df containing binary data on specific columns given a threshold of a continous variable.
 
    Outputs:
        Multiple boxplot

    """
    
    # Create a single figure and axis, specifying the desired size
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e']
    n_datasets = len(binary_data.columns)
    bar_width = 0.35
    index = np.arange(n_datasets)

    # Plot bars
    for i, (label, column) in enumerate(binary_data.iteritems()):
        percent_ones = column.mean()
        percent_zeros = 1 - percent_ones
        ax.bar(i, percent_zeros, bottom=percent_ones, 
               width=bar_width, color=colors[1], 
               label='0' if i == 0 else "", edgecolor='black')
        ax.bar(i, percent_ones, 
               width=bar_width, color=colors[0], 
               label='1' if i == 0 else "", edgecolor='black')

    # Set labels, title, and legends
    ax.set_ylabel('Porcentaje', fontsize=18, labelpad = 20)
    ax.set_title(title, fontsize=18, pad = 20)
    
    # Set x-ticks and x-labels, then rotate them
    ax.set_xticks(index)
    ax.set_xticklabels(binary_data.columns, rotation=40, fontsize=14)
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2)

    ax.legend(["50%","Si presenta", "No presenta"], loc='best', fontsize=12)
    
    # Adjust layout to prevent labels from getting cut off
    plt.tight_layout()
    plt.show()



#%% 

#############################
# Basic Plots of Clinical variables
#############################

plt.hist(covid_data.mmse_pnts, alpha = 0.6)
plt.hist(control_data.mmse_pnts, alpha = 0.6)
plt.xlabel('Puntuacion MOCA', fontsize=14)
plt.ylabel('Poblacion', fontsize=14)
plt.title('Histograma puntuacion MOCA', fontsize=16, fontweight='bold')
plt.grid()
plt.legend()
plt.show()


# Example usage with dummy data:
plot_histogram_with_binary_outcome(covid_data.moca_pnts, covid_data.persistencia_delirio, 10, "Puntuacion MOCA", "Delirio")
plot_histogram_with_binary_outcome(covid_data.moca_pnts, covid_data.duracion_fatiga, 10, "Puntuacion MOCA", "Fatiga")
plot_histogram_with_binary_outcome(covid_data.moca_pnts, covid_data.mov_anormales, 10, "Puntuacion Moca", "Mov. anormales")
plot_histogram_with_binary_outcome(covid_data.moca_pnts, covid_data.cefalea_postcovid, 10, "Puntuacion MOCA", "Cefalea")
plot_histogram_with_binary_outcome(covid_data.moca_pnts, covid_data.recupera_olfatogusto, 10, "Puntuacion MOCA", "Olfato y gusto")


plot_histogram_with_binary_outcome(covid_data.saturacion_ingreso, covid_data.persistencia_delirio, 10, "Saturacion ingreso", "Delirio")
plot_histogram_with_binary_outcome(covid_data.saturacion_ingreso, covid_data.duracion_fatiga, 10, "Saturacion ingreso", "Fatiga")
plot_histogram_with_binary_outcome(covid_data.saturacion_ingreso, covid_data.mov_anormales, 10, "Saturacion ingreso", "Mov. anormales")
plot_histogram_with_binary_outcome(covid_data.saturacion_ingreso, covid_data.cefalea_postcovid, 10, "Saturacion ingreso", "Cefalea")
plot_histogram_with_binary_outcome(covid_data.saturacion_ingreso, covid_data.recupera_olfatogusto, 10, "Saturacion ingreso", "Olfato y gusto")


moca_normal = covid_data[covid_data['moca_pnts'] >= 26]
plot_binary_data(moca_normal[['no_recupera_olfatogusto','cefalea_postcovid', 'persistencia_delirio',
                    'mov_anormales','duracion_fatiga']],
                 title="Variables binarias con MMSE bajo")


low_moca = covid_data[covid_data['moca_pnts'] <= 26]
plot_binary_data(low_moca[['no_recupera_olfatogusto','cefalea_postcovid', 'persistencia_delirio',
                    'mov_anormales','duracion_fatiga']],
                 title="Variables binarias con MMSE normal")



sat_bajo = covid_data[covid_data['spo2'] >= 70]
plot_binary_data(sat_bajo[['no_recupera_olfatogusto','cefalea_postcovid', 'persistencia_delirio',
                    'mov_anormales','duracion_fatiga']],
                 title= "Variables binarias con Saturacion baja")


sat_normal = covid_data[covid_data['spo2'] < 70]
plot_binary_data(sat_normal[['no_recupera_olfatogusto','cefalea_postcovid', 'persistencia_delirio',
                    'mov_anormales','duracion_fatiga']],
                 title="Variables binarias con Saturacion normal")


binary_cols = ['mov_anormales','no_recupera_olfatogusto','cefalea_postcovid','persistencia_delirio']

mmse_normal = covid_data[covid_data['mmse_pnts'] >= 27]
mmse_bajo = covid_data[covid_data['mmse_pnts'] < 27]

sat_normal = covid_data[covid_data['spo2'] >= 70]
sat_bajo = covid_data[covid_data['spo2'] < 70]


print( (mmse_normal[binary_cols].mean() * 100) ) 
print( mmse_bajo[binary_cols].mean() * 100 ) 

print( moca_normal[binary_cols].mean() * 100 ) 
print( moca_normal[binary_cols].mean() * 100 ) 

print((mmse_normal[binary_cols].mean() * 100).round(2).to_string(index=False, header=False))
print((mmse_bajo[binary_cols].mean() * 100).round(2).to_string(index=False, header=False))

print((sat_normal[binary_cols].mean() * 100).round(2).to_string(index=False, header=False))
print((sat_bajo[binary_cols].mean() * 100).round(2).to_string(index=False, header=False))


for col in binary_cols:
    # Count 0's and 1's for each group
    normal_0 = (sat_normal[col] == 0).sum()
    normal_1 = (sat_normal[col] == 1).sum()
    bajo_0   = (sat_bajo[col]   == 0).sum()
    bajo_1   = (sat_bajo[col]   == 1).sum()

    # Build 2×2 table for chi-square
    table = [[normal_0, normal_1],
             [bajo_0,   bajo_1]]

    chi2, p, dof, expected = chi2_contingency(table)

    print(f"{col}: chi2={chi2:.2f}, p={p:.4f}, dof={dof}")



#%%

#############################
# Functions for importing image metrics
#############################


def load_fa_adc_data(fa_adc_folder):
    """
    Loads four comma-separated .txt files from 'fa_adc' These .txt containing the averaging information obtained
    from a DTI image.
    To generate the .txt files one should run jhu_stats.sh
    ----------
    Inputs: 
        fa_adc_folder (string) : A string containing the path for the .txt files
 
    Outputs:
        control_adc_mean (df): A df containing the adc mean value for the DTI image in JHU atlas space for the control group 
        control_fa_mean (df): A df containing the fa mean value for the DTI image in JHU atlas space for the control group
        treatment_adc_mean (df): A df containing the adc mean value for the DTI image in JHU atlas space for the covid group
        treatment_fa_mean (df): A df containing the fa mean value for the DTI image in JHU atlas space for the covid group
    """
    import pandas as pd

    def read_file(name):
        df = pd.read_csv(f"{fa_adc_folder}/{name}.txt", sep=",", header=0, index_col=0)
        # Reset the index to get rid of "sub-XX" labels, using 0-based integers instead
        df.reset_index(drop=True, inplace=True)
        return df
    
    return read_file("control_adc_mean"),read_file("control_fa_mean"),read_file("treatment_adc_mean"),read_file("treatment_fa_mean")
    

def load_network_data(network_folder):
    """
    Loads three comma-separated .txt files from 'network' These .txt containing the network metrics from the structural
    connnectome derived from individual DWIs and Tractograns
    To generate the .txt files one should run network_metrics.m
    ----------
    Inputs: 
        network_folder (string) : A string containing the path for the .txt files
 
    Outputs:
        betw (df): A df containing the betweeness centrality network measure from the SC  
        cluster (df): A df containing the cluster coefficient network measure from the SC
        degree (df): A df containing the degree network measure from the SC
    """
    def read_file(name):
        return pd.read_csv(f"{network_folder}/{name}.txt", sep=",", header=None)
    
    return read_file("betw_w"),read_file("cluster_w"),read_file("degree_w")
    


def append_image_metrics(atlas, atlas_regions, atlas_region_names, target_df, origin_df, combine_rois=True):
    """
    Appends image metrics from an origin DataFrame to a target DataFrame.
    
    Parameters:
        atlas (str): Either "JHU" or "DK".
        atlas_regions (list of tuple): Pairs of region identifiers.
        atlas_region_names (list of str): Names for the ROIs.
        target_df (pd.DataFrame): DataFrame to which metrics will be appended.
        origin_df (pd.DataFrame): DataFrame from which to extract the image metrics.
        combine_rois (bool, optional): If True, combines left and right hemisphere data by averaging.
        
    Returns:
        pd.DataFrame: Updated DataFrame with appended image metrics.
    """
    # 1) Copy and reset index for both target and origin DataFrames to ensure proper row alignment.
    updated_df = target_df.copy().reset_index(drop=True)
    origin_df = origin_df.copy().reset_index(drop=True)

    # 2) Create or copy the "g" (global) column depending on atlas
    if atlas == "JHU":
        updated_df["g"] = origin_df["g"]
    elif atlas == "DK":
        if combine_rois:
            # Average each pair, then average across pairs
            pair_means = [origin_df[[col1, col2]].mean(axis=1) for col1, col2 in atlas_regions]
            pair_means_df = pd.concat(pair_means, axis=1)
            updated_df["g"] = pair_means_df.mean(axis=1)
        else:
            # Flatten all columns (L+R) and average
            all_cols = [c for (col1, col2) in atlas_regions for c in (col1, col2)]
            updated_df["g"] = origin_df[all_cols].mean(axis=1)
    else:
        raise ValueError("atlas must be either 'JHU' or 'DK'")

    # 3) Append the individual or averaged columns for each region.
    for (col1, col2), region_name in zip(atlas_regions, atlas_region_names):
        if combine_rois:
            updated_df[region_name] = origin_df[[col1, col2]].mean(axis=1)
        else:
            if atlas == "DK":
                updated_df[region_name + "_L"] = origin_df[col1]
                updated_df[region_name + "_R"] = origin_df[col2]
            else:  # For JHU or fallback.
                updated_df[region_name + "_R"] = origin_df[col1]
                updated_df[region_name + "_L"] = origin_df[col2]

    return updated_df


#%% Importing Control and Covid data already generated from replace_outliers and control_covid_df

#############################
# Importing Control and Covid db
#############################

covid_data_path = 'results/clinical_db/datos_clinicos_covidsp.xlsx'
control_data_path = 'results/clinical_db/datos_clinicos_control.xlsx'
save_path = 'results/clincal_db/'

covid_data, control_data = covid_control_df(covid_data_path, control_data_path, save_path)

del covid_data_path, control_data_path, save_path

#%% Importing DTI and Network Image metrics

#############################
# Importing and appending image metrics
#############################

path_adc_fa = 'results/fa_adc/'
path_network_metrics = 'results/sc_dk/weight/network'

control_mean_adc, control_mean_fa, treatment_mean_adc, treatment_mean_fa = load_fa_adc_data(path_adc_fa)
betw, cluster, degree = load_network_data(path_network_metrics)

del path_adc_fa, path_network_metrics



# Define columns to move
# 7,8 = Corticospinal tract (abnormal mov.)
# 15,16 = Cerebral peduncle (abnormal mov.)
# 37,38 = Cingulum (hippocampus) (spo2)
# 47,48 = Uncinate fasciculus (olfact)
jhu_regions = [("7", "8"), ("15", "16"), ("37", "38"), ("47", "48")]
jhu_region_names = ["Corticospinal_tract", "Cerebral_peduncle", "Hippocampus", "Unc_fasciculus"]

# 40,47 = Hippocampus
# 41,48 = Amygdala
# 23,72 = Precentral gyrus
# 35,84 = Cerebellum
# 60,11 = Lateral orbital frontal cortex
# 5,54 = Entorhinal
dk_regions = [(39,46),(40,47), (22,71), (34,83), (10,59), (4,50)]
dk_region_names = ["Hippocampus", "Amygdala", "Precentral_gyrus", "Cerebellum", "Lateral_orbital_frontal", "Entorhinal"]



control_data_fa = append_image_metrics("JHU",jhu_regions, jhu_region_names, control_data, control_mean_fa, combine_rois=True) 
control_data_adc = append_image_metrics("JHU",jhu_regions, jhu_region_names, control_data, control_mean_adc, combine_rois=True) 

covid_data_fa = append_image_metrics("JHU",jhu_regions, jhu_region_names, covid_data, treatment_mean_fa, combine_rois=True) 
covid_data_adc = append_image_metrics("JHU",jhu_regions, jhu_region_names, covid_data, treatment_mean_adc, combine_rois=True) 


control_data_degree = append_image_metrics("DK", dk_regions, dk_region_names, control_data, degree[:49], combine_rois=True) 
control_data_betw = append_image_metrics("DK", dk_regions, dk_region_names, control_data, betw[:49], combine_rois=True) 
control_data_cluster = append_image_metrics("DK", dk_regions, dk_region_names, control_data, cluster[:49], combine_rois=True) 

covid_data_degree = append_image_metrics("DK", dk_regions, dk_region_names, covid_data, degree[49:], combine_rois=True) 
covid_data_betw = append_image_metrics("DK", dk_regions, dk_region_names, covid_data, betw[49:], combine_rois=True) 
covid_data_cluster = append_image_metrics("DK", dk_regions, dk_region_names, covid_data, cluster[49:], combine_rois=True) 



del control_data, control_mean_fa, control_mean_adc, covid_data, treatment_mean_fa, treatment_mean_adc
del degree, cluster, betw
del dk_region_names, dk_regions, jhu_region_names, jhu_regions

#%%%

#############################
# Plot functions for clinical and image metrics
#############################

def plot_two_histograms(data1, data2, label1='Control', label2='Covid', title="Histograms with Means, Cohen's d & p-value"):
    """
    Appends image metrics from a origin df to a target df. Usually, the origin df will be an output of 
    load_fa_adc_data or load_network_data. Specific Rois and their names must be added. Can combine 
    Left and Right hemisphere data to generate a single mean.
    ----------
    Inputs: 
        atlas (string): Specify if the atlas is "JHU" or "DK"
        atlas_regions (list with pair of strings) : The numeric ID for the atlas. Found in specific .txt or .json
        atlas_region_names (list): Name of the ROIs in the columns of the df 
        origin (df): Dataframe from which to extract the image metrics
        target (df): Dataframe to appen the extracted image metrics
        combina_rois (boolean, True default): If true, combines the Left and Right hemisphere into a single average
        
    Outputs:
       histogram plot
    """

    # Convert to NumPy arrays (if they're Pandas Series)
    data1, data2 = np.array(data1), np.array(data2)

    # Basic stats
    mean1, mean2 = data1.mean(), data2.mean()
    var1, var2 = data1.var(ddof=1), data2.var(ddof=1)
    print(f"{label1} mean = {mean1:.3f}")
    print(f"{label1} variance = {var1:.3f}")
    print(f"{label2} mean = {mean2:.3f}")
    print(f"{label2} variance = {var2:.3f}")
    n1, n2 = len(data1), len(data2)

    # Cohen's d (pooled SD), using absolute value
    pooled_std = np.sqrt(((n1 - 1)*var1 + (n2 - 1)*var2) / (n1 + n2 - 2))
    cohens_d = abs((mean1 - mean2) / pooled_std)

    # Two-sided t-test
    _, p_val = ttest_ind(data1, data2, equal_var=True)
    stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''

    # Plot
    plt.figure(figsize=(10, 6))
    _, bins1, _ = plt.hist(data1, bins='auto', color='blue', alpha=0.55, label=label1)
    _, bins2, _ = plt.hist(data2, bins='auto', color='red',   alpha=0.55, label=label2)
    plt.axvline(mean1, color='blue', linestyle='--', label=f'{label1} mean')
    plt.axvline(mean2, color='red',   linestyle='--', label=f'{label2} mean')

    # Bracket position
    max_count1 = np.histogram(data1, bins=bins1)[0].max()
    max_count2 = np.histogram(data2, bins=bins2)[0].max()
    max_hist = max(max_count1, max_count2)
    bracket_y = max_hist * 1.2
    bracket_ht = max_hist * 0.05
    left_mean, right_mean = sorted([mean1, mean2])

    # Bracket & lines
    plt.hlines(bracket_y, left_mean, right_mean, color='k')
    plt.vlines([left_mean, right_mean], bracket_y - bracket_ht, bracket_y, color='k')

    # Text positions (Cohen's d above, p-value just below)
    plt.text((left_mean + right_mean)/2, bracket_y + bracket_ht*2, 
             f"Cohen's d = {cohens_d:.2f}", ha='center')
    plt.text((left_mean + right_mean)/2, bracket_y + bracket_ht, 
             f"p value: {p_val:.3f} {stars}", ha='center')

    # Final formatting
    plt.ylim(top=bracket_y + bracket_ht * 3.25)
    plt.grid(True)
    plt.xlabel("Value", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.show()



def plot_roc(data, continuous_col, outcome_col, invert_measure=False):
    """
    Appends image metrics from a origin df to a target df. Usually, the origin df will be an output of 
    load_fa_adc_data or load_network_data. Specific Rois and their names must be added. Can combine 
    Left and Right hemisphere data to generate a single mean.
    ----------
    Inputs: 
        atlas (string): Specify if the atlas is "JHU" or "DK"
        atlas_regions (list with pair of strings) : The numeric ID for the atlas. Found in specific .txt or .json
        atlas_region_names (list): Name of the ROIs in the columns of the df 
        origin (df): Dataframe from which to extract the image metrics
        target (df): Dataframe to appen the extracted image metrics
        combina_rois (boolean, True default): If true, combines the Left and Right hemisphere into a single average
        
    Outputs:
       updated_df (df): Df with the appended image metrics
    """
    
    test_measure = data[continuous_col].copy()
    if invert_measure:
        test_measure = -test_measure
    
    true_outcome = data[outcome_col].copy()

    fpr, tpr, thresholds = roc_curve(true_outcome, test_measure)
    roc_auc = auc(fpr, tpr)

    youden = tpr - fpr
    best_idx = np.argmax(youden)

    df_roc = pd.DataFrame({
        'threshold': thresholds,
        'fpr': fpr,
        'tpr': tpr,
        'youden': youden
    })

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.axvline(x=fpr[best_idx], color='red', linestyle='--', label='Youden Index')
    plt.plot(fpr[best_idx], tpr[best_idx], 'ro')
    plt.xlim([0, 1])
    plt.ylim([0, 1.02])
    plt.xlabel('FPR (1 - Specificity)', fontsize=15)
    plt.ylabel('TPR (Sensitivity)', fontsize=15)
    plt.title(f'ROC: {continuous_col}', fontsize=17)
    plt.grid()
    plt.legend()
    plt.show()

    return df_roc



def single_roc(continuous, binary, invert_measure=False):
    """Compute ROC metrics and return a DataFrame."""
    measure = continuous.copy()
    if invert_measure:
        measure = -measure
    fpr, tpr, thresholds = roc_curve(binary.copy(), measure)
    df_roc = pd.DataFrame({
        'threshold': thresholds,
        'fpr': fpr,
        'tpr': tpr,
        'youden': tpr - fpr
    })
    return df_roc

def multiple_roc(continuous1, continuous2, continuous3, binary, invert_measure=None, legend=None, title="Multiple ROC Curves"):
    """
    Plot multiple ROC curves (three by default) with custom legends, title, and styling.

    Parameters:
        continuous1, continuous2, continuous3: Continuous predictor data.
        binary: Common binary outcome data.
        invert_measure (list of bool, optional): A list of booleans for each continuous variable indicating 
            whether to invert its values. Defaults to [False, False, False].
        legend: List of strings to prefix series names (e.g., ["1", "2", "3"]).
        title: Plot title.

    Returns:
        dict: Dictionary with DataFrames for each ROC curve.
    """
    # Ensure invert_measure is a list of booleans for each curve.
    curves = [continuous1, continuous2, continuous3]
    if invert_measure is None:
        invert_measure = [False] * len(curves)
    if len(invert_measure) != len(curves):
        raise ValueError("invert_measure must be a list with three boolean values.")

    # Compute ROC DataFrames for each continuous predictor.
    df_rocs = [single_roc(c, binary, invert_measure[i]) for i, c in enumerate(curves)]
    colors = ['red', 'hotpink', 'purple']
    
    plt.figure(figsize=(8, 6))
    for idx, (df_roc, cont) in enumerate(zip(df_rocs, curves)):
        # Determine label and compute AUC.
        base_label = cont.name if hasattr(cont, 'name') and cont.name is not None else f'Curve {idx+1}'
        label_prefix = legend[idx] if legend and len(legend) > idx else ""
        auc_val = auc(df_roc['fpr'], df_roc['tpr'])
        label = f"{label_prefix}{base_label} (AUC = {auc_val:.2f})"
        
        # Plot ROC curve.
        plt.plot(df_roc['fpr'], df_roc['tpr'], label=label, color=colors[idx])
        
        # Mark and annotate maximum Youden's index.
        best_idx = df_roc['youden'].idxmax()
        plt.plot(df_roc.loc[best_idx, 'fpr'], df_roc.loc[best_idx, 'tpr'], 'o', color=colors[idx])
        plt.text(df_roc.loc[best_idx, 'fpr'] + 0.015, df_roc.loc[best_idx, 'tpr'] - 0.05,
                 f"Youden={df_roc.loc[best_idx, 'youden']:.2f}",
                 color=colors[idx], fontsize=12)
        #plt.axvline(x=df_roc.loc[best_idx, 'fpr'], color=colors[idx], linestyle='--')
    
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.xlabel('FPR (1 - Specificity)', fontsize=15)
    plt.ylabel('TPR (Sensitivity)', fontsize=15)
    plt.title(title, fontsize=17)
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
    
    return {'roc1': df_rocs[0], 'roc2': df_rocs[1], 'roc3': df_rocs[2]}


#%%

#############################
# Ploting mostly image metrics
#############################

plot_two_histograms(control_data_fa.g,covid_data_fa.g, title="FA global value")
plot_two_histograms(control_data_adc.g,covid_data_adc.g, title="ADC global value")
plot_two_histograms(control_data_degree.g, covid_data_degree.g, title="Degree global value")



#MMSE
olfact_roc_degree =multiple_roc(covid_data_adc["g"],covid_data_fa["g"],covid_data_degree["g"] ,
                                   binary=(covid_data_adc["mmse_pnts"] > 27).astype(int),
                                   legend=["[ADC] ","[FA] ","[Degree] "], invert_measure=[False,True,True],
                                   title="MMSE con ROI Global")

#Respiracion con g e Hipoccampo
olfact_roc_degree =multiple_roc(covid_data_adc["g"],covid_data_fa["g"],covid_data_degree["g"] ,
                                   binary=(covid_data_adc["spo2"] > 60).astype(int),
                                   legend=["[ADC] ","[FA] ","[Degree] "], invert_measure=[True,False,True],
                                   title="Oxigenacion(binaria) Spo2. con ROI Global")

olfact_roc_degree =multiple_roc(covid_data_adc["g"],covid_data_fa["g"],covid_data_degree["g"] ,
                                   binary=(covid_data_adc["saturacion_ingreso"] > 70).astype(int),
                                   legend=["[ADC] ","[FA] ","[Degree] "], invert_measure=[True,False,True],
                                   title="Oxigenacion (binaria) Sat. Ing. con ROI GLobal")

olfact_roc_degree =multiple_roc(covid_data_adc["Hippocampus"],covid_data_fa["Hippocampus"],covid_data_degree["Hippocampus"] ,
                                   binary=(covid_data_adc["saturacion_ingreso"] > 70).astype(int),
                                   legend=["[ADC] ","[FA] ","[Degree] "], invert_measure=[True,False,True],
                                   title="Oxigenacion (binaria) con Sat.Ing. con ROI Hipocampo")

#Cefalea postcovid
olfact_roc_degree =multiple_roc(covid_data_adc["g"],covid_data_fa["g"],covid_data_degree["g"] ,
                                   binary=covid_data_adc["cefalea_postcovid"], 
                                   legend=["[ADC] ","[FA] ","[Degree] "], invert_measure=[False,True,True],
                                   title="Cefalea post covid con ROI Global")

#Persistencia delirio
olfact_roc_degree =multiple_roc(covid_data_adc["g"],covid_data_fa["g"],covid_data_degree["g"] ,
                                   binary=covid_data_adc["persistencia_delirio"], 
                                   legend=["[ADC] ","[FA] ","[Degree] "], invert_measure=[False,True,False],
                                   title="Persistencia delirio con ROI Global")

#Movimientos abnormales
olfact_roc_degree =multiple_roc(covid_data_adc["Cerebral_peduncle"],covid_data_fa["Cerebral_peduncle"],covid_data_degree["Cerebellum"] ,
                                   binary=covid_data_adc["mov_anormales"], 
                                   legend=["[ADC] ","[FA] ","[Degree] "], invert_measure=[True,False,False],
                                   title="Movimientos abnormales con ROI Cerebelear")

olfact_roc_degree =multiple_roc(covid_data_adc["Corticospinal_tract"],covid_data_fa["Corticospinal_tract"],covid_data_degree["Precentral_gyrus"] ,
                                   binary=covid_data_adc["mov_anormales"], 
                                   legend=["[ADC] ","[FA] ","[Degree] "], invert_measure=[False,False,False],
                                   title="Movimientos abnormales con ROI Corticospinal")



#%%

#############################
# Helper function to compare two stats
#############################

def two_groups_compare_stats(col1, col2, title):
    """
    Compare two numeric Series (e.g., DataFrame columns) by calculating:
      - Mean of each
      - Standard deviation of each
      - p-value from an independent samples t-test
      - Cohen's d effect size

    Returns:
      mean1, std1, mean2, std2, pvalue, cohen_d
    """
    mean1 = col1.mean()
    std1 = col1.std()
    mean2 = col2.mean()
    std2 = col2.std()

    # Perform an independent t-test
    t_stat, p_value = ttest_ind(col1, col2, nan_policy='omit')

    # Calculate pooled standard deviation for Cohen's d
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    cohen_d = (mean1 - mean2) / pooled_std
    
    if p_value < 0.001:
       stars = "***"
    elif p_value < 0.01:
       stars = "**"
    elif p_value < 0.05:
       stars = "*"
    else:
       stars = ""


    print(title)
    print("Control mean: " + str(mean1))
    print("Control std: " + str(std1))
    print("Covid mean: " + str(mean2))
    print("Covid std: " + str(std2))
    print("p-value: ", f"{p_value} {stars}".strip())
    print("Cohen_d: " + str(cohen_d ))
    print("------------------------------- \n")
    return mean1, std1, mean2, std2, p_value, cohen_d



#%%

#############################
# Comparing lots of stats so I can build many tables. COntrol vs Covid
#############################

# JHU
# 7,8 = Corticospinal tract (abnormal mov.)
# 15,16 = Cerebral peduncle (abnormal mov.)
# 37,38 = Cingulum (hippocampus) (spo2)
# 47,48 = Uncinate fasciculus (olfact)

#############################
# 1) FA Control vs FA Treatment
#############################
two_groups_compare_stats(
    control_data_fa["Corticospinal_tract"], covid_data_fa["Corticospinal_tract"], title="FA Corticospinal tract")

two_groups_compare_stats(
    control_data_fa["Cerebral_peduncle"], covid_data_fa["Cerebral_peduncle"], title="FA Cerebral peduncle")

two_groups_compare_stats(
    control_data_fa["Hippocampus"], covid_data_fa["Hippocampus"], title="FA Hippocampus")

two_groups_compare_stats(
    control_data_fa["Unc_fasciculus"], covid_data_fa["Unc_fasciculus"], title="FA Uncinate fasciculus")
    

#############################
# 2) ADC Control vs ADC Treatment 
#############################
two_groups_compare_stats(
    control_data_adc["Corticospinal_tract"], covid_data_adc["Corticospinal_tract"], title="ADC Corticospinal tract")

two_groups_compare_stats(
    control_data_adc["Cerebral_peduncle"], covid_data_adc["Cerebral_peduncle"], title="ADC Cerebral peduncle")

two_groups_compare_stats(
    control_data_adc["Hippocampus"], covid_data_adc["Hippocampus"], title="ADC Hippocampus")

two_groups_compare_stats(
    control_data_adc["Unc_fasciculus"], covid_data_adc["Unc_fasciculus"], title="ADC Uncinate fasciculus")


# DK
# 40,47 = Hippocampus
# 41,48 = Amygdala
# 23,72 = Precentral gyrus
# 35,84 = Cerebellum
# 60,11 = Lateral orbital frontal cortex
# 5,54 = Entorhinal

#############################
# 3) Network Control vs Network Covid
#############################
two_groups_compare_stats(
    control_data_degree["Hippocampus"], covid_data_degree["Hippocampus"], title="Degree Hippocampus")

two_groups_compare_stats(
    control_data_degree["Amygdala"], covid_data_degree["Amygdala"],title="Degree Amygdala")

two_groups_compare_stats(
    control_data_degree["Precentral_gyrus"], covid_data_degree["Precentral_gyrus"], title="Degree Precentral Gyrus")

two_groups_compare_stats(
    control_data_degree["Cerebellum"], covid_data_degree["Cerebellum"], title="Degree Cerebellum")

two_groups_compare_stats(
    control_data_degree["Lateral_orbital_frontal"], covid_data_degree["Lateral_orbital_frontal"], title="Degree Lateral orbital frontal")

two_groups_compare_stats(
    control_data_degree["Entorhinal"], covid_data_degree["Entorhinal"], title="Degree Entorhinal")

#%%
#############################
# Comparing lots of stats so I can build many tables. Covid subgroups
#############################
#Abnormal movement group -------------------
two_groups_compare_stats(covid_data_fa.loc[covid_data_fa["mov_anormales"] == 1, "g"], 
                         covid_data_fa.loc[covid_data_fa["mov_anormales"] == 0, "g"], title="FA g Abn. mov.")

two_groups_compare_stats(covid_data_adc.loc[covid_data_fa["mov_anormales"] == 1, "g"], 
                         covid_data_adc.loc[covid_data_fa["mov_anormales"] == 0, "g"], title="ADC g Abn. mov.")

two_groups_compare_stats(covid_data_degree.loc[covid_data_degree["mov_anormales"] == 1, "g"], 
                         covid_data_degree.loc[covid_data_degree["mov_anormales"] == 0, "g"], title="Degree g Abn. mov")

two_groups_compare_stats(covid_data_fa.loc[covid_data_fa["mov_anormales"] == 1, "Corticospinal_tract"], 
                         covid_data_fa.loc[covid_data_fa["mov_anormales"] == 0, "Corticospinal_tract"], title="FA Corticospinal tract Abn. mov.")

two_groups_compare_stats(covid_data_adc.loc[covid_data_adc["mov_anormales"] == 1, "Corticospinal_tract"], 
                         covid_data_adc.loc[covid_data_adc["mov_anormales"] == 0, "Corticospinal_tract"], title="ADC Corticospinal tract Abn. mov.")

two_groups_compare_stats(covid_data_fa.loc[covid_data_fa["mov_anormales"] == 1, "Cerebral_peduncle"], 
                         covid_data_fa.loc[covid_data_fa["mov_anormales"] == 0, "Cerebral_peduncle"], title="FA Corticospinal tract Abn. mov.")

two_groups_compare_stats(covid_data_adc.loc[covid_data_adc["mov_anormales"] == 1, "Cerebral_peduncle"], 
                         covid_data_adc.loc[covid_data_adc["mov_anormales"] == 0, "Cerebral_peduncle"], title="ADC Corticospinal tract Abn. mov.")

two_groups_compare_stats(covid_data_degree.loc[covid_data_degree["mov_anormales"] == 1, "Cerebellum"], 
                         covid_data_degree.loc[covid_data_degree["mov_anormales"] == 0, "Cerebellum"], title="Degree Cerebellum Abn. mov.")

two_groups_compare_stats(covid_data_degree.loc[covid_data_degree["mov_anormales"] == 1, "Precentral_gyrus"], 
                         covid_data_degree.loc[covid_data_degree["mov_anormales"] == 0, "Precentral_gyrus"], title="Degree Precentral gyrus Abn. mov.")


#Fatiga group ---------------------------
two_groups_compare_stats(covid_data_fa.loc[covid_data_fa["duracion_fatiga"] == 1, "g"], 
                         covid_data_fa.loc[covid_data_fa["duracion_fatiga"] == 0, "g"], title="FA g duracion fatiga")

two_groups_compare_stats(covid_data_adc.loc[covid_data_adc["duracion_fatiga"] == 1, "g"], 
                         covid_data_adc.loc[covid_data_adc["duracion_fatiga"] == 0, "g"], title="ADC g duracion fatiga.")

two_groups_compare_stats(covid_data_degree.loc[covid_data_degree["duracion_fatiga"] == 1, "g"], 
                         covid_data_degree.loc[covid_data_degree["duracion_fatiga"] == 0, "g"], title="Degree g duracion fatiga")

#Cefalea group --------------------------
two_groups_compare_stats(covid_data_fa.loc[covid_data_fa["cefalea_postcovid"] == 1, "g"], 
                         covid_data_fa.loc[covid_data_fa["cefalea_postcovid"] == 0, "g"], title="FA g cefalea post covid")

two_groups_compare_stats(covid_data_adc.loc[covid_data_adc["cefalea_postcovid"] == 1, "g"], 
                         covid_data_adc.loc[covid_data_adc["cefalea_postcovid"] == 0, "g"], title="ADC g cefalea post covid")

two_groups_compare_stats(covid_data_degree.loc[covid_data_degree["cefalea_postcovid"] == 1, "g"], 
                         covid_data_degree.loc[covid_data_degree["cefalea_postcovid"] == 0, "g"], title="Degree g cefalea postcovid")


#Delirio group ----------------------------
two_groups_compare_stats(covid_data_fa.loc[covid_data_fa["persistencia_delirio"] == 1, "g"], 
                         covid_data_fa.loc[covid_data_fa["persistencia_delirio"] == 0, "g"], title="FA g delirio")

two_groups_compare_stats(covid_data_adc.loc[covid_data_degree["persistencia_delirio"] == 1, "g"], 
                         covid_data_adc.loc[covid_data_degree["persistencia_delirio"] == 0, "g"], title="ADC g delirio")

two_groups_compare_stats(covid_data_degree.loc[covid_data_degree["persistencia_delirio"] == 1, "g"], 
                         covid_data_degree.loc[covid_data_degree["persistencia_delirio"] == 0, "g"], title="Degree g delirio")


#Respiration related group ----------------
two_groups_compare_stats(covid_data_degree.loc[covid_data_degree["saturacion_ingreso"] <= 70, "Hippocampus"], 
                         covid_data_degree.loc[covid_data_degree["saturacion_ingreso"] > 70, "Hippocampus"], title="Degree Hippocampus sat")

two_groups_compare_stats(covid_data_degree.loc[covid_data_degree["saturacion_ingreso"] <= 70, "Amygdala"], 
                         covid_data_degree.loc[covid_data_degree["saturacion_ingreso"] > 70, "Amygdala"], title="Degree Amygdala sat")




two_groups_compare_stats(covid_data_fa.loc[covid_data_fa["spo2"] <= 70, "g"], 
                         covid_data_fa.loc[covid_data_fa["spo2"] > 70, "g"], title="FA Global spo2")

two_groups_compare_stats(covid_data_adc.loc[covid_data_adc["spo2"] <= 70, "g"], 
                         covid_data_adc.loc[covid_data_adc["spo2"] > 70, "g"], title="ADC Global spo2")

two_groups_compare_stats(covid_data_degree.loc[covid_data_degree["spo2"] <= 70, "g"], 
                         covid_data_degree.loc[covid_data_degree["spo2"] > 70, "g"], title="Degree Global spo2")

two_groups_compare_stats(covid_data_fa.loc[covid_data_fa["spo2"] <= 70, "Hippocampus"], 
                         covid_data_fa.loc[covid_data_fa["spo2"] > 70, "Hippocampus"], title="FA Hippocampus spo2")

two_groups_compare_stats(covid_data_adc.loc[covid_data_adc["spo2"] <= 70, "Hippocampus"], 
                         covid_data_adc.loc[covid_data_adc["spo2"] > 70, "Hippocampus"], title="ADC Hippocampus spo2")

two_groups_compare_stats(covid_data_degree.loc[covid_data_degree["spo2"] <= 70, "Hippocampus"], 
                         covid_data_degree.loc[covid_data_degree["spo2"] > 70, "Hippocampus"], title="Degree Hippocampus spo2")

two_groups_compare_stats(covid_data_degree.loc[covid_data_degree["spo2"] <= 70, "Amygdala"], 
                         covid_data_degree.loc[covid_data_degree["spo2"] > 70, "Amygdala"], title="Degree Amygdala spo2")


#Olfact group -------------------------------


two_groups_compare_stats(covid_data_fa.loc[covid_data_fa["no_recupera_olfatogusto"] == 1, "g"], 
                         covid_data_fa.loc[covid_data_fa["no_recupera_olfatogusto"] == 0, "g"], title="FA g no_recupera_olfatogusto.")

two_groups_compare_stats(covid_data_adc.loc[covid_data_fa["no_recupera_olfatogusto"] == 1, "g"], 
                         covid_data_adc.loc[covid_data_fa["no_recupera_olfatogusto"] == 0, "g"], title="ADC g no_recupera_olfatogusto")

two_groups_compare_stats(covid_data_degree.loc[covid_data_degree["no_recupera_olfatogusto"] == 1, "g"], 
                         covid_data_degree.loc[covid_data_degree["no_recupera_olfatogusto"] == 0, "g"], title="Degree g no_recupera_olfatogusto")


two_groups_compare_stats(covid_data_fa.loc[covid_data_fa["no_recupera_olfatogusto"] == 1, "Unc_fasciculus"], 
                         covid_data_fa.loc[covid_data_fa["no_recupera_olfatogusto"] == 0, "Unc_fasciculus"], title="FA Unc fasiculus Olfato")

two_groups_compare_stats(covid_data_adc.loc[covid_data_adc["no_recupera_olfatogusto"] == 1, "Unc_fasciculus"], 
                         covid_data_adc.loc[covid_data_adc["no_recupera_olfatogusto"] == 0, "Unc_fasciculus"], title="ADC Unc fasiculus Olfato")

two_groups_compare_stats(covid_data_degree.loc[covid_data_degree["no_recupera_olfatogusto"] == 1, "Entorhinal"], 
                         covid_data_degree.loc[covid_data_degree["no_recupera_olfatogusto"] == 0, "Entorhinal"], title="Degree Entorhinal Olfato")

two_groups_compare_stats(covid_data_degree.loc[covid_data_degree["no_recupera_olfatogusto"] == 1, "Lateral_orbital_frontal"], 
                         covid_data_degree.loc[covid_data_degree["no_recupera_olfatogusto"] == 0, "Lateral_orbital_frontal"]
                         ,title="Degree Entorhinal Olfato")


#MMSE -----------------------------------------

two_groups_compare_stats(covid_data_fa.loc[covid_data_fa["mmse_pnts"] <= 27, "g"], 
                         covid_data_fa.loc[covid_data_fa["mmse_pnts"] > 27, "g"], title="FA Corticospinal tract Abn. mov.")

two_groups_compare_stats(covid_data_adc.loc[covid_data_degree["mmse_pnts"] <= 27, "g"], 
                         covid_data_adc.loc[covid_data_degree["mmse_pnts"] > 27, "g"], title="ADC Corticospinal tract Abn. mov.")

two_groups_compare_stats(covid_data_degree.loc[covid_data_degree["mmse_pnts"] <= 27, "g"], 
                         covid_data_degree.loc[covid_data_degree["mmse_pnts"] > 27, "g"], title="Degree g cefalea postcovid")




#%%


results_data_path = 'results/final_metrics/'

control_data_fa.to_csv(results_data_path + 'control_data_fa.txt')
control_data_adc.to_csv(results_data_path + 'control_data_adc.txt')
control_data_degree.to_csv(results_data_path + 'control_data_degree.txt')
control_data_cluster.to_csv(results_data_path + 'control_data_cluster.txt')
control_data_betw.to_csv(results_data_path + 'control_data_betw.txt')

covid_data_fa.to_csv(results_data_path + 'covid_data_fa.txt')
covid_data_adc.to_csv(results_data_path + 'covid_data_adc.txt')
covid_data_degree.to_csv(results_data_path + 'covid_data_degree.txt')
covid_data_cluster.to_csv(results_data_path + 'covid_data_cluster.txt')
covid_data_betw.to_csv(results_data_path + 'covid_data_betw.txt')





#--------------------------------------------------------------------------------------
#Version v1.0.
#--------------------------------------------------------------------------------------
#Get the lastest version at:
#--------------------------------------------------------------------------------------
#script by Alejandro Garma Oehmichen
#--------------------------------------------------------------------------------------