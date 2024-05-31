import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

datos_clinicos_covid = pd.read_csv('covid_data.csv', index_col = 0)
datos_clinicos_covid['comorbilidades'] = datos_clinicos_covid[
    ['no_recupera_olfatogusto', 'cefalea_postcovid', 'persistencia_delirio', 'mov_anormales', 'duracion_fatiga']
].sum(axis=1)
pcr_data = pd.read_csv('pcr_data.csv', index_col = 0)


def multiple_roc(data_to_threshold, scores, thresholds):
    """
    Generate ROC curves for multiple thresholds.
    
    Parameters:
    - data_to_threshold: pd.Series or list, the data to be thresholded.
    - scores: pd.Series or list, the data to be compared with.
    - thresholds: list of int, the thresholds to apply to data_to_threshold.
    
    Returns:
    - None
    """
    plt.figure()
    
    for threshold in thresholds:
        # Create binary outcome based on the current threshold
        binary_outcome = (data_to_threshold < threshold).astype(int)
        
        # Calculate the ROC curve
        fpr, tpr, _ = roc_curve(binary_outcome, scores)
        roc_auc = roc_auc_score(binary_outcome, scores)
        
        # Plot the ROC curve
        plt.plot(fpr, tpr, lw=2, label=f'Threshold < {threshold} ROC curve (area = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize = 13)
    plt.ylabel('True Positive Rate', fontsize = 13)
    plt.title(f'Receiver Operating Characteristic with {data_to_threshold.name}')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

multiple_roc(datos_clinicos_covid['spo2urgencias'], datos_clinicos_covid['comorbilidades'], [20,50,80])
    
for column in pcr_data.columns:  
    multiple_roc(pcr_data[column], datos_clinicos_covid['comorbilidades'],pcr_data[column].quantile([0.25,0.5,0.75]))
    
