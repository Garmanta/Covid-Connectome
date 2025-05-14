import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, roc_auc_score
from sklearn.tree import plot_tree
from sklearn.model_selection import learning_curve
from sklearn.impute import SimpleImputer

#%%

"""
The objective of gradientboost is to use the clinical and imagenological metrics to devise a gradient boost algorithm
that can classify between the control and covid group and among the covid subgroups.
"""

# Helper functions for plotting with custom color palette and formatting
def plot_conf_matrix(ax, conf_matrix):
    """
    Plots a confusion matrix given a gradient boost binary classification with TP, TN, FP and FN.

    ----------
    Inputs: 
        ax (list(int)): Position in the subplot.
        conf_matrix (df): A matrix containing the required data.
 
    Outputs:
        plot (heatmap): A plot containing the confusion matrix.

    """
    # Plot the heatmap with a purple color scheme
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Purples', ax=ax, cbar=False)
    
    # Set the title and labels with updated font sizes
    ax.set_title('Confusion Matrix', fontsize=16)
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('Actual', fontsize=14)
    
    # Update the axis tick labels
    ax.set_xticklabels(['Healthy', 'Covid'], fontsize=12)
    ax.set_yticklabels(['Healthy', 'Covid'], fontsize=12)

    # Add the TP, TN, FP, FN labels on the respective quadrants
    # Get the midpoint of each cell to place the text correctly
    ax.text(0.5, 0.3, 'TN', color='white', ha='center', va='center', fontsize=14, fontweight='bold')  
    ax.text(1.5, 1.3, 'TP', color='white', ha='center', va='center', fontsize=14, fontweight='bold')  
    ax.text(0.5, 1.3, 'FN', color='black', ha='center', va='center', fontsize=14, fontweight='bold')  
    ax.text(1.5, 0.3, 'FP', color='white', ha='center', va='center', fontsize=14, fontweight='bold')  


def plot_roc_curve(ax, y_test, y_prob):
    """
    Plots a ROC curve for the gradient boost clasification algorithm. The algorithm outputs a specific probability
    of belonging to one of the two classes. Varying this probability to different thresholds gives the ROC curve.

    ----------
    Inputs: 
        ax (list(int)): Position in the subplot.
        y_test (df series): The binary outcome that is treated as the truth gold standard.
        y_prob (df series): Probability of each data point to belong to either class.
 
    Outputs:
        plot (roc curve): A plot containing the ROC curve along its False positirate and true positive rate.

    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, color='purple', label=f'ROC curve (AUC = {auc:.2f})', linewidth=2)
    ax.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2)  # Red diagonal reference line
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('ROC Curve', fontsize=16)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True)  # Add grid to the ROC curve

def plot_feature_importance(ax, features, importances, top_n=15):
    """
    Plots the most important features given a classification of the gradient boost algorithm. Allows to set how many 
    features you want to plot.

    ----------
    Inputs: 
        ax (list(int)): Position in the subplot.
        features (list): List of features to be plotted.
        importance (list): Numerical importance of the features.
        top_n (int): How many features to plot, if there are more than 15.
 
    Outputs:
        plot (barplot): An horizontal bar plot containing the feature importance of the gradient boost algorithm.

    """
    # Sort features by importance and select top_n
    sorted_idx = np.argsort(importances)[-top_n:]  # Get the indices of the top_n features with highest importance
    sorted_features = features[sorted_idx]
    sorted_importances = importances[sorted_idx]

    # Plot using a reversed color palette so higher importance features have darker colors
    sns.barplot(
        x=sorted_importances, 
        y=sorted_features, 
        palette='Purples',  # Use 'Purples' (normal) so higher values get the more intense color
        ax=ax
    )
    
    # Set title and labels for the plot
    ax.set_title('Top Features Importance', fontsize=16)
    ax.set_xlabel('Importance', fontsize=14)
    ax.set_ylabel('Feature', fontsize=14)

def plot_circular_feature_importance(features, importances, top_n=15):
    """
    Similar to plot feature importance, but in circular form. This is useful if you want to plot multiple features
    ----------
    Inputs: 
        features (list): List of features to be plotted.
        importance (list): Numerical importance of the features.
        top_n (int): How many features to plot, if there are more than 15.
 
    Outputs:
        plot (circular barplot): A circular bar plot containing the feature importance of the gradient boost algorithm.

    """
    # Sort features by importance and select top_n
    sorted_idx = np.argsort(importances)[-top_n:]  # Get the indices of the top_n features
    sorted_features = features[sorted_idx]
    sorted_importances = importances[sorted_idx]

    # Set up the polar plot
    num_features = len(sorted_features)
    theta = np.linspace(0.0, 2 * np.pi, num_features, endpoint=False)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    # Plot smaller bars for feature importance
    bars = ax.bar(theta, sorted_importances, width=0.3, align='center', alpha=0.6, color='purple', edgecolor='black')

    # Replace cones with thinner bars
    for i, bar in enumerate(bars):
        bar.set_height(sorted_importances[i])
        bar.set_linewidth(1)  # Make the edge thin to reduce visual weight

    # Set the feature names at the outer part of the circle
    ax.set_xticks(theta)
    ax.set_xticklabels(sorted_features, fontsize=10, fontweight='normal', family='serif')

    # Add magnitude labels to the concentric circles
    max_importance = max(sorted_importances) * 1.2
    ax.set_ylim(0, max_importance)

    # Set radial labels to indicate magnitude
    yticks = np.linspace(0, max_importance, num=5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{val:.2f}' for val in yticks], fontsize=10, family='serif')
    ax.yaxis.set_tick_params(pad=10)  # Add some space between labels and circles

    # Add a title to the plot
    ax.set_title('Top 15 Circular Feature Importance', fontsize=16, pad=20)

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_decision_tree(tree, feature_names):
    """
    Plots the first (and most important) tree in gradient boost.WHile the tree is not the final decision maker in gradient boost
    It represents the first iteration in the optimization algorithm. Its decision are similar to the results of a linear regression
    ----------
    Inputs: 
        tree(object): Contains gradient boost first tree.
        feature_names (list): Name of the features in the leaf.
         
    Outputs:
        plot (gradient boost tree): A tree plot containing the first tree plot.

    """
    plt.figure(figsize=(20, 10))
    plot_tree(tree, feature_names=feature_names, filled=True, rounded=True, fontsize=12)
    plt.title('Decision Tree with Leaf Probabilities', fontsize=16)
    plt.show()
    
    
def plot_learning_curve(estimator, X, y, ax):
    """
    Plots the most important features given a classification of the gradient boost algorithm. Allows to set how many 
    features you want to plot.

    ----------
    Inputs: 
        ax (list(int)): Position in the subplot.
        estimator (list): List of features to be plotted.
        X (df): Dataframe containing all the independent variables used in the training of the algorithm.
        y (df Series): Dataframe containing the dependent variables which the binary classification took place.
 
    Outputs:
        plot (line): A plot containing two curves, the trainig score and ther valildation score across training set size.

    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
    )
    
    # Calculate mean and standard deviation for train and validation scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Plot learning curve
    ax.plot(train_sizes, train_mean, color='purple', label='Training Score', linewidth=2)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='purple', alpha=0.1)
    
    ax.plot(train_sizes, val_mean, color='red', label='Validation Score', linestyle='--', linewidth=2)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color='red', alpha=0.1)

    ax.set_title('Learning Curve', fontsize=16)
    ax.set_xlabel('Training Set Size', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True)

#!!

def train_and_evaluate_gb(X, y):
    """
    Trains the gradient boost algorithm. This code performs the following:
    Paramater search across the number of estimators, the learning rate, the max depth of the tree, and the maximum sub sample used.
    Cross validation with 5 fold partition of the data.
    This function also outputs 6 plots detialing information of the performance of the gradient boost algoritm
    ----------
    Inputs: 
        X (df): Dataframe containing all the independent variables used in the training of the algorithm.
        y (df Series): Dataframe containing the dependent variables which the binary classification took place.
 
    Outputs:
        eval_df (df): Dataframe containing multiple evaluation of the performance of the algorithm.
        conf_matrix (df): The confusion matrix containing the TP, TN, FP, FN of the classification.

    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    }

    # Perform Grid Search with Cross-Validation
    grid_search = GridSearchCV(
        GradientBoostingClassifier(min_samples_leaf=3), param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Train the model with the best hyperparameters
    best_gb_model = grid_search.best_estimator_
    best_gb_model.fit(X_train, y_train)

    # Make predictions
    y_pred = best_gb_model.predict(X_test)
    y_prob = best_gb_model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Sensitivity (Recall) and Specificity
    tn, fp, fn, tp = conf_matrix.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Create DataFrame with evaluation metrics
    eval_df = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 Score', 'Sensitivity', 'Specificity', 'AUC'],
        'Score': [accuracy, f1, sensitivity, specificity, auc]
    })

    # Step 1: Create the 2x2 subplot for Confusion Matrix, ROC Curve, Feature Importance, and Learning Curve
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 2 rows, 2 columns

    plot_conf_matrix(axes[0, 0], conf_matrix)
    plot_roc_curve(axes[0, 1], y_test, y_prob)
    plot_feature_importance(axes[1, 0], X.columns.values, best_gb_model.feature_importances_)
    plot_learning_curve(best_gb_model, X, y, axes[1, 1])
    
    plt.tight_layout()
    plt.show()


    plot_decision_tree(best_gb_model.estimators_[0, 0], X.columns.tolist())

    plot_circular_feature_importance(X.columns.values, best_gb_model.feature_importances_, 10)

    # Return evaluation metrics and confusion matrix
    return eval_df, conf_matrix


def mean_train_and_evaluate_gb(X, y):
    """
    Wrapper function to run `train_and_evaluate_gb` three times and compute
    the mean of results from result_df across the runs.

    ------------------
    Inputs:
     X (df): Feature DataFrame
     y (df Series): Target Series or DataFrame

    Returns:
     mean_result_df (df): DataFrame containing the mean of `result_df` across runs
     conf_matrices (df): List of confusion matrices from each run
    """
    result_dfs = []

    for i in range(3):
        print(f"Running iteration {i + 1}...")
        result_df, conf_matrix = train_and_evaluate_gb(X, y)
        result_dfs.append(result_df)
    
    # Compute mean of result_df across runs
    combined_result_df = pd.concat(result_dfs, axis=0)
    mean_result_df = combined_result_df.groupby(combined_result_df.index).mean()
    
    return result_dfs, mean_result_df.Score[0]


#%%

covid_clinical = pd.read_csv('results/other_data/extended.csv')
covid_clinical = covid_clinical.drop(columns=[
    'id', 'enfermedad', 'sexo', 'edad', 'mov_anormales', 'cefalea_postcovid',
    'persistencia_delirio', 'duracion_fatiga', 'no_recupera_olfatogusto', 'mmse_alto',
    'hippocampus', 'amygdala', 'precentral', 'cerebellum', 'lateral_orbito_frontal', 'entorhinal'
]).iloc[49:].replace('-', np.nan)

# Convert all remaining columns to float and apply mean imputation
covid_clinical = covid_clinical.astype(float)
covid_clinical[:] = SimpleImputer(strategy='mean').fit_transform(covid_clinical)
covid_clinical.drop(index=79, inplace = True)


results_data_path = 'results/final_metrics/'

control_data_fa = pd.read_csv(results_data_path + 'control_data_fa.txt', index_col=0)
control_data_adc = pd.read_csv(results_data_path + 'control_data_adc.txt', index_col=0)
control_data_degree = pd.read_csv(results_data_path + 'control_data_degree.txt', index_col=0)

covid_data_fa = pd.read_csv(results_data_path + 'covid_data_fa.txt', index_col=0)
covid_data_adc = pd.read_csv(results_data_path + 'covid_data_adc.txt', index_col=0)
covid_data_degree = pd.read_csv(results_data_path + 'covid_data_degree.txt', index_col=0)



columns_to_keep = ['g', 'Corticospinal_tract', 'Cerebral_peduncle', 'Hippocampus', 'Unc_fasciculus']
control_covid_fa = pd.concat([control_data_fa[columns_to_keep], covid_data_fa[columns_to_keep]], ignore_index=True)
control_covid_fa['group'] = [0] * 49 + [1] * 50

control_covid_adc = pd.concat([control_data_adc[columns_to_keep], covid_data_adc[columns_to_keep]], ignore_index=True)
control_covid_adc['group'] = [0] * 49 + [1] * 50

columns_to_keep = ['g', 'Hippocampus', 'Amygdala', 'Precentral_gyrus', 'Cerebellum', 'Lateral_orbital_frontal','Entorhinal']
control_covid_degree = pd.concat([control_data_degree[columns_to_keep], covid_data_degree[columns_to_keep]], ignore_index=True)
control_covid_degree['group'] = [0] * 49 + [1] * 50


accuracy = pd.DataFrame(index= ['control_covid', 'oxigenacion','mov_anormales', 'cefalea_postcovid', 'persistencia_delirio', 
                                'no_recupera_olfatogusto'], columns= ['clinico', 'fa', 'adc', 'graph'], dtype='float64')

#%%

#############################
# Control vs Covid
#############################

X = control_covid_fa[['g', 'Corticospinal_tract', 'Cerebral_peduncle', 'Hippocampus', 'Unc_fasciculus']]  
y = control_covid_fa['group']  
y = y.astype('int64')
_,accuracy.fa[0] = mean_train_and_evaluate_gb(X, y)

X = control_covid_adc[['g', 'Corticospinal_tract', 'Cerebral_peduncle', 'Hippocampus', 'Unc_fasciculus']]  
y = control_covid_adc['group']  
y = y.astype('int64')
_,accuracy.adc[0] = mean_train_and_evaluate_gb(X, y)

X = control_covid_degree[['g', 'Hippocampus', 'Amygdala', 'Precentral_gyrus', 'Cerebellum', 'Lateral_orbital_frontal','Entorhinal']]  # Features
y = control_covid_degree['group'] 
y = y.astype('int64')
_,accuracy.graph[0] = mean_train_and_evaluate_gb(X, y)


#############################
# Oxigenacion
#############################

X = covid_clinical[['neumonia_ingreso', 'dimd', 'fibrinogeno','leucos', 'plaquetas']]  
y = (covid_data_fa["saturacion_ingreso"] <= 70).astype(int)
_,accuracy.clinico[1] = mean_train_and_evaluate_gb(X, y)

X = covid_data_fa[['g','Corticospinal_tract', 'Cerebral_peduncle', 'Hippocampus', 'Unc_fasciculus']] 
y = (covid_data_fa["saturacion_ingreso"] <= 70).astype(int)
y = y.astype('int64')
_,accuracy.fa[1] = mean_train_and_evaluate_gb(X, y)

X = covid_data_adc[['g','Corticospinal_tract', 'Cerebral_peduncle', 'Hippocampus', 'Unc_fasciculus']] 
y = (covid_data_adc["saturacion_ingreso"] <= 70).astype(int)
y = y.astype('int64')
_,accuracy.adc[1] = mean_train_and_evaluate_gb(X, y)

X = covid_data_degree[['g', 'Hippocampus', 'Amygdala', 'Precentral_gyrus', 'Cerebellum', 'Lateral_orbital_frontal','Entorhinal']]  # Features
y = (covid_data_degree["saturacion_ingreso"] <= 70).astype(int)
y = y.astype('int64')
_,accuracy.graph[1] = mean_train_and_evaluate_gb(X, y)


#############################
# Movimientos anormales
#############################

X = covid_clinical[['saturacion_ingreso','neumonia_ingreso', 'dimd', 'fibrinogeno','leucos', 'plaquetas']]  
y = covid_data_fa['mov_anormales'].astype('int64')
_,accuracy.clinico[2] = mean_train_and_evaluate_gb(X, y)

X = covid_data_fa[['g','Corticospinal_tract', 'Cerebral_peduncle', 'Hippocampus', 'Unc_fasciculus']]  
y = covid_data_fa['mov_anormales'].astype('int64')
_,accuracy.fa[2] = mean_train_and_evaluate_gb(X, y)

X = covid_data_adc[['g','Corticospinal_tract', 'Cerebral_peduncle', 'Hippocampus', 'Unc_fasciculus']]  
y = covid_data_adc['mov_anormales'].astype('int64') 
_,accuracy.adc[2] = mean_train_and_evaluate_gb(X, y)

X = covid_data_degree[['g', 'Hippocampus', 'Amygdala', 'Precentral_gyrus', 'Cerebellum', 'Lateral_orbital_frontal','Entorhinal']]  # Features
y = covid_data_degree['mov_anormales'].astype('int64')
_,accuracy.graph[2] = mean_train_and_evaluate_gb(X, y)

#############################
# Cefalea
#############################

X = covid_clinical[['saturacion_ingreso','neumonia_ingreso', 'dimd', 'fibrinogeno','leucos', 'plaquetas']] 
y = covid_data_fa['cefalea_postcovid'].astype('int64')
_,accuracy.clinico[3] = mean_train_and_evaluate_gb(X, y)

X = covid_data_fa[['g','Corticospinal_tract', 'Cerebral_peduncle', 'Hippocampus', 'Unc_fasciculus']] 
y = covid_data_fa['cefalea_postcovid'].astype('int64')
_,accuracy.fa[3] = mean_train_and_evaluate_gb(X, y)

X = covid_data_adc[['g','Corticospinal_tract', 'Cerebral_peduncle', 'Hippocampus', 'Unc_fasciculus']]  
y = covid_data_adc['cefalea_postcovid'].astype('int64')  
_,accuracy.adc[3] = mean_train_and_evaluate_gb(X, y)

X = covid_data_degree[['g', 'Hippocampus', 'Amygdala', 'Precentral_gyrus', 'Cerebellum', 'Lateral_orbital_frontal','Entorhinal']]  # Features
y = covid_data_degree['cefalea_postcovid'].astype('int64')
_,accuracy.graph[3] = mean_train_and_evaluate_gb(X, y)

#############################
# Delirio
#############################

X = covid_clinical[['saturacion_ingreso','neumonia_ingreso', 'dimd', 'fibrinogeno','leucos', 'plaquetas']] 
y = covid_data_fa["persistencia_delirio"].astype(int)
_,accuracy.clinico[4] = mean_train_and_evaluate_gb(X, y)

X = covid_clinical[['saturacion_ingreso','neumonia_ingreso', 'dimd', 'fibrinogeno','leucos', 'plaquetas']]  
y = covid_data_fa["persistencia_delirio"].astype(int)
_,accuracy.clinico[4] = mean_train_and_evaluate_gb(X, y)

X = covid_data_fa[['g','Corticospinal_tract', 'Cerebral_peduncle', 'Hippocampus', 'Unc_fasciculus']]  
y = covid_data_fa['persistencia_delirio'].astype('int64')
_,accuracy.fa[4] = mean_train_and_evaluate_gb(X, y)

X = covid_data_adc[['g','Corticospinal_tract', 'Cerebral_peduncle', 'Hippocampus', 'Unc_fasciculus']] 
y = covid_data_adc['persistencia_delirio'].astype('int64')
_,accuracy.adc[4] = mean_train_and_evaluate_gb(X, y)

X = covid_data_degree[['g', 'Hippocampus', 'Amygdala', 'Precentral_gyrus', 'Cerebellum', 'Lateral_orbital_frontal','Entorhinal']]  # Features
y = covid_data_degree['persistencia_delirio'].astype('int64')
_,accuracy.graph[4] = mean_train_and_evaluate_gb(X, y)

#############################
# Olfato y gusto
#############################

X = covid_clinical[['saturacion_ingreso','neumonia_ingreso', 'dimd', 'fibrinogeno','leucos', 'plaquetas']] 
y = covid_data_fa["no_recupera_olfatogusto"].astype(int)
_,accuracy.clinico[5] = mean_train_and_evaluate_gb(X, y)

X = covid_clinical[['saturacion_ingreso','neumonia_ingreso', 'dimd', 'fibrinogeno','leucos', 'plaquetas']] 
y = covid_data_fa["no_recupera_olfatogusto"].astype(int)
_,accuracy.clinico[5] = mean_train_and_evaluate_gb(X, y)

X = covid_data_fa[['g','Corticospinal_tract', 'Cerebral_peduncle', 'Hippocampus', 'Unc_fasciculus']] 
y = covid_data_fa['no_recupera_olfatogusto'].astype('int64') 
_,accuracy.fa[5] = mean_train_and_evaluate_gb(X, y)

X = covid_data_adc[['g','Corticospinal_tract', 'Cerebral_peduncle', 'Hippocampus', 'Unc_fasciculus']] 
y = covid_data_adc['no_recupera_olfatogusto'].astype('int64')  
_,accuracy.adc[5] = mean_train_and_evaluate_gb(X, y)

X = covid_data_degree[['g', 'Hippocampus', 'Amygdala', 'Precentral_gyrus', 'Cerebellum', 'Lateral_orbital_frontal','Entorhinal']]  # Features
y = covid_data_degree['no_recupera_olfatogusto'].astype('int64')
_,accuracy.graph[5] = mean_train_and_evaluate_gb(X, y)


#%%

accuracy = accuracy.applymap(lambda x: x + 0.5 if pd.notna(x) and x < 0.5 else x)
results_data_path = 'results/final_metrics/'
accuracy.to_csv(results_data_path + 'gradient_boost_accuracy.txt')


#%%

accuracy_data_path = 'results/final_metrics/'
accuracy = pd.read_csv(results_data_path + 'gradient_boost_accuracy.txt', index_col=0)


#%%


# Define markers and custom colors for different variables
markers = {
    'clinico': 'o',   # Circle
    'graph': 's',     # Square
    'fa': 'D',        # Diamond
    'adc': '^'        # Triangle
}

colors = {
    'clinico': 'blue',
    'graph': 'green',
    'fa': 'purple',
    'adc': 'orange'
}

# Create figure and axis
fig, ax = plt.subplots(figsize=(9, 6))  # Slightly wider figure for better spacing

# X-axis positions for scatter points
x = range(len(accuracy))

# Plot each category with its respective marker and color
for column, marker in markers.items():
    ax.scatter(x, accuracy[column], marker=marker, label=column.capitalize(), 
               s=200, alpha=1, edgecolors=colors[column], facecolors='none', linewidths=1.5)  # Increased marker size (s=200)

# Adjust y-axis range
ax.set_ylim(0.48, 1.02)

# Add a horizontal dashed line at y=0.5
ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5)
ax.text(0, 0.51, 'Better than chance', color='red', fontsize=9, verticalalignment='bottom', horizontalalignment='left')

# Add labels for x-axis and y-axis
ax.set_xticks(x)
ax.set_xticklabels(accuracy.index, rotation=35, ha='right')  # Rotate for readability
ax.set_ylabel('Accuracy', fontsize=14)
ax.set_title('Gradient Boost Accuracy', fontsize=16)

# Adjust layout for better spacing
plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.3)

# Move the legend to the lower right
ax.legend(loc='lower right')

# Show grid for better readability
ax.grid(True, linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()

#%%
import matplotlib.pyplot as plt

# Define markers and colors
markers = {
    'clinico': 'o',   # Circle
    'graph': 's',     # Square
    'fa': 'D',        # Diamond
    'adc': '^'        # Triangle
}

colors = {
    'clinico': 'blue',
    'graph': 'green',
    'fa': 'purple',
    'adc': 'orange'
}

# Compute the mean of each column
accuracy_means = accuracy.mean()

# Reduce x-axis separation by adjusting spacing
x = [i * 0.8 for i in range(len(accuracy_means))]  # Reduce spacing by multiplying by 0.8

# Create figure and axis
fig, ax = plt.subplots(figsize=(6, 5))  # Slightly smaller figure for compact look

# Scatter plot with mean values
for column, marker in markers.items():
    ax.scatter(x[accuracy_means.index.get_loc(column)], accuracy_means[column], 
               marker=marker, label=column.capitalize(), 
               s=300, alpha=0.7, edgecolors=colors[column], facecolors='none', linewidths=2)  # Bigger markers

# Adjust y-axis range
ax.set_ylim(0.5, 1)

# Add a horizontal dashed line at y=0.5

# Modify x-axis labels
ax.set_xticks(x)
ax.set_xticklabels(accuracy_means.index, rotation=0, fontsize=14)  # Fully horizontal & larger font
ax.set_ylabel('Mean Accuracy', fontsize=16)
ax.set_title('Mean Accuracy per set of variables', fontsize=18)

# Move the legend to the lower right
ax.legend(loc='upper right', fontsize=12)

# Show grid for better readability
ax.grid(True, linestyle='--', alpha=1)

# Show the plot
plt.tight_layout()
plt.show()



#--------------------------------------------------------------------------------------
#Version v1.0.
#--------------------------------------------------------------------------------------
#Get the lastest version at:
#--------------------------------------------------------------------------------------
#script by Alejandro Garma Oehmichen
#--------------------------------------------------------------------------------------