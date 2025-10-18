# (In src/visualization.py)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler # <-- Make sure this import is here
from .models import ricker_model, logistic_fun, exponential, gen_vb, linear, hill

def plot_growth_models(plant_id, dat, results_df):
    """
    Plots the observed data for a single plant against all fitted growth models.
    Performs local scaling on the data for accurate visualization.
    Includes AIC weights in the legend.
    
    Args:
        plant_id (str): The unique identifier for the plant.
        dat (pd.DataFrame): The full, UN-SCALED dataframe.
        results_df (pd.DataFrame): The dataframe with optimized parameters.
    """
    # Select the specific plant's data and use .copy() to avoid SettingWithCopyWarning
    plant_data = dat.loc[dat['Plant Info'] == plant_id].copy()
    
    # This ensures the plotted data is scaled from 0-1, exactly like the
    # data that was used during the model fitting loop.
    scaler = MinMaxScaler()
    plant_data['area_scaled_for_plot'] = scaler.fit_transform(plant_data[['area']])

    dd = plant_data["Days_Since_2024_05_26"].to_numpy()
    area_to_plot = plant_data["area_scaled_for_plot"].to_numpy()
    
    plant_params = results_df.loc[results_df['Plant Info'] == plant_id]
    
    if plant_params.empty:
        print(f"No optimized parameters found for plant {plant_id}.")
        return
    
    # Extracting model parameters
    W0_optimal, kg_optimal, m_optimal = plant_params[['W0_optimal_ricker', 'kg_optimal_ricker', 'm_optimal_ricker']].values[0]
    P0_optimal, r_optimal, K_optimal = plant_params[['P0_optimal_log', 'r_optimal_log', 'K_optimal_log']].values[0]
    m0_optimal_exp, k_optimal_exp = plant_params[['m0_optimal_exp', 'k_optimal_exp']].values[0]
    m0_optimal_gvb, k_optimal_gvb, f_optimal_gvb, A_optimal_gvb = plant_params[['m0_optimal_gvb', 'k_optimal_gvb', 'f_optimal_gvb', 'A_optimal_gvb']].values[0]
    m0_optimal_linear, k_optimal_linear = plant_params[['m0_optimal_linear', 'k_optimal_linear']].values[0]
    
    # Extracting AIC weights
    w_Ricker = plant_params['Ricker_AIC_Weight'].values[0]
    w_Logistic = plant_params['Logistic_AIC_Weight'].values[0]
    w_Exponential = plant_params['Exponential_AIC_Weight'].values[0]
    w_Gen_VB = plant_params['Gen_VB_AIC_Weight'].values[0]
    w_Linear = plant_params['Linear_AIC_Weight'].values[0]
    
    # Generate fitted model curves
    tvect = np.linspace(min(dd), max(dd), 100)
    
    model_ricker = ricker_model(tvect, [W0_optimal, kg_optimal, m_optimal])
    model_logistic = logistic_fun(tvect, [P0_optimal, r_optimal, K_optimal])
    model_exponential = exponential(tvect, [m0_optimal_exp, k_optimal_exp])
    model_gvb = gen_vb(tvect, [m0_optimal_gvb, k_optimal_gvb, f_optimal_gvb, A_optimal_gvb])
    model_linear = linear(tvect, [m0_optimal_linear, k_optimal_linear])

    # Plot data and fitted models
    plt.figure(figsize=(10, 6))
    plt.scatter(dd, area_to_plot, c='k', marker='o', label='Observed Data (Scaled 0-1)')
    
    plt.plot(tvect, model_linear, c='c', linestyle='--', label=f'Linear (W: {w_Linear:.2f})')
    plt.plot(tvect, model_ricker, c='b', label=f'Ricker (W: {w_Ricker:.2f})')
    plt.plot(tvect, model_logistic, c='r', label=f'Logistic (W: {w_Logistic:.2f})')
    plt.plot(tvect, model_exponential, c='g', label=f'Exponential (W: {w_Exponential:.2f})')
    plt.plot(tvect, model_gvb, c='y', label=f'Gen. VB (W: {w_Gen_VB:.2f})')
    
    plt.grid()
    plt.xlabel('Days')
    plt.ylabel('Scaled Area')
    plt.title(f'Growth Models - Plant {plant_id}')
    plt.legend(loc='best')
    plt.show()

def plot_global_models(daily_means_df, global_params):
    """
    Plots the aggregated daily mean data against all globally fitted models.
    
    Args:
        daily_means_df (pd.DataFrame): The dataframe of daily means.
        global_params (dict): A dictionary containing the optimized parameters 
                              for each global model. 
                              e.g., {'linear': [m0, k], 'logistic': [P0, r, K], ...}
    """
    dd = daily_means_df["Days_Since_2024_05_26"].to_numpy()
    area = daily_means_df["area"].to_numpy()
    
    tvect = np.linspace(0, max(dd) if len(dd) > 0 else 35, 100)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(dd, area, c='k', marker='o', label='Daily Mean Area')

    if 'linear' in global_params:
        model_linear = linear(tvect, global_params['linear'])
        plt.plot(tvect, model_linear, label='Global Linear')
        
    if 'logistic' in global_params:
        model_logistic = logistic_fun(tvect, global_params['logistic'])
        plt.plot(tvect, model_logistic, label='Global Logistic')
        
    if 'ricker' in global_params:
        model_ricker = ricker_model(tvect, global_params['ricker'])
        plt.plot(tvect, model_ricker, label='Global Ricker')

    if 'exponential' in global_params:
        model_exp = exponential(tvect, global_params['exponential'])
        plt.plot(tvect, model_exp, label='Global Exponential')

    if 'gen_vb' in global_params:
        model_gvb = gen_vb(tvect, global_params['gen_vb'])
        plt.plot(tvect, model_gvb, label='Global Gen. VB')
        
    if 'hill' in global_params:
        model_hill = hill(tvect, global_params['hill'])
        plt.plot(tvect, model_hill, label='Global Hill')

    plt.grid()
    plt.xlabel('Days')
    plt.ylabel('Scaled Area')
    plt.title('Global Growth Models - Daily Mean Area')
    plt.legend(loc='best')
    plt.show()

def plot_model_performance_curves(model_results_list):
    """
    Plots ROC and Precision-Recall curves for a list of model results.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    # ROC Curve Plot
    for i, results in enumerate(model_results_list):
        ax1.plot(results['fpr'], results['tpr'], color=colors[i], lw=2,
                 label=f'{results["feature_names"][0]} (AUC = {results["roc_auc"]:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True)
    
    # Precision-Recall Curve Plot
    for i, results in enumerate(model_results_list):
        ax2.plot(results['recall'], results['precision'], color=colors[i], lw=2,
                 label=f'{results["feature_names"][0]} (AUC = {results["pr_auc"]:.2f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    ax2.grid(True)
    
    plt.suptitle('Predicting High Vigor from Color Metrics', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()