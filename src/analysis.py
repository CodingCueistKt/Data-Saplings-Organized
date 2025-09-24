import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import f_oneway

def r2_adjusted(r2, n, m):
    """
    Calculates the adjusted R-squared value.
    
    Args:
        r2 (float): The R-squared value.
        n (int): The number of observations.
        m (int): The number of parameters (predictors).
        
    Returns:
        float: The adjusted R-squared value.
    """
    if (n - (m + 1)) == 0:
        return np.nan  # Avoid division by zero
    r2_adj = 1 - (1 - r2) * ((n - 1) / (n - (m + 1)))
    return r2_adj

# (In src/analysis.py)

def calculate_aic_weights(results_df):
    """
    Computes AIC weights for the fitted models.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing AIC values for different models.
    
    Returns:
        pd.DataFrame: Updated DataFrame with AIC weights added as new columns.
    """
    aic_cols = ["AIC_Linear", "AIC_Ricker", "AIC_Logistic", "AIC_Exponential", "AIC_Gen_VB"]
    delta_cols = [f"Delta_{col}" for col in aic_cols]
    
    weight_cols = [f"{col.replace('AIC_', '')}_AIC_Weight" for col in aic_cols]

    # Identify the minimum AIC value across the models for each row
    min_aic = results_df[aic_cols].min(axis=1)
    
    # Compute Delta_i(AIC) values
    for i, col in enumerate(aic_cols):
        results_df[delta_cols[i]] = results_df[col] - min_aic
    
    # Compute exp(-0.5 * Delta_i(AIC)) for each model
    exp_values = np.exp(-0.5 * results_df[delta_cols])
    
    # Compute the sum of these exponentials
    sum_exp = exp_values.sum(axis=1)
    
    # Compute AIC weights
    for i, col in enumerate(weight_cols):
        results_df[col] = exp_values[delta_cols[i]] / sum_exp

    # Drop intermediate Delta AIC columns
    results_df.drop(columns=delta_cols, inplace=True)

    return results_df

def calculate_heritability(results_df, trait_cols, trait_labels):
    """
    Performs ANOVA and calculates broad-sense heritability (H²) for given traits.
    
    Args:
        results_df (pd.DataFrame): DataFrame with plant genotypes and optimized trait parameters.
        trait_cols (list): List of column names in results_df to analyze as traits.
        trait_labels (dict): Dictionary mapping trait_cols to human-readable names.
        
    Returns:
        pd.DataFrame: A summary table of heritability calculations.
    """
    table_rows = []

    # Group by genotype
    grouped_by_genotype = results_df.groupby('Plant Genotype')

    for trait in trait_cols:
        # Prepare data for ANOVA: list of arrays, one for each genotype
        anova_data = [group[trait].dropna() for name, group in grouped_by_genotype]
        
        # Filter out empty groups (if any)
        anova_data = [group for group in anova_data if not group.empty]
        
        if len(anova_data) < 2:
            continue
            
        # Perform one-way ANOVA
        f_val, p_val = f_oneway(*anova_data)
        
        # Calculate sums of squares
        k = len(anova_data)  # Number of groups (genotypes)
        n_total = sum(len(group) for group in anova_data)
        
        grand_mean = results_df[trait].dropna().mean()
        
        ss_genotype = sum(len(group) * (group.mean() - grand_mean)**2 for group in anova_data)
        ss_error = sum(sum((x - group.mean())**2 for x in group) for group in anova_data)
        
        # Degrees of freedom
        df_genotype = k - 1
        df_error = n_total - k
        
        # Mean squares
        ms_genotype = ss_genotype / df_genotype if df_genotype > 0 else 0
        ms_error = ss_error / df_error if df_error > 0 else 0
        
        # Calculate n_r (harmonic mean of replications per genotype)
        n_r = k / sum(1.0 / len(group) for group in anova_data)
        
        # Variance components
        var_G = (ms_genotype - ms_error) / n_r if n_r != 0 else 0
        var_E = ms_error
        var_P = var_G + var_E
        
        # Ensure variances are non-negative
        var_G = max(0, var_G)
        var_P = max(0, var_P)

        # Heritability
        H2 = var_G / var_P if var_P > 0 else 0
        
        table_rows.append({
            'Trait': trait_labels.get(trait, trait),
            'MS Genotype': round(ms_genotype, 4),
            'MS Error': round(ms_error, 4),
            'n_r': round(n_r, 2),
            'Var(G)': round(var_G, 4),
            'Var(E)': round(var_E, 4),
            'Var(P)': round(var_P, 4),
            'H²': round(H2, 3)
        })

    # Create and return the summary DataFrame
    summary_df = pd.DataFrame(table_rows)
    return summary_df