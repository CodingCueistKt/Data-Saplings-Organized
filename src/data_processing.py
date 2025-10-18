import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

def load_data(features_path, barcode_path):
    """
    Loads the feature and barcode data from CSV files.
    
    Args:
        features_path (str): File path to the side_features.csv file.
        barcode_path (str): File path to the barcodes.csv file.
    
    Returns:
        tuple: A tuple containing two DataFrames (data_df, barcode_df).
    """
    try:
        data_df = pd.read_csv(features_path)
        barcode_df = pd.read_csv(barcode_path)
        return data_df, barcode_df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None

def preprocess_data(data_df, barcode_df, experiment_start_str='2024-05-26 00:00:00'):
    """
    Merges and preprocesses the dataframes.
    - Converts 'Analyse Date' to datetime objects.
    - Calculates 'Decimal Days' from the experiment start.
    - Calculates 'Days_Since_2024_05_26' as an integer day number.
    - Merges the two dataframes on plant identifiers.
    
    Args:
        data_df (pd.DataFrame): The main features dataframe.
        barcode_df (pd.DataFrame): The barcode dataframe.
        experiment_start_str (str): The timestamp of the experiment's start.

    Returns:
        pd.DataFrame: A merged and preprocessed dataframe.
    """
    # Convert Analyse Date to datetime
    data_df["Analyse Date"] = pd.to_datetime(data_df["Analyse Date"], errors="coerce")
    
    # Define experiment start time
    experiment_start = pd.Timestamp(experiment_start_str)
    
    # Compute decimal days from experiment start
    data_df['Decimal Days'] = (data_df['Analyse Date'] - experiment_start).dt.total_seconds() / 86400  # 86400 seconds in a day
    
    # Merge dataframes
    merged_df = data_df.merge(barcode_df, left_on="Plant Info", right_on="Plant.Info", how="inner")
    
    # Add integer day column for aggregation and per-plant fitting
    merged_df["Date"] = pd.to_datetime(merged_df["Analyse Date"], errors="coerce").dt.date
    reference_date = pd.to_datetime(experiment_start_str).date()
    merged_df["Date"] = pd.to_datetime(merged_df["Date"], errors="coerce")
    reference_date_dt = pd.to_datetime(reference_date)
    
    merged_df["Days_Since_2024_05_26"] = (merged_df['Date'] - reference_date_dt).dt.days
    
    return merged_df

def get_daily_means(merged_df):
    """
    Calculates the daily mean of all numeric features for global model fitting.
    
    Args:
        merged_df (pd.DataFrame): The preprocessed, merged dataframe.
        
    Returns:
        pd.DataFrame: A dataframe of daily means, with 'Days_Since_2024_05_26'.
    """
    # Select only numeric columns for aggregation
    numeric_cols = merged_df.select_dtypes(include=np.number).columns
    
    # Exclude id-like or date-like columns that shouldn't be averaged
    exclude_from_mean = ['Round Order', 'Decimal Days', 'Days_Since_2024_05_26']
    cols_to_mean = [col for col in numeric_cols if col not in exclude_from_mean]
    
    # Group by date and calculate mean
    daily_means_df = merged_df.groupby('Date')[cols_to_mean].mean().reset_index()
    
    # Add back the integer day column
    reference_date = pd.to_datetime("2024-05-26")
    daily_means_df['Date'] = pd.to_datetime(daily_means_df['Date'])
    daily_means_df['Days_Since_2024_05_26'] = (daily_means_df['Date'] - reference_date).dt.days
    
    return daily_means_df

def apply_scaling(df, scaler_name="MinMaxScaler", exclude_cols=None):
    """
    Applies a specified scaler to the dataframe.
    
    Args:
        df (pd.DataFrame): The dataframe to scale (e.g., daily means or plant data).
        scaler_name (str): 'MinMaxScaler', 'StandardScaler', 'RobustScaler', or None.
        exclude_cols (list): List of columns to exclude from scaling.
        
    Returns:
        tuple: (scaled_df, columns_to_scale, scaler_obj)
    """
    if scaler_name is None or scaler_name.lower() == 'unscaled':
        return df, [], None

    if exclude_cols is None:
        exclude_cols = ['Days_Since_2024_05_26', 'Date', 'Analyse Date', 'Plant Info', 
                          'File Path', 'Decimal Days', 'Plant.Genotype',
                          'Replication', 'Geno_Rep', 'Random', 'Plant.Info']

    columns_to_scale = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.int64, np.float64]]
    
    scalers = {
        'StandardScaler': StandardScaler(),
        'RobustScaler': RobustScaler(),
        'MinMaxScaler': MinMaxScaler()
    }
    
    if scaler_name not in scalers:
        raise ValueError(f"Scaler '{scaler_name}' not recognized. Choose from {list(scalers.keys())} or None.")
        
    scaler_obj = scalers[scaler_name]
    scaled_df = df.copy()
    
    if not columns_to_scale:
        return scaled_df, columns_to_scale, None
        
    scaled_data = scaler_obj.fit_transform(df[columns_to_scale])
    scaled_df[columns_to_scale] = scaled_data
    
    return scaled_df, columns_to_scale, scaler_obj

def calculate_deltas(df):
    """
    Calculates the change in metrics between consecutive rounds for each plant.
    """
    df_sorted = df.sort_values(by=['Plant Info', 'Days_Since_2024_05_26'])
    
    # Calculate the difference from the previous row within each plant group
    delta_cols = ['area', 'height', 'blue_yellow_mean', 'green_red_mean']
    for col in delta_cols:
        df_sorted[f'delta_{col}'] = df_sorted.groupby('Plant Info')[col].diff()
        
    return df_sorted.dropna(subset=[f'delta_{col}' for col in delta_cols])