# prepare_data.py
"""
Utility script to prepare and combine agricultural datasets for training.
This script helps integrate data from various sources into a consistent format.
"""

import os
import pandas as pd
import numpy as np
import glob
import argparse
import logging
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def find_csv_files(paths):
    """Find all CSV files in the given paths"""
    csv_files = []
    
    # Convert to list if a single path
    if isinstance(paths, str):
        paths = [paths]
    
    for path in paths:
        if os.path.isdir(path):
            # Walk through all subdirectories
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
        elif os.path.isfile(path) and path.endswith('.csv'):
            csv_files.append(path)
    
    return csv_files

def standardize_column_names(df):
    """Standardize column names across different datasets"""
    # Column name mappings
    column_mapping = {
        # Crop-related columns
        'crop': 'crop_label',
        'crop_type': 'crop_label',
        'crop_name': 'crop_label',
        'label': 'crop_label',
        
        # Fertilizer-related columns
        'fertilizer': 'fertilizer_label',
        'fert': 'fertilizer_label',
        'fertilizer_type': 'fertilizer_label',
        'fertilizer_name': 'fertilizer_label',
        'fertiliser': 'fertilizer_label',
        
        # NPK columns
        'n': 'feature_N',
        'p': 'feature_P',
        'k': 'feature_K',
        'n_content': 'feature_N',
        'p_content': 'feature_P',
        'k_content': 'feature_K',
        'nitrogen': 'feature_N',
        'phosphorus': 'feature_P',
        'potassium': 'feature_K',
        
        # pH column
        'ph': 'feature_pH',
        
        # Environmental columns
        'temperature': 'feature_temperature',
        'humidity': 'feature_humidity',
        'rainfall': 'feature_rainfall',
        'precipitation': 'feature_rainfall',
        
        # Soil columns
        'soil_moisture': 'feature_soil_moisture',
        'soil_type': 'feature_soil_type',
        'texture': 'feature_texture',
    }
    
    # Rename columns if they exist
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    # Convert other columns to feature format
    for col in df.columns:
        if col not in ['task_id', 'crop_label', 'fertilizer_label', 'climate_effect_label', 'yield_value'] and not col.startswith('feature_'):
            # Skip if it's already been handled
            if col in column_mapping.values():
                continue
                
            # Otherwise convert to feature format
            df = df.rename(columns={col: f'feature_{col}'})
    
    return df

def handle_categorical_features(df):
    """Convert categorical features to numeric"""
    result_df = df.copy()
    
    # Get all feature columns
    feature_cols = [col for col in result_df.columns if col.startswith('feature_')]
    
    # Handle each feature column
    for col in feature_cols:
        # If column is categorical
        if result_df[col].dtype == 'object':
            # Create a mapping for unique values
            unique_values = result_df[col].dropna().unique()
            value_map = {val: i for i, val in enumerate(unique_values)}
            
            # Apply mapping
            result_df[col] = result_df[col].map(value_map)
            
            logging.info(f"Converted categorical feature {col} with {len(unique_values)} unique values")
    
    return result_df

def standardize_categorical_labels(df, common_crops=None, common_fertilizers=None):
    """
    Standardize crop and fertilizer labels to common names.
    Helps merge datasets with slightly different naming conventions.
    """
    result_df = df.copy()
    
    # Default common names if not provided
    if common_crops is None:
        common_crops = {
            'rice': 'Rice',
            'paddy': 'Rice',
            'wheat': 'Wheat',
            'corn': 'Maize',
            'maize': 'Maize',
            'soyabean': 'Soybean',
            'soybean': 'Soybean',
            'cotton': 'Cotton',
            'sugarcane': 'Sugarcane',
            'potato': 'Potato',
            'tomato': 'Tomato',
            'onion': 'Onion',
            'mustard': 'Mustard',
            'barley': 'Barley',
            'pulses': 'Pulses',
            'gram': 'Chickpea',
            'chickpea': 'Chickpea',
            'lentil': 'Lentil'
        }
    
    if common_fertilizers is None:
        common_fertilizers = {
            'urea': 'Urea',
            'dap': 'DAP',
            'di-ammonium phosphate': 'DAP',
            'diammonium phosphate': 'DAP',
            'npk': 'NPK',
            'mop': 'MOP',
            'muriate of potash': 'MOP',
            'ammonium sulfate': 'Ammonium Sulfate',
            'ammonium sulphate': 'Ammonium Sulfate',
            'super phosphate': 'Super Phosphate',
            'potash': 'MOP',
            'calcium nitrate': 'Calcium Nitrate',
            'zinc sulfate': 'Micronutrient',
            'boron': 'Micronutrient',
            'organic': 'Organic Compost',
            'compost': 'Organic Compost',
            'manure': 'Organic Compost'
        }
    
    # Standardize crop labels
    if 'crop_label' in result_df.columns:
        result_df['crop_label'] = result_df['crop_label'].apply(
            lambda x: next(
                (common_crops[k] for k in common_crops if k in str(x).lower()),
                x
            )
        )
    
    # Standardize fertilizer labels
    if 'fertilizer_label' in result_df.columns:
        result_df['fertilizer_label'] = result_df['fertilizer_label'].apply(
            lambda x: next(
                (common_fertilizers[k] for k in common_fertilizers if k in str(x).lower()),
                x
            )
        )
    
    return result_df

def fill_missing_values(df):
    """Fill missing values in the dataframe"""
    result_df = df.copy()
    
    # Get feature columns
    feature_cols = [col for col in result_df.columns if col.startswith('feature_')]
    
    # Fill missing values in feature columns
    for col in feature_cols:
        # For numerical features, use median
        if result_df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            median_val = result_df[col].median()
            result_df[col] = result_df[col].fillna(median_val)
        # For categorical features, use most common value
        else:
            mode_val = result_df[col].mode().iloc[0] if not result_df[col].mode().empty else 0
            result_df[col] = result_df[col].fillna(mode_val)
    
    # Ensure required columns exist
    if 'crop_label' not in result_df.columns:
        result_df['crop_label'] = 'Unknown'
        
    if 'fertilizer_label' not in result_df.columns:
        result_df['fertilizer_label'] = 'Unknown'
        
    if 'task_id' not in result_df.columns:
        # Infer task_id based on columns
        if 'crop_label' in result_df.columns and result_df['crop_label'].nunique() > 1:
            result_df['task_id'] = 0  # Crop recommendation
        elif 'fertilizer_label' in result_df.columns and result_df['fertilizer_label'].nunique() > 1:
            result_df['task_id'] = 1  # Fertilizer recommendation
        else:
            result_df['task_id'] = 0  # Default
    
    return result_df

def infer_task_id_from_path(file_path):
    """Infer task ID from file path"""
    file_path = file_path.lower()
    
    if any(term in file_path for term in ['crop_recommendation', 'crop_recommend']):
        return 0  # Crop recommendation
    elif any(term in file_path for term in ['fertilizer', 'fert']):
        return 1  # Fertilizer recommendation
    elif any(term in file_path for term in ['climate', 'weather']):
        return 2  # Climate effect prediction
    elif any(term in file_path for term in ['yield', 'production']):
        return 3  # Yield prediction
    
    # Default to crop recommendation
    return 0

def process_file(file_path, encoding='utf-8', infer_task=True):
    """Process a single CSV file"""
    try:
        # Try to read with specified encoding
        df = pd.read_csv(file_path, encoding=encoding)
    except UnicodeDecodeError:
        # If that fails, try with a different encoding
        try:
            df = pd.read_csv(file_path, encoding='latin1')
        except Exception as e:
            logging.error(f"Failed to read {file_path}: {e}")
            return None
    
    # Apply preprocessing steps
    df = standardize_column_names(df)
    df = standardize_categorical_labels(df)
    df = handle_categorical_features(df)
    df = fill_missing_values(df)
    
    # Infer task_id if requested and not already present
    if infer_task and ('task_id' not in df.columns or df['task_id'].nunique() <= 1):
        task_id = infer_task_id_from_path(file_path)
        df['task_id'] = task_id
        logging.info(f"Inferred task_id={task_id} for {file_path}")
    
    return df

def combine_datasets(input_paths, output_path, encoding='utf-8', infer_task=True):
    """
    Combine multiple CSV datasets into a single standardized dataset for training.
    
    Args:
        input_paths (list): List of input paths (files or directories)
        output_path (str): Path to save the combined dataset
        encoding (str): Character encoding for CSV files
        infer_task (bool): Whether to infer task_id from file paths
    
    Returns:
        pd.DataFrame: The combined dataset
    """
    # Find all CSV files
    csv_files = find_csv_files(input_paths)
    logging.info(f"Found {len(csv_files)} CSV files")
    
    if not csv_files:
        logging.error("No CSV files found in the specified paths")
        return None
    
    # Process each file
    processed_dfs = []
    
    for file_path in tqdm(csv_files, desc="Processing files"):
        try:
            df = process_file(file_path, encoding=encoding, infer_task=infer_task)
            if df is not None and len(df) > 0:
                processed_dfs.append(df)
                logging.info(f"Processed {file_path}: {len(df)} rows")
            else:
                logging.warning(f"Skipping {file_path}: No valid data")
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
    
    if not processed_dfs:
        logging.error("No valid data found in any of the input files")
        return None
    
    # Combine all processed dataframes
    combined_df = pd.concat(processed_dfs, ignore_index=True)
    logging.info(f"Combined dataset has {len(combined_df)} rows")
    
    # Ensure all columns are the same type across all datasets
    for col in combined_df.columns:
        if combined_df[col].dtype == 'object':
            # For object columns, check if they're mostly numeric
            try:
                if combined_df[col].str.isnumeric().mean() > 0.5:
                    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
            except:
                pass
    
    # Handle duplicates
    combined_df = combined_df.drop_duplicates()
    logging.info(f"After removing duplicates: {len(combined_df)} rows")
    
    # Ensure feature columns are numerical
    feature_cols = [col for col in combined_df.columns if col.startswith('feature_')]
    for col in feature_cols:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        # Fill any remaining NaN values with column median
        combined_df[col] = combined_df[col].fillna(combined_df[col].median())
    
    # Save the combined dataset
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_df.to_csv(output_path, index=False)
        logging.info(f"Combined dataset saved to {output_path}")
    
    return combined_df

def analyze_dataset(df):
    """
    Analyze a dataset and print statistics.
    
    Args:
        df (pd.DataFrame): The dataset to analyze
    """
    print("\n===== Dataset Analysis =====")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Task distribution
    if 'task_id' in df.columns:
        task_counts = df['task_id'].value_counts()
        print("\nTask distribution:")
        for task_id, count in task_counts.items():
            task_name = "Crop recommendation" if task_id == 0 else \
                       "Fertilizer recommendation" if task_id == 1 else \
                       "Climate effect prediction" if task_id == 2 else \
                       "Yield prediction" if task_id == 3 else \
                       f"Task {task_id}"
            print(f"  {task_name}: {count} rows ({count/len(df)*100:.1f}%)")
    
    # Label distributions
    if 'crop_label' in df.columns:
        crop_counts = df['crop_label'].value_counts()
        print("\nTop crops:")
        for crop, count in crop_counts.head(10).items():
            print(f"  {crop}: {count} rows ({count/len(df)*100:.1f}%)")
        print(f"  Total unique crops: {df['crop_label'].nunique()}")
    
    if 'fertilizer_label' in df.columns:
        fert_counts = df['fertilizer_label'].value_counts()
        print("\nTop fertilizers:")
        for fert, count in fert_counts.head(10).items():
            print(f"  {fert}: {count} rows ({count/len(df)*100:.1f}%)")
        print(f"  Total unique fertilizers: {df['fertilizer_label'].nunique()}")
    
    # Feature statistics
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    if feature_cols:
        print("\nFeature statistics:")
        for col in feature_cols:
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                print(f"  {col}:")
                print(f"    Min: {df[col].min()}, Max: {df[col].max()}, Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}")
            else:
                print(f"  {col}: {df[col].nunique()} unique values (categorical)")

def main():
    parser = argparse.ArgumentParser(description="Prepare and combine agricultural datasets")
    parser.add_argument("--input", type=str, nargs='+', required=True, help="Input CSV files or directories")
    parser.add_argument("--output", type=str, help="Output path for combined dataset")
    parser.add_argument("--encoding", type=str, default="utf-8", help="Character encoding for CSV files")
    parser.add_argument("--no-infer-task", dest="infer_task", action="store_false", help="Don't infer task_id from file paths")
    parser.add_argument("--analyze", action="store_true", help="Analyze the dataset after combining")
    
    args = parser.parse_args()
    
    # Combine datasets
    combined_df = combine_datasets(
        input_paths=args.input,
        output_path=args.output,
        encoding=args.encoding,
        infer_task=args.infer_task
    )
    
    if combined_df is None:
        print("Failed to create combined dataset.")
        return 1
    
    # Analyze the dataset if requested
    if args.analyze:
        analyze_dataset(combined_df)
    
    return 0

if __name__ == "__main__":
    exit(main())