# preprocess.py
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def standardize_columns(df):
    """Standardize column names and formats"""
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Define common column aliases
    column_mapping = {
        # Crop-related columns
        'crop': 'crop_label',
        'crop_type': 'crop_label',
        'crop_name': 'crop_label',
        
        # Fertilizer-related columns
        'fertilizer': 'fertilizer_label',
        'fert': 'fertilizer_label',
        'fertilizer_type': 'fertilizer_label',
        'fertilizer_name': 'fertilizer_label',
        
        # Climate-related columns
        'climate_effect': 'climate_effect_label',
        'climate_impact': 'climate_effect_label',
        
        # Task ID column
        'task': 'task_id',
    }
    
    # Apply mappings if they exist in the dataframe
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    # Make sure all feature columns have a standard prefix
    feature_prefixes = ['feature', 'soil_', 'climate_', 'water_', 'ph_', 'nutrient_', 'n_', 'p_', 'k_']
    
    for col in df.columns:
        # Skip already standardized columns
        if col in ['task_id', 'crop_label', 'fertilizer_label', 'climate_effect_label']:
            continue
            
        # Skip columns that already have feature prefixes
        if any(col.startswith(prefix) for prefix in feature_prefixes):
            continue
            
        # Rename other columns to standard feature format
        if not col.startswith('feature'):
            df = df.rename(columns={col: f'feature_{col}'})
    
    # Convert numerical string columns to float
    for col in df.columns:
        if col not in ['task_id', 'crop_label', 'fertilizer_label', 'climate_effect_label']:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
    
    return df

def infer_task_id(file_path):
    """Infer task ID from file path or name"""
    # Extract filename from path
    filename = os.path.basename(file_path).lower()
    
    # Try to infer task from filename
    if any(term in filename for term in ['crop', 'plant', 'species']):
        return 0  # Crop recommendation task
    elif any(term in filename for term in ['fertilizer', 'fert', 'nutrient']):
        return 1  # Fertilizer recommendation task
    elif any(term in filename for term in ['climate', 'weather', 'temperature']):
        return 2  # Climate effect task
    elif any(term in filename for term in ['water', 'irrigation', 'moisture']):
        return 3  # Water management task
    elif any(term in filename for term in ['pest', 'disease', 'control']):
        return 4  # Pest control task
    else:
        # Default task ID
        return 0

def handle_missing_values(df):
    """Handle missing values in the dataframe"""
    # For feature columns, use median imputation
    feature_cols = [col for col in df.columns if col.startswith('feature')]
    
    for col in feature_cols:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    # For categorical columns, use mode imputation
    cat_cols = ['crop_label', 'fertilizer_label', 'climate_effect_label']
    for col in cat_cols:
        if col in df.columns:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "unknown"
            df[col] = df[col].fillna(mode_val)
    
    return df

def normalize_numerical_features(df):
    """Normalize numerical features to have zero mean and unit variance"""
    feature_cols = [col for col in df.columns if col.startswith('feature')]
    numeric_features = [col for col in feature_cols if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
    
    if numeric_features:
        scaler = StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    return df

def process_csv(file_path, output_path=None):
    """Process a CSV file and standardize it"""
    try:
        logger.info(f"Processing {file_path}")
        df = pd.read_csv(file_path)
        
        # Apply data preprocessing steps
        df = standardize_columns(df)
        df = handle_missing_values(df)
        
        # Add task_id if not present
        if 'task_id' not in df.columns:
            task_id = infer_task_id(file_path)
            df['task_id'] = task_id
            logger.info(f"Inferred task_id: {task_id} for {file_path}")
        
        # Ensure we have required label columns
        if 'crop_label' not in df.columns:
            logger.warning(f"No crop_label found in {file_path}, adding placeholder")
            df['crop_label'] = "unknown"
            
        if 'fertilizer_label' not in df.columns:
            logger.warning(f"No fertilizer_label found in {file_path}, adding placeholder")
            df['fertilizer_label'] = "unknown"
        
        # Normalize features (optional at this stage)
        # df = normalize_numerical_features(df)
        
        # Save processed file if output path specified
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Processed file saved to {output_path}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None

def process_directory(input_dir, output_dir=None):
    """Process all CSV files in a directory"""
    processed_files = 0
    failed_files = 0
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Walk through all files in the directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                
                # Create equivalent output path if needed
                if output_dir:
                    rel_path = os.path.relpath(file_path, input_dir)
                    output_path = os.path.join(output_dir, rel_path)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                else:
                    output_path = None
                
                # Process the file
                result = process_csv(file_path, output_path)
                
                if result is not None:
                    processed_files += 1
                else:
                    failed_files += 1
    
    logger.info(f"Processing complete: {processed_files} files processed, {failed_files} files failed")
    return processed_files, failed_files

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess CSV files for agricultural data")
    parser.add_argument("--input", type=str, required=True, help="Input directory or file")
    parser.add_argument("--output", type=str, default=None, help="Output directory (optional)")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        process_directory(args.input, args.output)
    elif os.path.isfile(args.input) and args.input.endswith('.csv'):
        process_csv(args.input, args.output)
    else:
        logger.error(f"Invalid input: {args.input}. Must be a CSV file or directory containing CSV files.")