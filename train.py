# train.py
"""
Enhanced training script for agricultural recommendation models.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
import logging
import time
from tqdm import tqdm
import glob
from collections import Counter
from model import EnhancedAgriRecommender, AgriMultiTaskModel

# Set environment variables to avoid multiprocessing issues
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

class AgriDataset(Dataset):
    """
    Enhanced dataset class for agricultural data with support for 
    different task types and data augmentation.
    """
    def __init__(self, dataframe, augment=False, noise_level=0.05):
        self.df = dataframe.copy()
        self.augment = augment
        self.noise_level = noise_level
        
        # Extract feature columns and labels
        self.feature_cols = [col for col in self.df.columns if col.startswith('feature')]
        
        # Initialize encoders
        self.label_enc_crop = LabelEncoder()
        self.label_enc_fert = LabelEncoder()
        
        # Encode target labels
        if 'crop_label' in self.df.columns:
            self.df['crop_label'] = self.label_enc_crop.fit_transform(self.df['crop_label'])
        else:
            self.df['crop_label'] = 0  # Default
            
        if 'fertilizer_label' in self.df.columns:
            self.df['fertilizer_label'] = self.label_enc_fert.fit_transform(self.df['fertilizer_label'])
        else:
            self.df['fertilizer_label'] = 0  # Default
            
        # Handle additional tasks if present
        if 'climate_effect_label' in self.df.columns:
            self.label_enc_climate = LabelEncoder()
            self.df['climate_effect_label'] = self.label_enc_climate.fit_transform(self.df['climate_effect_label'])
        
        if 'yield_value' in self.df.columns:
            # Normalize yield values
            self.yield_mean = self.df['yield_value'].mean()
            self.yield_std = self.df['yield_value'].std()
            self.df['yield_value'] = (self.df['yield_value'] - self.yield_mean) / (self.yield_std + 1e-8)
        
        # Normalize features
        self.scaler = StandardScaler()
        self.df[self.feature_cols] = self.scaler.fit_transform(self.df[self.feature_cols])
        
        # Save dimensions
        self.input_dim = len(self.feature_cols)
        self.crop_classes = len(self.label_enc_crop.classes_) if hasattr(self.label_enc_crop, 'classes_') else 1
        self.fert_classes = len(self.label_enc_fert.classes_) if hasattr(self.label_enc_fert, 'classes_') else 1
        self.climate_classes = len(self.label_enc_climate.classes_) if hasattr(self, 'label_enc_climate') and hasattr(self.label_enc_climate, 'classes_') else 0
        
        # Store class mappings
        self.crop_mapping = {i: cls for i, cls in enumerate(self.label_enc_crop.classes_)} if hasattr(self.label_enc_crop, 'classes_') else {}
        self.fert_mapping = {i: cls for i, cls in enumerate(self.label_enc_fert.classes_)} if hasattr(self.label_enc_fert, 'classes_') else {}
        
        # Check for class imbalance
        if 'crop_label' in self.df.columns:
            crop_counts = Counter(self.df['crop_label'])
            logging.info(f"Crop class distribution: {crop_counts}")
        
        if 'fertilizer_label' in self.df.columns:
            fert_counts = Counter(self.df['fertilizer_label'])
            logging.info(f"Fertilizer class distribution: {fert_counts}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get features and apply augmentation if enabled
        features = np.array(row[self.feature_cols].values, dtype=np.float32)
        
        if self.augment and np.random.random() < 0.5:
            # Apply random noise to features
            noise = np.random.normal(0, self.noise_level, size=features.shape)
            features = features + noise
        
        # Get task_id
        task_id = int(row['task_id']) if 'task_id' in row else 0
        
        # Get labels
        crop_target = int(row['crop_label']) if 'crop_label' in row else 0
        fert_target = int(row['fertilizer_label']) if 'fertilizer_label' in row else 0
        
        # Additional task targets
        climate_target = int(row['climate_effect_label']) if 'climate_effect_label' in row else 0
        yield_target = float(row['yield_value']) if 'yield_value' in row else 0.0
        
        # Convert to tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        task_id_tensor = torch.tensor(task_id, dtype=torch.long)
        crop_tensor = torch.tensor(crop_target, dtype=torch.long)
        fert_tensor = torch.tensor(fert_target, dtype=torch.long)
        climate_tensor = torch.tensor(climate_target, dtype=torch.long)
        yield_tensor = torch.tensor(yield_target, dtype=torch.float32)
        
        return {
            'features': features_tensor,
            'task_id': task_id_tensor,
            'crop_label': crop_tensor,
            'fertilizer_label': fert_tensor,
            'climate_label': climate_tensor,
            'yield_value': yield_tensor
        }

    def get_meta_info(self):
        """Return metadata about the dataset"""
        return {
            'input_dim': self.input_dim,
            'crop_classes': self.crop_classes,
            'fertilizer_classes': self.fert_classes,
            'climate_classes': self.climate_classes if hasattr(self, 'climate_classes') else 0,
            'feature_names': self.feature_cols,
            'crop_mapping': self.crop_mapping,
            'fertilizer_mapping': self.fert_mapping
        }


def find_csv_files(path):
    """Find all CSV files in a directory and its subdirectories"""
    csv_files = []
    
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
    elif os.path.isfile(path) and path.endswith('.csv'):
        csv_files.append(path)
    
    return csv_files

def infer_task_id(file_path):
    """Infer task ID from file path or content"""
    filename = os.path.basename(file_path).lower()
    
    if any(term in filename for term in ['crop_recommendation', 'crop_recommend']):
        return 0  # Crop recommendation
    elif any(term in filename for term in ['fertilizer', 'fert']):
        return 1  # Fertilizer recommendation
    elif any(term in filename for term in ['climate', 'weather']):
        return 2  # Climate effect prediction
    elif any(term in filename for term in ['yield', 'production']):
        return 3  # Yield prediction
    
    # If we can't determine from filename, return default
    return 0

def standardize_columns(df):
    """Standardize column names across different datasets"""
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

def load_dataset(paths, verbose=False):
    """
    Load and combine datasets from multiple CSV files or directories
    with smart preprocessing to handle different formats.
    """
    all_dfs = []
    
    # Convert to list if a single path
    if isinstance(paths, str):
        paths = [paths]
    
    for path in paths:
        # Find all CSV files
        csv_files = find_csv_files(path)
        logging.info(f"Found {len(csv_files)} CSV files in {path}")
        
        for file_path in tqdm(csv_files, desc="Loading CSVs"):
            try:
                # Load CSV
                df = pd.read_csv(file_path)
                
                if verbose:
                    logging.info(f"Loaded {file_path} with {len(df)} rows and columns: {df.columns.tolist()}")
                
                # Standardize column names
                df = standardize_columns(df)
                
                # Check if we have the necessary columns
                has_features = any(col.startswith('feature_') for col in df.columns)
                has_target = 'crop_label' in df.columns or 'fertilizer_label' in df.columns
                
                if not has_features or not has_target:
                    logging.warning(f"Skipping {file_path}: missing required columns")
                    continue
                
                # Infer task_id if not present
                if 'task_id' not in df.columns:
                    task_id = infer_task_id(file_path)
                    df['task_id'] = task_id
                    
                    if verbose:
                        logging.info(f"Inferred task_id={task_id} for {file_path}")
                
                # Handle categorical features - convert to numeric
                for col in df.columns:
                    if col.startswith('feature_') and df[col].dtype == 'object':
                        # Create a mapping for categorical features
                        unique_values = df[col].unique()
                        value_map = {val: i for i, val in enumerate(unique_values)}
                        df[col] = df[col].map(value_map)
                        
                        if verbose:
                            logging.info(f"Converted categorical feature {col} with mapping: {value_map}")
                
                # Fill missing values
                for col in df.columns:
                    if col.startswith('feature_') and df[col].isnull().any():
                        # For numerical features, use median
                        if df[col].dtype in ['int64', 'float64']:
                            median_val = df[col].median()
                            df[col] = df[col].fillna(median_val)
                        # For categorical features, use most common value
                        else:
                            mode_val = df[col].mode().iloc[0]
                            df[col] = df[col].fillna(mode_val)
                            
                # Append to list of dataframes
                all_dfs.append(df)
                
            except Exception as e:
                logging.error(f"Error loading {file_path}: {e}")
    
    if not all_dfs:
        raise ValueError("No valid dataframes could be loaded from the provided paths")
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logging.info(f"Combined dataset has {len(combined_df)} rows")
    
    # Make sure we have all required columns
    if 'crop_label' not in combined_df.columns:
        logging.warning("No 'crop_label' column found, adding placeholder")
        combined_df['crop_label'] = 'unknown'
        
    if 'fertilizer_label' not in combined_df.columns:
        logging.warning("No 'fertilizer_label' column found, adding placeholder")
        combined_df['fertilizer_label'] = 'unknown'
    
    # Ensure all feature columns are numeric
    feature_cols = [col for col in combined_df.columns if col.startswith('feature_')]
    for col in feature_cols:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        
    # Fill any remaining NaN values with column medians
    for col in feature_cols:
        if combined_df[col].isnull().any():
            combined_df[col] = combined_df[col].fillna(combined_df[col].median())
    
    return combined_df

def train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, device, epochs=30, patience=5):
    """
    Train the model with early stopping, learning rate scheduling, and gradient clipping
    """
    logging.info(f"Starting training for {epochs} epochs")
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    patience_counter = 0
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_crop_acc': [],
        'val_crop_acc': [],
        'train_fert_acc': [],
        'val_fert_acc': [],
        'learning_rate': []
    }
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_crop_correct = 0
        train_fert_correct = 0
        train_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in progress_bar:
            # Move batch to device
            features = batch['features'].to(device)
            task_ids = batch['task_id'].to(device)
            crop_labels = batch['crop_label'].to(device)
            fert_labels = batch['fertilizer_label'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features, task_ids)
            
            # Calculate loss - combined loss from both heads
            crop_loss = criterion['crop'](outputs['crop'], crop_labels)
            fert_loss = criterion['fertilizer'](outputs['fertilizer'], fert_labels)
            
            # Combine losses based on task_id
            # If task_id=0, focus more on crop loss; if task_id=1, focus more on fertilizer loss
            batch_loss = crop_loss + fert_loss
            
            # Backward pass
            batch_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Optimizer step
            optimizer.step()
            
            # Update statistics
            train_loss += batch_loss.item() * features.size(0)
            
            # Calculate accuracy
            _, crop_preds = torch.max(outputs['crop'], 1)
            _, fert_preds = torch.max(outputs['fertilizer'], 1)
            
            train_crop_correct += (crop_preds == crop_labels).sum().item()
            train_fert_correct += (fert_preds == fert_labels).sum().item()
            train_samples += features.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': batch_loss.item(),
                'crop_acc': train_crop_correct / train_samples,
                'fert_acc': train_fert_correct / train_samples
            })
        
        # Calculate average training metrics
        train_loss = train_loss / train_samples
        train_crop_acc = train_crop_correct / train_samples
        train_fert_acc = train_fert_correct / train_samples
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_crop_correct = 0
        val_fert_correct = 0
        val_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                # Move batch to device
                features = batch['features'].to(device)
                task_ids = batch['task_id'].to(device)
                crop_labels = batch['crop_label'].to(device)
                fert_labels = batch['fertilizer_label'].to(device)
                
                # Forward pass
                outputs = model(features, task_ids)
                
                # Calculate loss
                crop_loss = criterion['crop'](outputs['crop'], crop_labels)
                fert_loss = criterion['fertilizer'](outputs['fertilizer'], fert_labels)
                batch_loss = crop_loss + fert_loss
                
                # Update statistics
                val_loss += batch_loss.item() * features.size(0)
                
                # Calculate accuracy
                _, crop_preds = torch.max(outputs['crop'], 1)
                _, fert_preds = torch.max(outputs['fertilizer'], 1)
                
                val_crop_correct += (crop_preds == crop_labels).sum().item()
                val_fert_correct += (fert_preds == fert_labels).sum().item()
                val_samples += features.size(0)
        
        # Calculate average validation metrics
        val_loss = val_loss / val_samples
        val_crop_acc = val_crop_correct / val_samples
        val_fert_acc = val_fert_correct / val_samples
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        # Log metrics
        logging.info(f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Train Crop Acc: {train_crop_acc:.4f}, "
                    f"Val Crop Acc: {val_crop_acc:.4f}, "
                    f"Train Fert Acc: {train_fert_acc:.4f}, "
                    f"Val Fert Acc: {val_fert_acc:.4f}, "
                    f"LR: {current_lr:.6f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_crop_acc'].append(train_crop_acc)
        history['val_crop_acc'].append(val_crop_acc)
        history['train_fert_acc'].append(train_fert_acc)
        history['val_fert_acc'].append(val_fert_acc)
        history['learning_rate'].append(current_lr)
        
        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            logging.info(f"New best model at epoch {epoch+1}")
        else:
            patience_counter += 1
            logging.info(f"No improvement for {patience_counter} epochs")
            
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model
    model.load_state_dict(best_model_state)
    logging.info(f"Training completed. Best model was from epoch {best_epoch+1}")
    
    return model, history

def plot_training_history(history, save_path="training_history.png"):
    """Plot training history metrics"""
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot crop accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_crop_acc'], label='Train Crop Acc')
    plt.plot(history['val_crop_acc'], label='Val Crop Acc')
    plt.title('Crop Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot fertilizer accuracy
    plt.subplot(2, 2, 3)
    plt.plot(history['train_fert_acc'], label='Train Fert Acc')
    plt.plot(history['val_fert_acc'], label='Val Fert Acc')
    plt.title('Fertilizer Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate
    plt.subplot(2, 2, 4)
    plt.plot(history['learning_rate'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path)
    logging.info(f"Training history plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train agricultural recommendation model")
    parser.add_argument("--data", type=str, required=True, help="Path to data directory or CSV file")
    parser.add_argument("--model-type", type=str, default="enhanced", choices=["enhanced", "multitask"], help="Model type to use")
    parser.add_argument("--output-dir", type=str, default="models/agri_model", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--emb-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--augment", action="store_true", help="Use data augmentation")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and preprocess data
    data_df = load_dataset(args.data, verbose=args.verbose)
    logging.info(f"Loaded dataset with shape: {data_df.shape}")
    
    # Create dataset
    dataset = AgriDataset(data_df, augment=args.augment)
    
    # Get metadata
    meta_info = dataset.get_meta_info()
    logging.info(f"Dataset metadata: {meta_info}")
    
    # Split into train and validation sets
    train_size = int((1 - args.test_size) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logging.info(f"Training set size: {len(train_dataset)}")
    logging.info(f"Validation set size: {len(val_dataset)}")
    
    # Create data loaders with 0 workers to avoid multiprocessing issues
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,  # Changed from 4 to 0 to fix pickling error
        pin_memory=device.type == "cuda"
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0,  # Changed from 4 to 0 to fix pickling error
        pin_memory=device.type == "cuda"
    )
    
    # Initialize model
    if args.model_type == "enhanced":
        model = EnhancedAgriRecommender(
            input_dim=meta_info['input_dim'],
            emb_dim=args.emb_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            crop_classes=meta_info['crop_classes'],
            fert_classes=meta_info['fertilizer_classes'],
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    else:  # multitask
        model = AgriMultiTaskModel(
            input_dim=meta_info['input_dim'],
            emb_dim=args.emb_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            crop_classes=meta_info['crop_classes'],
            fert_classes=meta_info['fertilizer_classes'],
            climate_effect_classes=meta_info.get('climate_classes', 3),
            yield_bins=5,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    
    model.to(device)
    logging.info(f"Initialized {args.model_type} model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize loss functions with class weighting if needed
    criterion = {
        'crop': nn.CrossEntropyLoss(),
        'fertilizer': nn.CrossEntropyLoss()
    }
    
    # Initialize optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Initialize learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Train model
    trained_model, history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        patience=args.patience
    )
    
    # Save model
    model_path = os.path.join(args.output_dir, "best_model.pt")
    torch.save(trained_model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")
    
    # Save model info
    model_info = {
        "input_dim": meta_info['input_dim'],
        "crop_classes": meta_info['crop_classes'],
        "fertilizer_classes": meta_info['fertilizer_classes'],
        "feature_names": meta_info['feature_names'],
        "crop_mapping": meta_info['crop_mapping'],
        "fertilizer_mapping": meta_info['fert_mapping'],
        "model_params": {
            "emb_dim": args.emb_dim,
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dropout": args.dropout
        },
        "training_args": vars(args)
    }
    
    with open(os.path.join(args.output_dir, "model_info.json"), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Save training history
    with open(os.path.join(args.output_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    # Plot training history
    plot_training_history(history, save_path=os.path.join(args.output_dir, "training_history.png"))
    
    # Save final model as well
    final_model_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved to {final_model_path}")
    
    # Print final metrics
    logging.info(f"Training completed with final metrics:")
    logging.info(f"  Best validation loss: {min(history['val_loss']):.4f}")
    logging.info(f"  Best crop accuracy: {max(history['val_crop_acc']):.4f}")
    logging.info(f"  Best fertilizer accuracy: {max(history['val_fert_acc']):.4f}")
    
    return 0

if __name__ == "__main__":
    exit(main())