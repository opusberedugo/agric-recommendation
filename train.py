# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
import os
import sys
import time
import logging
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Import local modules
from model import AgriRecommender
from data_loader import AgriDataset
from preprocess import standardize_columns, infer_task_id, process_csv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=5, min_delta=0.001, checkpoint_path='best_model.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.checkpoint_path = checkpoint_path
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            logging.info(f"Validation loss decreased to {val_loss:.4f}. Saving model...")
            return False
        else:
            self.counter += 1
            logging.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                logging.info(f"Early stopping triggered. Best val loss: {self.best_loss:.4f}")
                return True
        return False

def load_all_csvs(root_folder):
    """Load and compile all CSV files recursively from a directory"""
    csv_paths = []
    for dirpath, _, filenames in os.walk(root_folder):
        for f in filenames:
            if f.endswith(".csv"):
                csv_paths.append(os.path.join(dirpath, f))
    return csv_paths

def preprocess_dataframe(df, file_path):
    """Apply preprocessing to a dataframe before creating dataset"""
    # Standardize columns 
    df = standardize_columns(df)
    
    # Add or infer task_id
    task_id = infer_task_id(file_path)
    df['task_id'] = task_id
    
    # Add default labels if not present
    if 'crop_label' not in df.columns:
        df['crop_label'] = "unknown"  # Placeholder
    if 'fertilizer_label' not in df.columns:
        df['fertilizer_label'] = "unknown"  # Placeholder
        
    return df

def create_datasets(csv_files, test_size=0.2, val_size=0.1):
    """Create and split datasets from CSV files"""
    datasets = []
    input_dim = None
    n_crop_classes = 0
    n_fert_classes = 0
    
    for csv_path in tqdm(csv_files, desc="Loading datasets"):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                # Preprocess the dataframe
                df = preprocess_dataframe(df, csv_path)
                
                try:
                    dataset = AgriDataset(df)
                    datasets.append(dataset)
                    
                    # Update input dimension
                    if input_dim is None:
                        input_dim = dataset.input_dim
                    
                    # Track the number of classes
                    n_crop_classes = max(n_crop_classes, len(dataset.label_enc_crop.classes_))
                    n_fert_classes = max(n_fert_classes, len(dataset.label_enc_fert.classes_))
                    
                    logging.info(f"Loaded: {csv_path} - {len(dataset)} samples")
                except Exception as e:
                    logging.error(f"Error creating dataset from {csv_path}: {e}")
        except Exception as e:
            logging.error(f"Skipping {csv_path}, error: {e}")

    if not datasets:
        logging.error("No valid datasets found. Please check your CSV files.")
        return None, None, None, None, None, None
    
    # Combine all datasets
    full_dataset = ConcatDataset(datasets)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    test_size_abs = int(test_size * total_size)
    val_size_abs = int(val_size * total_size)
    train_size = total_size - test_size_abs - val_size_abs
    
    # Split into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size_abs, test_size_abs],
        generator=torch.Generator().manual_seed(42)
    )
    
    logging.info(f"Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, input_dim, n_crop_classes, n_fert_classes

def train_epoch(model, dataloader, optimizer, crop_criterion, fert_criterion, device, clip_value=1.0):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    crop_losses = 0
    fert_losses = 0
    correct_crops = 0
    correct_ferts = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        x, task_id, crop_target, fert_target = batch
        x, task_id = x.to(device), task_id.to(device)
        crop_target, fert_target = crop_target.to(device), fert_target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(x, task_id)
        
        # Calculate losses
        crop_loss = crop_criterion(outputs['crop'], crop_target)
        fert_loss = fert_criterion(outputs['fertilizer'], fert_target)
        
        # Combined loss - weighted by task importance
        loss = crop_loss + fert_loss
        
        # Backward pass and optimize
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()
        
        # Track metrics
        epoch_loss += loss.item() * x.size(0)
        crop_losses += crop_loss.item() * x.size(0)
        fert_losses += fert_loss.item() * x.size(0)
        
        # Calculate accuracy
        _, crop_preds = torch.max(outputs['crop'], 1)
        _, fert_preds = torch.max(outputs['fertilizer'], 1)
        correct_crops += (crop_preds == crop_target).sum().item()
        correct_ferts += (fert_preds == fert_target).sum().item()
        total_samples += x.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'crop_acc': correct_crops / total_samples,
            'fert_acc': correct_ferts / total_samples
        })
    
    # Calculate epoch metrics
    epoch_loss = epoch_loss / total_samples
    crop_losses = crop_losses / total_samples
    fert_losses = fert_losses / total_samples
    crop_acc = correct_crops / total_samples
    fert_acc = correct_ferts / total_samples
    
    return {
        'loss': epoch_loss,
        'crop_loss': crop_losses,
        'fert_loss': fert_losses,
        'crop_acc': crop_acc,
        'fert_acc': fert_acc
    }

def validate(model, dataloader, crop_criterion, fert_criterion, device):
    """Validate the model"""
    model.eval()
    val_loss = 0
    crop_losses = 0
    fert_losses = 0
    correct_crops = 0
    correct_ferts = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            x, task_id, crop_target, fert_target = batch
            x, task_id = x.to(device), task_id.to(device)
            crop_target, fert_target = crop_target.to(device), fert_target.to(device)
            
            # Forward pass
            outputs = model(x, task_id)
            
            # Calculate losses
            crop_loss = crop_criterion(outputs['crop'], crop_target)
            fert_loss = fert_criterion(outputs['fertilizer'], fert_target)
            loss = crop_loss + fert_loss
            
            # Track metrics
            val_loss += loss.item() * x.size(0)
            crop_losses += crop_loss.item() * x.size(0)
            fert_losses += fert_loss.item() * x.size(0)
            
            # Calculate accuracy
            _, crop_preds = torch.max(outputs['crop'], 1)
            _, fert_preds = torch.max(outputs['fertilizer'], 1)
            correct_crops += (crop_preds == crop_target).sum().item()
            correct_ferts += (fert_preds == fert_target).sum().item()
            total_samples += x.size(0)
    
    # Calculate validation metrics
    val_loss = val_loss / total_samples
    crop_losses = crop_losses / total_samples
    fert_losses = fert_losses / total_samples
    crop_acc = correct_crops / total_samples
    fert_acc = correct_ferts / total_samples
    
    return {
        'loss': val_loss,
        'crop_loss': crop_losses,
        'fert_loss': fert_losses,
        'crop_acc': crop_acc,
        'fert_acc': fert_acc
    }

def test(model, test_dataloader, device):
    """Evaluate model on test set"""
    model.eval()
    correct_crops = 0
    correct_ferts = 0
    total_samples = 0
    
    all_crop_preds = []
    all_crop_targets = []
    all_fert_preds = []
    all_fert_targets = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            x, task_id, crop_target, fert_target = batch
            x, task_id = x.to(device), task_id.to(device)
            crop_target, fert_target = crop_target.to(device), fert_target.to(device)
            
            # Forward pass
            outputs = model(x, task_id)
            
            # Calculate accuracy
            _, crop_preds = torch.max(outputs['crop'], 1)
            _, fert_preds = torch.max(outputs['fertilizer'], 1)
            
            correct_crops += (crop_preds == crop_target).sum().item()
            correct_ferts += (fert_preds == fert_target).sum().item()
            total_samples += x.size(0)
            
            # Store predictions and targets for analysis
            all_crop_preds.extend(crop_preds.cpu().numpy())
            all_crop_targets.extend(crop_target.cpu().numpy())
            all_fert_preds.extend(fert_preds.cpu().numpy())
            all_fert_targets.extend(fert_target.cpu().numpy())
    
    # Calculate test metrics
    crop_acc = correct_crops / total_samples
    fert_acc = correct_ferts / total_samples
    
    logging.info(f"Test Results - Crop Accuracy: {crop_acc:.4f}, Fertilizer Accuracy: {fert_acc:.4f}")
    
    # Convert to numpy arrays for analysis
    all_crop_preds = np.array(all_crop_preds)
    all_crop_targets = np.array(all_crop_targets)
    all_fert_preds = np.array(all_fert_preds)
    all_fert_targets = np.array(all_fert_targets)
    
    # Save predictions for further analysis
    np.savez(
        'test_predictions.npz',
        crop_preds=all_crop_preds,
        crop_targets=all_crop_targets,
        fert_preds=all_fert_preds,
        fert_targets=all_fert_targets
    )
    
    return crop_acc, fert_acc

def plot_training_history(history, save_path='training_history.png'):
    """Plot training and validation metrics"""
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot crop accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_crop_acc'], label='Train Crop Acc')
    plt.plot(history['val_crop_acc'], label='Val Crop Acc')
    plt.title('Crop Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot fertilizer accuracy
    plt.subplot(2, 2, 3)
    plt.plot(history['train_fert_acc'], label='Train Fert Acc')
    plt.plot(history['val_fert_acc'], label='Val Fert Acc')
    plt.title('Fertilizer Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot learning rate
    plt.subplot(2, 2, 4)
    plt.plot(history['learning_rate'], label='Learning Rate')
    plt.title('Learning Rate Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_model_info(model, input_dim, n_crop_classes, n_fert_classes, train_history, output_dir="model_output"):
    """Save model information and training history"""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Save model architecture info
    model_info = {
        "input_dim": input_dim,
        "crop_classes": n_crop_classes,
        "fertilizer_classes": n_fert_classes,
        "model_params": {
            "emb_dim": model.feature_embed[0].out_features,
            "num_layers": len(model.transformer_blocks),
            "hidden_dim": model.crop_head[2].out_features
        },
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save model architecture information
    with open(os.path.join(output_dir, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=4)
    
    # Save training history
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(train_history, f, indent=4)
    
    logging.info(f"Model information saved to {output_dir}")


def train_model(csv_folder, output_dir="model_output", config=None):
    """Main training function"""
    # Set default configuration if not provided
    if config is None:
        config = {
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 50,
            "patience": 10,
            "weight_decay": 1e-5,
            "emb_dim": 128,
            "hidden_dim": 256,
            "num_heads": 4,
            "num_layers": 2,
            "task_emb_dim": 32,
            "dropout": 0.2,
            "clip_value": 1.0,
            "test_size": 0.2,
            "val_size": 0.1,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
    
    # Set output directory
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Save configuration
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # Log configuration
    logging.info(f"Training configuration: {config}")
    logging.info(f"Using device: {config['device']}")
    
    # Load all CSV files
    csv_files = load_all_csvs(csv_folder)
    logging.info(f"Found {len(csv_files)} CSV files in {csv_folder}")
    
    # Create datasets
    train_dataset, val_dataset, test_dataset, input_dim, n_crop_classes, n_fert_classes = create_datasets(
        csv_files, 
        test_size=config["test_size"], 
        val_size=config["val_size"]
    )
    
    if train_dataset is None:
        logging.error("Failed to create datasets, exiting")
        return
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=4,
        pin_memory=True if config["device"] == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True if config["device"] == "cuda" else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    model = AgriRecommender(
        input_dim=input_dim,
        emb_dim=config["emb_dim"],
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        crop_classes=n_crop_classes,
        fert_classes=n_fert_classes,
        num_layers=config["num_layers"],
        task_embedding_dim=config["task_emb_dim"],
        dropout=config["dropout"]
    )
    model.to(config["device"])
    
    # Loss functions and optimizer
    crop_criterion = nn.CrossEntropyLoss()
    fert_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Early stopping
    checkpoint_path = os.path.join(output_dir, "best_model.pt")
    early_stopping = EarlyStopping(patience=config["patience"], checkpoint_path=checkpoint_path)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_crop_loss': [],
        'val_crop_loss': [],
        'train_fert_loss': [],
        'val_fert_loss': [],
        'train_crop_acc': [],
        'val_crop_acc': [],
        'train_fert_acc': [],
        'val_fert_acc': [],
        'learning_rate': []
    }
    
    # Training loop
    logging.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(config["epochs"]):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, 
            train_loader, 
            optimizer, 
            crop_criterion, 
            fert_criterion, 
            config["device"],
            config["clip_value"]
        )
        
        # Validate
        val_metrics = validate(
            model, 
            val_loader, 
            crop_criterion, 
            fert_criterion, 
            config["device"]
        )
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        epoch_time = time.time() - epoch_start
        logging.info(f"Epoch {epoch+1}/{config['epochs']} - {epoch_time:.2f}s - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Train Crop Acc: {train_metrics['crop_acc']:.4f}, "
                    f"Val Crop Acc: {val_metrics['crop_acc']:.4f}, "
                    f"Train Fert Acc: {train_metrics['fert_acc']:.4f}, "
                    f"Val Fert Acc: {val_metrics['fert_acc']:.4f}, "
                    f"LR: {current_lr:.6f}")
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_crop_loss'].append(train_metrics['crop_loss'])
        history['val_crop_loss'].append(val_metrics['crop_loss'])
        history['train_fert_loss'].append(train_metrics['fert_loss'])
        history['val_fert_loss'].append(val_metrics['fert_loss'])
        history['train_crop_acc'].append(train_metrics['crop_acc'])
        history['val_crop_acc'].append(val_metrics['crop_acc'])
        history['train_fert_acc'].append(train_metrics['fert_acc'])
        history['val_fert_acc'].append(val_metrics['fert_acc'])
        history['learning_rate'].append(current_lr)
        
        # Check early stopping
        if early_stopping(val_metrics['loss'], model):
            logging.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Calculate total training time
    total_time = time.time() - start_time
    logging.info(f"Training completed in {total_time:.2f} seconds")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(checkpoint_path))
    
    # Evaluate on test set
    crop_acc, fert_acc = test(model, test_loader, config["device"])
    
    # Save training history plot
    plot_path = os.path.join(output_dir, "training_history.png")
    plot_training_history(history, save_path=plot_path)
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    
    # Save model info and history
    save_model_info(model, input_dim, n_crop_classes, n_fert_classes, history, output_dir)
    
    # Export model to torchscript format for deployment
    script_model_path = os.path.join(output_dir, "model_scripted.pt")
    try:
        # Create example inputs for tracing
        example_inputs = (
            torch.randn(1, input_dim, device=config["device"]),
            torch.tensor([0], device=config["device"])
        )
        script_model = torch.jit.trace(model, example_inputs)
        script_model.save(script_model_path)
        logging.info(f"TorchScript model saved to {script_model_path}")
    except Exception as e:
        logging.error(f"Failed to export TorchScript model: {e}")
    
    return model, history, (crop_acc, fert_acc)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train agricultural recommendation model")
    parser.add_argument("--data", type=str, required=True, help="Path to folder containing CSV files")
    parser.add_argument("--output", type=str, default="model_output", help="Output directory for model and results")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file (optional)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Create config from command line args
        config = {
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "patience": 10,
            "weight_decay": 1e-5,
            "emb_dim": 128,
            "hidden_dim": 256,
            "num_heads": 4,
            "num_layers": 2,
            "task_emb_dim": 32,
            "dropout": 0.2,
            "clip_value": 1.0,
            "test_size": 0.2,
            "val_size": 0.1,
            "device": "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        }
    
    # Run training
    train_model(args.data, args.output, config)