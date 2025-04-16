# inference.py
import torch
import pandas as pd
import json
import os
from model import AgriRecommender
from data_loader import AgriDataset
import sys


def load_model_info(model_output_dir):
    """Load model information from the output directory"""
    model_info_path = os.path.join(model_output_dir, "model_info.json")
    if not os.path.exists(model_info_path):
        raise FileNotFoundError(f"Model info file not found at {model_info_path}")
    
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    return model_info


def load_model(model_path, model_info, device):
    """Load the trained model using parameters from model_info"""
    # Extract parameters from model_info
    input_dim = model_info["input_dim"]
    crop_classes = model_info["crop_classes"]
    fert_classes = model_info["fertilizer_classes"]
    
    # Extract model architecture parameters
    emb_dim = model_info["model_params"]["emb_dim"]
    num_layers = model_info["model_params"]["num_layers"]
    hidden_dim = model_info["model_params"]["hidden_dim"]
    
    # Initialize model with the same architecture as the trained model
    model = AgriRecommender(
        input_dim=input_dim,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_heads=4,  # Default value if not in model_info
        crop_classes=crop_classes,
        fert_classes=fert_classes,
        num_layers=num_layers,
        task_embedding_dim=32,  # Default value if not in model_info
        dropout=0.0  # Use 0 dropout for inference
    )
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model


def predict(model, sample_tensor, task_id_tensor, device):
    """Make predictions using the model"""
    sample_tensor = sample_tensor.unsqueeze(0).to(device)
    task_id_tensor = task_id_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(sample_tensor, task_id_tensor)
    
    crop_pred = torch.argmax(outputs['crop'], dim=1).item()
    fert_pred = torch.argmax(outputs['fertilizer'], dim=1).item()
    
    # Get confidence scores
    crop_probs = torch.nn.functional.softmax(outputs['crop'], dim=1)
    fert_probs = torch.nn.functional.softmax(outputs['fertilizer'], dim=1)
    
    crop_confidence = crop_probs[0, crop_pred].item()
    fert_confidence = fert_probs[0, fert_pred].item()
    
    return crop_pred, fert_pred, crop_confidence, fert_confidence


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference with trained agricultural recommendation model")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing trained model files")
    parser.add_argument("--input-csv", type=str, required=True, help="Input CSV file with sample data")
    parser.add_argument("--task-id", type=int, default=0, help="Task ID (0: crop recommendation, 1: fertilizer)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model info and model
    model_info = load_model_info(args.model_dir)
    model_path = os.path.join(args.model_dir, "best_model.pt")
    
    print(f"Loading model from {model_path}")
    model = load_model(model_path, model_info, device)
    print(f"Model loaded successfully")
    
    # Load and preprocess input data
    print(f"Loading input data from {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    dataset = AgriDataset(df)
    
    # Make predictions for each sample in the dataset
    results = []
    
    print("Running inference...")
    for i in range(len(dataset)):
        features, _, crop_target, fert_target = dataset[i]
        task_id = torch.tensor(args.task_id, dtype=torch.long)
        
        crop_pred, fert_pred, crop_conf, fert_conf = predict(model, features, task_id, device)
        
        results.append({
            'sample_idx': i,
            'crop_prediction': crop_pred,
            'crop_confidence': crop_conf,
            'fertilizer_prediction': fert_pred,
            'fertilizer_confidence': fert_conf
        })
    
    # Print results
    print("\nInference Results:")
    print("------------------")
    for res in results:
        print(f"Sample {res['sample_idx']}: "
              f"Crop={res['crop_prediction']} (conf={res['crop_confidence']:.4f}), "
              f"Fertilizer={res['fertilizer_prediction']} (conf={res['fertilizer_confidence']:.4f})")