# prompt.py
"""
Enhanced prompt processing for agricultural recommendations using pure AI approach.
This version uses NLP techniques to understand and process natural language queries
without relying on a database of predefined values.
"""

import torch
import argparse
import os
import json
import sys
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
from model import AgriRecommender

# Path configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "agri_model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")
MODEL_INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")

# Common crops and fertilizers for text matching
COMMON_CROPS = [
    "rice", "wheat", "maize", "corn", "soybean", "cotton", 
    "sugarcane", "potato", "tomato", "onion", "mustard", "strawberry"
]

COMMON_FERTILIZERS = [
    "urea", "dap", "npk", "mop", "ammonium sulfate", "ammonium sulphate", 
    "super phosphate", "potash", "calcium nitrate"
]

def load_model_info(model_info_path=None):
    """Load model architecture information"""
    info_path = model_info_path or MODEL_INFO_PATH
    
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Model info file not found at {info_path}. Please train the model first.")
    
    with open(info_path, 'r') as f:
        model_info = json.load(f)
    
    return model_info

def load_model(model_path=None, model_info_path=None, device="cpu"):
    """Load the trained PyTorch model"""
    # Use provided paths or defaults
    path = model_path or MODEL_PATH
    
    # Get model architecture parameters
    model_info = load_model_info(model_info_path)
    
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
    
    # Check if model file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}. Please train the model first.")
    
    # Load the trained weights
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, model_info

def analyze_query(query_text):
    """
    Analyze a natural language query to understand intent and extract key information.
    
    Args:
        query_text (str): User's natural language query
        
    Returns:
        dict: Analysis results including query type, entities, and task_id
    """
    query_text = query_text.lower()
    analysis = {
        'query_type': None,
        'found_crop': None,
        'found_fertilizer': None,
        'soil_params': {'n': None, 'p': None, 'k': None, 'ph': None},
        'task_id': None
    }
    
    # Detect query type
    if any(term in query_text for term in ['fertilizer', 'fertiliser', 'nutrient']):
        analysis['query_type'] = 'fertilizer_recommendation'
        analysis['task_id'] = 1
    elif any(term in query_text for term in ['crop', 'plant', 'grow']):
        analysis['query_type'] = 'crop_recommendation'
        analysis['task_id'] = 0
    else:
        # Default to crop recommendation if unclear
        analysis['query_type'] = 'crop_recommendation'
        analysis['task_id'] = 0
    
    # Extract crop mentions
    for crop in COMMON_CROPS:
        if crop in query_text:
            pattern = r'\b' + re.escape(crop) + r'\b'
            if re.search(pattern, query_text):
                analysis['found_crop'] = crop
                break
    
    # Extract fertilizer mentions
    for fertilizer in COMMON_FERTILIZERS:
        if fertilizer in query_text:
            pattern = r'\b' + re.escape(fertilizer) + r'\b'
            if re.search(pattern, query_text):
                analysis['found_fertilizer'] = fertilizer
                break
    
    # Look for NPK pattern
    npk_match = re.search(r'npk\s*\d+[-\s]*\d+[-\s]*\d+', query_text)
    if npk_match and not analysis['found_fertilizer']:
        analysis['found_fertilizer'] = 'npk'
    
    # Extract soil parameters using regex patterns
    param_patterns = {
        'n': [r'(?:nitrogen|n)\s*(?::|is|=)\s*(\d+\.?\d*)', r'n\s*[-:=]\s*(\d+\.?\d*)'],
        'p': [r'(?:phosphorus|p)\s*(?::|is|=)\s*(\d+\.?\d*)', r'p\s*[-:=]\s*(\d+\.?\d*)'],
        'k': [r'(?:potassium|k)\s*(?::|is|=)\s*(\d+\.?\d*)', r'k\s*[-:=]\s*(\d+\.?\d*)'],
        'ph': [r'(?:ph)\s*(?::|is|=)\s*(\d+\.?\d*)', r'ph\s*[-:=]\s*(\d+\.?\d*)']
    }
    
    for param, patterns in param_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, query_text)
            if match:
                analysis['soil_params'][param] = float(match.group(1))
                break
    
    # If a fertilizer question with a specific crop was asked, make the task fertilizer
    if analysis['found_crop'] and 'fertilizer' in query_text:
        analysis['task_id'] = 1  # Set to fertilizer task
    
    return analysis

def generate_features_from_query(query_text, input_dim=4):
    """
    Generate feature vector from natural language query.
    Uses adaptive feature generation based on query analysis.
    
    Args:
        query_text (str): User's natural language query
        input_dim (int): Expected input dimension for the model
        
    Returns:
        tuple: (feature_vector, task_id, query_analysis)
    """
    # Analyze the query
    query_analysis = analyze_query(query_text)
    
    # Initialize feature vector with default values
    feature_vector = [50.0, 30.0, 20.0, 6.5]  # Default N, P, K, pH
    
    # Update with any found soil parameters
    for i, param in enumerate(['n', 'p', 'k', 'ph']):
        if query_analysis['soil_params'][param] is not None:
            feature_vector[i] = query_analysis['soil_params'][param]
    
    # If the query is about a specific crop, adjust the feature vector
    # Instead of using fixed database values, we'll modify features based on crop type
    if query_analysis['found_crop']:
        crop = query_analysis['found_crop']
        
        # Apply crop-specific adjustments (this is a heuristic approach)
        if crop == 'rice':
            if feature_vector[0] == 50.0:  # Only adjust if N wasn't explicitly mentioned
                feature_vector[0] = 80.0    # Rice needs more nitrogen
            if feature_vector[3] == 6.5:    # Only adjust if pH wasn't explicitly mentioned
                feature_vector[3] = 6.0     # Rice prefers slightly acidic soil
        
        elif crop in ['wheat', 'maize', 'corn']:
            if feature_vector[0] == 50.0:
                feature_vector[0] = 100.0   # Grain crops need high nitrogen
        
        elif crop in ['potato', 'tomato']:
            if feature_vector[2] == 20.0:
                feature_vector[2] = 60.0    # Root/fruit crops need more potassium
        
        elif crop == 'soybean':
            if feature_vector[0] == 50.0:
                feature_vector[0] = 30.0    # Legumes need less nitrogen
            if feature_vector[1] == 30.0:
                feature_vector[1] = 60.0    # But need more phosphorus
                
        elif crop == 'strawberry':
            if feature_vector[0] == 50.0:
                feature_vector[0] = 35.0    # Strawberries need moderate nitrogen
            if feature_vector[2] == 20.0:
                feature_vector[2] = 70.0    # But high potassium
            if feature_vector[3] == 6.5:
                feature_vector[3] = 5.8     # Slightly acidic soil
    
    # If query is about a specific fertilizer, adjust the feature vector
    if query_analysis['found_fertilizer']:
        fert = query_analysis['found_fertilizer']
        
        # Apply fertilizer-specific adjustments
        if fert == 'urea':
            if feature_vector[0] == 50.0:   # Urea is high in nitrogen
                feature_vector[0] = 90.0
        
        elif fert == 'dap':                # DAP is high in phosphorus
            if feature_vector[1] == 30.0:
                feature_vector[1] = 70.0
        
        elif fert == 'mop':                # MOP is high in potassium
            if feature_vector[2] == 20.0:
                feature_vector[2] = 70.0
    
    # Make sure the feature vector has the correct dimension
    if len(feature_vector) < input_dim:
        feature_vector.extend([0.0] * (input_dim - len(feature_vector)))
    elif len(feature_vector) > input_dim:
        feature_vector = feature_vector[:input_dim]
    
    # Normalize the feature vector with StandardScaler
    scaler = StandardScaler()
    normalized_vector = scaler.fit_transform([feature_vector])[0]
    
    print(f"Generated features from query: {feature_vector}")
    
    return torch.tensor(normalized_vector, dtype=torch.float32), query_analysis['task_id'], query_analysis

def predict(query_text, task_id=None, device="cpu", model_path=None, model_info_path=None):
    """
    Get agricultural recommendations from natural language query.
    
    Args:
        query_text (str): Natural language query
        task_id (int, optional): Force specific task (0=crop, 1=fertilizer)
        
    Returns:
        dict: Prediction results and extracted context
    """
    try:
        # Get model dimensions
        model_info = load_model_info(model_info_path)
        input_dim = model_info["input_dim"]
        
        # Generate features from query
        features, auto_task_id, query_analysis = generate_features_from_query(query_text, input_dim)
        
        # Use provided task_id if specified, otherwise use auto-detected
        if task_id is not None:
            task_id_to_use = task_id
        else:
            task_id_to_use = auto_task_id
        
        # Load model
        model, _ = load_model(model_path, model_info_path, device)
        
        # Convert to PyTorch tensors and add batch dimension
        features = features.unsqueeze(0).to(device)
        task_id_tensor = torch.tensor(task_id_to_use, dtype=torch.long).unsqueeze(0).to(device)
        
        # Run prediction
        with torch.no_grad():
            outputs = model(features, task_id_tensor)
        
        # Get probabilities
        crop_probs = torch.nn.functional.softmax(outputs['crop'], dim=1)
        fert_probs = torch.nn.functional.softmax(outputs['fertilizer'], dim=1)
        
        # Get top predictions
        crop_pred = torch.argmax(outputs['crop'], dim=1).item()
        fert_pred = torch.argmax(outputs['fertilizer'], dim=1).item()
        
        crop_conf = crop_probs[0, crop_pred].item()
        fert_conf = fert_probs[0, fert_pred].item()
        
        # Get top k predictions (where k is min of 3 and number of classes)
        num_crop_classes = crop_probs.size(1)
        num_fert_classes = fert_probs.size(1)
        
        k_crops = min(3, num_crop_classes)
        k_ferts = min(3, num_fert_classes)
        
        _, top_crop_indices = torch.topk(crop_probs, k_crops, dim=1)
        _, top_fert_indices = torch.topk(fert_probs, k_ferts, dim=1)
        
        top_crops = [(idx.item(), crop_probs[0, idx].item()) for idx in top_crop_indices[0]]
        top_ferts = [(idx.item(), fert_probs[0, idx].item()) for idx in top_fert_indices[0]]
        
        # Construct response based on task
        if task_id_to_use == 0:  # Crop recommendation
            return {
                "recommendation_type": "crop",
                "prediction": crop_pred,
                "confidence": crop_conf,
                "top_predictions": top_crops,
                "query_analysis": query_analysis,
                "embedding": outputs['embedding'].cpu().numpy().tolist() if 'embedding' in outputs else None
            }
        else:  # Fertilizer recommendation
            return {
                "recommendation_type": "fertilizer",
                "prediction": fert_pred,
                "confidence": fert_conf,
                "top_predictions": top_ferts,
                "query_analysis": query_analysis,
                "embedding": outputs['embedding'].cpu().numpy().tolist() if 'embedding' in outputs else None
            }
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        raise

def get_crop_name(crop_id):
    """Map crop ID to human-readable crop name"""
    # This mapping should be updated based on your specific crop IDs
    crop_names = {
        0: "Rice", 1: "Wheat", 2: "Maize", 3: "Soybean", 4: "Cotton",
        5: "Sugarcane", 6: "Potato", 7: "Tomato", 8: "Onion", 9: "Mustard"
    }
    return crop_names.get(crop_id, f"Crop {crop_id}")

def get_fertilizer_name(fert_id):
    """Map fertilizer ID to human-readable fertilizer name"""
    # This mapping should be updated based on your specific fertilizer IDs
    fertilizer_names = {
        0: "Urea", 1: "DAP", 2: "NPK", 3: "MOP", 4: "Ammonium Sulfate"
    }
    return fertilizer_names.get(fert_id, f"Fertilizer {fert_id}")

def main():
    parser = argparse.ArgumentParser(description="Agricultural recommendation system using natural language queries.")
    parser.add_argument("--question", type=str, required=True, help="Your agricultural question (e.g., 'What's the best fertilizer for rice?')")
    parser.add_argument("--task", type=int, default=None, help="Force task ID (0: crop recommendation, 1: fertilizer recommendation, None: auto-detect)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    parser.add_argument("--model-dir", type=str, default=None, help="Custom model directory path")
    args = parser.parse_args()

    # Set up model paths based on arguments
    model_path = None
    model_info_path = None
    
    if args.model_dir:
        model_path = os.path.join(args.model_dir, "best_model.pt")
        model_info_path = os.path.join(args.model_dir, "model_info.json")

    try:
        # Check if models directory exists
        if not os.path.exists(MODEL_DIR):
            print(f"❌ Model directory not found at {MODEL_DIR}")
            print("Please make sure your directory structure is correct or provide a valid model directory with --model-dir")
            sys.exit(1)
            
        # Run prediction with the query
        result = predict(args.question, args.task, args.device, model_path, model_info_path)
        
        # Get human-readable names
        if result["recommendation_type"] == "crop":
            prediction_name = get_crop_name(result["prediction"])
            rec_type = "Crop"
        else:
            prediction_name = get_fertilizer_name(result["prediction"])
            rec_type = "Fertilizer"
        
        # Check what was found in the query
        query_analysis = result.get("query_analysis", {})
        found_crop = query_analysis.get("found_crop")
        found_fertilizer = query_analysis.get("found_fertilizer")
            
        # Print user-friendly output
        print("\n" + "="*50)
        print(f"🧠 {rec_type} Recommendation for: '{args.question}'")
        print("="*50)
        
        # If the user asked about a specific crop/fertilizer, acknowledge it
        if found_crop:
            print(f"► Detected crop in query: {found_crop.capitalize()}")
        if found_fertilizer:
            print(f"► Detected fertilizer in query: {found_fertilizer.upper() if found_fertilizer.lower() == 'npk' else found_fertilizer.capitalize()}")
            
        # Main recommendation
        print(f"► Recommended: {prediction_name}")
        print(f"► Confidence: {result['confidence']:.2f}")
        print("="*50)
        
        # Print top recommendations
        if "top_predictions" in result:
            print("\nTop Recommendations:")
            for i, (pred_id, confidence) in enumerate(result["top_predictions"]):
                name = get_crop_name(pred_id) if result["recommendation_type"] == "crop" else get_fertilizer_name(pred_id)
                print(f"{i+1}. {name} (confidence: {confidence:.2f})")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()