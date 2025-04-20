# api.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import os
import pickle
import numpy as np
from model import AgriRecommender
import json

# Paths
MODEL_DIR = "models/agri_model"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")
MODEL_INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")
TOKENIZER_PATH = "tokenizer.pkl"

# Check if model and info files exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(MODEL_INFO_PATH):
    raise FileNotFoundError(f"Model files not found at {MODEL_PATH} or {MODEL_INFO_PATH}. Please train the model first.")

# Load model info
with open(MODEL_INFO_PATH, 'r') as f:
    model_info = json.load(f)

# Extract model parameters
input_dim = model_info["input_dim"]
crop_classes = model_info["crop_classes"]
fert_classes = model_info["fertilizer_classes"]
emb_dim = model_info["model_params"]["emb_dim"]
num_layers = model_info["model_params"]["num_layers"]
hidden_dim = model_info["model_params"]["hidden_dim"]

# Initialize model with the same architecture as the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Load tokenizer if available
tokenizer = None
if os.path.exists(TOKENIZER_PATH):
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)

# FastAPI app setup
app = FastAPI(title="Deeproot Agri API")

# Enable CORS for React UI (localhost development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your React app domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Common crops and fertilizers for text matching
COMMON_CROPS = [
    "rice", "wheat", "maize", "corn", "soybean", "cotton", 
    "sugarcane", "potato", "tomato", "onion", "mustard", "strawberry"
]

COMMON_FERTILIZERS = [
    "urea", "dap", "npk", "mop", "ammonium sulfate", "ammonium sulphate", 
    "super phosphate", "potash", "calcium nitrate"
]

# Request schema
class PromptRequest(BaseModel):
    input: str
    task_id: int = None  # Optional: force a specific task (0=crop, 1=fertilizer)

# Helper functions adapted from prompt.py
def analyze_query(query_text):
    """Extract intent and key information from the query."""
    import re
    
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
    else:
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
    
    # Extract soil parameters
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
    
    return analysis

def generate_features(query_text):
    """Generate feature vector from query text."""
    from sklearn.preprocessing import StandardScaler
    
    # Analyze the query
    query_analysis = analyze_query(query_text)
    
    # Initialize feature vector with default values
    feature_vector = [50.0, 30.0, 20.0, 6.5]  # Default N, P, K, pH
    
    # Update with any found soil parameters
    for i, param in enumerate(['n', 'p', 'k', 'ph']):
        if query_analysis['soil_params'][param] is not None:
            feature_vector[i] = query_analysis['soil_params'][param]
    
    # Adjust for specific crops
    if query_analysis['found_crop']:
        crop = query_analysis['found_crop']
        
        if crop == 'rice':
            if feature_vector[0] == 50.0:
                feature_vector[0] = 80.0
            if feature_vector[3] == 6.5:
                feature_vector[3] = 6.0
        
        elif crop in ['wheat', 'maize', 'corn']:
            if feature_vector[0] == 50.0:
                feature_vector[0] = 100.0
        
        # ... more crop-specific adjustments
    
    # Make sure the feature vector has the correct dimension
    if len(feature_vector) < input_dim:
        feature_vector.extend([0.0] * (input_dim - len(feature_vector)))
    elif len(feature_vector) > input_dim:
        feature_vector = feature_vector[:input_dim]
    
    # Normalize the feature vector
    scaler = StandardScaler()
    normalized_vector = scaler.fit_transform([feature_vector])[0]
    
    return torch.tensor(normalized_vector, dtype=torch.float32), query_analysis['task_id'], query_analysis

def get_crop_name(crop_id):
    """Map crop ID to human-readable crop name"""
    crop_names = {
        0: "Rice", 1: "Wheat", 2: "Maize", 3: "Soybean", 4: "Cotton",
        5: "Sugarcane", 6: "Potato", 7: "Tomato", 8: "Onion", 9: "Mustard"
    }
    return crop_names.get(crop_id, f"Crop {crop_id}")

def get_fertilizer_name(fert_id):
    """Map fertilizer ID to human-readable fertilizer name"""
    fertilizer_names = {
        0: "Urea", 1: "DAP", 2: "NPK", 3: "MOP", 4: "Ammonium Sulfate"
    }
    return fertilizer_names.get(fert_id, f"Fertilizer {fert_id}")

@app.post("/predict")
def get_prediction(data: PromptRequest):
    try:
        # Process input text to extract features
        features, auto_task_id, query_analysis = generate_features(data.input)
        
        # Use provided task_id if specified, otherwise use auto-detected
        task_id_to_use = data.task_id if data.task_id is not None else auto_task_id
        
        # Prepare tensors for prediction
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
        
        # Construct response based on task
        if task_id_to_use == 0:  # Crop recommendation
            recommendation = get_crop_name(crop_pred)
            confidence = crop_conf
            rec_type = "crop"
        else:  # Fertilizer recommendation
            recommendation = get_fertilizer_name(fert_pred)
            confidence = fert_conf
            rec_type = "fertilizer"
        
        # Return formatted response
        return {
            "recommendation": recommendation,
            "confidence": confidence,
            "type": rec_type,
            "query_analysis": {
                "found_crop": query_analysis.get("found_crop"),
                "found_fertilizer": query_analysis.get("found_fertilizer"),
                "soil_params": query_analysis.get("soil_params")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))