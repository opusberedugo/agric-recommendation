# api.py

from fastapi import FastAPI, HTTPException, Query, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union, Any
import torch
import os
import pickle
import numpy as np
import json
import re
import logging
from model import EnhancedAgriRecommender, AgriMultiTaskModel  # Fixed import path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Paths
MODEL_DIR = "models/agri_model"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")
MODEL_INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")

# Check if model files exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(MODEL_INFO_PATH):
    raise FileNotFoundError(f"Model files not found at {MODEL_PATH} or {MODEL_INFO_PATH}. Please train the model first.")

# Load model info
with open(MODEL_INFO_PATH, 'r') as f:
    model_info = json.load(f)

# Extract model parameters
input_dim = model_info["input_dim"]
crop_classes = model_info["crop_classes"]
fert_classes = model_info["fertilizer_classes"]
feature_names = model_info.get("feature_names", [])
crop_mapping = model_info.get("crop_mapping", {})
fert_mapping = model_info.get("fertilizer_mapping", {})
model_params = model_info.get("model_params", {})

# Convert string keys to integers in mappings
crop_mapping = {int(k): v for k, v in crop_mapping.items()}
fert_mapping = {int(k): v for k, v in fert_mapping.items()}

# If feature names not available, create default ones
if not feature_names:
    feature_names = [f"feature_{i}" for i in range(input_dim)]

# Initialize model with the same architecture as the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnhancedAgriRecommender(
    input_dim=input_dim,
    emb_dim=model_params.get("emb_dim", 128),
    hidden_dim=model_params.get("hidden_dim", 256),
    num_heads=model_params.get("num_heads", 4),
    crop_classes=crop_classes,
    fert_classes=fert_classes,
    num_layers=model_params.get("num_layers", 3),
    dropout=0.0  # Use 0 dropout for inference
)

# Load the trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Crop and fertilizer name mappings
CROP_NAMES = {i: name for i, name in crop_mapping.items()} if crop_mapping else {
    0: "Rice", 1: "Wheat", 2: "Maize", 3: "Soybean", 4: "Cotton", 
    5: "Sugarcane", 6: "Potato", 7: "Tomato", 8: "Onion", 9: "Mustard",
    10: "Barley", 11: "Chickpea", 12: "Cucumber", 13: "Lentil", 14: "Watermelon"
}

FERTILIZER_NAMES = {i: name for i, name in fert_mapping.items()} if fert_mapping else {
    0: "Urea", 1: "DAP", 2: "NPK", 3: "MOP", 4: "Ammonium Sulfate",
    5: "Super Phosphate", 6: "Calcium Nitrate", 7: "Micronutrient Mixture", 8: "Organic Compost"
}

# Common crops and fertilizers for text matching
COMMON_CROPS = [
    "rice", "wheat", "maize", "corn", "soybean", "cotton", 
    "sugarcane", "potato", "tomato", "onion", "mustard", "strawberry",
    "barley", "chickpea", "lentil", "cucumber", "watermelon"
]

COMMON_FERTILIZERS = [
    "urea", "dap", "npk", "mop", "ammonium sulfate", "ammonium sulphate", 
    "super phosphate", "potash", "calcium nitrate", "organic", "compost", "micronutrient"
]

# Default soil-crop recommendations (scientific background info)
CROP_RECOMMENDATIONS = {
    "rice": {
        "description": "Rice thrives in waterlogged conditions and requires high nitrogen.",
        "optimal_conditions": {
            "N": "80-100 kg/ha", "P": "30-50 kg/ha", "K": "30-40 kg/ha", "pH": "5.5-6.5",
            "temperature": "20-35°C", "humidity": "60-80%", "rainfall": "1000-1500 mm"
        }
    },
    "wheat": {
        "description": "Wheat prefers moderate moisture and high nitrogen for grain production.",
        "optimal_conditions": {
            "N": "100-120 kg/ha", "P": "50-60 kg/ha", "K": "25-30 kg/ha", "pH": "6.0-7.0",
            "temperature": "15-25°C", "humidity": "50-70%", "rainfall": "500-700 mm"
        }
    },
    "maize": {
        "description": "Maize needs good drainage and benefits from balanced NPK.",
        "optimal_conditions": {
            "N": "120-150 kg/ha", "P": "60-80 kg/ha", "K": "60-80 kg/ha", "pH": "5.8-7.0",
            "temperature": "18-30°C", "humidity": "50-75%", "rainfall": "500-800 mm"
        }
    },
    "potato": {
        "description": "Potatoes thrive in well-drained soil with high potassium.",
        "optimal_conditions": {
            "N": "100-120 kg/ha", "P": "80-100 kg/ha", "K": "100-150 kg/ha", "pH": "5.0-6.5",
            "temperature": "15-25°C", "humidity": "60-70%", "rainfall": "500-700 mm"
        }
    }
}

# FastAPI app setup
app = FastAPI(
    title="Deeproot AgriPredict API",
    description="Advanced agricultural recommendation API for crop and fertilizer recommendations",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class SoilParams(BaseModel):
    N: Optional[float] = Field(None, description="Nitrogen content (mg/kg)")
    P: Optional[float] = Field(None, description="Phosphorus content (mg/kg)")
    K: Optional[float] = Field(None, description="Potassium content (mg/kg)")
    pH: Optional[float] = Field(None, description="Soil pH level")
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    humidity: Optional[float] = Field(None, description="Humidity percentage")
    rainfall: Optional[float] = Field(None, description="Rainfall in mm")
    
    @validator('N', 'P', 'K', pre=True, always=True)
    def validate_nutrients(cls, v):
        if v is not None and v < 0:
            raise ValueError("Nutrient values must be non-negative")
        return v
    
    @validator('pH', pre=True, always=True)
    def validate_ph(cls, v):
        if v is not None and (v < 0 or v > 14):
            raise ValueError("pH must be between 0 and 14")
        return v
    
    @validator('humidity', pre=True, always=True)
    def validate_humidity(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError("Humidity must be between 0 and 100")
        return v

class PromptRequest(BaseModel):
    input: str = Field(..., description="Natural language query text")
    task_id: Optional[int] = Field(None, description="Optional task ID (0=crop, 1=fertilizer)")
    soil_params: Optional[SoilParams] = Field(None, description="Optional explicit soil parameters")

class RecommendationResponse(BaseModel):
    recommendation: str
    confidence: float
    type: str
    alternatives: List[Dict[str, Any]]
    soil_analysis: Dict[str, Any]
    explanation: str
    scientific_details: Optional[Dict[str, Any]] = None

# Helper functions
def analyze_query(query_text):
    """Extract intent and key information from the query."""
    query_text = query_text.lower()
    analysis = {
        'query_type': None,
        'found_crop': None,
        'found_fertilizer': None,
        'soil_params': {'n': None, 'p': None, 'k': None, 'ph': None, 'temperature': None, 'humidity': None, 'rainfall': None},
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
    
    # Extract soil parameters using regex patterns
    param_patterns = {
        'n': [r'(?:nitrogen|n)\s*(?::|is|=)\s*(\d+\.?\d*)', r'n\s*[-:=]\s*(\d+\.?\d*)'],
        'p': [r'(?:phosphorus|p)\s*(?::|is|=)\s*(\d+\.?\d*)', r'p\s*[-:=]\s*(\d+\.?\d*)'],
        'k': [r'(?:potassium|k)\s*(?::|is|=)\s*(\d+\.?\d*)', r'k\s*[-:=]\s*(\d+\.?\d*)'],
        'ph': [r'(?:ph)\s*(?::|is|=)\s*(\d+\.?\d*)', r'ph\s*[-:=]\s*(\d+\.?\d*)'],
        'temperature': [r'(?:temperature|temp)\s*(?::|is|=)\s*(\d+\.?\d*)', r'temperature\s*[-:=]\s*(\d+\.?\d*)'],
        'humidity': [r'(?:humidity|hum)\s*(?::|is|=)\s*(\d+\.?\d*)', r'humidity\s*[-:=]\s*(\d+\.?\d*)'],
        'rainfall': [r'(?:rainfall|rain)\s*(?::|is|=)\s*(\d+\.?\d*)', r'rainfall\s*[-:=]\s*(\d+\.?\d*)']
    }
    
    for param, patterns in param_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, query_text)
            if match:
                analysis['soil_params'][param] = float(match.group(1))
                break
    
    # Check for contextual clues about soil quality
    if 'acidic' in query_text and analysis['soil_params']['ph'] is None:
        analysis['soil_params']['ph'] = 5.5  # Approximate acidic pH
    elif 'alkaline' in query_text and analysis['soil_params']['ph'] is None:
        analysis['soil_params']['ph'] = 8.0  # Approximate alkaline pH
        
    if 'fertile' in query_text and analysis['soil_params']['n'] is None:
        analysis['soil_params']['n'] = 80.0  # Approximate fertile soil N level
        
    if 'dry' in query_text and analysis['soil_params']['humidity'] is None:
        analysis['soil_params']['humidity'] = 30.0  # Approximate dry conditions
    elif 'humid' in query_text and analysis['soil_params']['humidity'] is None:
        analysis['soil_params']['humidity'] = 75.0  # Approximate humid conditions
    
    return analysis

def generate_features(query_text, explicit_params=None):
    """Generate feature vector from query text and optional explicit parameters."""
    # Analyze the query
    query_analysis = analyze_query(query_text)
    
    # Initialize feature vector with default values
    feature_vector = [50.0, 30.0, 20.0, 6.5, 25.0, 65.0, 1000.0]  # Default N, P, K, pH, temp, humidity, rainfall
    
    # Update with any found soil parameters from text
    analysis_params = query_analysis['soil_params']
    params_to_update = ['n', 'p', 'k', 'ph', 'temperature', 'humidity', 'rainfall']
    
    for i, param in enumerate(params_to_update):
        if analysis_params[param] is not None:
            feature_vector[i] = analysis_params[param]
    
    # Override with explicit parameters if provided
    if explicit_params:
        if explicit_params.N is not None:
            feature_vector[0] = explicit_params.N
        if explicit_params.P is not None:
            feature_vector[1] = explicit_params.P
        if explicit_params.K is not None:
            feature_vector[2] = explicit_params.K
        if explicit_params.pH is not None:
            feature_vector[3] = explicit_params.pH
        if explicit_params.temperature is not None:
            feature_vector[4] = explicit_params.temperature
        if explicit_params.humidity is not None:
            feature_vector[5] = explicit_params.humidity
        if explicit_params.rainfall is not None:
            feature_vector[6] = explicit_params.rainfall
    
    # Adjust for specific crops if found in query
    if query_analysis['found_crop']:
        crop = query_analysis['found_crop']
        
        # Use knowledge-based heuristics for crop-specific adjustments
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
                
        elif crop == 'cotton':
            if feature_vector[1] == 30.0:
                feature_vector[1] = 50.0    # Cotton needs more phosphorus
                
        elif crop == 'sugarcane':
            if feature_vector[0] == 50.0:
                feature_vector[0] = 120.0   # Sugarcane needs high nitrogen
            if feature_vector[2] == 20.0:
                feature_vector[2] = 80.0    # And high potassium
    
    # Make sure the feature vector has the correct dimension
    if len(feature_vector) < input_dim:
        feature_vector.extend([0.0] * (input_dim - len(feature_vector)))
    elif len(feature_vector) > input_dim:
        feature_vector = feature_vector[:input_dim]
    
    # Normalize the feature vector using simple standardization
    # We use a heuristic approach since we don't have the training set statistics
    # These values approximate typical ranges for agricultural soil parameters
    means = [80.0, 40.0, 40.0, 6.5, 25.0, 60.0, 1000.0]
    stds = [40.0, 20.0, 20.0, 1.0, 8.0, 20.0, 500.0]
    
    # Only normalize the dimensions we know about
    normalize_dims = min(len(feature_vector), len(means))
    normalized_vector = feature_vector.copy()
    
    for i in range(normalize_dims):
        normalized_vector[i] = (feature_vector[i] - means[i]) / stds[i]
    
    return torch.tensor(normalized_vector, dtype=torch.float32), query_analysis['task_id'], query_analysis

def get_crop_name(crop_id):
    """Map crop ID to human-readable crop name"""
    return CROP_NAMES.get(crop_id, f"Crop {crop_id}")

def get_fertilizer_name(fert_id):
    """Map fertilizer ID to human-readable fertilizer name"""
    return FERTILIZER_NAMES.get(fert_id, f"Fertilizer {fert_id}")

def generate_explanation(recommendation_type, recommendation, soil_params, found_crop=None):
    """Generate a natural language explanation for the recommendation"""
    
    if recommendation_type == "crop":
        crop_name = recommendation
        
        # Start with a basic explanation
        explanation = f"{crop_name} is recommended based on the soil parameters provided."
        
        # Add specific details based on soil parameters
        details = []
        if soil_params.get('n') is not None:
            if soil_params['n'] > 80:
                details.append(f"The high nitrogen level ({soil_params['n']} mg/kg) is suitable for {crop_name}, which is a nitrogen-loving crop.")
            elif soil_params['n'] < 40:
                if crop_name.lower() in ["soybean", "chickpea", "lentil"]:
                    details.append(f"The lower nitrogen level ({soil_params['n']} mg/kg) works well for {crop_name}, which can fix its own nitrogen from the atmosphere.")
                else:
                    details.append(f"The nitrogen level ({soil_params['n']} mg/kg) is on the lower side, but {crop_name} can still perform well with proper fertilization.")
        
        if soil_params.get('ph') is not None:
            if 5.5 <= soil_params['ph'] <= 7.0:
                details.append(f"The soil pH of {soil_params['ph']} is in the ideal range for {crop_name}.")
            elif soil_params['ph'] < 5.5:
                details.append(f"The soil is acidic (pH {soil_params['ph']}), which {crop_name} can tolerate, though liming might improve yields.")
            else:
                details.append(f"The soil is alkaline (pH {soil_params['ph']}), which {crop_name} can adapt to, though some amendments might help.")
                
        # Add details about weather conditions if available
        if soil_params.get('temperature') is not None:
            if 20 <= soil_params['temperature'] <= 30:
                details.append(f"The temperature of {soil_params['temperature']}°C provides good growing conditions for {crop_name}.")
            elif soil_params['temperature'] < 20:
                details.append(f"The temperature of {soil_params['temperature']}°C is cool, but {crop_name} can still thrive in these conditions.")
            else:
                details.append(f"The temperature of {soil_params['temperature']}°C is warm, but {crop_name} is heat-tolerant.")
        
        # Add more details about the recommended crop
        if crop_name.lower() == "rice":
            details.append("Rice thrives in water-logged conditions and requires good water management.")
        elif crop_name.lower() == "wheat":
            details.append("Wheat is adaptable to various soil types but requires good drainage and moderate fertility.")
        elif crop_name.lower() in ["maize", "corn"]:
            details.append("Maize/Corn has deep roots and benefits from well-drained soils with good organic matter content.")
        elif crop_name.lower() == "potato":
            details.append("Potatoes prefer loose, well-drained soils and benefit from high potassium levels for tuber development.")
        
        # Combine details into a coherent explanation
        if details:
            explanation += " " + " ".join(details)
        
    else:  # fertilizer recommendation
        fert_name = recommendation
        
        # Start with a basic explanation
        if found_crop:
            explanation = f"{fert_name} is recommended for {found_crop} based on the soil analysis."
        else:
            explanation = f"{fert_name} is recommended based on the soil parameters provided."
        
        # Add specific details based on soil parameters and fertilizer type
        details = []
        if fert_name.lower() == "urea" or "nitrogen" in fert_name.lower():
            if soil_params.get('n') is not None and soil_params['n'] < 60:
                details.append(f"The soil shows lower nitrogen levels ({soil_params['n']} mg/kg), and {fert_name} will provide the necessary nitrogen supplement.")
            else:
                details.append(f"{fert_name} provides essential nitrogen for plant growth and leaf development.")
                
        elif fert_name.lower() == "dap" or "phosphate" in fert_name.lower() or "phosphorus" in fert_name.lower():
            if soil_params.get('p') is not None and soil_params['p'] < 25:
                details.append(f"The soil is deficient in phosphorus ({soil_params['p']} mg/kg), and {fert_name} will boost root development and flowering.")
            else:
                details.append(f"{fert_name} provides phosphorus which is crucial for root development, flowering, and fruiting.")
                
        elif fert_name.lower() == "mop" or "potash" in fert_name.lower() or "potassium" in fert_name.lower():
            if soil_params.get('k') is not None and soil_params['k'] < 25:
                details.append(f"The soil has lower potassium levels ({soil_params['k']} mg/kg), and {fert_name} will improve crop quality and stress resistance.")
            else:
                details.append(f"{fert_name} provides potassium which strengthens plants, improves quality, and increases resistance to pests and diseases.")
                
        elif fert_name.lower() == "npk":
            details.append(f"{fert_name} provides a balanced mixture of nitrogen, phosphorus, and potassium, addressing multiple nutrient needs simultaneously.")
        
        # Add advice based on pH if available
        if soil_params.get('ph') is not None:
            if soil_params['ph'] < 5.5:
                details.append(f"Since your soil is acidic (pH {soil_params['ph']}), consider using lime alongside {fert_name} to improve nutrient availability.")
            elif soil_params['ph'] > 7.5:
                details.append(f"Since your soil is alkaline (pH {soil_params['ph']}), organic matter additions alongside {fert_name} may improve its effectiveness.")