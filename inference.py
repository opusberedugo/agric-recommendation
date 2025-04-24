# enhanced_inference.py
"""
Enhanced inference script for agricultural recommendations using the trained model.
This script provides more advanced features and better explanations compared to the basic inference.
"""

import torch
import argparse
import os
import json
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
import logging
from tqdm import tqdm
import sys
from enhanced_model import EnhancedAgriRecommender, AgriMultiTaskModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Path configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "agri_model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")
MODEL_INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")

# Constants for natural language processing
COMMON_CROPS = [
    "rice", "wheat", "maize", "corn", "soybean", "cotton", 
    "sugarcane", "potato", "tomato", "onion", "mustard", "strawberry",
    "barley", "chickpea", "lentil", "cucumber", "watermelon"
]

COMMON_FERTILIZERS = [
    "urea", "dap", "npk", "mop", "ammonium sulfate", "ammonium sulphate", 
    "super phosphate", "potash", "calcium nitrate", "organic", "compost", "micronutrient"
]

# Crop-fertilizer knowledge base
CROP_FERTILIZER_GUIDELINES = {
    "rice": {
        "recommended_npk": "120-60-40",
        "nitrogen_timing": "40% as basal, 30% at tillering, 30% at panicle initiation",
        "specific_needs": "Zinc is often required in rice cultivation"
    },
    "wheat": {
        "recommended_npk": "120-60-40",
        "nitrogen_timing": "50% as basal, 25% at crown root initiation, 25% at tillering",
        "specific_needs": "Responds well to micronutrients like zinc and iron"
    },
    "maize": {
        "recommended_npk": "150-75-75",
        "nitrogen_timing": "30% as basal, 40% at knee-high stage, 30% at tasseling",
        "specific_needs": "Requires good zinc availability for kernel development"
    },
    "potato": {
        "recommended_npk": "100-150-100",
        "nitrogen_timing": "40% as basal, 60% at earthing up",
        "specific_needs": "Higher potassium needs for tuber formation"
    },
    "tomato": {
        "recommended_npk": "100-80-100",
        "nitrogen_timing": "30% as basal, 40% at flowering, 30% at fruit set",
        "specific_needs": "Calcium is important to prevent blossom end rot"
    },
    "cotton": {
        "recommended_npk": "120-60-60",
        "nitrogen_timing": "40% as basal, 30% at squaring, 30% at flowering",
        "specific_needs": "Boron is often required for proper boll development"
    },
    "soybean": {
        "recommended_npk": "30-80-40",
        "nitrogen_timing": "All as basal (can fix own N if inoculated)",
        "specific_needs": "Benefits from molybdenum for nitrogen fixation"
    }
}

def load_model_info(model_path=None):
    """Load model info from JSON file"""
    info_path = model_path or MODEL_INFO_PATH
    
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Model info file not found at {info_path}")
    
    with open(info_path, 'r') as f:
        model_info = json.load(f)
    
    return model_info

def load_model(model_path=None, model_info_path=None, device="cpu"):
    """Load the trained model and necessary metadata"""
    # Load model info
    model_info = load_model_info(model_info_path)
    
    # Extract model parameters
    input_dim = model_info["input_dim"]
    crop_classes = model_info["crop_classes"]
    fert_classes = model_info["fertilizer_classes"]
    model_params = model_info.get("model_params", {})
    crop_mapping = model_info.get("crop_mapping", {})
    fert_mapping = model_info.get("fertilizer_mapping", {})
    
    # Convert string keys to integers
    crop_mapping = {int(k): v for k, v in crop_mapping.items()} if crop_mapping else {}
    fert_mapping = {int(k): v for k, v in fert_mapping.items()} if fert_mapping else {}
    
    # Default mappings if not available in model info
    if not crop_mapping:
        crop_mapping = {
            0: "Rice", 1: "Wheat", 2: "Maize", 3: "Soybean", 4: "Cotton", 
            5: "Sugarcane", 6: "Potato", 7: "Tomato", 8: "Onion", 9: "Mustard"
        }
    
    if not fert_mapping:
        fert_mapping = {
            0: "Urea", 1: "DAP", 2: "NPK", 3: "MOP", 4: "Ammonium Sulfate"
        }
    
    # Initialize model
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
    
    # Load weights
    path = model_path or MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, crop_mapping, fert_mapping, model_info

def analyze_query(query_text):
    """
    Analyze a natural language query to extract relevant information.
    """
    query_text = query_text.lower()
    analysis = {
        'query_type': None,
        'found_crop': None,
        'found_fertilizer': None,
        'soil_params': {'n': None, 'p': None, 'k': None, 'ph': None, 'temperature': None, 'humidity': None, 'rainfall': None},
        'task_id': None,
        'location': None,
        'season': None
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
        'temperature': [r'(?:temperature|temp)\s*(?::|is|=)\s*(\d+\.?\d*)', r'(?:temperature|temp)(?:\s+of)?\s+(\d+\.?\d*)'],
        'humidity': [r'(?:humidity|hum)\s*(?::|is|=)\s*(\d+\.?\d*)', r'(?:humidity|hum)(?:\s+of)?\s+(\d+\.?\d*)'],
        'rainfall': [r'(?:rainfall|rain)\s*(?::|is|=)\s*(\d+\.?\d*)', r'(?:rainfall|rain)(?:\s+of)?\s+(\d+\.?\d*)']
    }
    
    for param, patterns in param_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, query_text)
            if match:
                analysis['soil_params'][param] = float(match.group(1))
                break
    
    # Look for NPK pattern (e.g., "NPK 14-14-14")
    npk_match = re.search(r'npk\s*(\d+)[-\s]*(\d+)[-\s]*(\d+)', query_text)
    if npk_match and not analysis['found_fertilizer']:
        analysis['found_fertilizer'] = 'npk'
        # Extract NPK values if not already set
        if analysis['soil_params']['n'] is None:
            analysis['soil_params']['n'] = float(npk_match.group(1))
        if analysis['soil_params']['p'] is None:
            analysis['soil_params']['p'] = float(npk_match.group(2))
        if analysis['soil_params']['k'] is None:
            analysis['soil_params']['k'] = float(npk_match.group(3))
    
    # Detect location mentions
    location_patterns = [
        r'in\s+([A-Za-z\s]+)(?:region|state|province|district|area)',
        r'(?:region|state|province|district|area)\s+of\s+([A-Za-z\s]+)',
        r'(?:location|place)(?:\s+is)?\s+([A-Za-z\s]+)'
    ]
    
    for pattern in location_patterns:
        match = re.search(pattern, query_text)
        if match:
            analysis['location'] = match.group(1).strip()
            break
    
    # Detect season mentions
    seasons = ['summer', 'winter', 'spring', 'autumn', 'fall', 'monsoon', 'rainy', 'dry']
    for season in seasons:
        if season in query_text:
            analysis['season'] = season
            break
    
    # If a fertilizer question with a specific crop was asked, make the task fertilizer
    if analysis['found_crop'] and 'fertilizer' in query_text:
        analysis['task_id'] = 1  # Set to fertilizer task
    
    return analysis

def generate_features(query_text, soil_params=None):
    """
    Generate feature vector from natural language query or explicit soil parameters.
    
    Args:
        query_text (str): Natural language query text
        soil_params (dict, optional): Explicitly provided soil parameters
    
    Returns:
        tuple: (feature_tensor, task_id, query_analysis)
    """
    # Analyze query
    query_analysis = analyze_query(query_text)
    
    # Initialize feature vector with default values
    # [N, P, K, pH, temperature, humidity, rainfall]
    feature_vector = [50.0, 30.0, 20.0, 6.5, 25.0, 65.0, 1000.0]
    
    # Update with values from query analysis
    analysis_params = query_analysis['soil_params']
    for i, param in enumerate(['n', 'p', 'k', 'ph', 'temperature', 'humidity', 'rainfall']):
        if analysis_params[param] is not None:
            feature_vector[i] = analysis_params[param]
    
    # Update with explicit parameters if provided
    if soil_params:
        for i, param in enumerate(['n', 'p', 'k', 'ph', 'temperature', 'humidity', 'rainfall']):
            if param in soil_params and soil_params[param] is not None:
                feature_vector[i] = soil_params[param]
    
    # Apply crop-specific adjustments if a crop was mentioned
    if query_analysis['found_crop']:
        crop = query_analysis['found_crop']
        
        # Apply domain knowledge about crop requirements
        if crop == 'rice':
            if feature_vector[0] == 50.0:  # N
                feature_vector[0] = 80.0  # Rice needs more nitrogen
            if feature_vector[3] == 6.5:  # pH
                feature_vector[3] = 6.0  # Rice prefers slightly acidic soil
            if feature_vector[6] == 1000.0:  # Rainfall
                feature_vector[6] = 1500.0  # Rice needs more water
        
        elif crop in ['wheat', 'maize', 'corn']:
            if feature_vector[0] == 50.0:  # N
                feature_vector[0] = 100.0  # Grain crops need high nitrogen
            if feature_vector[3] == 6.5:  # pH
                feature_vector[3] = 6.8  # Prefer slightly alkaline soil
        
        elif crop in ['potato', 'tomato']:
            if feature_vector[2] == 20.0:  # K
                feature_vector[2] = 60.0  # Root/fruit crops need more potassium
            if feature_vector[3] == 6.5:  # pH
                feature_vector[3] = 5.8  # Prefer slightly acidic soil
        
        elif crop == 'soybean':
            if feature_vector[0] == 50.0:  # N
                feature_vector[0] = 30.0  # Legumes need less nitrogen
            if feature_vector[1] == 30.0:  # P
                feature_vector[1] = 60.0  # But need more phosphorus
    
    # Apply season-specific adjustments if mentioned
    if query_analysis['season']:
        season = query_analysis['season']
        
        if season in ['summer', 'dry']:
            if feature_vector[4] == 25.0:  # Temperature
                feature_vector[4] = 30.0  # Higher temperature
            if feature_vector[5] == 65.0:  # Humidity
                feature_vector[5] = 50.0  # Lower humidity
            if feature_vector[6] == 1000.0:  # Rainfall
                feature_vector[6] = 500.0  # Lower rainfall
        
        elif season in ['winter']:
            if feature_vector[4] == 25.0:  # Temperature
                feature_vector[4] = 15.0  # Lower temperature
        
        elif season in ['monsoon', 'rainy']:
            if feature_vector[5] == 65.0:  # Humidity
                feature_vector[5] = 80.0  # Higher humidity
            if feature_vector[6] == 1000.0:  # Rainfall
                feature_vector[6] = 2000.0  # Higher rainfall
    
    # Normalize the features using simple standardization
    # These are approximate means and standard deviations for agricultural parameters
    means = [80.0, 40.0, 40.0, 6.5, 25.0, 60.0, 1000.0]
    stds = [40.0, 20.0, 20.0, 1.0, 8.0, 20.0, 500.0]
    
    normalized_vector = []
    for i, val in enumerate(feature_vector[:len(means)]):
        normalized_vector.append((val - means[i]) / stds[i])
    
    # Fill remaining features with zeros if needed
    if len(normalized_vector) < input_dim:
        normalized_vector.extend([0.0] * (input_dim - len(normalized_vector)))
    
    return torch.tensor(normalized_vector, dtype=torch.float32), query_analysis['task_id'], query_analysis

def predict_recommendation(model, query_text, soil_params=None, task_id=None, device="cpu"):
    """
    Generate agricultural recommendations from a natural language query.
    
    Args:
        model: The trained recommendation model
        query_text (str): Natural language query
        soil_params (dict, optional): Explicit soil parameters
        task_id (int, optional): Force a specific task (0=crop, 1=fertilizer)
        device (str): Computation device
    
    Returns:
        dict: Prediction results and analysis
    """
    try:
        # Generate features
        features, auto_task_id, query_analysis = generate_features(query_text, soil_params)
        
        # Use provided task_id if specified, otherwise use auto-detected
        task_id_to_use = task_id if task_id is not None else auto_task_id
        
        # Prepare tensors
        features = features.unsqueeze(0).to(device)
        task_id_tensor = torch.tensor(task_id_to_use, dtype=torch.long).unsqueeze(0).to(device)
        
        # Run prediction
        with torch.no_grad():
            outputs = model(features, task_id_tensor)
        
        # Get probabilities
        crop_probs = torch.nn.functional.softmax(outputs['crop'], dim=1)
        fert_probs = torch.nn.functional.softmax(outputs['fertilizer'], dim=1)
        
        # Get predictions
        crop_pred = torch.argmax(outputs['crop'], dim=1).item()
        fert_pred = torch.argmax(outputs['fertilizer'], dim=1).item()
        
        crop_conf = crop_probs[0, crop_pred].item()
        fert_conf = fert_probs[0, fert_pred].item()
        
        # Get top k predictions
        k = 3  # Number of alternatives to provide
        _, top_crop_indices = torch.topk(crop_probs, min(k, crop_probs.size(1)), dim=1)
        _, top_fert_indices = torch.topk(fert_probs, min(k, fert_probs.size(1)), dim=1)
        
        top_crops = [(idx.item(), crop_probs[0, idx].item()) for idx in top_crop_indices[0]]
        top_ferts = [(idx.item(), fert_probs[0, idx].item()) for idx in top_fert_indices[0]]
        
        # Get embeddings for future similarity calculations
        embeddings = outputs['embedding'].cpu().numpy()
        
        # Return comprehensive results
        return {
            "recommendation_type": "crop" if task_id_to_use == 0 else "fertilizer",
            "crop_prediction": crop_pred,
            "crop_confidence": crop_conf,
            "fertilizer_prediction": fert_pred,
            "fertilizer_confidence": fert_conf,
            "top_crops": top_crops,
            "top_fertilizers": top_ferts,
            "query_analysis": query_analysis,
            "embeddings": embeddings,
            "task_id": task_id_to_use
        }
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        raise

def generate_explanation(result, crop_mapping, fert_mapping):
    """
    Generate a detailed explanation for the recommendation.
    
    Args:
        result (dict): Prediction results
        crop_mapping (dict): Mapping from crop IDs to names
        fert_mapping (dict): Mapping from fertilizer IDs to names
    
    Returns:
        str: Detailed explanation
    """
    # Get the recommendation type and prediction
    rec_type = result["recommendation_type"]
    query_analysis = result["query_analysis"]
    soil_params = query_analysis["soil_params"]
    
    if rec_type == "crop":
        crop_id = result["crop_prediction"]
        crop_name = crop_mapping.get(crop_id, f"Crop {crop_id}")
        confidence = result["crop_confidence"] * 100
        
        explanation = [
            f"Based on your soil parameters, I recommend growing {crop_name} (confidence: {confidence:.1f}%)."
        ]
        
        # Add details about soil parameters
        soil_details = []
        if soil_params['n'] is not None:
            soil_details.append(f"Nitrogen level: {soil_params['n']} mg/kg")
        if soil_params['p'] is not None:
            soil_details.append(f"Phosphorus level: {soil_params['p']} mg/kg")
        if soil_params['k'] is not None:
            soil_details.append(f"Potassium level: {soil_params['k']} mg/kg")
        if soil_params['ph'] is not None:
            soil_details.append(f"pH level: {soil_params['ph']}")
        
        if soil_details:
            explanation.append("Soil analysis:")
            explanation.extend(["  - " + detail for detail in soil_details])
        
        # Add crop-specific recommendations if available
        crop_key = crop_name.lower()
        if crop_key in CROP_FERTILIZER_GUIDELINES:
            guidelines = CROP_FERTILIZER_GUIDELINES[crop_key]
            explanation.append(f"\nRecommended fertilizer for {crop_name}:")
            explanation.append(f"  - NPK ratio: {guidelines['recommended_npk']}")
            explanation.append(f"  - Application timing: {guidelines['nitrogen_timing']}")
            explanation.append(f"  - Special needs: {guidelines['specific_needs']}")
        
        # Add alternative crops
        if len(result["top_crops"]) > 1:
            alternatives = []
            for idx, (crop_id, conf) in enumerate(result["top_crops"][1:], 1):
                crop = crop_mapping.get(crop_id, f"Crop {crop_id}")
                alternatives.append(f"{crop} ({conf*100:.1f}%)")
            
            explanation.append("\nAlternative crops that would also grow well:")
            explanation.extend(["  - " + alt for alt in alternatives])
        
    else:  # fertilizer recommendation
        fert_id = result["fertilizer_prediction"]
        fert_name = fert_mapping.get(fert_id, f"Fertilizer {fert_id}")
        confidence = result["fertilizer_confidence"] * 100
        
        explanation = [
            f"Based on your soil parameters, I recommend using {fert_name} (confidence: {confidence:.1f}%)."
        ]
        
        # Add soil details
        soil_details = []
        if soil_params['n'] is not None:
            soil_details.append(f"Nitrogen level: {soil_params['n']} mg/kg")
            if soil_params['n'] < 40:
                soil_details.append("  - This is low; nitrogen supplementation is needed")
            elif soil_params['n'] > 80:
                soil_details.append("  - This is high; reduce nitrogen application")
        
        if soil_params['p'] is not None:
            soil_details.append(f"Phosphorus level: {soil_params['p']} mg/kg")
            if soil_params['p'] < 20:
                soil_details.append("  - This is low; phosphorus supplementation is needed")
            elif soil_params['p'] > 60:
                soil_details.append("  - This is high; reduce phosphorus application")
        
        if soil_params['k'] is not None:
            soil_details.append(f"Potassium level: {soil_params['k']} mg/kg")
            if soil_params['k'] < 20:
                soil_details.append("  - This is low; potassium supplementation is needed")
            elif soil_params['k'] > 60:
                soil_details.append("  - This is high; reduce potassium application")
        
        if soil_params['ph'] is not None:
            soil_details.append(f"pH level: {soil_params['ph']}")
            if soil_params['ph'] < 5.5:
                soil_details.append("  - This is acidic; consider liming to raise pH")
            elif soil_params['ph'] > 7.5:
                soil_details.append("  - This is alkaline; consider adding organic matter to lower pH")
        
        if soil_details:
            explanation.append("\nSoil analysis:")
            explanation.extend(soil_details)
        
        # Add fertilizer-specific recommendations
        if fert_name.lower() == "urea":
            explanation.append("\nApplication guidelines:")
            explanation.append("  - High nitrogen content (46-0-0)")
            explanation.append("  - Apply in split doses for better efficiency")
            explanation.append("  - Keep soil moist after application to prevent volatilization")
        
        elif fert_name.lower() == "dap":
            explanation.append("\nApplication guidelines:")
            explanation.append("  - Contains both nitrogen and phosphorus (18-46-0)")
            explanation.append("  - Good starter fertilizer for most crops")
            explanation.append("  - Apply at planting or during land preparation")
        
        elif fert_name.lower() == "npk":
            explanation.append("\nApplication guidelines:")
            explanation.append("  - Balanced nutrition with nitrogen, phosphorus, and potassium")
            explanation.append("  - Apply based on specific crop requirements")
            explanation.append("  - Consider doing a soil test to determine specific NPK formula needed")
        
        # Add crop-specific fertilizer recommendations if a crop was mentioned
        if query_analysis['found_crop'] and query_analysis['found_crop'] in CROP_FERTILIZER_GUIDELINES:
            crop = query_analysis['found_crop']
            guidelines = CROP_FERTILIZER_GUIDELINES[crop]
            explanation.append(f"\nSpecific recommendations for {crop}:")
            explanation.append(f"  - Recommended NPK ratio: {guidelines['recommended_npk']}")
            explanation.append(f"  - Application timing: {guidelines['nitrogen_timing']}")
            
        # Add alternative fertilizers
        if len(result["top_fertilizers"]) > 1:
            alternatives = []
            for fert_id, conf in result["top_fertilizers"][1:]:
                fert = fert_mapping.get(fert_id, f"Fertilizer {fert_id}")
                alternatives.append(f"{fert} ({conf*100:.1f}%)")
            
            explanation.append("\nAlternative fertilizers that would also work well:")
            explanation.extend(["  - " + alt for alt in alternatives])
    
    return "\n".join(explanation)

def visualize_prediction(result, crop_mapping, fert_mapping, save_path=None):
    """
    Create visualizations for the prediction results.
    
    Args:
        result (dict): Prediction results
        crop_mapping (dict): Mapping from crop IDs to names
        fert_mapping (dict): Mapping from fertilizer IDs to names
        save_path (str, optional): Path to save the visualization
    """
    plt.figure(figsize=(15, 10))
    
    # Create a 2x2 grid of subplots
    plt.subplot(2, 2, 1)
    
    # Visualize top crop predictions
    crop_ids = [crop_id for crop_id, _ in result["top_crops"][:5]]
    crop_confs = [conf for _, conf in result["top_crops"][:5]]
    crop_names = [crop_mapping.get(crop_id, f"Crop {crop_id}") for crop_id in crop_ids]
    
    bars = plt.bar(crop_names, crop_confs)
    plt.title('Top Crop Recommendations')
    plt.ylabel('Confidence')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    
    # Color the bars (green for the top pick)
    for i, bar in enumerate(bars):
        if i == 0:
            bar.set_color('green')
        else:
            bar.set_color('lightgreen')
    
    # Visualize top fertilizer predictions
    plt.subplot(2, 2, 2)
    fert_ids = [fert_id for fert_id, _ in result["top_fertilizers"][:5]]
    fert_confs = [conf for _, conf in result["top_fertilizers"][:5]]
    fert_names = [fert_mapping.get(fert_id, f"Fertilizer {fert_id}") for fert_id in fert_ids]
    
    bars = plt.bar(fert_names, fert_confs)
    plt.title('Top Fertilizer Recommendations')
    plt.ylabel('Confidence')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    
    # Color the bars (blue for the top pick)
    for i, bar in enumerate(bars):
        if i == 0:
            bar.set_color('blue')
        else:
            bar.set_color('lightblue')
    
    # Visualize soil parameters
    plt.subplot(2, 1, 2)
    soil_params = result["query_analysis"]["soil_params"]
    param_names = ['N', 'P', 'K', 'pH', 'Temp', 'Humidity', 'Rainfall']
    param_values = [
        soil_params['n'] or 0, 
        soil_params['p'] or 0, 
        soil_params['k'] or 0,
        soil_params['ph'] or 0,
        soil_params['temperature'] or 0,
        soil_params['humidity'] or 0,
        soil_params['rainfall'] or 0
    ]
    
    # Normalize values for better visualization
    max_vals = [150, 100, 100, 14, 40, 100, 2000]
    normalized_values = [val / max_val for val, max_val in zip(param_values, max_vals)]
    
    bars = plt.bar(param_names, normalized_values)
    plt.title('Soil Parameters (Normalized)')
    plt.ylabel('Value (normalized)')
    plt.ylim(0, 1.0)
    
    # Color the bars based on parameter type
    colors = ['#8B4513', '#A0522D', '#D2691E', '#CD853F', '#FFD700', '#87CEEB', '#1E90FF']
    for i, bar in enumerate(bars):
        bar.set_color(colors[i])
    
    # Add actual values as text on bars
    for i, v in enumerate(param_values):
        if v > 0:  # Only show if value is present
            plt.text(i, normalized_values[i] + 0.02, f"{v:.1f}", ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Visualization saved to {save_path}")
    else:
        plt.show()

def process_csv_input(file_path, model, crop_mapping, fert_mapping, output_path=None, task_id=None, device="cpu"):
    """
    Process a CSV file with soil parameters and generate recommendations for each row.
    
    Args:
        file_path (str): Path to the CSV file
        model: The trained model
        crop_mapping (dict): Mapping from crop IDs to names
        fert_mapping (dict): Mapping from fertilizer IDs to names
        output_path (str, optional): Path to save results
        task_id (int, optional): Force a specific task
        device (str): Computation device
    
    Returns:
        pd.DataFrame: DataFrame with original data and recommendations
    """
    try:
        # Load CSV
        df = pd.read_csv(file_path)
        logging.info(f"Loaded CSV with {len(df)} rows and columns: {df.columns.tolist()}")
        
        # Check for required columns
        required_columns = ['n', 'p', 'k', 'ph']
        lowercase_columns = [col.lower() for col in df.columns]
        
        # Map between common column name variations and standard names
        column_mapping = {
            'n': ['n', 'nitrogen', 'n_content'],
            'p': ['p', 'phosphorus', 'p_content', 'p2o5'],
            'k': ['k', 'potassium', 'k_content', 'k2o'],
            'ph': ['ph', 'soil_ph', 'ph_value'],
            'temperature': ['temperature', 'temp', 'air_temperature'],
            'humidity': ['humidity', 'hum', 'air_humidity'],
            'rainfall': ['rainfall', 'rain', 'precipitation', 'annual_rainfall']
        }
        
        # Standardize column names
        renamed_columns = {}
        for std_name, variations in column_mapping.items():
            for col in df.columns:
                if col.lower() in variations:
                    renamed_columns[col] = std_name
        
        # Rename columns if needed
        if renamed_columns:
            df = df.rename(columns=renamed_columns)
        
        # Check if we have the minimum required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.warning(f"Missing required columns: {missing_columns}")
            logging.warning("Will use default values for missing columns")
        
        # Initialize results DataFrame
        results_df = df.copy()
        results_df['crop_recommendation'] = None
        results_df['crop_confidence'] = None
        results_df['fertilizer_recommendation'] = None
        results_df['fertilizer_confidence'] = None
        results_df['alternative_crops'] = None
        results_df['alternative_fertilizers'] = None
        
        # Process each row
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            # Extract soil parameters
            soil_params = {}
            for param in ['n', 'p', 'k', 'ph', 'temperature', 'humidity', 'rainfall']:
                if param in row:
                    soil_params[param] = row[param]
            
            # Generate a dummy query for feature generation
            query = "Recommend crops for my soil"
            if soil_params.get('n') is not None:
                query += f" with nitrogen {soil_params['n']}"
            if soil_params.get('p') is not None:
                query += f", phosphorus {soil_params['p']}"
            if soil_params.get('k') is not None:
                query += f", potassium {soil_params['k']}"
            if soil_params.get('ph') is not None:
                query += f", pH {soil_params['ph']}"
            
            # Make prediction
            try:
                result = predict_recommendation(
                    model=model, 
                    query_text=query, 
                    soil_params=soil_params,
                    task_id=task_id,
                    device=device
                )
                
                # Extract results
                crop_id = result['crop_prediction']
                crop_name = crop_mapping.get(crop_id, f"Crop {crop_id}")
                crop_confidence = result['crop_confidence']
                
                fert_id = result['fertilizer_prediction']
                fert_name = fert_mapping.get(fert_id, f"Fertilizer {fert_id}")
                fert_confidence = result['fertilizer_confidence']
                
                # Get alternatives
                alt_crops = []
                for alt_id, alt_conf in result['top_crops'][1:3]:  # Top 2 alternatives
                    alt_name = crop_mapping.get(alt_id, f"Crop {alt_id}")
                    alt_crops.append(f"{alt_name} ({alt_conf:.2f})")
                
                alt_ferts = []
                for alt_id, alt_conf in result['top_fertilizers'][1:3]:  # Top 2 alternatives
                    alt_name = fert_mapping.get(alt_id, f"Fertilizer {alt_id}")
                    alt_ferts.append(f"{alt_name} ({alt_conf:.2f})")
                
                # Update results DataFrame
                results_df.at[i, 'crop_recommendation'] = crop_name
                results_df.at[i, 'crop_confidence'] = crop_confidence
                results_df.at[i, 'fertilizer_recommendation'] = fert_name
                results_df.at[i, 'fertilizer_confidence'] = fert_confidence
                results_df.at[i, 'alternative_crops'] = ', '.join(alt_crops)
                results_df.at[i, 'alternative_fertilizers'] = ', '.join(alt_ferts)
                
            except Exception as e:
                logging.error(f"Error processing row {i}: {e}")
                results_df.at[i, 'crop_recommendation'] = "Error"
                results_df.at[i, 'fertilizer_recommendation'] = "Error"
        
        # Save results if output path specified
        if output_path:
            results_df.to_csv(output_path, index=False)
            logging.info(f"Results saved to {output_path}")
        
        return results_df

def interactive_mode(model, crop_mapping, fert_mapping, device="cpu"):
    """
    Run the model in interactive mode, allowing users to enter queries.
    
    Args:
        model: The trained model
        crop_mapping (dict): Mapping from crop IDs to names
        fert_mapping (dict): Mapping from fertilizer IDs to names
        device (str): Computation device
    """
    print("\n===== DeepRoot Agricultural Recommendation System =====")
    print("Enter your query about crops or fertilizers. Type 'exit' to quit.")
    print("Example queries:")
    print("  - What crop should I plant with nitrogen=40, phosphorus=60, potassium=30, pH=6.2?")
    print("  - What fertilizer should I use for my rice crop?")
    print("  - Recommend a crop for acidic soil with low nitrogen")
    print("  - Best fertilizer for sandy soil with tomato plants")
    
    while True:
        print("\n" + "="*60)
        query = input("Your query: ")
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
        
        try:
            # Make prediction
            result = predict_recommendation(model, query, device=device)
            
            # Generate explanation
            explanation = generate_explanation(result, crop_mapping, fert_mapping)
            
            # Display results
            print("\nRECOMMENDATION RESULTS:")
            print("-" * 40)
            print(explanation)
            
            # Generate visualization
            visualize_prediction(result, crop_mapping, fert_mapping)
            
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Advanced agricultural recommendation inference")
    parser.add_argument("--model-dir", type=str, default="models/agri_model", help="Directory containing model files")
    parser.add_argument("--query", type=str, help="Natural language query for recommendation")
    parser.add_argument("--input-csv", type=str, help="Input CSV file with soil parameters")
    parser.add_argument("--output-csv", type=str, help="Output CSV file for batch results")
    parser.add_argument("--task-id", type=int, choices=[0, 1], help="Force specific task (0=crop, 1=fertilizer)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Computation device")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--viz-output", type=str, help="Output path for visualization")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load model and mappings
    model_path = os.path.join(args.model_dir, "best_model.pt")
    model_info_path = os.path.join(args.model_dir, "model_info.json")
    
    try:
        model, crop_mapping, fert_mapping, model_info = load_model(
            model_path=model_path,
            model_info_path=model_info_path,
            device=device
        )
        
        logging.info(f"Model loaded successfully from {model_path}")
        logging.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        sys.exit(1)
    
    # Interactive mode
    if args.interactive:
        interactive_mode(model, crop_mapping, fert_mapping, device)
        return
    
    # Process CSV if provided
    if args.input_csv:
        if not os.path.exists(args.input_csv):
            logging.error(f"Input CSV file not found: {args.input_csv}")
            sys.exit(1)
        
        try:
            results_df = process_csv_input(
                file_path=args.input_csv,
                model=model,
                crop_mapping=crop_mapping,
                fert_mapping=fert_mapping,
                output_path=args.output_csv,
                task_id=args.task_id,
                device=device
            )
            
            # Print summary
            print("\nProcessing summary:")
            print(f"  - Total rows processed: {len(results_df)}")
            print(f"  - Top crop recommendations:")
            top_crops = results_df['crop_recommendation'].value_counts().head(5)
            for crop, count in top_crops.items():
                print(f"    * {crop}: {count} rows")
                
            print(f"  - Top fertilizer recommendations:")
            top_ferts = results_df['fertilizer_recommendation'].value_counts().head(5)
            for fert, count in top_ferts.items():
                print(f"    * {fert}: {count} rows")
                
        except Exception as e:
            logging.error(f"Error processing CSV: {e}")
            sys.exit(1)
        
        return
    
    # Process single query if provided
    if args.query:
        try:
            # Make prediction
            result = predict_recommendation(
                model=model,
                query_text=args.query,
                task_id=args.task_id,
                device=device
            )
            
            # Generate explanation
            explanation = generate_explanation(result, crop_mapping, fert_mapping)
            
            # Display results
            print("\nRECOMMENDATION RESULTS:")
            print("-" * 40)
            print(explanation)
            
            # Generate visualization if requested
            if args.visualize:
                visualize_prediction(
                    result=result,
                    crop_mapping=crop_mapping,
                    fert_mapping=fert_mapping,
                    save_path=args.viz_output
                )
            
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            sys.exit(1)
        
        return
    
    # If we get here, no valid action was specified
    logging.error("No action specified. Use --query, --input-csv, or --interactive")
    parser.print_help()

if __name__ == "__main__":
    main()