# prompt.py
"""
Agricultural recommendation system that processes natural language queries.
Uses a PyTorch model for recommendations and an LSTM-based NLP system for responses.
"""

import torch
import argparse
import os
import json
import sys
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from model import AgriRecommender

# Path configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "agri_model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")
MODEL_INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")
NLP_MODEL_PATH = os.path.join(BASE_DIR, "agric_nlp_model.pt")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")

# Constants
MAX_LEN = 100  # Maximum sequence length for NLP model
MAX_RESPONSE_LEN = 50  # Maximum length of generated responses

# Common crops and fertilizers for text matching
COMMON_CROPS = [
    "rice", "wheat", "maize", "corn", "soybean", "cotton", 
    "sugarcane", "potato", "tomato", "onion", "mustard", "strawberry"
]

COMMON_FERTILIZERS = [
    "urea", "dap", "npk", "mop", "ammonium sulfate", "ammonium sulphate", 
    "super phosphate", "potash", "calcium nitrate"
]

class SimpleNLPModel(torch.nn.Module):
    """A simple PyTorch-based NLP model for text generation"""
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, output_dim=None):
        super(SimpleNLPModel, self).__init__()
        self.output_dim = output_dim or vocab_size
        
        # Embedding layer
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = torch.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Output layer
        self.output = torch.nn.Linear(hidden_dim, self.output_dim)
        
    def forward(self, x, hidden=None):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        if hidden is None:
            output, hidden = self.lstm(embedded)
        else:
            output, hidden = self.lstm(embedded, hidden)
        
        # Output layer
        predictions = self.output(output)
        
        return predictions, hidden


def pad_sequences(sequences, maxlen, padding='post', value=0):
    """Simple implementation of padding sequences"""
    padded_sequences = []
    for seq in sequences:
        if len(seq) > maxlen:
            # Truncate
            padded_seq = seq[:maxlen]
        else:
            # Pad
            if padding == 'post':
                padded_seq = seq + [value] * (maxlen - len(seq))
            else:  # 'pre'
                padded_seq = [value] * (maxlen - len(seq)) + seq
        padded_sequences.append(padded_seq)
    return padded_sequences


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

def load_nlp_model():
    """Load the PyTorch NLP model and tokenizer if available"""
    # Check if the NLP model exists
    if not os.path.exists(NLP_MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        print("NLP model or tokenizer not found. Only using base recommendation model.")
        return None, None
    
    try:
        # Load the tokenizer first to get vocabulary size
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
        
        # Create and load the model
        vocab_size = len(tokenizer.word_index) + 1  # +1 for padding/unknown
        nlp_model = SimpleNLPModel(vocab_size)
        nlp_model.load_state_dict(torch.load(NLP_MODEL_PATH))
        nlp_model.eval()
        
        return nlp_model, tokenizer
    except Exception as e:
        print(f"Error loading NLP model: {e}")
        return None, None

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

def generate_nlp_response(query, tokenizer, nlp_model, max_length=MAX_RESPONSE_LEN):
    """
    Generate a natural language response using the PyTorch NLP model.
    
    Args:
        query (str): The user's query
        tokenizer: The tokenizer used for text processing
        nlp_model: The NLP model for text generation
        max_length (int): Maximum response length
        
    Returns:
        str: Generated natural language response
    """
    if nlp_model is None or tokenizer is None:
        return None
    
    try:
        # Convert query to sequence
        input_seq = tokenizer.texts_to_sequences([query.lower()])
        # Pad the sequence
        input_pad = pad_sequences(input_seq, maxlen=MAX_LEN, padding='post')
        
        # Convert to PyTorch tensor
        input_tensor = torch.tensor(input_pad, dtype=torch.long)
        
        # Initialize the response
        response = []
        
        # Generate the response word by word
        with torch.no_grad():
            for i in range(max_length):
                # Predict the next token
                prediction, hidden = nlp_model(input_tensor)
                
                # Get the prediction from the last time step
                last_time_step = prediction[:, -1, :]
                output_idx = torch.argmax(last_time_step, dim=1).item()
                
                # Convert the token ID to a word
                word = None
                for token, idx in tokenizer.word_index.items():
                    if idx == output_idx:
                        word = token
                        break
                
                # Stop if we generated an end token or couldn't find the word
                if word is None or word == '<END>':
                    break
                    
                # Add the predicted word to the response
                if word != '<START>':
                    response.append(word)
                
                # Update the sequence for the next prediction
                new_input = tokenizer.texts_to_sequences([' '.join(response)])
                new_input_pad = pad_sequences(new_input, maxlen=MAX_LEN, padding='post')
                input_tensor = torch.tensor(new_input_pad, dtype=torch.long)
        
        # Join the words to form the response
        return ' '.join(response)
    except Exception as e:
        print(f"Error generating NLP response: {e}")
        return None

def predict_recommendation(query_text, task_id=None, device="cpu", model_path=None, model_info_path=None):
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

def generate_complete_response(query, recommendation_result, nlp_response):
    """
    Generate a complete response combining the model recommendation with NLP explanation.
    
    Args:
        query (str): Original user query
        recommendation_result (dict): Result from the recommendation model
        nlp_response (str): Generated NLP response or None
        
    Returns:
        str: Complete formatted response
    """
    # Get recommendation type and name
    if recommendation_result["recommendation_type"] == "crop":
        prediction_name = get_crop_name(recommendation_result["prediction"])
        rec_type = "Crop"
    else:
        prediction_name = get_fertilizer_name(recommendation_result["prediction"])
        rec_type = "Fertilizer"
    
    # Extract detected entities
    query_analysis = recommendation_result.get("query_analysis", {})
    found_crop = query_analysis.get("found_crop")
    found_fertilizer = query_analysis.get("found_fertilizer")
    
    # Start building the response
    response = [f"📊 {rec_type} Recommendation"]
    
    # Mention detected crop/fertilizer if any
    if found_crop:
        response.append(f"I noticed you mentioned {found_crop.capitalize()}.")
    if found_fertilizer:
        response.append(f"I noticed you mentioned {found_fertilizer.upper() if found_fertilizer.lower() == 'npk' else found_fertilizer.capitalize()}.")
    
    # Add the main recommendation
    response.append(f"Based on the information provided, I recommend: {prediction_name} (confidence: {recommendation_result['confidence']:.2f})")
    
    # Add top alternative recommendations
    alternatives = []
    for i, (pred_id, confidence) in enumerate(recommendation_result["top_predictions"][1:], 1):  # Skip the first one as it's already shown
        name = get_crop_name(pred_id) if recommendation_result["recommendation_type"] == "crop" else get_fertilizer_name(pred_id)
        alternatives.append(f"{name} (confidence: {confidence:.2f})")
    
    if alternatives:
        response.append("Alternative recommendations: " + ", ".join(alternatives))
    
    # Add NLP explanation if available
    if nlp_response:
        response.append("\n💬 Additional Information:")
        response.append(nlp_response)
    
    # Return the complete response
    return "\n".join(response)