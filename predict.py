# predict.py
"""
Standalone script for generating agricultural recommendations using the NLP model.
This script can be used for testing the NLP model or for interactive sessions.
"""

import os
import pickle
import numpy as np
import sys

# Constants
MAX_LEN = 100  # Maximum sequence length for input
MAX_RESPONSE_LEN = 50  # Maximum length of generated responses
MODEL_PATH = "agric_nlp_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"

def check_requirements():
    """Check if TensorFlow is installed"""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        print(f"TensorFlow version: {tf.__version__}")
        return True
    except ImportError:
        print("TensorFlow not found. Please install it with: pip install tensorflow")
        return False

def load_nlp_components():
    """Load the NLP model and tokenizer"""
    # Check if the required files exist
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}")
        return None, None
    
    if not os.path.exists(TOKENIZER_PATH):
        print(f"Tokenizer file not found at {TOKENIZER_PATH}")
        return None, None
    
    try:
        # Import TensorFlow locally to avoid issues if not installed
        from tensorflow.keras.models import load_model
        
        # Load the model
        print(f"Loading NLP model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        
        # Load the tokenizer
        print(f"Loading tokenizer from {TOKENIZER_PATH}...")
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading NLP components: {e}")
        return None, None

def generate_response(query, tokenizer, model, max_length=MAX_RESPONSE_LEN):    
    """
    Generate a response to a query using the NLP model.
    
    Args:
        query (str): The user's query
        tokenizer: The tokenizer used for text processing
        model: The NLP model for text generation
        max_length (int): Maximum response length
        
    Returns:
        str: Generated response
    """
    try:
        # Import TensorFlow locally to avoid issues if not installed
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        # Convert query to sequence and pad it
        query = query.lower()
        input_seq = tokenizer.texts_to_sequences([query])
        input_pad = pad_sequences(input_seq, maxlen=MAX_LEN, padding='post')
        
        # Generate response
        print("Generating response...")
        
        # Start with an empty response
        response_words = []
        
        # Generate the response word by word
        for _ in range(max_length):
            # Predict next token
            prediction = model.predict(input_pad, verbose=0)
            predicted_token = np.argmax(prediction, axis=1)[0]
            
            # If end of sequence or unknown token, stop
            if predicted_token == 0:
                break
                
            # Find the corresponding word for the predicted token
            word = None
            for token, idx in tokenizer.word_index.items():
                if idx == predicted_token:
                    word = token
                    break
            
            # If we couldn't find a word or hit the end token, stop
            if word is None or word == '<END>':
                break
                
            # Add the word to our response if it's not the start token
            if word != '<START>':
                response_words.append(word)
            
            # Update the input for the next prediction
            # We'll use the current response as the context for the next word
            new_input = tokenizer.texts_to_sequences([' '.join(response_words)])
            input_pad = pad_sequences(new_input, maxlen=MAX_LEN, padding='post')