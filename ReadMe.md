# DeepRoot Agricultural Recommendation System

An AI-powered system for recommending crops and fertilizers based on soil parameters, environmental conditions, and natural language queries.

## Overview

DeepRoot uses deep learning to analyze soil and environmental data to make accurate recommendations for:

1. **Crop Selection** - Find the best crops to grow based on your soil parameters
2. **Fertilizer Recommendation** - Get personalized fertilizer suggestions for your crops and soil
3. **Natural Language Interface** - Ask questions in plain English to get recommendations

The system leverages a transformer-based architecture with specialized components that understand the relationships between soil nutrients, pH levels, environmental factors, and agricultural requirements.

## Features

- Process natural language queries about crops and fertilizers
- Analyze soil parameters (N, P, K, pH, etc.)
- Consider environmental factors (temperature, humidity, rainfall)
- Provide confidence scores for recommendations
- Generate detailed explanations with scientific background
- Support batch processing through CSV files
- Interactive command-line interface
- FastAPI-based web API for integration

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- FastAPI (for API server)
- Pandas, NumPy, Scikit-learn

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-username/agric-recommendation.git
   cd agric-recommendation
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Prepare your data (optional if you want to train your own model):
   ```
   python prepare_data.py --input dataset/Crop\ Recommendation dataset/Fertilizer\ Recommendation --output dataset/combined_dataset.csv --analyze
   ```

4. Train the model (optional if you want to use the pre-trained model):
   ```
   python enhanced_train.py --data dataset/combined_dataset.csv --output-dir models/agri_model --batch-size 32 --epochs 50
   ```

## Usage

### Interactive Mode

Run the system in interactive mode to ask questions directly:

```
python enhanced_inference.py --interactive
```

Example queries:
- "What crop should I plant with nitrogen=40, phosphorus=60, potassium=30, pH=6.2?"
- "What fertilizer should I use for my rice crop?"
- "Recommend a crop for acidic soil with low nitrogen"

### Process CSV Input

Process a batch of soil samples from a CSV file:

```
python enhanced_inference.py --input-csv soil_samples.csv --output-csv recommendations.csv
```

The input CSV should contain columns for soil parameters (n, p, k, ph), and can optionally include temperature, humidity, and rainfall.

### Single Query

Get a recommendation for a specific query:

```
python enhanced_inference.py --query "What crop is best for soil with nitrogen=40, phosphorus=30, potassium=20, and pH=6.5?"
```

### Web API

Start the API server:

```
uvicorn enhanced_api:app --host 0.0.0.0 --port 8000
```

The API provides the following endpoints:
- `GET /` - Root endpoint
- `GET /info` - Get information about the model
- `GET /crops` - List available crops
- `GET /fertilizers` - List available fertilizers
- `POST /predict` - Get recommendations from natural language query

Example API call:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"input": "What crop should I plant with nitrogen=40, phosphorus=60, potassium=30, pH=6.2?"}'
```

## Training Your Own Model

1. Prepare your dataset using the `prepare_data.py` script
2. Train the model using the `enhanced_train.py` script
3. The trained model will be saved to the specified output directory

You can customize the model architecture and training parameters using command-line arguments.

## Model Architecture

The system uses an enhanced transformer architecture with:

- **Soil Feature Encoder** - Specialized for soil parameter relationships
- **Multi-head Attention** - Captures relationships between features
- **Task-specific Components** - Specialized for crop and fertilizer recommendations
- **Natural Language Processing** - For understanding queries and generating explanations

## File Structure

```
agric-recommendation/
├── enhanced_model.py       # Enhanced AI model architecture
├── enhanced_train.py       # Training script for the model
├── enhanced_inference.py   # Inference script for predictions
├── enhanced_api.py         # FastAPI server implementation
├── prepare_data.py         # Data preparation utility
├── models/                 # Directory for trained models
│   └── agri_model/         # Default model directory
│       ├── best_model.pt   # Model weights
│       └── model_info.json # Model metadata
└── dataset/                # Directory for training data
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

# Key Improvements to Agricultural Recommendation System

## Enhanced Model Architecture

1. **Specialized Soil Feature Encoder**
   - Customized encoder that understands relationships between soil nutrients (N, P, K) and environmental factors
   - Uses domain-specific knowledge about agricultural requirements
   - Handles both categorical and numerical features effectively

2. **Improved Transformer Architecture**
   - Added multi-head attention mechanism to capture complex relationships
   - Implemented more layers for deeper representation learning
   - Included residual connections and layer normalization for better training stability

3. **Task-specific Prediction Heads**
   - Specialized layers for crop recommendations based on nutrient profiles
   - Dedicated components for fertilizer recommendations based on soil deficiencies
   - Better handling of different agricultural tasks (crop recommendation, fertilizer recommendation)

## Enhanced Training Process

1. **Better Data Handling**
   - Improved preprocessing of diverse agricultural datasets
   - Standardized column naming and feature normalization
   - Handling of categorical features (soil types, textures) using embeddings

2. **Advanced Training Techniques**
   - Implemented early stopping to prevent overfitting
   - Added learning rate scheduling for better convergence
   - Used gradient clipping to stabilize training
   - Employed class weighting to handle imbalanced datasets

3. **Model Evaluation**
   - More comprehensive evaluation metrics
   - Cross-validation for more robust performance assessment
   - Confidence scoring for recommendations

## Improved Inference and User Experience

1. **Natural Language Processing**
   - Advanced query analysis to extract key information
   - Better extraction of soil parameters from text
   - Recognition of crop names, fertilizer names, and agricultural terms

2. **Detailed Explanations**
   - Scientific explanations for recommendations
   - Crop-specific fertilizer guidelines
   - Context-aware explanations based on query and soil parameters

3. **Multiple Interfaces**
   - Enhanced API with more comprehensive responses
   - Interactive command-line interface for easy testing
   - Batch processing capabilities for CSV data

## Domain-specific Knowledge Integration

1. **Crop-Soil Relationships**
   - Encoded knowledge about optimal growing conditions for different crops
   - Understanding of how crops interact with different soil types
   - Consideration of environmental factors like temperature and rainfall

2. **Fertilizer Effectiveness**
   - Model understands nutrient content of different fertilizers
   - Can recommend proper application timing and methods
   - Takes into account soil pH and its effect on nutrient availability

3. **Agricultural Best Practices**
   - Includes recommendations aligned with sustainable farming practices
   - Considers local growing conditions when available
   - Provides alternative recommendations with confidence scores

## Technical Infrastructure Improvements

1. **Code Structure**
   - More modular and maintainable codebase
   - Better separation of concerns (model, training, inference, API)
   - Comprehensive logging and error handling

2. **Performance Optimizations**
   - More efficient data processing pipeline
   - Optimized model for faster inference
   - Support for both CPU and GPU execution

3. **Extensibility**
   - Easy to train on new datasets
   - Configurable model architecture and hyperparameters
   - Well-documented API for integration with other systems