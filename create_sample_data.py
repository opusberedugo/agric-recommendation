# create_sample_data.py
"""
Create sample training data for the NLP model.
This script generates a CSV file with prompt-response pairs for agricultural queries.
"""

import os
import pandas as pd
import random

# Create the dataset directory if it doesn't exist
os.makedirs("dataset", exist_ok=True)

# Define a set of templates for different types of agricultural queries
QUERY_TEMPLATES = {
    "crop_recommendation": [
        "What crop should I plant in soil with pH {ph}?",
        "Which crop grows best in soil with nitrogen {n}, phosphorus {p}, and potassium {k}?",
        "What's the best crop for soil with pH {ph} and nitrogen {n}?",
        "Recommend a crop for my field with N:{n}, P:{p}, K:{k}, pH:{ph}",
        "What can I grow in soil with high nitrogen ({n}) and low pH ({ph})?",
        "Best crop for sandy soil with pH {ph}?",
        "What to plant in a field with nitrogen {n} and potassium {k}?",
        "Suggest crops for clay soil with pH {ph}",
        "What crop is suitable for soil with nitrogen level {n}?",
        "I have soil with pH {ph} and potassium {k}, what should I plant?"
    ],
    "fertilizer_recommendation": [
        "What fertilizer should I use for {crop}?",
        "Best fertilizer for growing {crop} in soil with pH {ph}?",
        "Recommend a fertilizer for {crop} in soil with N:{n}, P:{p}, K:{k}",
        "What's the optimal fertilizer for {crop} in acidic soil (pH {ph})?",
        "Which fertilizer is best for {crop} with low nitrogen ({n})?",
        "How should I fertilize my {crop} plants?",
        "What nutrients does {crop} need most?",
        "Fertilizer recommendation for {crop} in clay soil",
        "What NPK ratio is best for {crop}?",
        "How to improve soil for {crop} cultivation?"
    ]
}

# Define response templates for different query types
RESPONSE_TEMPLATES = {
    "crop_recommendation": [
        "Based on your soil parameters (N:{n}, P:{p}, K:{k}, pH:{ph}), {crop} would be an excellent choice. {crop} thrives in these conditions and should give you a good yield.",
        "For soil with N:{n}, P:{p}, K:{k}, and pH:{ph}, I recommend planting {crop}. This crop is well-suited to these soil conditions and will likely perform well.",
        "Given your soil's characteristics (Nitrogen:{n}, Phosphorus:{p}, Potassium:{k}, pH:{ph}), {crop} is the most suitable option. It has been shown to grow well in similar conditions.",
        "The soil parameters you provided (N:{n}, P:{p}, K:{k}, pH:{ph}) are ideal for growing {crop}. This crop thrives in these soil conditions and would be my recommendation.",
        "With nitrogen {n}, phosphorus {p}, potassium {k}, and pH {ph}, your soil is best suited for {crop}. This crop requires these specific nutrient levels to thrive."
    ],
    "fertilizer_recommendation": [
        "For {crop} cultivation with your soil parameters (N:{n}, P:{p}, K:{k}, pH:{ph}), {fertilizer} would be the most effective fertilizer. It provides the necessary nutrients that {crop} needs.",
        "To maximize your {crop} yield in soil with N:{n}, P:{p}, K:{k}, and pH:{ph}, I recommend using {fertilizer}. This fertilizer will balance the nutrient levels in your soil for optimal {crop} growth.",
        "Given your soil's nutrient profile (Nitrogen:{n}, Phosphorus:{p}, Potassium:{k}, pH:{ph}) and that you're growing {crop}, {fertilizer} would be the best choice. It will address any deficiencies and promote healthy growth.",
        "For growing {crop} in soil with the parameters you provided, {fertilizer} is recommended. This fertilizer will help maintain the optimal nutrient balance needed for {crop}.",
        "To successfully grow {crop} in your soil conditions (N:{n}, P:{p}, K:{k}, pH:{ph}), apply {fertilizer}. This will ensure your {crop} plants receive the proper nutrients for healthy development."
    ]
}

# Define ranges for soil parameters
SOIL_PARAMS = {
    "n": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
    "p": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "k": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "ph": [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
}

# Define crops and their recommended parameters (simplified)
CROPS = {
    "rice": {"n": [70, 80, 90, 100], "p": [30, 40, 50], "k": [30, 40, 50], "ph": [5.5, 6.0, 6.5]},
    "wheat": {"n": [80, 90, 100, 110], "p": [50, 60, 70], "k": [30, 40, 50], "ph": [6.0, 6.5, 7.0]},
    "maize": {"n": [100, 110, 120, 130], "p": [50, 60, 70], "k": [50, 60, 70], "ph": [5.5, 6.0, 6.5]},
    "soybean": {"n": [20, 30, 40], "p": [50, 60, 70], "k": [30, 40, 50], "ph": [6.0, 6.5, 7.0]},
    "cotton": {"n": [70, 80, 90], "p": [30, 40, 50], "k": [50, 60, 70], "ph": [5.5, 6.0, 6.5]},
    "sugarcane": {"n": [90, 100, 110], "p": [50, 60, 70], "k": [70, 80, 90], "ph": [6.0, 6.5, 7.0]},
    "potato": {"n": [50, 60, 70], "p": [70, 80, 90], "k": [90, 100, 110], "ph": [4.5, 5.0, 5.5]},
    "tomato": {"n": [50, 60, 70], "p": [50, 60, 70], "k": [50, 60, 70], "ph": [5.5, 6.0, 6.5]},
    "onion": {"n": [30, 40, 50], "p": [50, 60, 70], "k": [30, 40, 50], "ph": [5.5, 6.0, 6.5]},
    "mustard": {"n": [50, 60, 70], "p": [30, 40, 50], "k": [30, 40, 50], "ph": [6.0, 6.5, 7.0]}
}

# Define fertilizers and their compositions (simplified)
FERTILIZERS = {
    "Urea": {"n": "high", "p": "low", "k": "low", "best_for": ["rice", "wheat", "maize", "sugarcane"]},
    "DAP": {"n": "medium", "p": "high", "k": "low", "best_for": ["wheat", "maize", "soybean"]},
    "NPK 10-10-10": {"n": "medium", "p": "medium", "k": "medium", "best_for": ["tomato", "potato", "onion"]},
    "NPK 15-15-15": {"n": "medium", "p": "medium", "k": "medium", "best_for": ["cotton", "maize"]},
    "MOP": {"n": "low", "p": "low", "k": "high", "best_for": ["potato", "sugarcane", "cotton"]},
    "Ammonium Sulfate": {"n": "medium", "p": "low", "k": "low", "best_for": ["rice", "potato", "onion"]}
}

def get_suitable_crop(n, p, k, ph):
    """Determine a suitable crop based on soil parameters."""
    suitable_crops = []
    
    for crop, params in CROPS.items():
        if (n in params["n"] or abs(n - sum(params["n"])/len(params["n"])) < 20) and \
           (p in params["p"] or abs(p - sum(params["p"])/len(params["p"])) < 20) and \
           (k in params["k"] or abs(k - sum(params["k"])/len(params["k"])) < 20) and \
           (ph in params["ph"] or abs(ph - sum(params["ph"])/len(params["ph"])) < 1.0):
            suitable_crops.append(crop)
    
    if suitable_crops:
        return random.choice(suitable_crops)
    else:
        # If no perfect match, return the closest match
        return random.choice(list(CROPS.keys()))

def get_suitable_fertilizer(crop, n, p, k):
    """Determine a suitable fertilizer based on crop and soil parameters."""
    suitable_fertilizers = []
    
    # Check if there are specific fertilizers recommended for this crop
    for fertilizer, info in FERTILIZERS.items():
        if crop in info["best_for"]:
            suitable_fertilizers.append(fertilizer)
    
    # If no specific recommendations, use nutrient levels to decide
    if not suitable_fertilizers:
        if n < 50:  # Low nitrogen
            suitable_fertilizers.extend(["Urea", "NPK 15-15-15"])
        if p < 50:  # Low phosphorus
            suitable_fertilizers.extend(["DAP", "NPK 10-10-10"])
        if k < 50:  # Low potassium
            suitable_fertilizers.extend(["MOP", "NPK 15-15-15"])
        
        # Remove duplicates
        suitable_fertilizers = list(set(suitable_fertilizers))
    
    if suitable_fertilizers:
        return random.choice(suitable_fertilizers)
    else:
        # If still no match, return a random fertilizer
        return random.choice(list(FERTILIZERS.keys()))

def generate_data(num_samples=500):
    """Generate sample data for training."""
    data = []
    
    for _ in range(num_samples):
        # Randomly select query type
        query_type = random.choice(["crop_recommendation", "fertilizer_recommendation"])
        
        # Generate random soil parameters
        n = random.choice(SOIL_PARAMS["n"])
        p = random.choice(SOIL_PARAMS["p"])
        k = random.choice(SOIL_PARAMS["k"])
        ph = random.choice(SOIL_PARAMS["ph"])
        
        # For crop recommendation queries
        if query_type == "crop_recommendation":
            # Get a suitable crop for these parameters
            crop = get_suitable_crop(n, p, k, ph)
            
            # Choose a random query template and fill in parameters
            query_template = random.choice(QUERY_TEMPLATES[query_type])
            query = query_template.format(n=n, p=p, k=k, ph=ph)
            
            # Choose a random response template and fill in parameters
            response_template = random.choice(RESPONSE_TEMPLATES[query_type])
            response = response_template.format(n=n, p=p, k=k, ph=ph, crop=crop)
        
        # For fertilizer recommendation queries
        else:
            # Choose a random crop
            crop = random.choice(list(CROPS.keys()))
            
            # Get a suitable fertilizer for this crop and soil
            fertilizer = get_suitable_fertilizer(crop, n, p, k)
            
            # Choose a random query template and fill in parameters
            query_template = random.choice(QUERY_TEMPLATES[query_type])
            query = query_template.format(crop=crop, n=n, p=p, k=k, ph=ph)
            
            # Choose a random response template and fill in parameters
            response_template = random.choice(RESPONSE_TEMPLATES[query_type])
            response = response_template.format(crop=crop, n=n, p=p, k=k, ph=ph, fertilizer=fertilizer)
        
        # Add to data
        data.append({"prompt": query, "response": response, "type": query_type})
    
    return pd.DataFrame(data)

def main():
    """Main function to generate and save the data."""
    print("Generating sample training data...")
    df = generate_data(1000)  # Generate 1000 samples
    
    # Print some statistics
    print(f"Generated {len(df)} samples")
    print(f"Crop recommendation queries: {len(df[df['type'] == 'crop_recommendation'])}")
    print(f"Fertilizer recommendation queries: {len(df[df['type'] == 'fertilizer_recommendation'])}")
    
    # Save to CSV
    output_path = os.path.join("dataset", "agricultural_nlp_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    
    # Display some examples
    print("\nExample data:")
    for i in range(min(5, len(df))):
        print(f"\nQuery {i+1}: {df.iloc[i]['prompt']}")
        print(f"Response {i+1}: {df.iloc[i]['response']}")

if __name__ == "__main__":
    main()