# api.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os

# Paths
MODEL_PATH = "models/agri_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

# Load model and vectorizer once on startup
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("Model or vectorizer file not found. Please train the model first.")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

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

# Request schema
class PromptRequest(BaseModel):
    input: str

@app.post("/predict")
def get_prediction(data: PromptRequest):
    try:
        input_text = data.input
        X = vectorizer.transform([input_text])
        prediction = model.predict(X)
        return {"recommendation": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
