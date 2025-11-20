#!/usr/bin/env python3
"""
Ames Housing Price Prediction API
Serves predictions for one observation via a FastAPI endpoint.
"""

import joblib
import pandas as pd
import numpy as np
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

# Import preprocessing classes BEFORE loading the model
from preprocessing_utils import MissingValueHandler, OrdinalMapper, create_features

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load model and preprocessor
MODEL_PATH = "models/house_price_model.pkl"

try:
    artifacts = joblib.load(MODEL_PATH)
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    logger.info("✅ Model and preprocessor loaded successfully.")
except Exception as e:
    logger.exception("❌ Failed to load model.")
    raise e

# Define FastAPI app
app = FastAPI(
    title="🏡 Ames Housing Price Prediction API",
    description="Predict house sale prices using the Ames dataset model.",
    version="1.0.0"
)

class HouseFeatures(BaseModel):
    features: Dict[str, Any]

def predict_price(input_dict: Dict[str, any]) -> float:
    """Predict price for one observation (expects raw features)"""
    try:
        df = pd.DataFrame([input_dict])
        df = create_features(df)
        X_processed = preprocessor.transform(df)
        log_pred = model.predict(X_processed)[0]
        price = float(np.expm1(log_pred))
        return price
    except Exception as e:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
def home():
    return {"message": "🏠 Ames Housing Price Prediction API is running!"}

@app.post("/predict")
def predict(data: HouseFeatures):
    """POST endpoint for single prediction"""
    price = predict_price(data.features)
    return {"predicted_price": round(price, 2)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)