import pickle
import uvicorn
from fastapi import FastAPI
from typing import Dict, Any

app = FastAPI(title="Churn Prediction Service")

with open('pipeline_v2.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)

@app.post("/predict")
def predict(customer: Dict[str, Any]):
    prob = predict_single(customer)

    return {
        "churn_probability": prob,
        "churn": bool(prob >= 0.5)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)