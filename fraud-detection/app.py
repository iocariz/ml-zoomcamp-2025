#!/usr/bin/env python3
"""
FastAPI application for fraud detection system
Provides REST API endpoints for model predictions and evaluation
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import traceback

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import components from trainer.py
import sys
sys.path.insert(0, str(Path(__file__).parent))

from trainer import (
    EnsembleSystem, 
    DataProcessor,
    zscore_apply,
    load_ensemble_config,
    safe_auc,
    evaluate_scores
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="REST API for credit card fraud detection using ensemble ML models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model caching
MODEL_CACHE = {}
CONFIG_CACHE = {}
PROCESSOR_CACHE = {}

# =============================================================================
# Pydantic Models
# =============================================================================

class RawTransactionRequest(BaseModel):
    """Request model for raw credit card transaction (before feature engineering)"""
    Time: float = Field(..., description="Seconds elapsed between this and first transaction")
    V1: float = Field(..., description="PCA component 1")
    V2: float = Field(..., description="PCA component 2")
    V3: float = Field(..., description="PCA component 3")
    V4: float = Field(..., description="PCA component 4")
    V5: float = Field(..., description="PCA component 5")
    V6: float = Field(..., description="PCA component 6")
    V7: float = Field(..., description="PCA component 7")
    V8: float = Field(..., description="PCA component 8")
    V9: float = Field(..., description="PCA component 9")
    V10: float = Field(..., description="PCA component 10")
    V11: float = Field(..., description="PCA component 11")
    V12: float = Field(..., description="PCA component 12")
    V13: float = Field(..., description="PCA component 13")
    V14: float = Field(..., description="PCA component 14")
    V15: float = Field(..., description="PCA component 15")
    V16: float = Field(..., description="PCA component 16")
    V17: float = Field(..., description="PCA component 17")
    V18: float = Field(..., description="PCA component 18")
    V19: float = Field(..., description="PCA component 19")
    V20: float = Field(..., description="PCA component 20")
    V21: float = Field(..., description="PCA component 21")
    V22: float = Field(..., description="PCA component 22")
    V23: float = Field(..., description="PCA component 23")
    V24: float = Field(..., description="PCA component 24")
    V25: float = Field(..., description="PCA component 25")
    V26: float = Field(..., description="PCA component 26")
    V27: float = Field(..., description="PCA component 27")
    V28: float = Field(..., description="PCA component 28")
    Amount: float = Field(..., description="Transaction amount")
    strategy: str = Field("weighted", description="Ensemble strategy to use")
    
    class Config:
        schema_extra = {
            "example": {
                "Time": 0.0,
                "V1": -1.359807,
                "V2": -0.072781,
                "V3": 2.536347,
                "V4": 1.378155,
                "V5": -0.338321,
                "V6": 0.462388,
                "V7": 0.239599,
                "V8": 0.098698,
                "V9": 0.363787,
                "V10": 0.090794,
                "V11": -0.551600,
                "V12": -0.617801,
                "V13": -0.991390,
                "V14": -0.311169,
                "V15": 1.468177,
                "V16": -0.470401,
                "V17": 0.207971,
                "V18": 0.025791,
                "V19": 0.403993,
                "V20": 0.251412,
                "V21": -0.018307,
                "V22": 0.277838,
                "V23": -0.110474,
                "V24": 0.066928,
                "V25": 0.128539,
                "V26": -0.189115,
                "V27": 0.133558,
                "V28": -0.021053,
                "Amount": 149.62,
                "strategy": "weighted"
            }
        }

class PredictionRequest(BaseModel):
    """Request model for single transaction prediction"""
    features: List[float] = Field(..., description="Feature vector for a single transaction")
    strategy: str = Field("weighted", description="Ensemble strategy to use")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [0.0] * 29,  # Example with 29 features
                "strategy": "weighted"
            }
        }

class BatchRawTransactionRequest(BaseModel):
    """Request model for batch raw transactions"""
    transactions: List[Dict[str, float]] = Field(..., description="List of raw transaction dictionaries")
    strategy: str = Field("weighted", description="Ensemble strategy to use")
    return_scores: bool = Field(False, description="Whether to return anomaly scores")

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    features: List[List[float]] = Field(..., description="2D array of features for multiple transactions")
    strategy: str = Field("weighted", description="Ensemble strategy to use")
    return_scores: bool = Field(False, description="Whether to return anomaly scores along with predictions")

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    is_fraud: bool
    fraud_probability: float
    anomaly_score: float
    threshold: float
    strategy: str
    timestamp: str

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[Dict[str, Any]]
    summary: Dict[str, Any]
    timestamp: str

class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    available_models: List[str]
    available_strategies: List[str]
    threshold_percentile: float
    created_at: str
    model_versions: Dict[str, Any]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    models_loaded: bool
    config_loaded: bool

# =============================================================================
# Helper Functions
# =============================================================================

def preprocess_transaction(
    raw_data: dict,
    processor: DataProcessor
) -> np.ndarray:
    """Preprocess raw transaction data through feature engineering and scaling"""
    
    # Convert to DataFrame
    df = pd.DataFrame([raw_data])
    
    # Add a dummy Class column (required by engineer function)
    df['Class'] = 0
    
    # Apply feature engineering
    df_engineered = processor.engineer(df)
    
    # Get feature columns (excluding Class and optionally Time)
    drop_cols = ['Class']
    if processor.exclude_time:
        drop_cols.append('Time')
    
    # Use the same feature names as during training
    if processor.feature_names:
        feature_cols = processor.feature_names
    else:
        feature_cols = [c for c in df_engineered.columns if c not in drop_cols]
    
    # Extract features
    X = df_engineered[feature_cols].values
    
    # Apply scaling if scaler is available
    if processor.scaler:
        X = processor.scaler.transform(X)
    else:
        logger.warning("No scaler available - using unscaled features")
    
    return X

def preprocess_batch(
    raw_data_list: list,
    processor: DataProcessor
) -> np.ndarray:
    """Preprocess multiple raw transactions"""
    
    # Convert to DataFrame
    df = pd.DataFrame(raw_data_list)
    
    # Add dummy Class column
    df['Class'] = 0
    
    # Apply feature engineering
    df_engineered = processor.engineer(df)
    
    # Get feature columns
    drop_cols = ['Class']
    if processor.exclude_time:
        drop_cols.append('Time')
    
    if processor.feature_names:
        feature_cols = processor.feature_names
    else:
        feature_cols = [c for c in df_engineered.columns if c not in drop_cols]
    
    # Extract features
    X = df_engineered[feature_cols].values
    
    # Apply scaling
    if processor.scaler:
        X = processor.scaler.transform(X)
    else:
        logger.warning("No scaler available - using unscaled features")
    
    return X

def load_models_and_config(
    models_dir: str = "models/production",
    config_path: str = "models/production/ensemble_config.json",
    artifacts_dir: str = "artifacts",
    version: Optional[str] = None
) -> tuple:
    """Load models, configuration, and data processor with caching"""
    global MODEL_CACHE, CONFIG_CACHE, PROCESSOR_CACHE
    
    cache_key = f"{models_dir}_{config_path}_{version}"
    
    # Check cache
    if cache_key in MODEL_CACHE and cache_key in CONFIG_CACHE and cache_key in PROCESSOR_CACHE:
        logger.info("Using cached models, config, and processor")
        return MODEL_CACHE[cache_key], CONFIG_CACHE[cache_key], PROCESSOR_CACHE[cache_key]
    
    try:
        # Load configuration
        logger.info(f"Loading config from {config_path}")
        cfg = load_ensemble_config(Path(config_path))
        
        # Load ensemble system
        logger.info(f"Loading models from {models_dir}")
        system = EnsembleSystem(Path(models_dir), version=version)
        
        # Load meta-model if present
        if cfg.get("stacking") and cfg["stacking"].get("meta_model_path"):
            import pickle
            with open(cfg["stacking"]["meta_model_path"], "rb") as f:
                system.meta_model = pickle.load(f)
                logger.info("Loaded stacking meta-model")
        
        # Load weights if present
        if cfg.get("weights"):
            system.weights = {k: float(v) for k, v in cfg["weights"].items()}
            logger.info(f"Loaded ensemble weights: {system.weights}")
        
        # Load the data processor with fitted scaler
        logger.info(f"Loading data processor from {artifacts_dir}")
        processor = DataProcessor(exclude_time=True)  # Models trained without Time
        
        # Load the fitted scaler
        scaler_path = Path(artifacts_dir) / "scaler.pkl"
        if scaler_path.exists():
            import pickle
            with open(scaler_path, "rb") as f:
                processor.scaler = pickle.load(f)
            logger.info("Loaded fitted RobustScaler")
        else:
            logger.warning(f"Scaler not found at {scaler_path} - predictions may be incorrect!")
        
        # Load feature names if available
        feature_names_path = Path(artifacts_dir) / "feature_names.json"
        if feature_names_path.exists():
            with open(feature_names_path, "r") as f:
                import json
                processor.feature_names = json.load(f)
            logger.info(f"Loaded {len(processor.feature_names)} feature names")
        
        # Cache everything
        MODEL_CACHE[cache_key] = system
        CONFIG_CACHE[cache_key] = cfg
        PROCESSOR_CACHE[cache_key] = processor
        
        return system, cfg, processor
        
    except Exception as e:
        logger.error(f"Failed to load models/config/processor: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")

def predict_transaction(
    features: np.ndarray,
    system: EnsembleSystem,
    config: dict,
    strategy: str = "weighted",
    is_preprocessed: bool = True
) -> dict:
    """
    Make prediction for a single transaction
    
    Args:
        features: Either preprocessed features or raw transaction data
        system: Loaded ensemble system
        config: Loaded configuration
        strategy: Ensemble strategy to use
        is_preprocessed: Whether features are already preprocessed
    """
    
    # Ensure features is 2D
    if features.ndim == 1:
        features = features.reshape(1, -1)
    
    # Get raw scores from all models
    raw = system.raw_scores(features)
    
    # Get normalizers from config
    normalizers = config["normalizers"]
    
    # Check if strategy is available
    if strategy not in config["ensemble_thresholds"]:
        available = list(config["ensemble_thresholds"].keys())
        raise ValueError(f"Strategy '{strategy}' not available. Choose from: {available}")
    
    # Get threshold for the strategy
    threshold = float(config["ensemble_thresholds"][strategy])
    
    # Calculate ensemble score
    scores = system.ensemble_score(raw, normalizers, strategy)
    score = float(scores[0]) if len(scores) > 0 else 0.0
    
    # Make prediction
    is_fraud = score > threshold
    
    # Calculate a pseudo-probability (normalize score to 0-1 range)
    # This is a simplified approach - you might want to use calibration
    fraud_prob = 1 / (1 + np.exp(-score))  # Sigmoid transformation
    
    return {
        "is_fraud": bool(is_fraud),
        "fraud_probability": float(fraud_prob),
        "anomaly_score": score,
        "threshold": threshold,
        "strategy": strategy,
        "timestamp": datetime.now().isoformat()
    }

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", tags=["root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint"""
    models_loaded = len(MODEL_CACHE) > 0
    config_loaded = len(CONFIG_CACHE) > 0
    processor_loaded = len(PROCESSOR_CACHE) > 0
    
    all_loaded = models_loaded and config_loaded and processor_loaded
    
    return HealthResponse(
        status="healthy" if all_loaded else "initializing",
        timestamp=datetime.now().isoformat(),
        models_loaded=models_loaded,
        config_loaded=config_loaded
    )

@app.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
async def model_info(
    models_dir: str = "models/production",
    config_path: str = "models/production/ensemble_config.json",
    artifacts_dir: str = "artifacts"
):
    """Get information about loaded models"""
    try:
        system, cfg, processor = load_models_and_config(models_dir, config_path, artifacts_dir)
        
        return ModelInfoResponse(
            available_models=cfg.get("available_models", []),
            available_strategies=list(cfg.get("ensemble_thresholds", {}).keys()),
            threshold_percentile=cfg.get("threshold_percentile", 99.0),
            created_at=cfg.get("created_at", "unknown"),
            model_versions={
                "isolation_forest": "sklearn",
                "autoencoder": "pytorch",
                "vae": "pytorch", 
                "deep_svdd": "pytorch"
            }
        )
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/raw", response_model=PredictionResponse, tags=["prediction"])
async def predict_raw_transaction(
    request: RawTransactionRequest,
    models_dir: str = "models/production",
    config_path: str = "models/production/ensemble_config.json",
    artifacts_dir: str = "artifacts"
):
    """
    Predict fraud for a single RAW credit card transaction.
    This endpoint handles feature engineering and scaling automatically.
    """
    try:
        # Load models, config, and processor
        system, cfg, processor = load_models_and_config(models_dir, config_path, artifacts_dir)
        
        # Convert request to dict (excluding strategy)
        raw_data = request.dict(exclude={'strategy'})
        
        # Preprocess the raw transaction
        features = preprocess_transaction(raw_data, processor)
        
        # Make prediction
        result = predict_transaction(features, system, cfg, request.strategy)
        
        return PredictionResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Raw prediction error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(
    request: PredictionRequest,
    models_dir: str = "models/production",
    config_path: str = "models/production/ensemble_config.json",
    artifacts_dir: str = "artifacts"
):
    """
    Predict fraud for a single transaction using PREPROCESSED features.
    Use /predict/raw if you have raw transaction data.
    """
    try:
        # Load models, config, and processor
        system, cfg, processor = load_models_and_config(models_dir, config_path, artifacts_dir)
        
        # Prepare features
        features = np.array(request.features).reshape(1, -1)
        
        # Make prediction
        result = predict_transaction(features, system, cfg, request.strategy)
        
        return PredictionResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
async def predict_batch(
    request: BatchPredictionRequest,
    models_dir: str = "models/production",
    config_path: str = "models/production/ensemble_config.json",
    artifacts_dir: str = "artifacts"
):
    """
    Predict fraud for multiple transactions using PREPROCESSED features.
    For raw transaction data, use /predict/batch/raw
    """
    try:
        # Load models, config, and processor
        system, cfg, processor = load_models_and_config(models_dir, config_path, artifacts_dir)
        
        # Prepare features
        features = np.array(request.features)
        n_samples = len(features)
        
        # Get raw scores for all samples
        raw = system.raw_scores(features)
        normalizers = cfg["normalizers"]
        
        # Check strategy
        if request.strategy not in cfg["ensemble_thresholds"]:
            available = list(cfg["ensemble_thresholds"].keys())
            raise ValueError(f"Strategy '{request.strategy}' not available. Choose from: {available}")
        
        # Get threshold and scores
        threshold = float(cfg["ensemble_thresholds"][request.strategy])
        scores = system.ensemble_score(raw, normalizers, request.strategy)
        
        # Make predictions
        is_fraud = (scores > threshold).astype(int)
        fraud_probs = 1 / (1 + np.exp(-scores))  # Sigmoid transformation
        
        # Build response
        predictions = []
        for i in range(n_samples):
            pred_dict = {
                "index": i,
                "is_fraud": bool(is_fraud[i]),
                "fraud_probability": float(fraud_probs[i])
            }
            if request.return_scores:
                pred_dict["anomaly_score"] = float(scores[i])
                pred_dict["threshold"] = threshold
            predictions.append(pred_dict)
        
        # Summary statistics
        summary = {
            "total_transactions": n_samples,
            "fraudulent_count": int(is_fraud.sum()),
            "fraud_rate": float(is_fraud.mean()),
            "avg_fraud_probability": float(fraud_probs.mean()),
            "strategy": request.strategy,
            "threshold": threshold
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary,
            timestamp=datetime.now().isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/predict/batch/raw", response_model=BatchPredictionResponse, tags=["prediction"])
async def predict_batch_raw(
    request: BatchRawTransactionRequest,
    models_dir: str = "models/production",
    config_path: str = "models/production/ensemble_config.json",
    artifacts_dir: str = "artifacts"
):
    """
    Predict fraud for multiple RAW credit card transactions.
    This endpoint handles feature engineering and scaling automatically.
    """
    try:
        # Load models, config, and processor
        system, cfg, processor = load_models_and_config(models_dir, config_path, artifacts_dir)
        
        # Preprocess all transactions
        features = preprocess_batch(request.transactions, processor)
        n_samples = len(features)
        
        # Get raw scores for all samples
        raw = system.raw_scores(features)
        normalizers = cfg["normalizers"]
        
        # Check strategy
        if request.strategy not in cfg["ensemble_thresholds"]:
            available = list(cfg["ensemble_thresholds"].keys())
            raise ValueError(f"Strategy '{request.strategy}' not available. Choose from: {available}")
        
        # Get threshold and scores
        threshold = float(cfg["ensemble_thresholds"][request.strategy])
        scores = system.ensemble_score(raw, normalizers, request.strategy)
        
        # Make predictions
        is_fraud = (scores > threshold).astype(int)
        fraud_probs = 1 / (1 + np.exp(-scores))  # Sigmoid transformation
        
        # Build response
        predictions = []
        for i in range(n_samples):
            pred_dict = {
                "index": i,
                "is_fraud": bool(is_fraud[i]),
                "fraud_probability": float(fraud_probs[i])
            }
            if request.return_scores:
                pred_dict["anomaly_score"] = float(scores[i])
                pred_dict["threshold"] = threshold
            predictions.append(pred_dict)
        
        # Summary statistics
        summary = {
            "total_transactions": n_samples,
            "fraudulent_count": int(is_fraud.sum()),
            "fraud_rate": float(is_fraud.mean()),
            "avg_fraud_probability": float(fraud_probs.mean()),
            "strategy": request.strategy,
            "threshold": threshold
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary,
            timestamp=datetime.now().isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch raw prediction error: {str(e)}\n{traceback.format_exc()}")

@app.post("/predict/csv", tags=["prediction"])
async def predict_from_csv(
    file: UploadFile = File(...),
    strategy: str = "weighted",
    models_dir: str = "models/production", 
    config_path: str = "models/production/ensemble_config.json",
    artifacts_dir: str = "artifacts"
):
    """
    Predict fraud from CSV file upload.
    CSV should contain columns: Time, V1-V28, Amount
    Optionally can include Class column for evaluation.
    """
    try:
        # Read CSV file
        contents = await file.read()
        import io
        df = pd.read_csv(io.BytesIO(contents))
        
        # Load models, config, and processor
        system, cfg, processor = load_models_and_config(models_dir, config_path, artifacts_dir)
        
        # Check if this is raw transaction data or already processed
        has_pca_features = "V1" in df.columns
        has_class = "Class" in df.columns
        
        if has_pca_features:
            # Raw credit card transaction data
            logger.info("Processing raw transaction data from CSV")
            
            # Save true labels if present
            y_true = df["Class"].values if has_class else None
            
            # Engineer features
            df_engineered = processor.engineer(df)
            
            # Get feature columns
            drop_cols = ["Class"]
            if processor.exclude_time:
                drop_cols.append("Time")
            
            if processor.feature_names:
                feature_cols = processor.feature_names
            else:
                feature_cols = [c for c in df_engineered.columns if c not in drop_cols]
            
            # Extract and scale features
            X = df_engineered[feature_cols].values
            if processor.scaler:
                X = processor.scaler.transform(X)
            else:
                logger.warning("No scaler available - using unscaled features")
        else:
            # Assume features are already preprocessed
            logger.info("Using preprocessed features from CSV")
            X = df.values
            y_true = None
        
        # Make predictions
        raw = system.raw_scores(X)
        normalizers = cfg["normalizers"]
        
        if strategy not in cfg["ensemble_thresholds"]:
            available = list(cfg["ensemble_thresholds"].keys())
            raise ValueError(f"Strategy '{strategy}' not available. Choose from: {available}")
        
        threshold = float(cfg["ensemble_thresholds"][strategy])
        scores = system.ensemble_score(raw, normalizers, strategy)
        is_fraud = (scores > threshold).astype(int)
        
        # Prepare results
        results = {
            "n_transactions": len(X),
            "n_fraudulent": int(is_fraud.sum()),
            "fraud_rate": float(is_fraud.mean()),
            "strategy": strategy,
            "threshold": threshold
        }
        
        # Add evaluation metrics if true labels are available
        if y_true is not None:
            metrics = evaluate_scores(y_true, scores, threshold)
            results["evaluation"] = metrics
        
        # Add predictions to dataframe
        df["fraud_prediction"] = is_fraud
        df["anomaly_score"] = scores
        
        # Convert to CSV for download
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=predictions_{file.filename}"}
        )
        
    except Exception as e:
        logger.error(f"CSV prediction error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"CSV prediction failed: {str(e)}")

@app.post("/evaluate", tags=["evaluation"])
async def evaluate(
    file: UploadFile = File(...),
    strategy: str = "weighted",
    models_dir: str = "models/production",
    config_path: str = "models/production/ensemble_config.json",
    artifacts_dir: str = "artifacts"
):
    """
    Evaluate model performance on labeled data.
    CSV must contain columns: Time, V1-V28, Amount, Class
    """
    try:
        # Read CSV file
        contents = await file.read()
        import io
        df = pd.read_csv(io.BytesIO(contents))
        
        if "Class" not in df.columns:
            raise ValueError("CSV must contain 'Class' column for evaluation")
        
        # Load models, config, and processor
        system, cfg, processor = load_models_and_config(models_dir, config_path, artifacts_dir)
        
        # Save true labels
        y_true = df["Class"].values
        
        # Engineer features
        df_engineered = processor.engineer(df)
        
        # Get feature columns
        drop_cols = ["Class"]
        if processor.exclude_time:
            drop_cols.append("Time")
        
        if processor.feature_names:
            feature_cols = processor.feature_names
        else:
            feature_cols = [c for c in df_engineered.columns if c not in drop_cols]
        
        # Extract and scale features
        X = df_engineered[feature_cols].values
        if processor.scaler:
            X = processor.scaler.transform(X)
        else:
            logger.warning("No scaler available - using unscaled features")
        
        # Make predictions
        raw = system.raw_scores(X)
        normalizers = cfg["normalizers"]
        
        # Evaluate all strategies
        results = {}
        for strat in cfg["ensemble_thresholds"].keys():
            threshold = float(cfg["ensemble_thresholds"][strat])
            scores = system.ensemble_score(raw, normalizers, strat)
            metrics = evaluate_scores(y_true, scores, threshold)
            results[strat] = metrics
        
        # Also evaluate individual models
        model_results = {}
        for name, thr in cfg["model_thresholds"].items():
            z = zscore_apply(raw[name], normalizers[name]["mu"], normalizers[name]["sigma"])
            metrics = evaluate_scores(y_true, z, float(thr))
            model_results[name] = metrics
        
        return JSONResponse({
            "ensemble_strategies": results,
            "individual_models": model_results,
            "best_strategy": max(results.items(), key=lambda x: x[1]["auroc"])[0],
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

# =============================================================================
# Background Tasks
# =============================================================================

async def warm_up_models(models_dir: str, config_path: str, artifacts_dir: str):
    """Background task to warm up model loading"""
    try:
        logger.info("Starting model warm-up...")
        load_models_and_config(models_dir, config_path, artifacts_dir)
        logger.info("Model warm-up completed")
    except Exception as e:
        logger.error(f"Model warm-up failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("Starting Fraud Detection API...")
    
    # Schedule model warm-up if default paths exist
    default_models = "models/production"
    default_config = "models/production/ensemble_config.json"
    default_artifacts = "artifacts"
    
    if Path(default_models).exists() and Path(default_config).exists():
        background_tasks = BackgroundTasks()
        background_tasks.add_task(warm_up_models, default_models, default_config, default_artifacts)
        logger.info("Scheduled model warm-up")

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Run with: python app.py
    # Or use: uvicorn app:app --reload --host 0.0.0.0 --port 8000
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )