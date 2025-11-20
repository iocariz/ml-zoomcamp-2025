#!/usr/bin/env python3
"""
Ames Housing Price Prediction Script
Loads the trained model and predicts the sale price for one observation.
"""

import argparse
import joblib
import pandas as pd
import numpy as np
import logging
import os
from preprocessing_utils import MissingValueHandler, OrdinalMapper, create_features

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_artifacts(model_path: str):
    """Load model and preprocessor from disk"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    logger.info(f"Loading model from {model_path}")

    artifacts = joblib.load(model_path)
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    return model, preprocessor


def predict_price(model, preprocessor, input_dict: dict):
    """Predict price for one observation (expects raw features)"""
    df = pd.DataFrame([input_dict])      
    df = create_features(df)             

    logger.info("Preprocessing input...")
    X_processed = preprocessor.transform(df)

    log_pred = model.predict(X_processed)[0]
    price = np.expm1(log_pred)  # inverse of log1p
    return price


def main():
    parser = argparse.ArgumentParser(description="Predict house price for one sample")
    parser.add_argument("--model", type=str, default="models/house_price_model.pkl",
                        help="Path to saved model file")
    parser.add_argument("--input", type=str, default=None,
                        help="Optional path to CSV or JSON with one observation")

    args = parser.parse_args()

    # Load model and preprocessor
    model, preprocessor = load_artifacts(args.model)

    # --- Option 1: load observation from file (CSV/JSON) ---
    if args.input:
        logger.info(f"Loading observation from {args.input}")
        if args.input.endswith(".csv"):
            df = pd.read_csv(args.input)
        elif args.input.endswith(".json"):
            df = pd.read_json(args.input)
        else:
            raise ValueError("Input must be a .csv or .json file")
        obs = df.iloc[0].to_dict()

    # --- Option 2: hardcoded sample ---
    else:
        logger.warning("No input provided. Using a sample observation.")
        obs = {
            "MSSubClass": 20,
            "MSZoning": "RL",
            "LotFrontage": 80.0,
            "LotArea": 9600,
            "Street": "Pave",
            "Alley": "None",
            "LotShape": "Reg",
            "LandContour": "Lvl",
            "Utilities": "AllPub",
            "LotConfig": "Inside",
            "Neighborhood": "CollgCr",
            "Condition1": "Norm",
            "Condition2": "Norm",
            "BldgType": "1Fam",
            "HouseStyle": "2Story",
            "OverallQual": 7,
            "OverallCond": 5,
            "YearBuilt": 2003,
            "YearRemodAdd": 2003,
            "RoofStyle": "Gable",
            "RoofMatl": "CompShg",
            "Exterior1st": "VinylSd",
            "Exterior2nd": "VinylSd",
            "MasVnrType": "BrkFace",
            "MasVnrArea": 196.0,
            "ExterQual": "Gd",
            "ExterCond": "TA",
            "Foundation": "PConc",
            "BsmtQual": "Gd",
            "BsmtCond": "TA",
            "BsmtExposure": "No",
            "BsmtFinType1": "GLQ",
            "BsmtFinSF1": 706,
            "BsmtFinType2": "Unf",
            "BsmtFinSF2": 0,
            "BsmtUnfSF": 150,
            "TotalBsmtSF": 856,
            "Heating": "GasA",
            "HeatingQC": "Ex",
            "CentralAir": "Y",
            "Electrical": "SBrkr",
            "1stFlrSF": 856,
            "2ndFlrSF": 854,
            "LowQualFinSF": 0,
            "GrLivArea": 1710,
            "BsmtFullBath": 1,
            "BsmtHalfBath": 0,
            "FullBath": 2,
            "HalfBath": 1,
            "BedroomAbvGr": 3,
            "KitchenAbvGr": 1,
            "KitchenQual": "Gd",
            "TotRmsAbvGrd": 8,
            "Fireplaces": 1,
            "FireplaceQu": "TA",
            "GarageType": "Attchd",
            "GarageYrBlt": 2003,
            "GarageFinish": "RFn",
            "GarageCars": 2,
            "GarageArea": 548,
            "GarageQual": "TA",
            "GarageCond": "TA",
            "PavedDrive": "Y",
            "WoodDeckSF": 0,
            "OpenPorchSF": 61,
            "EnclosedPorch": 0,
            "3SsnPorch": 0,
            "ScreenPorch": 0,
            "PoolArea": 0,
            "Fence": "None",
            "MiscFeature": "None",
            "MiscVal": 0,
            "MoSold": 2,
            "YrSold": 2008,
            "SaleType": "WD",
            "SaleCondition": "Normal",
            "Functional": "Typ",
            "LandSlope": "Gtl",
            "PoolQC": "None"
        }

    logger.info("Running prediction...")
    predicted_price = predict_price(model, preprocessor, obs)
    print("\n🏡 Predicted Sale Price: ${:,.0f}".format(predicted_price))


if __name__ == "__main__":
    main()
