#!/usr/bin/env python3
"""
Ames Housing Price Prediction Model Training Script
This script trains a machine learning model to predict house prices using the Ames Housing Dataset.
"""

import os
import argparse
import logging
import warnings
import joblib
import numpy as np
import pandas as pd

from typing import List, Dict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, PowerTransformer
from xgboost import XGBRegressor

from preprocessing_utils import MissingValueHandler, OrdinalMapper, create_features, quality_map

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Load and preprocess data
# -----------------------------------------------------------------------------
def load_and_prepare_data(filepath: str):
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} samples with {df.shape[1]} columns")

    df = create_features(df)

    if "SalePrice" not in df.columns:
        raise ValueError("Target column 'SalePrice' not found")

    X = df.drop(columns=["SalePrice"], errors="ignore")
    y = df["SalePrice"]

    if "Id" in X.columns:
        X = X.drop(columns=["Id"])

    y_log = np.log1p(y)
    return X, y_log, y


def create_preprocessor(X: pd.DataFrame):
    """Create preprocessing pipeline"""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Define ordinal columns present in the dataset
    ordinal_cols_candidates = [
        "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
        "HeatingQC", "KitchenQual", "FireplaceQu",
        "GarageQual", "GarageCond", "PoolQC"
    ]
    ordinal_cols = [c for c in ordinal_cols_candidates if c in X.columns]

    nominal_cols = [c for c in categorical_cols if c not in ordinal_cols]

    logger.info(f"Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}, Ordinal: {len(ordinal_cols)}")

    num_pipe = Pipeline([
        ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
        ("scaler", RobustScaler())
    ])

    cat_pipe = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    ord_pipe = Pipeline([
        ("ordmap", OrdinalMapper(cols=ordinal_cols, mapping=quality_map)),
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, nominal_cols),
        ("ord", ord_pipe, ordinal_cols)
    ], remainder="drop")

    full_pipeline = Pipeline([
        ("missing_handler", MissingValueHandler()),
        ("preprocessor", preprocessor)
    ])

    return full_pipeline, numeric_cols, categorical_cols


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------
def train_model(X_train, y_train, X_val, y_val, tune_hyperparameters=True):
    if tune_hyperparameters:
        logger.info("Starting hyperparameter tuning...")
        param_grid = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }

        model = XGBRegressor(random_state=42, verbosity=0)
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
    else:
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)

    y_pred = np.expm1(model.predict(X_val))
    y_val_exp = np.expm1(y_val)

    rmse = np.sqrt(mean_squared_error(y_val_exp, y_pred))
    mae = mean_absolute_error(y_val_exp, y_pred)
    r2 = r2_score(y_val_exp, y_pred)

    logger.info(f"Validation RMSE: ${rmse:,.2f}")
    logger.info(f"Validation MAE: ${mae:,.2f}")
    logger.info(f"Validation R²: {r2:.4f}")

    return model, {"rmse": rmse, "mae": mae, "r2": r2}


# -----------------------------------------------------------------------------
# Save model
# -----------------------------------------------------------------------------
def save_model(model, preprocessor, metrics, numeric_cols, categorical_cols, output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)

    model_artifacts = {
        "model": model,
        "preprocessor": preprocessor,
        "model_name": "XGBoost",
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "metrics": metrics
    }

    model_path = os.path.join(output_dir, "house_price_model.pkl")
    joblib.dump(model_artifacts, model_path)
    logger.info(f"Model saved to {model_path}")
    return model_path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train house price prediction model")
    parser.add_argument("--data", type=str, default="data/train.csv", help="Path to training data CSV file")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save trained model")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation set size (0-1)")
    parser.add_argument("--no-tuning", action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")

    args = parser.parse_args()

    try:
        X, y_log, y_original = load_and_prepare_data(args.data)
        preprocessor, num_cols, cat_cols = create_preprocessor(X)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y_log, test_size=args.test_size, random_state=args.random_state
        )

        logger.info(f"Train set: {X_train.shape}, Validation set: {X_val.shape}")

        X_train_prep = preprocessor.fit_transform(X_train)
        X_val_prep = preprocessor.transform(X_val)

        model, metrics = train_model(X_train_prep, y_train, X_val_prep, y_val, tune_hyperparameters=not args.no_tuning)

        save_model(model, preprocessor, metrics, num_cols, cat_cols, args.output_dir)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.exception(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
