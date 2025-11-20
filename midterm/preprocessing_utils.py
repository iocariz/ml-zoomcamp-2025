#!/usr/bin/env python3
"""
Preprocessing utilities for Ames Housing Dataset
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.base import BaseEstimator, TransformerMixin


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features based on domain knowledge"""
    df_new = df.copy()

    if {'TotalBsmtSF', '1stFlrSF', '2ndFlrSF'}.issubset(df.columns):
        df_new["TotalSF"] = df_new["TotalBsmtSF"] + df_new["1stFlrSF"] + df_new["2ndFlrSF"]

    if {"FullBath", "HalfBath"}.issubset(df.columns):
        df_new["TotalBath"] = df_new["FullBath"] + 0.5 * df_new["HalfBath"]
    if {"BsmtFullBath", "BsmtHalfBath"}.issubset(df.columns):
        df_new["TotalBsmtBath"] = df_new["BsmtFullBath"] + 0.5 * df_new["BsmtHalfBath"]

    if {"YearBuilt", "YrSold"}.issubset(df.columns):
        df_new["HouseAge"] = df_new["YrSold"] - df_new["YearBuilt"]

    if {"YearRemodAdd", "YrSold"}.issubset(df.columns):
        df_new["YearsSinceRemod"] = df_new["YrSold"] - df_new["YearRemodAdd"]

    porch_cols = ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
    available_porch = [c for c in porch_cols if c in df.columns]
    if available_porch:
        df_new["TotalPorchSF"] = df_new[available_porch].sum(axis=1)

    if "PoolArea" in df.columns:
        df_new["HasPool"] = (df_new["PoolArea"] > 0).astype(int)
    if "GarageArea" in df.columns:
        df_new["HasGarage"] = (df_new["GarageArea"] > 0).astype(int)
    if "TotalBsmtSF" in df.columns:
        df_new["HasBasement"] = (df_new["TotalBsmtSF"] > 0).astype(int)
    if "Fireplaces" in df.columns:
        df_new["HasFireplace"] = (df_new["Fireplaces"] > 0).astype(int)

    if "OverallQual" in df.columns:
        df_new["OverallQual2"] = df_new["OverallQual"] ** 2

    return df_new


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Custom transformer for Ames dataset that applies domain-specific missing value logic."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        none_features = [
            "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
            "GarageType", "GarageFinish", "GarageQual", "GarageCond",
            "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
            "MasVnrType"
        ]
        for f in none_features:
            if f in df.columns:
                df[f] = df[f].fillna("None")

        zero_features = [
            "GarageYrBlt", "GarageArea", "GarageCars",
            "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
            "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"
        ]
        for f in zero_features:
            if f in df.columns:
                df[f] = df[f].fillna(0)

        mode_features = [
            "MSZoning", "Electrical", "KitchenQual", "Exterior1st",
            "Exterior2nd", "SaleType", "Functional"
        ]
        for f in mode_features:
            if f in df.columns:
                mode_val = df[f].mode().iloc[0] if not df[f].mode().empty else "Unknown"
                df[f] = df[f].fillna(mode_val)

        if {"LotFrontage", "Neighborhood"}.issubset(df.columns):
            df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
            df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

        return df


class OrdinalMapper(BaseEstimator, TransformerMixin):
    def __init__(self, cols: List[str], mapping: Dict[str, int]):
        self.cols = cols
        self.mapping = mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            if c in X.columns:
                X[c] = X[c].fillna("None").map(self.mapping).astype(float)
        return X


# Quality mapping for ordinal features
quality_map = {"Ex": 5, "Gd": 4, "TA": 3, "Ta": 3, "Fa": 2, "Po": 1, "None": 0, np.nan: 0}