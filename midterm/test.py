import requests

#url = 'http://localhost:8000/predict'
url = 'https://twilight-water-1732.fly.dev/predict'

sample_features = { 
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

# Wrap features in the expected format
payload = {"features": sample_features}

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        predicted_price_data = response.json()
        predicted_price = predicted_price_data["predicted_price"]
        print(f"\n🏡 Predicted Sale Price: ${predicted_price:,.0f}")
    else:
        print(f"Error: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("❌ Connection failed. Make sure the API is running on http://localhost:8000")
except requests.exceptions.JSONDecodeError as e:
    print(f"❌ Invalid JSON response: {e}")
    print(f"Response text: {response.text}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
