# 🏡 Ames Housing Price Prediction

A machine learning project that predicts house prices using the Ames Housing Dataset. This project includes comprehensive exploratory data analysis, advanced feature engineering, model training with XGBoost, and deployment as a REST API on Fly.io.

## 🌟 Features

- **Advanced Feature Engineering**: Domain-specific features including house age, total square footage, bathroom counts, and quality indicators
- **Robust Preprocessing**: Custom transformers for handling missing values with domain knowledge
- **XGBoost Model**: Optimized gradient boosting model with hyperparameter tuning
- **REST API**: FastAPI-based prediction service with automatic documentation
- **Cloud Deployment**: Containerized application deployed on Fly.io
- **Production Ready**: Docker containerization, dependency management with UV, and comprehensive testing

## 📊 Model Performance

- **Validation RMSE**: Optimized for minimal prediction error
- **R² Score**: High explained variance in house prices
- **MAE**: Competitive mean absolute error for price predictions

## 🏗️ Project Structure

```
ames-housing-prediction/
│
├── data/                      # Data directory
│   └── train.csv             # Ames housing dataset
│
├── models/                    # Trained models
│   └── house_price_model.pkl # Serialized model and preprocessor
│
├── notebook.ipynb            # EDA and model experimentation
├── train_model.py           # Model training pipeline
├── predict.py               # Local prediction script
├── preprocessing_utils.py   # Custom preprocessing transformers
├── app.py                   # FastAPI application
├── test.py                  # API testing script
├── test.txt                 # cURL test commands
│
├── Dockerfile               # Container configuration
├── fly.toml                # Fly.io deployment config
├── pyproject.toml          # Project dependencies
├── uv.lock                 # Locked dependencies
└── README.md              # Project documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- UV package manager (or pip)
- Docker (for containerization)
- Fly.io CLI (for deployment)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd ames-housing-prediction
```

2. **Install dependencies with UV**
```bash
uv sync
```

Or with pip:
```bash
pip install -r requirements.txt
```

3. **Download the Ames Housing Dataset**
```bash
# Place train.csv in the data/ directory
mkdir -p data
# Download from Kaggle or your data source
```

## 📈 Model Training

### Train a New Model

```bash
python train_model.py --data data/train.csv --output-dir models
```

### Training Options

- `--data`: Path to training data (default: `data/train.csv`)
- `--output-dir`: Directory to save model (default: `models`)
- `--test-size`: Validation split ratio (default: 0.2)
- `--no-tuning`: Skip hyperparameter tuning for faster training
- `--random-state`: Random seed for reproducibility (default: 42)

### Example with Custom Settings

```bash
python train_model.py \
    --data data/train.csv \
    --output-dir models \
    --test-size 0.15 \
    --random-state 123
```

## 🔮 Making Predictions

### Local Prediction

Run predictions on a single observation:

```bash
python predict.py --model models/house_price_model.pkl
```

With custom input:
```bash
python predict.py --model models/house_price_model.pkl --input sample_house.json
```

### API Prediction

1. **Start the API locally**:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

2. **Test with Python**:
```bash
python test.py
```

3. **Test with cURL**:
```bash
bash test.txt
```

## 🌐 API Documentation

### Base URL

- **Local**: `http://localhost:8000`
- **Production**: `https://twilight-water-1732.fly.dev`

### Endpoints

#### Health Check
```
GET /
```

Response:
```json
{
  "message": "🏠 Ames Housing Price Prediction API is running!"
}
```

#### Predict Price
```
POST /predict
```

Request Body:
```json
{
  "features": {
    "MSSubClass": 20,
    "MSZoning": "RL",
    "LotFrontage": 80.0,
    "LotArea": 9600,
    "OverallQual": 7,
    "YearBuilt": 2003,
    "GrLivArea": 1710,
    "FullBath": 2,
    "GarageCars": 2,
    ...
  }
}
```

Response:
```json
{
  "predicted_price": 205000.00
}
```

### Interactive API Documentation

When running locally, access:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t ames-housing-api .
```

### Run Container Locally

```bash
docker run -p 8000:8000 ames-housing-api
```

### Test Container

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d @test_request.json
```

## ☁️ Cloud Deployment (Fly.io)

### Prerequisites

1. Install Fly.io CLI:
```bash
curl -L https://fly.io/install.sh | sh
```

2. Login to Fly.io:
```bash
fly auth login
```

### Deploy to Fly.io

1. **Initialize app** (if not already done):
```bash
fly launch --name twilight-water-1732
```

2. **Deploy**:
```bash
fly deploy
```

3. **Monitor logs**:
```bash
fly logs
```

4. **Check status**:
```bash
fly status
```

### Production API

The API is live at: `https://twilight-water-1732.fly.dev`

Test it:
```bash
python test.py  # Already configured for production URL
```

## 🧪 Testing

### Unit Tests
```bash
pytest tests/
```

### API Integration Test
```bash
# Local
python test.py --url http://localhost:8000

# Production
python test.py --url https://twilight-water-1732.fly.dev
```

### Load Testing
```bash
locust -f locustfile.py --host https://twilight-water-1732.fly.dev
```

## 📊 Feature Engineering

The project implements sophisticated feature engineering:

### Engineered Features
- **TotalSF**: Total square footage (basement + 1st + 2nd floor)
- **TotalBath**: Combined full and half bathrooms
- **HouseAge**: Years since construction
- **YearsSinceRemod**: Years since last remodeling
- **TotalPorchSF**: Combined porch areas
- **Binary Indicators**: HasPool, HasGarage, HasBasement, HasFireplace
- **Polynomial Features**: OverallQual²

### Missing Value Strategy
- **Categorical "None"**: For features like PoolQC, Alley (absence is meaningful)
- **Zero Imputation**: For numeric features related to absent structures
- **Mode Imputation**: For truly missing categorical values
- **Neighborhood Median**: For LotFrontage based on location

## 🔧 Configuration

### Environment Variables

Create a `.env` file:
```env
MODEL_PATH=models/house_price_model.pkl
API_PORT=8000
LOG_LEVEL=INFO
```

### Model Configuration

Edit `train_model.py` for different algorithms:
```python
# Current: XGBoost
model = XGBRegressor(...)

# Alternative: LightGBM
from lightgbm import LGBMRegressor
model = LGBMRegressor(...)

# Alternative: CatBoost
from catboost import CatBoostRegressor
model = CatBoostRegressor(...)
```

## 📦 Dependencies

Key dependencies:
- **FastAPI**: Modern web framework for building APIs
- **XGBoost**: Gradient boosting framework
- **scikit-learn**: Machine learning utilities
- **pandas/numpy**: Data manipulation
- **uvicorn**: ASGI server
- **joblib**: Model serialization

Full list in `pyproject.toml`

## 🛠️ Development

### Setting up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
uv sync --dev

# Install pre-commit hooks
pre-commit install
```

### Code Quality

```bash
# Format code
black .

# Lint
flake8 .

# Type checking
mypy .
```

## 📝 Notebooks

The `notebook.ipynb` contains:
- Comprehensive EDA with visualizations
- Feature importance analysis
- Model comparison (XGBoost, LightGBM, CatBoost)
- Hyperparameter optimization with Optuna
- SHAP value interpretation
- Cross-validation strategies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ames Housing Dataset**: Dean De Cock, Truman State University
- **Kaggle Community**: For insights and kernels
- **scikit-learn**: For preprocessing utilities
- **XGBoost**: For the gradient boosting implementation

## 📧 Contact

For questions or support, please open an issue in the repository.

---

**Live API**: [https://twilight-water-1732.fly.dev](https://twilight-water-1732.fly.dev)

**Documentation**: [https://twilight-water-1732.fly.dev/docs](https://twilight-water-1732.fly.dev/docs)
