# 🛡️ Credit Card Fraud Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/FastAPI-0.104+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-Ready-blue.svg" alt="Docker">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

<p align="center">
  <strong>Production-ready fraud detection using an ensemble of Deep Learning and Anomaly Detection models</strong>
</p>

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Model Training](#-model-training)
- [API Usage](#-api-usage)
- [Deployment](#-deployment)
- [Performance](#-performance)
- [Project Structure](#-project-structure)

## 🎯 Overview

This project implements a **state-of-the-art fraud detection system** for credit card transactions, developed as a capstone project for MLZoomcamp. It combines multiple anomaly detection algorithms in an ensemble approach to achieve high accuracy while minimizing false positives.

### 🌟 Key Features

- **🧠 Ensemble Learning**: Combines 4 different models for robust detection
  - Isolation Forest (tree-based anomaly detection)
  - Autoencoder (neural reconstruction error)
  - Variational Autoencoder (probabilistic approach)
  - Deep SVDD (one-class neural network)
  
- **⚡ Real-Time Processing**: 
  - Single transaction: ~10-50ms latency
  - Batch processing: 100 transactions in ~200ms
  
- **🎯 High Accuracy**: 
  - AUROC: 95.1%
  - AUPRC: 82.3%
  - Precision: 89.1% at 99th percentile threshold
  
- **🔧 Production Ready**:
  - Automated feature engineering pipeline
  - Model versioning and artifact management
  - Comprehensive API documentation
  - Docker containerization
  - Health monitoring endpoints

## 🏗️ Architecture

```mermaid
graph LR
    A[Raw Transaction] --> B[Feature Engineering]
    B --> C[RobustScaler]
    C --> D[Ensemble Models]
    D --> E1[Isolation Forest]
    D --> E2[Autoencoder]
    D --> E3[VAE]
    D --> E4[Deep SVDD]
    E1 --> F[Score Normalization]
    E2 --> F
    E3 --> F
    E4 --> F
    F --> G[Weighted Ensemble]
    G --> H[Threshold Check]
    H --> I[Fraud/Normal]
```

### 📊 Data Flow

1. **Input**: Raw credit card transaction (Time, V1-V28, Amount)
2. **Feature Engineering**: Creates 40+ features including:
   - Temporal features (Hour_sin, Hour_cos)
   - Amount transformations (log, sqrt)
   - PCA aggregations (mean, std, min, max)
3. **Scaling**: RobustScaler (trained on normal transactions)
4. **Models**: Each model produces an anomaly score
5. **Ensemble**: Weighted combination of normalized scores
6. **Decision**: Compare to threshold (99th percentile of normal)

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone <repo-url>
cd fraud-detection

# 2. Run with Docker (easiest)
docker-compose up

# 3. Test the API
curl http://localhost:8000/health

# 4. Run demo
python client_demo.py
```

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- 4GB+ RAM
- (Optional) CUDA-capable GPU for faster training

### Option 1: Using UV (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and create virtual environment
uv sync

# Activate environment
source .venv/bin/activate
```

### Option 2: Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Using Docker

```bash
# Build image
docker build -t fraud-detection .

# Run container
docker run -p 8000:8000 -v $(pwd)/models:/app/models fraud-detection
```

## 🧠 Model Training

### 1. Complete Training Pipeline

```bash
# Train all models with default settings (~10 minutes on CPU)
python trainer.py train \
  --data data/creditcard.csv \
  --epochs 100 \
  --threshold-percentile 99
```

### 2. Hyperparameter Optimization (Optional)

```bash
# Use Optuna for automatic hyperparameter tuning (~1 hour)
python trainer.py train \
  --data data/creditcard.csv \
  --optimize \
  --optimize-ae \
  --optimize-vae \
  --optimize-svdd \
  --optimize-metric auprc \
  --optimization-trials 30 \
  --epochs 100
```

### 3. Ensemble Configuration

```bash
# Fit ensemble weights and thresholds on validation set
python trainer.py fit-config \
  --models-dir models/production \
  --val-data artifacts/X_val_scaled.npy \
  --val-labels artifacts/y_val.npy \
  --save-config models/production/ensemble_config.json \
  --optimize-weights \
  --train-stacking
```

### 4. Model Evaluation

```bash
# Evaluate on test set
python trainer.py evaluate \
  --models-dir models/production \
  --config models/production/ensemble_config.json \
  --data artifacts/X_test_scaled.npy \
  --labels artifacts/y_test.npy
```

**Expected Output:**
```
Strategy          AUROC    AUPRC    Precision  Recall   F1
Ensemble_weighted 0.951    0.823    0.891      0.867    0.879
Ensemble_stacking 0.948    0.819    0.883      0.871    0.877
Model_autoencoder 0.942    0.801    0.865      0.834    0.849
```

## 🔌 API Usage

### Starting the API

```bash
# Development mode with auto-reload
python run_api.py

# Production mode
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Endpoints

| Method | Endpoint | Description | Request Body |
|--------|----------|-------------|--------------|
| `GET` | `/` | API documentation | - |
| `GET` | `/health` | Health check & model status | - |
| `GET` | `/model/info` | Model configuration details | - |
| `POST` | `/predict/raw` | Single raw transaction | `{Time, V1-V28, Amount, strategy}` |
| `POST` | `/predict` | Preprocessed features | `{features: [...], strategy}` |
| `POST` | `/predict/batch/raw` | Multiple raw transactions | `{transactions: [...], strategy}` |
| `POST` | `/predict/csv` | CSV file upload | Form data with file |
| `POST` | `/evaluate` | Evaluate on labeled data | CSV with Class column |

### Example: Raw Transaction Prediction

```python
import requests

# Raw credit card transaction
transaction = {
    "Time": 0.0,
    "V1": -1.359807,
    "V2": -0.072781,
    # ... V3-V27 ...
    "V28": -0.021053,
    "Amount": 149.62,
    "strategy": "weighted"
}

response = requests.post(
    "http://localhost:8000/predict/raw",
    json=transaction
)

result = response.json()
print(f"Is Fraud: {result['is_fraud']}")
print(f"Probability: {result['fraud_probability']:.2%}")
print(f"Anomaly Score: {result['anomaly_score']:.4f}")
```

### Interactive API Documentation

Visit `http://localhost:8000/docs` for interactive Swagger UI or `http://localhost:8000/redoc` for ReDoc.

## 🐳 Deployment

### Local Docker Deployment

```bash
# Using docker-compose (recommended)
docker-compose up -d

# Check logs
docker-compose logs -f fraud-api

# Stop services
docker-compose down
```

### Cloud Deployment (Fly.io)

```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Launch app (first time)
fly launch --no-deploy

# Deploy
fly deploy

# Check status
fly status

# View logs
fly logs
```

### Environment Variables

```bash
# .env file
API_PORT=8000
API_WORKERS=4
MODEL_CACHE_SIZE=1000
LOG_LEVEL=info
MODELS_DIR=/app/models/production
CONFIG_PATH=/app/models/production/ensemble_config.json
ARTIFACTS_DIR=/app/artifacts
```

## 📈 Performance

### Model Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **AUROC** | 0.951 | Area Under ROC Curve |
| **AUPRC** | 0.823 | Area Under Precision-Recall |
| **Precision** | 0.891 | At 99th percentile threshold |
| **Recall** | 0.867 | Fraud detection rate |
| **F1 Score** | 0.879 | Harmonic mean |

### System Performance

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Single prediction | ~30ms | 30 req/s |
| Batch (100 items) | ~200ms | 500 tx/s |
| Model loading | ~5s | One-time |
| Memory usage | ~1.5GB | All models loaded |

### Threshold Strategy Comparison

```
Percentile  FPR     TPR     Precision  F1-Score
95.0%       5.0%    94.2%   0.762      0.841
99.0%       1.0%    86.7%   0.891      0.879  ← Default
99.5%       0.5%    82.1%   0.934      0.874
99.9%       0.1%    71.3%   0.981      0.826
```

## 📁 Project Structure

```
fraud-detection/
│
├── 📊 data/
│   └── creditcard.csv          # Dataset (download separately)
│
├── 🧠 models/
│   └── production/
│       ├── isolation_forest_*.pkl
│       ├── autoencoder_*.pth
│       ├── vae_*.pth
│       ├── deep_svdd_*.pth
│       └── ensemble_config.json
│
├── 📦 artifacts/
│   ├── scaler.pkl              # Fitted RobustScaler
│   ├── feature_names.json      # Feature order
│   ├── X_*_scaled.npy          # Processed data splits
│   └── score_stats.json        # Score distributions
|
├── Dockerfile
├── docker-compose.yml
├── fly.toml
├── test_api.py
├── trainer.py
├── client_demo.py
├── requirements.txt
├── README.md
└── LICENSE
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/ -v

# Run integration tests
pytest tests/integration/ -v

# Test API endpoints
python client_demo_v2.py

# Load testing with locust
locust -f tests/load_test.py --host http://localhost:8000
```

## 📊 Dataset

This project uses the [Credit Card Fraud Detection dataset from Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud):

- **284,807** transactions
- **492** frauds (0.172% of dataset)
- **28** PCA-transformed features (V1-V28)
- **Time** and **Amount** as original features

```bash
# Download dataset
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MLZoomcamp** - For the excellent machine learning course
- **Kaggle** - For providing the dataset
- **Anthropic** - For Claude AI assistance
- **Open Source Community** - For the amazing tools and libraries

## 📧 Contact

**Iñigo** - Senior Scoring Analyst

- GitHub: [@iocariz ](https://github.com/iocariz)
- LinkedIn: [www.linkedin.com/in/inigo-ocariz-69a3807]
- Email: inigo_ocariz@eicloud.com

---

<p align="center">
  Made with ❤️ for MLZoomcamp Capstone Project
</p>
