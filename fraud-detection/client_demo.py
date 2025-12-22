#!/usr/bin/env python3
"""
Updated client for the Fraud Detection API
Demonstrates both raw and preprocessed transaction predictions
"""

import requests
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import time

class FraudDetectionClient:
    """Client for interacting with the Fraud Detection API"""
    
    def __init__(self, base_url: str = "https://fraud-detection-demo.fly.dev/"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        response = self.session.get(f"{self.base_url}/model/info")
        response.raise_for_status()
        return response.json()
    
    def predict_raw_transaction(self, transaction: Dict[str, float], strategy: str = "weighted") -> Dict[str, Any]:
        """Predict fraud for a RAW credit card transaction"""
        payload = {**transaction, "strategy": strategy}
        response = self.session.post(f"{self.base_url}/predict/raw", json=payload)
        response.raise_for_status()
        return response.json()
    
    def predict_batch_raw(self, transactions: List[Dict[str, float]], 
                         strategy: str = "weighted", return_scores: bool = False) -> Dict[str, Any]:
        """Predict fraud for multiple RAW transactions"""
        payload = {
            "transactions": transactions,
            "strategy": strategy,
            "return_scores": return_scores
        }
        response = self.session.post(f"{self.base_url}/predict/batch/raw", json=payload)
        response.raise_for_status()
        return response.json()
    
    def predict_preprocessed(self, features: List[float], strategy: str = "weighted") -> Dict[str, Any]:
        """Predict fraud for preprocessed features"""
        payload = {
            "features": features,
            "strategy": strategy
        }
        response = self.session.post(f"{self.base_url}/predict", json=payload)
        response.raise_for_status()
        return response.json()

def demo_raw_transaction():
    """Demo: Raw credit card transaction prediction"""
    print("\n" + "="*60)
    print("DEMO: Raw Credit Card Transaction Prediction")
    print("="*60)
    
    client = FraudDetectionClient()
    
    # Create a sample raw transaction (similar to creditcard.csv format)
    sample_transaction = {
        "Time": 0.0,
        "V1": -1.359807134,
        "V2": -0.072781173,
        "V3": 2.536346738,
        "V4": 1.378155224,
        "V5": -0.338320769,
        "V6": 0.462387778,
        "V7": 0.239598554,
        "V8": 0.098697901,
        "V9": 0.363786970,
        "V10": 0.090794172,
        "V11": -0.551599533,
        "V12": -0.617800856,
        "V13": -0.991389847,
        "V14": -0.311169354,
        "V15": 1.468176972,
        "V16": -0.470400525,
        "V17": 0.207971242,
        "V18": 0.025790644,
        "V19": 0.403992960,
        "V20": 0.251412098,
        "V21": -0.018306778,
        "V22": 0.277837576,
        "V23": -0.110473910,
        "V24": 0.066928075,
        "V25": 0.128539358,
        "V26": -0.189114844,
        "V27": 0.133558377,
        "V28": -0.021053053,
        "Amount": 149.62
    }
    
    try:
        result = client.predict_raw_transaction(sample_transaction, strategy="weighted")
        print(f"\nTransaction Analysis:")
        print(f"  Amount: ${sample_transaction['Amount']:.2f}")
        print(f"  Time: {sample_transaction['Time']:.1f} seconds from start")
        print(f"\nPrediction Result:")
        print(f"  ✓ Is Fraud: {result['is_fraud']}")
        print(f"  ✓ Fraud Probability: {result['fraud_probability']:.2%}")
        print(f"  ✓ Anomaly Score: {result['anomaly_score']:.4f}")
        print(f"  ✓ Threshold: {result['threshold']:.4f}")
        print(f"  ✓ Strategy: {result['strategy']}")
        
        # Interpretation
        if result['is_fraud']:
            print(f"\n⚠️  ALERT: Transaction flagged as FRAUDULENT!")
        else:
            print(f"\n✅ Transaction appears NORMAL")
            
    except Exception as e:
        print(f"Error: {e}")

def demo_batch_raw_transactions():
    """Demo: Batch raw transaction predictions"""
    print("\n" + "="*60)
    print("DEMO: Batch Raw Transaction Predictions")
    print("="*60)
    
    client = FraudDetectionClient()
    
    # Generate sample transactions with varying patterns
    transactions = []
    
    # Normal-looking transaction
    normal_tx = {
        "Time": float(i * 100) if 'i' in locals() else 0.0,
        **{f"V{j}": np.random.randn() * 0.5 for j in range(1, 29)},
        "Amount": np.random.uniform(10, 200)
    }
    transactions.append(normal_tx)
    
    # Suspicious transaction (higher variance in PCA components)
    suspicious_tx = {
        "Time": 3600.0,
        **{f"V{j}": np.random.randn() * 2.0 for j in range(1, 29)},
        "Amount": 2500.0  # Unusually high amount
    }
    transactions.append(suspicious_tx)
    
    # Another normal transaction
    normal_tx2 = {
        "Time": 7200.0,
        **{f"V{j}": np.random.randn() * 0.3 for j in range(1, 29)},
        "Amount": 50.0
    }
    transactions.append(normal_tx2)
    
    try:
        result = client.predict_batch_raw(transactions, strategy="weighted", return_scores=True)
        
        print(f"\nBatch Summary:")
        print(f"  Total Transactions: {result['summary']['total_transactions']}")
        print(f"  Fraudulent Count: {result['summary']['fraudulent_count']}")
        print(f"  Fraud Rate: {result['summary']['fraud_rate']:.2%}")
        print(f"  Strategy: {result['summary']['strategy']}")
        print(f"  Threshold: {result['summary']['threshold']:.4f}")
        
        print(f"\nIndividual Results:")
        for i, pred in enumerate(result['predictions']):
            amount = transactions[i]['Amount']
            status = "🚨 FRAUD" if pred['is_fraud'] else "✅ Normal"
            print(f"  Transaction {pred['index'] + 1}: Amount=${amount:.2f}")
            print(f"    Status: {status}")
            print(f"    Fraud Probability: {pred['fraud_probability']:.2%}")
            if 'anomaly_score' in pred:
                print(f"    Anomaly Score: {pred['anomaly_score']:.4f}")
            print()
            
    except Exception as e:
        print(f"Error: {e}")

def demo_normal_vs_fraud_pattern():
    """Demo: Compare normal vs fraudulent transaction patterns"""
    print("\n" + "="*60)
    print("DEMO: Normal vs Fraudulent Transaction Patterns")
    print("="*60)
    
    client = FraudDetectionClient()
    
    # Create a normal-looking transaction
    normal_transaction = {
        "Time": 1000.0,
        **{f"V{j}": np.random.normal(0, 0.5) for j in range(1, 29)},
        "Amount": 75.50
    }
    
    # Create a fraud-pattern transaction
    # Based on typical fraud patterns: unusual PCA components and high amount
    fraud_transaction = {
        "Time": 1000.0,
        "V1": -4.5,  # Extreme value
        "V2": 3.2,   # Extreme value
        **{f"V{j}": np.random.normal(0, 2.5) for j in range(3, 15)},  # High variance
        **{f"V{j}": np.random.normal(0, 0.5) for j in range(15, 29)},
        "Amount": 3500.0  # Unusually high
    }
    
    try:
        # Predict both
        normal_result = client.predict_raw_transaction(normal_transaction)
        fraud_result = client.predict_raw_transaction(fraud_transaction)
        
        print("\n📊 COMPARISON:")
        print("-" * 50)
        print(f"{'Metric':<25} {'Normal TX':<15} {'Fraud TX':<15}")
        print("-" * 50)
        print(f"{'Amount':<25} ${normal_transaction['Amount']:<14.2f} ${fraud_transaction['Amount']:<14.2f}")
        print(f"{'Is Fraud':<25} {str(normal_result['is_fraud']):<15} {str(fraud_result['is_fraud']):<15}")
        print(f"{'Fraud Probability':<25} {normal_result['fraud_probability']:<14.2%} {fraud_result['fraud_probability']:<14.2%}")
        print(f"{'Anomaly Score':<25} {normal_result['anomaly_score']:<14.4f} {fraud_result['anomaly_score']:<14.4f}")
        print(f"{'Above Threshold':<25} {'No':<15} {'Yes' if fraud_result['is_fraud'] else 'No':<15}")
        print("-" * 50)
        
        # Interpretation
        print("\n💡 INSIGHTS:")
        score_diff = fraud_result['anomaly_score'] - normal_result['anomaly_score']
        print(f"  • Fraud pattern scores {abs(score_diff):.2f} points higher")
        print(f"  • Threshold is set at {normal_result['threshold']:.4f}")
        print(f"  • Normal transaction is {normal_result['threshold'] - normal_result['anomaly_score']:.2f} points below threshold")
        if fraud_result['is_fraud']:
            print(f"  • Fraud transaction is {fraud_result['anomaly_score'] - fraud_result['threshold']:.2f} points above threshold")
        
    except Exception as e:
        print(f"Error: {e}")

def demo_different_strategies():
    """Demo: Compare different ensemble strategies"""
    print("\n" + "="*60)
    print("DEMO: Comparing Different Ensemble Strategies")
    print("="*60)
    
    client = FraudDetectionClient()
    
    # Create a borderline transaction
    borderline_transaction = {
        "Time": 5000.0,
        **{f"V{j}": np.random.normal(0, 1.2) for j in range(1, 29)},
        "Amount": 500.0
    }
    
    try:
        # Get available strategies
        info = client.get_model_info()
        strategies = info.get('available_strategies', ['weighted', 'average', 'max'])
        
        print(f"\nTesting borderline transaction (Amount: ${borderline_transaction['Amount']:.2f})")
        print(f"Available strategies: {', '.join(strategies)}\n")
        
        results = {}
        for strategy in strategies:
            try:
                result = client.predict_raw_transaction(borderline_transaction, strategy=strategy)
                results[strategy] = result
            except:
                continue
        
        if results:
            print(f"{'Strategy':<15} {'Is Fraud':<12} {'Probability':<15} {'Score':<12} {'Threshold':<12}")
            print("-" * 70)
            for strategy, result in results.items():
                print(f"{strategy:<15} {str(result['is_fraud']):<12} "
                      f"{result['fraud_probability']:<14.2%} "
                      f"{result['anomaly_score']:<11.4f} "
                      f"{result['threshold']:<11.4f}")
            
            print("\n💡 Strategy Characteristics:")
            print("  • 'weighted': Optimized weights for each model")
            print("  • 'average': Equal weight to all models")
            print("  • 'max': Most conservative (highest score wins)")
            if 'stacking' in strategies:
                print("  • 'stacking': Meta-model combines all outputs")
                
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Run all demos"""
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + " "*15 + "FRAUD DETECTION API CLIENT DEMO V2" + " "*18 + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    # Wait for API to be ready
    client = FraudDetectionClient()
    max_retries = 5
    
    print("\n🔍 Checking API availability...")
    for i in range(max_retries):
        try:
            health = client.health_check()
            if health['status'] == 'healthy':
                print("✅ API is ready!")
                info = client.get_model_info()
                print(f"📊 Models loaded: {', '.join(info['available_models'])}")
                print(f"🎯 Strategies available: {', '.join(info['available_strategies'])}")
                break
            else:
                print(f"⏳ API is {health['status']}... (attempt {i+1}/{max_retries})")
                time.sleep(2)
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to API at {client.base_url} (attempt {i+1}/{max_retries})")
            if i < max_retries - 1:
                time.sleep(2)
            else:
                print("\n⚠️  Please make sure the API is running:")
                print("  python run_api.py")
                return
        except Exception as e:
            print(f"Error: {e}")
            return
    
    # Run demos
    try:
        demo_raw_transaction()
        demo_batch_raw_transactions()
        demo_normal_vs_fraud_pattern()
        demo_different_strategies()
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("\n⚠️  Make sure you have trained models and config available:")
        print("  1. Train models: python trainer.py train --data data/creditcard.csv")
        print("  2. Fit config: python trainer.py fit-config ...")
    
    print("\n" + "#"*70)
    print("#" + " "*20 + "DEMO COMPLETED SUCCESSFULLY" + " "*21 + "#")
    print("#"*70)
    print("\n💡 Key Takeaways:")
    print("  1. Use /predict/raw for raw credit card transactions")
    print("  2. The API handles feature engineering automatically")
    print("  3. Different strategies have different sensitivity levels")
    print("  4. Anomaly scores > threshold trigger fraud alerts")
    print("\n📚 Check the API docs at http://localhost:8000/docs for more!")

if __name__ == "__main__":
    main()