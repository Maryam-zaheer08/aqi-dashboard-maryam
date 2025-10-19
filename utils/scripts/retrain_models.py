#!/usr/bin/env python3
"""
Automated model retraining script for GitHub Actions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.models import EnhancedAQIPredictor
from utils.feature_store import FeatureStoreManager

def main():
    print("Starting automated model retraining...")
    
    # Initialize components
    predictor = EnhancedAQIPredictor()
    feature_store = FeatureStoreManager()
    
    # Get latest features
    feature_data = feature_store.get_historical_features(days=90)
    
    # Retrain models
    print("Training XGBoost model...")
    predictor.train_xgboost()
    
    print("Training LightGBM model...")
    predictor.train_lightgbm()
    
    print("Training LSTM model...")
    predictor.train_lstm()
    
    print("Training hazardous classifier...")
    predictor.train_hazardous_classifier()
    
    print("Model retraining completed successfully!")

if __name__ == "__main__":
    main()