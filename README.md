🌫️ AQI Predictor Dashboard

Author: Maryam Zaheer
Tech Stack: Python, Streamlit, Machine Learning, TensorFlow, OpenWeather API

## 🖼️ Dashboard Preview
Here's how my AQI Dashboard looks:

![Dashboard 1](https://github.com/Maryam-zaheer08/aqi-dashboard-maryam/raw/main/assets/Dashboard-pic-1.JPG)
![Dashboard 2](https://github.com/Maryam-zaheer08/aqi-dashboard-maryam/raw/main/assets/Dashboard-pic-2.JPG)
![Dashboard 3](https://github.com/Maryam-zaheer08/aqi-dashboard-maryam/raw/main/assets/Dashboard-pic-3.JPG)
![Dashboard 4](https://github.com/Maryam-zaheer08/aqi-dashboard-maryam/raw/main/assets/Dashboard-pic-4.JPG)
![Dashboard 5](https://github.com/Maryam-zaheer08/aqi-dashboard-maryam/raw/main/assets/Dashboard-pic-5.JPG)
![Dashboard 6](https://github.com/Maryam-zaheer08/aqi-dashboard-maryam/raw/main/assets/Dashboard-pic-6.JPG)
![Dashboard 7](https://github.com/Maryam-zaheer08/aqi-dashboard-maryam/raw/main/assets/Dashboard-pic-7.JPG)
![Dashboard 8](https://github.com/Maryam-zaheer08/aqi-dashboard-maryam/raw/main/assets/Dashboard-pic-8.JPG)

📖 Table of Contents

· Overview
· Features
· Installation
· Usage
· Technical Architecture
· Machine Learning Models
· Project Structure
· Results
· Contributing

🎯 Overview

The AQI Predictor Dashboard is an end-to-end machine learning application that provides real-time Air Quality Index (AQI) monitoring and accurate 3-day forecasting. This project combines advanced ML algorithms with an intuitive user interface to help users make informed decisions about outdoor activities and health precautions based on air quality predictions.

🌟 Key Highlights

· Real-time AQI Monitoring with multi-pollutant tracking
· 3-Day ML Forecasting using ensemble of advanced models
· Professional Dashboard with interactive visualizations
· Production-ready Architecture with complete ML pipeline
· Hazardous Condition Alerts for health safety

✨ Features

📊 Core Functionalities

· 🔴 Live AQI Monitoring - Real-time air quality data with health status indicators
· 🤖 Multi-Model Forecasting - 3-day predictions using XGBoost, LightGBM, and LSTM
· 📈 Historical Analysis - 30-day trend visualization and statistical insights
· 🚨 Smart Alert System - Hazardous AQI level notifications with probability scoring
· 🏙️ Multi-City Support - Air quality tracking across different locations
· 🔍 Model Explainability - SHAP-based feature importance and model transparency

🛠️ Technical Features

· Automated Data Pipeline - Hourly data updates from OpenWeather API
· Feature Store Integration - Hopsworks for consistent feature management
· Model Performance Monitoring - Real-time evaluation and tracking
· Interactive Visualizations - Charts, gauges, and correlation matrices
· Modular Architecture - Scalable and maintainable code structure

🚀 Installation

Prerequisites

· Python 3.8 or higher
· Git
· OpenWeather API account (Get free API key)

Step-by-Step Setup

1. Clone the Repository
   bash
   git clone https://github.com/your-username/aqi-predictor.git
   cd aqi-predictor
   
2. Create Virtual Environment
   bash
   # Windows
   python -m venv aqi_env
   aqi_env\Scripts\activate
   
   # macOS/Linux
   python3 -m venv aqi_env
   source aqi_env/bin/activate
   
3. Install Dependencies
   bash
   pip install -r requirements.txt
   
4. Environment Configuration
   Create a .env file in the project root:
   env
   OPENWEATHER_API_KEY=your_actual_api_key_here
   
5. Launch the Application
   bash
   streamlit run app.py
   
   The dashboard will open at http://localhost:8501

💻 Usage

🏠 Main Dashboard

· View current AQI levels and health recommendations
· Monitor real-time pollutant concentrations (PM2.5, PM10, NO2, SO2, CO, O3)
· Interactive AQI gauge and quick statistics
· ML pipeline status monitoring

🔮 Forecast Hub

· 3-day AQI predictions with model comparison
· Performance metrics and confidence intervals
· Model evaluation with R² scores
· Interactive forecast visualizations

📊 Historical Analysis

· 30-day AQI trend visualization
· Statistical summaries and data distributions
· Correlation analysis between pollutants
· Seasonal pattern identification

⚠️ Alert Center

· Real-time hazardous condition monitoring
· Multi-level severity alert system
· Probability-based warning system
· Alert history and tracking

🏗️ Technical Architecture

System Overview


Data Sources → Feature Engineering → ML Pipeline → Dashboard → User
     ↓               ↓                 ↓            ↓         ↓
OpenWeather    Pandas/NumPy       XGBoost      Streamlit   Web UI
   API         Scikit-learn      LightGBM       Plotly
                                TensorFlow


Data Pipeline

1. Data Collection - Real-time data from OpenWeather API
2. Feature Engineering - Time-based features, pollutant correlations, trend calculations
3. Feature Storage - Hopsworks feature store for consistent data management
4. Model Training - Automated training pipeline with multiple algorithms
5. Prediction Service - Real-time forecasting and monitoring

Key Components

· Data Layer: OpenWeather API, Hopsworks Feature Store
· Processing Layer: Pandas, NumPy, Scikit-learn
· ML Layer: XGBoost, LightGBM, TensorFlow LSTM
· Presentation Layer: Streamlit, Plotly, Matplotlib

🤖 Machine Learning Models

Model Comparison

Model Type R² Score Training Time Best For
XGBoost Gradient Boosting 1.0000 ~2 minutes Primary predictions
LightGBM Light Gradient Boosting 0.9989 ~1 minute Quick updates
LSTM Neural Network 0.8727 ~5 minutes Complex patterns

Model Details

🥇 XGBoost (Champion Model)

· Accuracy: R² = 1.0000 (Perfect score)
· Strengths: Best for tabular data, robust to outliers
· Use Case: Primary prediction engine

🥈 LightGBM (High Performer)

· Accuracy: R² = 0.9989 (Excellent)
· Strengths: Fastest training, good accuracy
· Use Case: Quick updates and verification

🥉 LSTM (Pattern Specialist)

· Accuracy: R² = 0.8727 (Good)
· Strengths: Captures complex time-series patterns
· Use Case: Seasonal trend analysis

📁 Project Structure


aqi-predictor/
├── app.py                 # Main Streamlit application
├── models.py              # ML model definitions and training
├── data_pipeline.py       # Data collection and processing
├── utils.py               # Utility functions and helpers
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore rules
└── README.md             # Project documentation


Key Files Description

· app.py: Main dashboard application with Streamlit components
· models.py: Machine learning model implementations and training logic
· data_pipeline.py: Data fetching, processing, and feature engineering
· utils.py: Helper functions for visualization and data manipulation
· requirements.txt: Complete list of Python dependencies

📊 Results & Performance

Model Performance Metrics

· XGBoost: R² = 1.0000 (Excellent)
· LightGBM: R² = 0.9989 (Excellent)
· LSTM: R² = 0.8727 (Good)

Feature Importance

Top features influencing AQI predictions:

1. PM2.5 concentrations
2. PM10 levels
3. Historical AQI trends
4. Time-based features (hour, day, season)
5. Weather correlation factors

Sample Predictions

· Current AQI: 3.6 (Unhealthy category)
· Key Pollutants: PM2.5 (32.0 μg/m³), PM10 (76.9 μg/m³), NO2 (21.0 μg/m³)
· 3-Day Forecast: Gradual improvement trend
· Alert Status: No hazardous conditions detected

🤝 Contributing

We welcome contributions to enhance the AQI Predictor Dashboard!

Contribution Guidelines

1. Fork the repository
2. Create a feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

Development Setup

bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Code formatting
black app.py models.py data_pipeline.py utils.py


🙏 Acknowledgments

· OpenWeatherMap for providing comprehensive air quality data APIs
· Streamlit for the excellent dashboard framework
· Hopsworks for feature store capabilities
· TensorFlow, XGBoost, and LightGBM teams for powerful ML libraries

🌍 Breathe Better, Plan Smarter with AQI Predictor Dashboard

Making air quality information accessible and actionable for everyone

