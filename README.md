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

📊 Project Overview

The AQI Predictor Dashboard is a comprehensive real-time air quality monitoring and forecasting system that leverages advanced machine learning to predict Air Quality Index (AQI) levels. This production-grade application provides accurate 3-day AQI forecasts, historical trend analysis, and hazardous condition alerts to help users make informed decisions about outdoor activities and health precautions.

🎯 Key Features

· 🔴 Real-time AQI Monitoring - Live air quality data with pollutant breakdown
· 🤖 Multi-Model ML Forecasting - 3-day predictions using XGBoost, LightGBM & LSTM
· 📈 Historical Analysis - 30-day trend visualization and statistical insights
· 🚨 Smart Alert System - Hazardous AQI level notifications with probability scoring
· 🏙️ Multi-City Support - Air quality tracking across multiple locations
· 🔍 Model Explainability - SHAP-based feature importance and model transparency


🛠️ Installation & Local Development

Prerequisites

· Python 3.8 or higher
· Git
· OpenWeather API account (Get free API key)
. Run vs code as Administrator

🚀 Quick Start

1. Clone the repository
   ```bash
   git clone: https://github.com/Maryam-zaheer08/aqi-dashboard-maryam.git
   ```
2. Set up virtual environment
   ```bash
   # Windows
   python -m venv aqi_env
   aqi_env\Scripts\activate
   
   # macOS/Linux
   python3 -m venv aqi_env
   source aqi_env/bin/activate
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt

4. Launch the dashboard
   ```bash
   streamlit run app.py
   ```

The dashboard will open automatically in your default browser at http://localhost:8501

🏗️ System Architecture

```
Data Sources → Feature Engineering → ML Pipeline → Dashboard → User
     ↓               ↓                 ↓            ↓         ↓
OpenWeather    Pandas/NumPy       XGBoost      Streamlit   Web UI
   API                           LightGBM       Plotly
                                TensorFlow
```

🤖 Machine Learning Models

· XGBoost Regressor (R²: 1.000) - High accuracy ensemble method
· LightGBM Regressor (R²: 0.9989) - Fast gradient boosting framework
· LSTM Neural Network (R²: 0.8727) - Temporal pattern recognition

📈 Dashboard Sections

🏠 Main Dashboard

· Current AQI levels and health recommendations
· Real-time pollutant concentrations (PM2.5, PM10, NO2, SO2, CO, O3)
· Interactive AQI gauge and quick statistics

🔮 Forecast Hub

· 3-day AQI predictions with model comparison
· Performance metrics and confidence intervals
· Model evaluation with R² scores

📊 Historical Analysis

· 30-day AQI trend visualization
· Statistical summaries and data distributions
· Correlation analysis between pollutants

⚠️ Alert Center

· Real-time hazardous condition monitoring
· Multi-level severity alert system
· Probability-based warning system

⚙️ Advanced Analytics

· ML pipeline status monitoring
· Feature store information (Hopsworks integration)
· Model training and performance tracking

🌐 Deployment

This project is deployment-ready with:

· GitHub Actions for CI/CD automation
· Streamlit Cloud for seamless web deployment
· Environment variable security for API keys
· Automated model retraining pipelines

📋 Requirements

Core dependencies include:

· streamlit>=1.28.0 - Web application framework
· tensorflow>=2.13.0 - Deep learning capabilities
· pandas>=2.0.0 - Data manipulation and analysis
· scikit-learn>=1.3.0 - Machine learning utilities
· plotly>=5.15.0 - Interactive visualizations
· xgboost>=1.7.0 - Gradient boosting models
· lightgbm>=4.0.0 - Light gradient boosting

See requirements.txt for complete dependency list

🎯 Use Cases

· Health-conscious individuals - Plan outdoor activities based on air quality
· Environmental researchers - Analyze pollution patterns and trends
· Urban planners - Monitor city-wide air quality metrics
· Healthcare providers - Alert sensitive patients about hazardous conditions

🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest new features.

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
