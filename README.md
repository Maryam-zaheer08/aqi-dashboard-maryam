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

Overview

Features

Installation

Usage

Technical Architecture

Machine Learning Models

Project Structure

Results

Contributing

Support & Contact

Acknowledgments

🎯 Overview

The AQI Predictor Dashboard is an end-to-end machine learning application that provides real-time Air Quality Index (AQI) monitoring and 3-day forecasting.
This project merges advanced AI models with a clean, intuitive interface to help users make data-driven decisions about outdoor activities and health precautions based on air quality trends.

🌟 Key Highlights

🌫 Real-Time AQI Monitoring with multi-pollutant tracking

🤖 3-Day ML Forecasting using ensemble models (XGBoost, LightGBM, LSTM)

📊 Professional Dashboard with interactive visualizations

🧩 Production-Ready Architecture with full ML pipeline

🚨 Health Alerts for hazardous air quality conditions

✨ Features
📊 Core Functionalities

🔴 Live AQI Monitoring – Real-time air quality data with health indicators

🤖 Multi-Model Forecasting – 3-day predictions using XGBoost, LightGBM, and LSTM

📈 Historical Analysis – 30-day trend visualization and statistical insights

🚨 Smart Alert System – Hazardous AQI notifications with probability scoring

🏙️ Multi-City Support – Track air quality across multiple locations

🔍 Model Explainability – SHAP-based feature importance for transparency

🛠️ Technical Features

⚙️ Automated Data Pipeline – Hourly updates via OpenWeather API

🧠 Feature Store Integration – Hopsworks for consistent data handling

📡 Model Performance Monitoring – Real-time model evaluation

📊 Interactive Visuals – Charts, gauges, and correlation matrices

🧩 Modular Architecture – Scalable and maintainable code structure

🚀 Installation
🧾 Prerequisites

Python 3.8+

Git

OpenWeather API key

🪜 Setup Steps

Clone the Repository

git clone https://github.com/Maryam-zaheer08/aqi-dashboard-maryam.git
cd aqi-dashboard-maryam


Create Virtual Environment

# Windows
python -m venv aqi_env
aqi_env\Scripts\activate

# macOS/Linux
python3 -m venv aqi_env
source aqi_env/bin/activate


Install Dependencies

pip install -r requirements.txt


Configure Environment
Create a .env file in the project root:

OPENWEATHER_API_KEY=your_actual_api_key_here


Launch Application

streamlit run app.py


Visit → http://localhost:8501

💻 Usage
🏠 Main Dashboard

View current AQI levels and health recommendations

Monitor pollutant concentrations (PM2.5, PM10, NO₂, SO₂, CO, O₃)

Observe interactive gauges and live statistics

Track ML pipeline and model performance

🔮 Forecast Hub

Get 3-day AQI predictions from multiple models

Compare R² scores and confidence intervals

View forecast charts and error analysis

📊 Historical Analysis

Visualize 30-day AQI trends

Identify seasonal and correlation patterns

Explore data distributions and pollutant dependencies

⚠️ Alert Center

Receive real-time air quality alerts

View severity levels and probability warnings

Access alert history and trend tracking

🏗️ Technical Architecture
🔍 System Overview
Data Sources → Feature Engineering → ML Pipeline → Dashboard → User
     ↓               ↓                 ↓            ↓         ↓
OpenWeather     Pandas/NumPy       XGBoost      Streamlit   Web UI
   API          Scikit-learn      LightGBM       Plotly
                                 TensorFlow

⚙️ Data Pipeline

Data Collection – OpenWeather API (real-time updates)

Feature Engineering – Temporal and pollutant-based transformations

Feature Storage – Managed via Hopsworks

Model Training – Automated multi-model training

Prediction Service – Real-time forecasting and monitoring

🧱 Layers Overview
Layer	Components
Data Layer	OpenWeather API, Hopsworks
Processing Layer	Pandas, NumPy, Scikit-learn
ML Layer	XGBoost, LightGBM, TensorFlow LSTM
Presentation Layer	Streamlit, Plotly, Matplotlib
🤖 Machine Learning Models
Model	Type	R² Score	Training Time	Best For
🥇 XGBoost	Gradient Boosting	1.0000	~2 min	Primary predictions
🥈 LightGBM	Light Gradient Boosting	0.9989	~1 min	Quick updates
🥉 LSTM	Neural Network	0.8727	~5 min	Complex temporal patterns
Model Details

XGBoost (Champion Model)

Accuracy: R² = 1.0000

Strengths: Excellent for tabular data, robust performance

Use Case: Primary prediction engine

LightGBM (High Performer)

Accuracy: R² = 0.9989

Strengths: Fast training, great accuracy

Use Case: Quick update model

LSTM (Pattern Specialist)

Accuracy: R² = 0.8727

Strengths: Captures long-term temporal patterns

Use Case: Seasonal AQI trend analysis

📁 Project Structure
aqi-dashboard-maryam/
├── app.py               # Main Streamlit application
├── models.py            # ML model definitions and training
├── data_pipeline.py     # Data collection and processing
├── utils.py             # Helper functions and visualizations
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation

📘 File Descriptions

app.py: Streamlit dashboard logic

models.py: Machine learning model implementations

data_pipeline.py: Data fetching, processing, feature engineering

utils.py: Helper utilities for visualization and metrics

requirements.txt: Required Python libraries

📊 Results & Performance
Model Evaluation

✅ XGBoost: R² = 1.0000

✅ LightGBM: R² = 0.9989

⚙️ LSTM: R² = 0.8727

🔍 Top Influential Features

PM2.5 concentration

PM10 levels

Historical AQI trends

Time-based features (hour, day, season)

Weather-related correlations

📈 Sample Output
Current AQI: 3.6 (Unhealthy)
Pollutants: PM2.5 (32.0 μg/m³), PM10 (76.9 μg/m³), NO₂ (21.0 μg/m³)
Forecast: Gradual improvement expected
Alerts: No hazardous conditions detected

🤝 Contributing

We welcome all contributions to enhance this project!

🧩 Contribution Steps

Fork the repository

Create a branch

git checkout -b feature/YourFeature


Commit changes

git commit -m "Add YourFeature"


Push and open a Pull Request on GitHub

🧪 Development Setup
pip install -r requirements.txt
python -m pytest tests/
black app.py models.py data_pipeline.py utils.py

📞 Support & Contact

For questions, feedback, or suggestions:

🐛 Issues: GitHub Issues

💬 Discussions: GitHub Discussions

🙏 Acknowledgments

🌤 OpenWeatherMap – For providing AQI and weather APIs

🖥 Streamlit – For an intuitive dashboard framework

🧠 Hopsworks – For feature store management

⚡ TensorFlow, XGBoost, LightGBM – For powerful ML capabilities

🌍 Breathe Better, Plan Smarter with AQI Predictor Dashboard
