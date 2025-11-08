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

Results & Performance

Contributing

Support & Contact

Acknowledgments



---

🎯 Overview

The AQI Predictor Dashboard is an end-to-end machine learning application that provides real-time Air Quality Index (AQI) monitoring and 3-day forecasting.
It integrates advanced ML algorithms with an intuitive, interactive dashboard—helping users make data-driven decisions about outdoor activities and health precautions based on air quality trends.

🌟 Key Highlights

Real-Time AQI Monitoring with multi-pollutant tracking

3-Day ML Forecasting using ensemble models (XGBoost, LightGBM, LSTM)

Professional Dashboard with interactive visualizations

Production-Ready Architecture with complete ML pipeline

Health Alerts for hazardous air conditions



---

✨ Features

📊 Core Functionalities

🔴 Live AQI Monitoring – Real-time air quality data with health indicators

🤖 Multi-Model Forecasting – 3-day predictions using XGBoost, LightGBM, and LSTM

📈 Historical Analysis – 30-day trend visualization and statistical insights

🚨 Smart Alert System – Hazardous AQI notifications with probability scoring

🏙️ Multi-City Support – Track air quality across multiple locations

🔍 Model Explainability – SHAP-based feature importance and transparency


🛠️ Technical Features

⚙️ Automated Data Pipeline – Hourly updates from OpenWeather API

🧠 Feature Store Integration – Hopsworks for consistent data handling

📡 Model Performance Monitoring – Real-time model evaluation

📊 Interactive Visuals – Charts, gauges, and correlation matrices

🧩 Modular Architecture – Scalable and maintainable code structure



---

🚀 Installation

🧾 Prerequisites

Python 3.8+

Git

OpenWeather API key


🪜 Setup Steps

1. Clone the Repository

git clone https://github.com/Maryam-zaheer08/aqi-dashboard-maryam.git
cd aqi-dashboard-maryam


2. Create Virtual Environment

# Windows
python -m venv aqi_env
aqi_env\Scripts\activate

# macOS/Linux
python3 -m venv aqi_env
source aqi_env/bin/activate


3. Install Dependencies

pip install -r requirements.txt


4. Configure Environment Create a .env file in the project root:

OPENWEATHER_API_KEY=your_actual_api_key_here


5. Launch the Application

streamlit run app.py

Open in your browser: http://localhost:8501




---

💻 Usage

🏠 Main Dashboard

View current AQI levels and health recommendations

Monitor pollutant concentrations (PM2.5, PM10, NO₂, SO₂, CO, O₃)

Explore interactive AQI gauges and live statistics

Track ML pipeline performance


🔮 Forecast Hub

View 3-day AQI predictions from multiple models

Compare R² scores and confidence intervals

Visualize forecast charts and model performance


📊 Historical Analysis

Analyze 30-day AQI trends

Identify seasonal and correlation patterns

Study data distributions and pollutant relationships


⚠️ Alert Center

Receive real-time hazardous air quality alerts

View multi-level severity warnings

Track alert history and probability trends



---

🏗️ Technical Architecture

🔍 System Overview

Data Sources → Feature Engineering → ML Pipeline → Dashboard → User
     ↓               ↓                 ↓            ↓         ↓
OpenWeather     Pandas/NumPy       XGBoost      Streamlit   Web UI
   API          Scikit-learn      LightGBM       Plotly
                                 TensorFlow

⚙️ Data Pipeline

1. Data Collection: Real-time data from OpenWeather API


2. Feature Engineering: Time-based features, correlations, trend calculations


3. Feature Storage: Hopsworks for reliable data management


4. Model Training: Automated multi-model training pipeline


5. Prediction Service: Real-time AQI forecasting



Architecture Layers

Data Layer: OpenWeather API, Hopsworks

Processing Layer: Pandas, NumPy, Scikit-learn

ML Layer: XGBoost, LightGBM, TensorFlow LSTM

Presentation Layer: Streamlit, Plotly, Matplotlib



---

🤖 Machine Learning Models

Model	Type	R² Score	Training Time	Best For

🥇 XGBoost	Gradient Boosting	1.0000	~2 min	Primary predictions
🥈 LightGBM	Light Gradient Boosting	0.9989	~1 min	Quick updates
🥉 LSTM	Neural Network	0.8727	~5 min	Complex patterns


🧠 Model Insights

XGBoost (Champion Model)

Accuracy: R² = 1.0000

Strengths: Excellent for structured data, robust and reliable

Use Case: Primary prediction engine


LightGBM (High Performer)

Accuracy: R² = 0.9989

Strengths: Extremely fast, efficient, and accurate

Use Case: Quick model updates


LSTM (Pattern Specialist)

Accuracy: R² = 0.8727

Strengths: Captures long-term temporal and seasonal variations

Use Case: Time-series and seasonal pattern forecasting



---

📁 Project Structure

aqi-dashboard-maryam/
├── app.py               # Main Streamlit application
├── models.py            # ML model definitions and training
├── data_pipeline.py     # Data collection and preprocessing
├── utils.py             # Helper functions and visualizations
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation

📘 File Descriptions

app.py: Streamlit dashboard logic and UI

models.py: ML model implementations and training functions

data_pipeline.py: Data fetching, processing, and feature engineering

utils.py: Helper utilities for visualization and data manipulation

requirements.txt: Python dependencies required for the project



---

📊 Results & Performance

⚡ Model Evaluation

XGBoost: R² = 1.0000 ✅

LightGBM: R² = 0.9989 ✅

LSTM: R² = 0.8727 ⚙️


🔍 Top Influential Features

1. PM2.5 concentration


2. PM10 levels


3. Historical AQI trends


4. Time-based features (hour, day, season)


5. Weather correlation factors



📈 Sample Prediction

Current AQI: 3.6 (Unhealthy)

Pollutants: PM2.5 (32.0 μg/m³), PM10 (76.9 μg/m³), NO₂ (21.0 μg/m³)

3-Day Forecast: Gradual improvement trend

Alert Status: No hazardous conditions detected



---

🤝 Contributing

We welcome contributions to improve the AQI Predictor Dashboard!

🧩 Steps to Contribute

1. Fork this repository


2. Create a new branch

git checkout -b feature/YourFeature


3. Commit your changes

git commit -m "Add YourFeature"


4. Push and open a Pull Request



🧪 Development Setup

pip install -r requirements.txt
python -m pytest tests/
black app.py models.py data_pipeline.py utils.py


---

📞 Support & Contact

For questions, feedback, or collaboration:

🐛 Issues: GitHub Issues

💬 Discussions: GitHub Discussions



---

🙏 Acknowledgments

OpenWeatherMap – for providing comprehensive air quality data APIs

Streamlit – for an elegant and simple dashboard framework

Hopsworks – for feature store integration

TensorFlow, XGBoost, LightGBM – for advanced machine learning libraries



---

🌍 Breathe Better, Plan Smarter with AQI Predictor Dashboard

Empowering everyone with accurate, accessible, and actionable air quality insights.
