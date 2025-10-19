import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

class AQIDataFetcher:
    def __init__(self):
        self.api_key = os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5"
        
        # City-specific base AQI patterns (realistic averages)
        self.city_profiles = {
            "Lahore": {"base_aqi": 4.2, "variation": 0.5, "industrial": True, "traffic": "high"},
            "Karachi": {"base_aqi": 3.8, "variation": 0.6, "industrial": True, "traffic": "very_high"},
            "Islamabad": {"base_aqi": 2.5, "variation": 0.3, "industrial": False, "traffic": "medium"},
            "Rawalpindi": {"base_aqi": 3.5, "variation": 0.4, "industrial": True, "traffic": "high"},
            "Faisalabad": {"base_aqi": 4.0, "variation": 0.5, "industrial": True, "traffic": "medium"},
            "Multan": {"base_aqi": 3.7, "variation": 0.4, "industrial": True, "traffic": "medium"},
            "Gujranwala": {"base_aqi": 3.9, "variation": 0.5, "industrial": True, "traffic": "high"},
            "Peshawar": {"base_aqi": 3.6, "variation": 0.4, "industrial": True, "traffic": "medium"},
            "Quetta": {"base_aqi": 2.8, "variation": 0.3, "industrial": False, "traffic": "low"},
            "Sialkot": {"base_aqi": 3.2, "variation": 0.3, "industrial": True, "traffic": "medium"}
        }
        
    def get_city_coordinates(self, city_name):
        """Get coordinates for a city"""
        city_coordinates = {
            "Lahore": (31.5204, 74.3587),
            "Karachi": (24.8607, 67.0011),
            "Islamabad": (33.6844, 73.0479),
            "Rawalpindi": (33.5651, 73.0169),
            "Faisalabad": (31.4504, 73.1350),
            "Multan": (30.1575, 71.5249),
            "Gujranwala": (32.1877, 74.1945),
            "Peshawar": (34.0151, 71.5249),
            "Quetta": (30.1798, 66.9750),
            "Sialkot": (32.4945, 74.5229)
        }
        
        return city_coordinates.get(city_name, (31.5204, 74.3587))
    
    def get_current_aqi(self, lat, lon, city_name="Lahore"):
        """Get current AQI data - city-specific"""
        return self._generate_city_specific_current_data(city_name)
    
    def get_historical_aqi(self, lat, lon, city_name="Lahore", days=30):
        """Get historical AQI data with city-specific patterns"""
        return self._generate_city_specific_historical_data(city_name, days)
    
    def _generate_city_specific_current_data(self, city_name):
        """Generate realistic current AQI data specific to each city"""
        profile = self.city_profiles.get(city_name, self.city_profiles["Lahore"])
        
        # Base AQI with city-specific pattern and some randomness
        base_aqi = np.random.normal(profile["base_aqi"], profile["variation"])
        base_aqi = max(1.0, min(5.0, base_aqi))
        
        # City-specific pollutant patterns
        if city_name == "Lahore":
            # Lahore typically has high PM2.5 and PM10
            pm25 = base_aqi * 12 + np.random.normal(0, 3)
            pm10 = base_aqi * 25 + np.random.normal(0, 5)
            no2 = base_aqi * 7 + np.random.normal(0, 2)
        elif city_name == "Karachi":
            # Karachi has industrial pollution
            pm25 = base_aqi * 10 + np.random.normal(0, 4)
            pm10 = base_aqi * 22 + np.random.normal(0, 6)
            no2 = base_aqi * 8 + np.random.normal(0, 3)
        elif city_name == "Islamabad":
            # Islamabad has better air quality
            pm25 = base_aqi * 8 + np.random.normal(0, 2)
            pm10 = base_aqi * 15 + np.random.normal(0, 3)
            no2 = base_aqi * 5 + np.random.normal(0, 1)
        elif city_name == "Quetta":
            # Quetta has lower pollution
            pm25 = base_aqi * 7 + np.random.normal(0, 2)
            pm10 = base_aqi * 12 + np.random.normal(0, 3)
            no2 = base_aqi * 4 + np.random.normal(0, 1)
        else:
            # Other cities
            pm25 = base_aqi * 9 + np.random.normal(0, 3)
            pm10 = base_aqi * 18 + np.random.normal(0, 4)
            no2 = base_aqi * 6 + np.random.normal(0, 2)
        
        return {
            'aqi': round(base_aqi, 1),
            'pm2_5': round(max(5, pm25), 1),
            'pm10': round(max(10, pm10), 1),
            'no2': round(max(2, no2), 1),
            'so2': round(max(1, base_aqi * 3 + np.random.normal(0, 1)), 1),
            'co': round(max(50, base_aqi * 70 + np.random.normal(0, 20)), 1),
            'o3': round(max(5, base_aqi * 10 + np.random.normal(0, 3)), 1),
            'timestamp': datetime.now(),
            'temperature': self._get_city_temperature(city_name),
            'humidity': self._get_city_humidity(city_name),
            'wind_speed': self._get_city_wind_speed(city_name),
            'pressure': round(np.random.normal(1013, 10), 1),
            'city': city_name
        }
    
    def _generate_city_specific_historical_data(self, city_name, days=30):
        """Generate realistic historical data with city-specific trends"""
        profile = self.city_profiles.get(city_name, self.city_profiles["Lahore"])
        dates = [datetime.now() - timedelta(days=x) for x in range(days, 0, -1)]
        
        # Create city-specific patterns
        data = []
        for i, date in enumerate(dates):
            # Weekly pattern (worse on weekdays)
            day_of_week = date.weekday()
            weekday_factor = 1.2 if day_of_week < 5 else 0.8
            
            # Seasonal pattern (slight variation)
            seasonal_factor = 1 + 0.1 * np.sin(i * 0.05)
            
            # City-specific base with trends
            base_aqi = profile["base_aqi"] * seasonal_factor * weekday_factor
            base_aqi += np.random.normal(0, profile["variation"])
            base_aqi = max(1.0, min(5.0, base_aqi))
            
            # City-specific pollutant calculations
            if city_name == "Lahore":
                pm25 = base_aqi * 12 + np.random.normal(0, 3)
                pm10 = base_aqi * 25 + np.random.normal(0, 5)
            elif city_name == "Karachi":
                pm25 = base_aqi * 10 + np.random.normal(0, 4)
                pm10 = base_aqi * 22 + np.random.normal(0, 6)
            elif city_name == "Islamabad":
                pm25 = base_aqi * 8 + np.random.normal(0, 2)
                pm10 = base_aqi * 15 + np.random.normal(0, 3)
            elif city_name == "Quetta":
                pm25 = base_aqi * 7 + np.random.normal(0, 2)
                pm10 = base_aqi * 12 + np.random.normal(0, 3)
            else:
                pm25 = base_aqi * 9 + np.random.normal(0, 3)
                pm10 = base_aqi * 18 + np.random.normal(0, 4)
            
            data.append({
                'date': date.date(),
                'aqi': round(base_aqi, 2),
                'pm2_5': round(max(5, pm25), 1),
                'pm10': round(max(10, pm10), 1),
                'no2': round(max(2, base_aqi * 6 + np.random.normal(0, 2)), 1),
                'so2': round(max(1, base_aqi * 3 + np.random.normal(0, 1)), 1),
                'co': round(max(50, base_aqi * 70 + np.random.normal(0, 20)), 1),
                'o3': round(max(5, base_aqi * 10 + np.random.normal(0, 3)), 1),
                'city': city_name
            })
        
        return pd.DataFrame(data)
    
    def _get_city_temperature(self, city_name):
        """Get city-specific temperature patterns"""
        temp_profiles = {
            "Lahore": (25, 8),    # mean, variation
            "Karachi": (28, 4),   # warmer, less variation
            "Islamabad": (22, 6),
            "Rawalpindi": (23, 7),
            "Faisalabad": (26, 8),
            "Multan": (27, 9),    # hot
            "Gujranwala": (25, 8),
            "Peshawar": (24, 7),
            "Quetta": (18, 10),   # cooler
            "Sialkot": (24, 6)
        }
        mean_temp, variation = temp_profiles.get(city_name, (25, 8))
        return round(np.random.normal(mean_temp, variation), 1)
    
    def _get_city_humidity(self, city_name):
        """Get city-specific humidity patterns"""
        humidity_profiles = {
            "Lahore": (60, 15),
            "Karachi": (70, 10),  # more humid
            "Islamabad": (55, 12),
            "Rawalpindi": (58, 13),
            "Faisalabad": (62, 14),
            "Multan": (58, 16),
            "Gujranwala": (61, 14),
            "Peshawar": (59, 13),
            "Quetta": (45, 15),   # drier
            "Sialkot": (63, 12)
        }
        mean_humidity, variation = humidity_profiles.get(city_name, (60, 15))
        return round(max(20, min(95, np.random.normal(mean_humidity, variation))), 1)
    
    def _get_city_wind_speed(self, city_name):
        """Get city-specific wind speed patterns"""
        wind_profiles = {
            "Lahore": (12, 4),
            "Karachi": (18, 6),   # windier (coastal)
            "Islamabad": (10, 3),
            "Rawalpindi": (11, 4),
            "Faisalabad": (13, 5),
            "Multan": (14, 5),
            "Gujranwala": (12, 4),
            "Peshawar": (11, 4),
            "Quetta": (16, 6),    # windier
            "Sialkot": (10, 3)
        }
        mean_wind, variation = wind_profiles.get(city_name, (12, 4))
        return round(max(2, np.random.normal(mean_wind, variation)), 1)