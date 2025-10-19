import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class AQIVisualizer:
    def __init__(self):
        # Dark Navy Blue Color Scheme
        self.colors = {
            'primary': '#1E40AF',      # Navy Blue
            'secondary': '#3B82F6',    # Royal Blue
            'accent': '#60A5FA',       # Light Blue
            'background': '#0F172A',   # Dark Navy
            'card_bg': '#1E293B',      # Card Background
            'text': '#E8EAFF',         # Light Text
            'text_secondary': '#94A3B8', # Secondary Text
            'success': '#10B981',      # Emerald Green
            'warning': '#F59E0B',      # Amber
            'danger': '#EF4444',       # Red
            'grid': '#374151'          # Grid Lines
        }
    
    def create_gauge_chart(self, aqi_value):
        """Create AQI gauge chart with new colors"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = aqi_value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Current AQI", 'font': {'color': self.colors['text'], 'size': 24, 'family': 'Inter'}},
            delta = {'reference': 2.5, 'increasing': {'color': self.colors['danger']}},
            gauge = {
                'axis': {'range': [1, 5], 'tickwidth': 1, 'tickcolor': self.colors['text'], 'tickfont': {'color': self.colors['text']}},
                'bar': {'color': self.colors['primary']},
                'bgcolor': self.colors['card_bg'],
                'borderwidth': 2,
                'bordercolor': self.colors['primary'],
                'steps': [
                    {'range': [1, 2], 'color': self.colors['success']},
                    {'range': [2, 3], 'color': self.colors['warning']},
                    {'range': [3, 5], 'color': self.colors['danger']}],
                'threshold': {
                    'line': {'color': self.colors['text'], 'width': 4},
                    'thickness': 0.75,
                    'value': 4.5}}))
        
        fig.update_layout(
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['card_bg'],
            font={'color': self.colors['text'], 'family': "Inter"},
            height=350,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    def create_historical_trend(self, historical_data):
        """Create historical AQI trend chart with new colors"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data['aqi'],
            mode='lines+markers',
            name='AQI Trend',
            line=dict(color=self.colors['primary'], width=4),
            marker=dict(size=8, color=self.colors['secondary'])
        ))
        
        fig.update_layout(
            title={'text': 'Historical AQI Trend (30 Days)', 'font': {'color': self.colors['text'], 'size': 20}},
            xaxis_title='Date',
            yaxis_title='AQI Level',
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['card_bg'],
            font=dict(color=self.colors['text'], family="Inter"),
            height=450,
            xaxis=dict(gridcolor=self.colors['grid'], linecolor=self.colors['grid']),
            yaxis=dict(gridcolor=self.colors['grid'], linecolor=self.colors['grid']),
            showlegend=False
        )
        
        return fig
    
    def create_forecast_chart(self, forecast_data):
        """Create forecast comparison chart with new colors"""
        days = ['Today', 'Tomorrow', 'Day After']
        fig = go.Figure()
        
        # Define colors for different models
        model_colors = {
            'xgboost': self.colors['primary'],
            'lightgbm': self.colors['secondary'], 
            'lstm': self.colors['accent']
        }
        
        for model, values in forecast_data.items():
            fig.add_trace(go.Scatter(
                x=days,
                y=values,
                mode='lines+markers',
                name=model.upper(),
                line=dict(width=4, color=model_colors.get(model, self.colors['primary'])),
                marker=dict(size=10)
            ))
        
        fig.update_layout(
            title={'text': '3-Day AQI Forecast - Model Comparison', 'font': {'color': self.colors['text'], 'size': 20}},
            xaxis_title='Day',
            yaxis_title='Predicted AQI',
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['card_bg'],
            font=dict(color=self.colors['text'], family="Inter"),
            height=450,
            xaxis=dict(gridcolor=self.colors['grid'], linecolor=self.colors['grid']),
            yaxis=dict(gridcolor=self.colors['grid'], linecolor=self.colors['grid']),
            legend=dict(
                bgcolor=self.colors['card_bg'],
                bordercolor=self.colors['grid'],
                borderwidth=1
            )
        )
        
        return fig
    
    def create_pollutants_chart(self, current_data):
        """Create pollutants breakdown chart with new colors"""
        pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        values = [
            current_data['pm2_5'],
            current_data['pm10'],
            current_data['no2'],
            current_data['so2'],
            current_data['co'],
            current_data['o3']
        ]
        
        fig = px.bar(
            x=pollutants,
            y=values,
            color=pollutants,
            color_discrete_sequence=[
                self.colors['primary'],
                self.colors['secondary'],
                self.colors['accent'],
                self.colors['success'],
                self.colors['warning'],
                self.colors['danger']
            ]
        )
        
        fig.update_layout(
            title={'text': 'Pollutants Concentration', 'font': {'color': self.colors['text'], 'size': 20}},
            xaxis_title='Pollutant',
            yaxis_title='Concentration (μg/m³)',
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['card_bg'],
            font=dict(color=self.colors['text'], family="Inter"),
            showlegend=False,
            height=450,
            xaxis=dict(gridcolor=self.colors['grid'], linecolor=self.colors['grid']),
            yaxis=dict(gridcolor=self.colors['grid'], linecolor=self.colors['grid'])
        )
        
        return fig
    
    def create_model_performance(self, performance_data):
        """Create model performance comparison with new colors"""
        models = list(performance_data.keys())
        scores = list(performance_data.values())
        
        colors = [
            self.colors['success'] if score > 0.9 
            else self.colors['warning'] if score > 0.8 
            else self.colors['danger']
            for score in scores
        ]
        
        fig = go.Figure(data=[
            go.Bar(
                x=models,
                y=scores,
                marker_color=colors,
                text=[f'{score:.4f}' for score in scores],
                textposition='auto',
                textfont=dict(color=self.colors['text'], size=14)
            )
        ])
        
        fig.update_layout(
            title={
                'text': 'Model Performance (R² Score)',
                'font': {'color': self.colors['text'], 'size': 20}
            },
            xaxis_title='Model',
            yaxis_title='R² Score',
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['card_bg'],
            font=dict(color=self.colors['text'], family="Inter"),
            height=450,
            xaxis=dict(
                gridcolor=self.colors['grid'],
                linecolor=self.colors['grid']
            ),
            yaxis=dict(
                range=[0, 1.1],
                gridcolor=self.colors['grid'],
                linecolor=self.colors['grid']
            )
        )
        
        return fig