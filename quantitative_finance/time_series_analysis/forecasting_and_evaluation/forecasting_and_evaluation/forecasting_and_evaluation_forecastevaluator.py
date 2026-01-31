from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

class ForecastEvaluator:
    """Comprehensive forecast evaluation metrics"""
    
    def __init__(self):
        pass
    
    def mae(self, y_true, y_pred):
        """Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))
    
    def rmse(self, y_true, y_pred):
        """Root Mean Squared Error"""
        return np.sqrt(np.mean((y_true - y_pred)**2))
    
    def mape(self, y_true, y_pred):
        """Mean Absolute Percentage Error"""
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return np.inf
        return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    
    def smape(self, y_true, y_pred):
        """Symmetric Mean Absolute Percentage Error"""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        if not np.any(mask):
            return 0
        return 100 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])
    
    def mase(self, y_true, y_pred, y_train, seasonal_period=1):
        """Mean Absolute Scaled Error"""
        # Naive forecast MAE (in-sample)
        if seasonal_period == 1:
            naive_mae = np.mean(np.abs(np.diff(y_train)))
        else:
            naive_mae = np.mean(np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period]))
        
        if naive_mae == 0:
            return np.inf
        
        mae_forecast = self.mae(y_true, y_pred)
        return mae_forecast / naive_mae
    
    def theil_u(self, y_true, y_pred, y_train):
        """Theil's U Statistic"""
        # Naive forecast: last observation
        naive_pred = np.full(len(y_true), y_train[-1])
        
        rmse_model = self.rmse(y_true, y_pred)
        rmse_naive = self.rmse(y_true, naive_pred)
        
        if rmse_naive == 0:
            return np.inf
        
        return rmse_model / rmse_naive
    
    def mda(self, y_true, y_pred):
        """Mean Directional Accuracy"""
        if len(y_true) < 2:
            return np.nan
        
        # Direction: sign of change
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        
        # Proportion correct
        return np.mean(true_direction == pred_direction)
    
    def coverage_probability(self, y_true, lower, upper):
        """Prediction interval coverage"""
        in_interval = (y_true >= lower) & (y_true <= upper)
        return np.mean(in_interval)
    
    def mean_interval_width(self, lower, upper):
        """Average width of prediction intervals"""
        return np.mean(upper - lower)
    
    def winkler_score(self, y_true, lower, upper, alpha=0.05):
        """Winkler score for interval forecasts"""
        width = upper - lower
        penalty_lower = (2/alpha) * (lower - y_true) * (y_true < lower)
        penalty_upper = (2/alpha) * (y_true - upper) * (y_true > upper)
        
        return np.mean(width + penalty_lower + penalty_upper)
    
    def summary(self, y_true, y_pred, y_train, seasonal_period=1):
        """Compute all metrics"""
        metrics = {
            'MAE': self.mae(y_true, y_pred),
            'RMSE': self.rmse(y_true, y_pred),
            'MAPE': self.mape(y_true, y_pred),
            'sMAPE': self.smape(y_true, y_pred),
            'MASE': self.mase(y_true, y_pred, y_train, seasonal_period),
            'Theil_U': self.theil_u(y_true, y_pred, y_train),
            'MDA': self.mda(y_true, y_pred)
        }
        return metrics
