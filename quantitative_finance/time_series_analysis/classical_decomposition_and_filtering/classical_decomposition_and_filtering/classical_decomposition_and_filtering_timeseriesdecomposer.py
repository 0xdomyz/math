from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

class TimeSeriesDecomposer:
    """Classical decomposition methods"""
    
    def __init__(self):
        pass
    
    def additive_decomposition(self, y, period=12):
        """
        Classical additive decomposition: Y = T + S + I
        
        Parameters:
        - y: Time series (array)
        - period: Seasonal period (e.g., 12 for monthly)
        """
        n = len(y)
        
        # Step 1: Estimate trend using centered moving average
        if period % 2 == 0:
            # Even period: 2×m MA
            weights = np.ones(period) / period
            weights[0] = weights[-1] = 0.5 / period
            trend = np.convolve(y, weights, mode='same')
            # Fix endpoints
            half_window = period // 2
            trend[:half_window] = np.nan
            trend[-half_window:] = np.nan
        else:
            # Odd period: simple MA
            half_window = period // 2
            trend = np.full(n, np.nan)
            for t in range(half_window, n - half_window):
                trend[t] = np.mean(y[t-half_window:t+half_window+1])
        
        # Step 2: Detrend
        detrended = y - trend
        
        # Step 3: Estimate seasonal component
        seasonal = np.full(n, np.nan)
        seasonal_avg = np.zeros(period)
        
        for s in range(period):
            # Average all observations in season s
            season_vals = detrended[s::period]
            season_vals = season_vals[~np.isnan(season_vals)]
            if len(season_vals) > 0:
                seasonal_avg[s] = np.mean(season_vals)
        
        # Normalize (sum to zero)
        seasonal_avg -= np.mean(seasonal_avg)
        
        # Replicate pattern
        for t in range(n):
            seasonal[t] = seasonal_avg[t % period]
        
        # Step 4: Irregular component
        irregular = y - trend - seasonal
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'irregular': irregular,
            'seasonal_avg': seasonal_avg
        }
    
    def multiplicative_decomposition(self, y, period=12):
        """
        Classical multiplicative decomposition: Y = T × S × I
        Convert to additive via log transform
        """
        # Take logs (ensure positive)
        y_log = np.log(np.maximum(y, 1e-10))
        
        # Additive decomposition on logs
        decomp_log = self.additive_decomposition(y_log, period)
        
        # Back-transform
        trend = np.exp(decomp_log['trend'])
        seasonal = np.exp(decomp_log['seasonal'])
        irregular = np.exp(decomp_log['irregular'])
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'irregular': irregular,
            'seasonal_avg': np.exp(decomp_log['seasonal_avg'])
        }
    
    def stl_decomposition(self, y, period=12, n_s=7, n_t=None, n_l=13, n_i=2, n_o=0):
        """
        STL: Seasonal and Trend decomposition using LOESS
        Simplified implementation
        
        Parameters:
        - n_s: Seasonal smoothing parameter (odd, ≥3)
        - n_t: Trend smoothing parameter (odd)
        - n_l: Low-pass filter length
        - n_i: Inner loop iterations
        - n_o: Outer loop iterations (robustness)
        """
        if n_t is None:
            n_t = int(np.ceil((1.5 * period) / (1 - 1.5 / n_s)))
            if n_t % 2 == 0:
                n_t += 1
        
        n = len(y)
        seasonal = np.zeros(n)
        trend = np.zeros(n)
        weights = np.ones(n)
        
        for outer in range(max(1, n_o)):
            for inner in range(n_i):
                # Step 1: Detrend
                detrended = y - trend
                
                # Step 2: Cycle-subseries smoothing
                seasonal_temp = np.zeros(n)
                for s in range(period):
                    # Extract subseries for season s
                    indices = np.arange(s, n, period)
                    sub_series = detrended[indices]
                    
                    # LOESS smooth (simplified: moving average)
                    smoothed = self._moving_average_smooth(sub_series, n_s, weights[indices])
                    seasonal_temp[indices] = smoothed
                
                # Step 3: Low-pass filter on seasonal
                seasonal = self._moving_average_smooth(seasonal_temp, n_l, weights)
                
                # Step 4: Deseasonalize and smooth for trend
                deseasonalized = y - seasonal
                trend = self._moving_average_smooth(deseasonalized, n_t, weights)
            
            # Outer loop: Compute robustness weights
            if n_o > 0 and outer < n_o - 1:
                remainder = y - trend - seasonal
                weights = self._bisquare_weights(remainder)
        
        irregular = y - trend - seasonal
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'irregular': irregular
        }
    
    def _moving_average_smooth(self, y, window, weights=None):
        """Weighted moving average"""
        if weights is None:
            weights = np.ones(len(y))
        
        n = len(y)
        smoothed = np.zeros(n)
        half_window = window // 2
        
        for i in range(n):
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            
            window_vals = y[start:end]
            window_weights = weights[start:end]
            
            if np.sum(window_weights) > 0:
                smoothed[i] = np.sum(window_vals * window_weights) / np.sum(window_weights)
            else:
                smoothed[i] = y[i]
        
        return smoothed
    
    def _bisquare_weights(self, residuals):
        """Compute bisquare robustness weights"""
        abs_res = np.abs(residuals)
        median_abs = np.median(abs_res)
        
        if median_abs == 0:
            return np.ones(len(residuals))
        
        standardized = abs_res / (6 * median_abs)
        weights = np.where(standardized < 1, (1 - standardized**2)**2, 0)
        
        return weights
