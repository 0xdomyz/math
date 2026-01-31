from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

class TimeSeriesFilter:
    """Various filtering methods"""
    
    def moving_average(self, y, window=5, center=True):
        """Simple moving average filter"""
        if center:
            # Centered MA (symmetric)
            smoothed = np.convolve(y, np.ones(window)/window, mode='same')
        else:
            # Trailing MA (causal, real-time)
            smoothed = np.zeros(len(y))
            for t in range(len(y)):
                start = max(0, t - window + 1)
                smoothed[t] = np.mean(y[start:t+1])
        
        return smoothed
    
    def exponential_smoothing(self, y, alpha=0.3):
        """Simple exponential smoothing"""
        n = len(y)
        smoothed = np.zeros(n)
        smoothed[0] = y[0]
        
        for t in range(1, n):
            smoothed[t] = alpha * y[t] + (1 - alpha) * smoothed[t-1]
        
        return smoothed
    
    def hodrick_prescott(self, y, lam=1600):
        """
        Hodrick-Prescott filter
        Minimize: Σ(y_t - τ_t)² + λ Σ((τ_{t+1} - τ_t) - (τ_t - τ_{t-1}))²
        """
        n = len(y)
        
        # Build second difference matrix K
        # K @ τ computes second differences
        diag_vals = np.array([1, -2, 1])
        offsets = np.array([0, 1, 2])
        K = diags(diag_vals, offsets, shape=(n-2, n)).tocsr()
        
        # Solve: (I + λ K'K) τ = y
        I = diags([1], [0], shape=(n, n))
        A = I + lam * K.T @ K
        
        trend = spsolve(A, y)
        cycle = y - trend
        
        return trend, cycle
    
    def butterworth_filter(self, y, cutoff_freq, fs=1.0, order=5, btype='low'):
        """
        Butterworth filter
        
        Parameters:
        - cutoff_freq: Cutoff frequency (normalized, 0-0.5)
        - fs: Sampling frequency
        - order: Filter order (higher = sharper)
        - btype: 'low', 'high', or 'band'
        """
        nyquist = 0.5 * fs
        normal_cutoff = cutoff_freq / nyquist
        
        # Design Butterworth filter
        b, a = signal.butter(order, normal_cutoff, btype=btype, analog=False)
        
        # Apply zero-phase filter (forward-backward)
        filtered = signal.filtfilt(b, a, y)
        
        return filtered
    
    def band_pass_filter(self, y, low_freq, high_freq, fs=1.0):
        """
        Band-pass filter (extract specific frequency range)
        Uses FFT
        """
        n = len(y)
        
        # FFT
        Y = fft(y)
        freqs = fftfreq(n, 1/fs)
        
        # Create mask
        mask = np.zeros(n)
        mask[(np.abs(freqs) >= low_freq) & (np.abs(freqs) <= high_freq)] = 1
        
        # Apply mask and inverse FFT
        Y_filtered = Y * mask
        filtered = np.real(ifft(Y_filtered))
        
        return filtered
