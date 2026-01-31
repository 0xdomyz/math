from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import chi2
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import warnings

class SpectralAnalyzer:
    """Spectral analysis tools for time series"""
    
    def __init__(self, fs=1.0):
        """
        Parameters:
        - fs: Sampling frequency (samples per unit time)
        """
        self.fs = fs
    
    def periodogram(self, x, detrend='constant', window='boxcar'):
        """
        Compute periodogram (raw spectral estimate)
        
        Parameters:
        - x: Time series
        - detrend: 'constant', 'linear', or None
        - window: Window function ('boxcar', 'hann', 'hamming')
        """
        n = len(x)
        
        # Detrend
        if detrend == 'constant':
            x = x - np.mean(x)
        elif detrend == 'linear':
            t = np.arange(n)
            coef = np.polyfit(t, x, 1)
            x = x - np.polyval(coef, t)
        
        # Apply window
        if window == 'hann':
            w = np.hanning(n)
        elif window == 'hamming':
            w = np.hamming(n)
        else:
            w = np.ones(n)
        
        # Window normalization
        w_norm = w / np.sqrt(np.mean(w**2))
        
        # Compute FFT
        X = fft(x * w_norm)
        
        # Periodogram: |X|^2 / N
        pgram = (np.abs(X)**2) / n
        
        # Frequencies
        freqs = fftfreq(n, 1/self.fs)
        
        # Positive frequencies only
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        pgram = pgram[pos_mask]
        
        # Scale (double power for positive freq, except DC and Nyquist)
        pgram[1:-1] *= 2
        
        return freqs, pgram
    
    def welch(self, x, nperseg=None, noverlap=None, window='hann'):
        """
        Welch's method: Averaged periodogram over segments
        
        Parameters:
        - nperseg: Segment length (default: N/8)
        - noverlap: Overlap length (default: nperseg/2)
        """
        n = len(x)
        
        if nperseg is None:
            nperseg = n // 8
        if noverlap is None:
            noverlap = nperseg // 2
        
        # Use scipy implementation
        freqs, psd = signal.welch(x, fs=self.fs, nperseg=nperseg, 
                                   noverlap=noverlap, window=window,
                                   detrend='constant', scaling='density')
        
        return freqs, psd
    
    def multitaper(self, x, nw=4, k=None):
        """
        Multitaper spectral estimation (simplified)
        
        Parameters:
        - nw: Time-bandwidth product
        - k: Number of tapers (default: 2*nw - 1)
        """
        n = len(x)
        
        if k is None:
            k = int(2 * nw - 1)
        
        # Get Slepian sequences (DPSS)
        tapers = signal.windows.dpss(n, nw, k)
        
        # Compute spectrum for each taper
        spectra = []
        for taper in tapers:
            # Apply taper
            x_tapered = x * taper
            
            # FFT
            X = fft(x_tapered)
            
            # Power spectrum
            spec = np.abs(X)**2 / n
            spectra.append(spec)
        
        # Average across tapers
        psd = np.mean(spectra, axis=0)
        
        # Frequencies
        freqs = fftfreq(n, 1/self.fs)
        
        # Positive frequencies
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        psd = psd[pos_mask]
        psd[1:-1] *= 2
        
        return freqs, psd, k
    
    def ar_spectrum(self, x, order=None):
        """
        Parametric AR spectrum estimation
        
        Parameters:
        - order: AR order (default: AIC selection)
        """
        
        # Remove mean
        x_centered = x - np.mean(x)
        
        # Fit AR model
        if order is None:
            # Try orders 1-20, select by AIC
            best_aic = np.inf
            best_order = 1
            for p in range(1, min(21, len(x)//10)):
                try:
                    model = AutoReg(x_centered, lags=p, old_names=False)
                    result = model.fit()
                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_order = p
                except:
                    pass
            order = best_order
        
        # Fit final model
        model = AutoReg(x_centered, lags=order, old_names=False)
        result = model.fit()
        
        # Extract AR coefficients
        ar_coefs = result.params[1:]  # Exclude intercept
        sigma2 = result.sigma2
        
        # Compute spectrum: S(ω) = σ² / |1 - Σφ_i e^{-iωi}|²
        n_freqs = 512
        freqs = np.linspace(0, self.fs/2, n_freqs)
        omega = 2 * np.pi * freqs / self.fs
        
        # AR polynomial: 1 - φ_1 e^{-iω} - φ_2 e^{-2iω} - ...
        ar_poly = np.ones(n_freqs, dtype=complex)
        for i, phi in enumerate(ar_coefs, start=1):
            ar_poly -= phi * np.exp(-1j * omega * i)
        
        # Spectrum
        psd = sigma2 / (np.abs(ar_poly)**2)
        
        return freqs, psd, order
    
    def find_peaks(self, freqs, psd, height_percentile=90):
        """
        Detect spectral peaks
        
        Parameters:
        - height_percentile: Minimum height as percentile of PSD
        """
        # Threshold
        threshold = np.percentile(psd, height_percentile)
        
        # Find peaks using scipy
        peak_indices, properties = signal.find_peaks(psd, height=threshold)
        
        peak_freqs = freqs[peak_indices]
        peak_powers = psd[peak_indices]
        
        return peak_freqs, peak_powers, peak_indices
    
    def fishers_test(self, psd):
        """
        Fisher's test for periodicity
        Test if largest peak is significant
        
        H0: White noise (no peaks)
        """
        # Test statistic: max / sum
        g = np.max(psd) / np.sum(psd)
        
        # Number of frequencies
        n_freqs = len(psd)
        
        # P-value approximation (for large n)
        # Under H0, g has known distribution
        p_value = 1 - (1 - np.exp(-g * n_freqs))**n_freqs
        
        return g, p_value
    
    def coherence(self, x, y, nperseg=None):
        """
        Coherence between two time series
        
        Returns:
        - freqs: Frequencies
        - coh: Coherence (0-1)
        - phase: Phase spectrum (radians)
        """
        n = len(x)
        if nperseg is None:
            nperseg = n // 8
        
        # Cross-spectral density
        freqs, Pxy = signal.csd(x, y, fs=self.fs, nperseg=nperseg)
        
        # Auto-spectral densities
        _, Pxx = signal.welch(x, fs=self.fs, nperseg=nperseg)
        _, Pyy = signal.welch(y, fs=self.fs, nperseg=nperseg)
        
        # Coherence: |Pxy|^2 / (Pxx * Pyy)
        coh = np.abs(Pxy)**2 / (Pxx * Pyy)
        
        # Phase
        phase = np.angle(Pxy)
        
        return freqs, coh, phase
