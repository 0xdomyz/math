from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import chi2
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import warnings

class WaveletAnalyzer:
    """Time-frequency analysis via wavelets"""
    
    def __init__(self, fs=1.0):
        self.fs = fs
    
    def cwt(self, x, scales, wavelet='morl'):
        """
        Continuous Wavelet Transform
        
        Parameters:
        - x: Time series
        - scales: Array of scales (~ 1/frequency)
        - wavelet: Wavelet type ('morl', 'mexh', 'cgau5')
        """
        
        # Compute CWT
        coeffs, freqs = pywt.cwt(x, scales, wavelet, sampling_period=1/self.fs)
        
        # Power (scalogram)
        power = np.abs(coeffs)**2
        
        return freqs, power
    
    def spectrogram(self, x, nperseg=256, noverlap=None):
        """
        Short-Time Fourier Transform spectrogram
        """
        if noverlap is None:
            noverlap = nperseg // 2
        
        freqs, times, Sxx = signal.spectrogram(x, fs=self.fs, nperseg=nperseg,
                                                noverlap=noverlap)
        
        return times, freqs, Sxx

# Scenario 1: Synthetic signal with known frequencies
print("\n" + "="*80)
print("SCENARIO 1: Synthetic Signal - Multiple Periodicities")
print("="*80)

# Generate signal: 3 sinusoids + noise
fs = 100  # 100 Hz sampling
t = np.linspace(0, 10, 1000)  # 10 seconds
f1, f2, f3 = 5, 15, 25  # Hz
signal1 = 2 * np.sin(2*np.pi*f1*t)
signal2 = 1 * np.sin(2*np.pi*f2*t)
signal3 = 0.5 * np.sin(2*np.pi*f3*t)
noise = np.random.normal(0, 0.5, len(t))
x = signal1 + signal2 + signal3 + noise

print(f"\nGenerated Signal:")
print(f"  Duration: 10 seconds, Sampling: {fs} Hz")
print(f"  Components: {f1} Hz (amp=2), {f2} Hz (amp=1), {f3} Hz (amp=0.5)")
print(f"  SNR: ~10 dB")

analyzer = SpectralAnalyzer(fs=fs)

# Periodogram
freqs_pgram, psd_pgram = analyzer.periodogram(x, window='hann')

# Welch's method
freqs_welch, psd_welch = analyzer.welch(x, nperseg=256)

# Multitaper
freqs_mt, psd_mt, k_tapers = analyzer.multitaper(x, nw=4)

# AR spectrum
freqs_ar, psd_ar, ar_order = analyzer.ar_spectrum(x, order=20)

# Find peaks in Welch spectrum
peak_freqs, peak_powers, peak_idx = analyzer.find_peaks(freqs_welch, psd_welch, 
                                                         height_percentile=95)

print(f"\nDetected Peaks (Welch's Method):")
for i, (freq, power) in enumerate(zip(peak_freqs, peak_powers)):
    print(f"  Peak {i+1}: {freq:.2f} Hz (Power: {power:.4f})")

# Fisher's test
g_stat, p_value = analyzer.fishers_test(psd_welch)
print(f"\nFisher's Test for Periodicity:")
print(f"  g-statistic: {g_stat:.4f}")
print(f"  p-value: {p_value:.4e}")
print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")

# Scenario 2: Real-world data - Sunspot numbers
print("\n" + "="*80)
print("SCENARIO 2: Sunspot Data - Solar Cycle Detection")
print("="*80)

# Simulate sunspot-like data (11-year cycle)
years = np.arange(300)  # 300 years
cycle_period = 11  # years
sunspots = (50 + 40 * np.sin(2*np.pi*years/cycle_period) + 
            np.random.normal(0, 10, len(years)))
sunspots = np.maximum(sunspots, 0)  # No negative sunspots

analyzer_sun = SpectralAnalyzer(fs=1.0)  # Annual sampling

# Periodogram
freqs_sun, psd_sun = analyzer_sun.periodogram(sunspots, detrend='linear', window='hann')

# Welch
freqs_sun_w, psd_sun_w = analyzer_sun.welch(sunspots, nperseg=64)

# Find dominant period
peak_freqs_sun, peak_powers_sun, _ = analyzer_sun.find_peaks(freqs_sun_w, psd_sun_w,
                                                              height_percentile=98)

if len(peak_freqs_sun) > 0:
    dominant_freq = peak_freqs_sun[np.argmax(peak_powers_sun)]
    dominant_period = 1 / dominant_freq if dominant_freq > 0 else np.inf
    
    print(f"\nSunspot Cycle Analysis:")
    print(f"  Dominant frequency: {dominant_freq:.4f} cycles/year")
    print(f"  Period: {dominant_period:.2f} years (Expected: 11 years)")
    print(f"  Relative power: {np.max(peak_powers_sun) / np.mean(psd_sun_w):.1f}× mean")

# Scenario 3: Comparison of methods - variance
print("\n" + "="*80)
print("SCENARIO 3: Method Comparison - Variance Reduction")
print("="*80)

# Generate pure signal (no noise) for reference
x_clean = signal1 + signal2 + signal3

# Compare variance at known peaks
freq_tolerance = 0.5  # Hz

methods = {
    'Periodogram': (freqs_pgram, psd_pgram),
    'Welch': (freqs_welch, psd_welch),
    'Multitaper': (freqs_mt, psd_mt)
}

print(f"\nPower Estimates at True Frequencies:")
print(f"{'Method':<15} {'5 Hz':<12} {'15 Hz':<12} {'25 Hz':<12} {'Variance':<10}")
print("-" * 61)

for method_name, (freqs, psd) in methods.items():
    powers = []
    for true_f in [f1, f2, f3]:
        # Find closest frequency
        idx = np.argmin(np.abs(freqs - true_f))
        if np.abs(freqs[idx] - true_f) < freq_tolerance:
            powers.append(psd[idx])
        else:
            powers.append(np.nan)
    
    # Variance (std of estimate across frequencies away from peaks)
    # Use frequencies away from peaks
    mask = np.ones(len(freqs), dtype=bool)
    for true_f in [f1, f2, f3]:
        mask &= np.abs(freqs - true_f) > 2  # More than 2 Hz away
    
    variance = np.std(psd[mask]) if np.any(mask) else 0
    
    print(f"{method_name:<15} {powers[0]:<12.4f} {powers[1]:<12.4f} {powers[2]:<12.4f} {variance:<10.4f}")

# Scenario 4: Coherence between two series
print("\n" + "="*80)
print("SCENARIO 4: Coherence Analysis - Bivariate Relationship")
print("="*80)

# Generate two related series
x1 = 2 * np.sin(2*np.pi*10*t) + np.random.normal(0, 0.3, len(t))
x2 = 1.5 * np.sin(2*np.pi*10*t + np.pi/4) + np.random.normal(0, 0.3, len(t))  # Phase shifted

# Add independent components
x1 += 0.5 * np.sin(2*np.pi*20*t) + np.random.normal(0, 0.2, len(t))
x2 += 0.5 * np.sin(2*np.pi*30*t) + np.random.normal(0, 0.2, len(t))

# Compute coherence
freqs_coh, coh, phase = analyzer.coherence(x1, x2, nperseg=128)

# Find high coherence frequencies
high_coh_mask = coh > 0.7
high_coh_freqs = freqs_coh[high_coh_mask]

print(f"\nCoherence Analysis:")
print(f"  Frequencies with coherence > 0.7:")
for freq in high_coh_freqs[:5]:  # Top 5
    idx = np.argmin(np.abs(freqs_coh - freq))
    print(f"    {freq:.2f} Hz: Coherence = {coh[idx]:.3f}, Phase = {phase[idx]:.3f} rad")

# Scenario 5: Time-frequency analysis (non-stationary)
print("\n" + "="*80)
print("SCENARIO 5: Time-Frequency Analysis - Non-Stationary Signal")
print("="*80)

# Generate chirp signal (frequency increases over time)
t_chirp = np.linspace(0, 10, 1000)
f_start, f_end = 5, 30
chirp = signal.chirp(t_chirp, f_start, t_chirp[-1], f_end)
chirp += np.random.normal(0, 0.3, len(t_chirp))

print(f"\nChirp Signal: Frequency sweeps from {f_start} Hz to {f_end} Hz")

wavelet_analyzer = WaveletAnalyzer(fs=fs)

# Spectrogram (STFT)
times_stft, freqs_stft, Sxx_stft = wavelet_analyzer.spectrogram(chirp, nperseg=128)

print(f"  STFT computed: {len(times_stft)} time windows, {len(freqs_stft)} frequencies")

# Compare to global spectrum (misleading for non-stationary)
freqs_global, psd_global = analyzer.periodogram(chirp)

print(f"  Global spectrum shows broad peak (misleading - averages over time)")

# Scenario 6: Band-pass filtering
print("\n" + "="*80)
print("SCENARIO 6: Spectral Filtering - Extract Specific Frequency Band")
print("="*80)

# Original signal (from Scenario 1)
print(f"\nOriginal signal has components at {f1}, {f2}, {f3} Hz")

# Design band-pass filter to extract 15 Hz component
lowcut, highcut = 12, 18
nyquist = fs / 2
low = lowcut / nyquist
high = highcut / nyquist

# Butterworth filter
b, a = signal.butter(4, [low, high], btype='band')
x_filtered = signal.filtfilt(b, a, x)

# Compute spectrum of filtered signal
freqs_filt, psd_filt = analyzer.periodogram(x_filtered, window='hann')

print(f"\nBand-pass filter [{lowcut}-{highcut} Hz] applied")
print(f"  Original signal RMS: {np.std(x):.3f}")
print(f"  Filtered signal RMS: {np.std(x_filtered):.3f}")

# Correlation with pure 15 Hz component
corr_15hz = np.corrcoef(x_filtered, signal2)[0, 1]
print(f"  Correlation with true 15 Hz component: {corr_15hz:.3f}")

# Visualizations
fig, axes = plt.subplots(3, 3, figsize=(18, 14))

# Plot 1: Time series
ax = axes[0, 0]
ax.plot(t[:200], x[:200], 'b-', linewidth=1, alpha=0.7)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title('Synthetic Signal (First 2 seconds)')
ax.grid(alpha=0.3)

# Plot 2: Periodogram vs Welch
ax = axes[0, 1]
ax.semilogy(freqs_pgram, psd_pgram, 'b-', alpha=0.5, linewidth=1, label='Periodogram')
ax.semilogy(freqs_welch, psd_welch, 'r-', linewidth=2, label='Welch')
ax.axvline(f1, color='g', linestyle='--', alpha=0.7, label=f'{f1} Hz')
ax.axvline(f2, color='g', linestyle='--', alpha=0.7)
ax.axvline(f3, color='g', linestyle='--', alpha=0.7)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD')
ax.set_title('Periodogram vs Welch (Log Scale)')
ax.set_xlim(0, 50)
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: All methods comparison
ax = axes[0, 2]
ax.plot(freqs_pgram, psd_pgram, 'b-', alpha=0.4, linewidth=1, label='Periodogram')
ax.plot(freqs_welch, psd_welch, 'r-', linewidth=2, label='Welch')
ax.plot(freqs_mt, psd_mt, 'g-', linewidth=2, label=f'Multitaper (k={k_tapers})')
ax.plot(freqs_ar, psd_ar, 'm--', linewidth=1.5, label=f'AR({ar_order})')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD')
ax.set_title('Method Comparison')
ax.set_xlim(0, 50)
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Sunspot spectrum
ax = axes[1, 0]
ax.plot(1/freqs_sun_w[1:], psd_sun_w[1:], 'b-', linewidth=2)
ax.axvline(11, color='r', linestyle='--', linewidth=2, label='Expected (11 years)')
if len(peak_freqs_sun) > 0 and dominant_freq > 0:
    ax.axvline(dominant_period, color='g', linestyle=':', linewidth=2, label=f'Detected ({dominant_period:.1f} years)')
ax.set_xlabel('Period (years)')
ax.set_ylabel('PSD')
ax.set_title('Sunspot Cycle - Power Spectrum')
ax.set_xlim(0, 50)
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Sunspot time series
ax = axes[1, 1]
ax.plot(years, sunspots, 'b-', linewidth=1, alpha=0.7)
ax.set_xlabel('Year')
ax.set_ylabel('Sunspot Count')
ax.set_title('Sunspot Time Series (Simulated)')
ax.grid(alpha=0.3)

# Plot 6: Coherence
ax = axes[1, 2]
ax.plot(freqs_coh, coh, 'b-', linewidth=2)
ax.axhline(0.7, color='r', linestyle='--', alpha=0.7, label='Threshold (0.7)')
ax.axvline(10, color='g', linestyle=':', alpha=0.7, label='Shared 10 Hz')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Coherence')
ax.set_title('Coherence Between Two Series')
ax.set_ylim(0, 1)
ax.legend()
ax.grid(alpha=0.3)

# Plot 7: Phase spectrum
ax = axes[2, 0]
ax.plot(freqs_coh, phase, 'b-', linewidth=2)
ax.axhline(0, color='k', linestyle='-', alpha=0.3)
ax.axvline(10, color='g', linestyle=':', alpha=0.7)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Phase (radians)')
ax.set_title('Phase Spectrum')
ax.grid(alpha=0.3)

# Plot 8: Spectrogram (STFT)
ax = axes[2, 1]
im = ax.pcolormesh(times_stft, freqs_stft, 10*np.log10(Sxx_stft + 1e-10), 
                   shading='gouraud', cmap='viridis')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Frequency (Hz)')
ax.set_title('Spectrogram (STFT) - Chirp Signal')
ax.set_ylim(0, 50)
plt.colorbar(im, ax=ax, label='PSD (dB)')

# Plot 9: Filtered signal
ax = axes[2, 2]
ax.plot(t[:200], x[:200], 'gray', alpha=0.4, linewidth=1, label='Original')
ax.plot(t[:200], x_filtered[:200], 'r-', linewidth=2, label=f'Filtered [{lowcut}-{highcut} Hz]')
ax.plot(t[:200], signal2[:200], 'g--', linewidth=1.5, alpha=0.7, label=f'True 15 Hz')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title('Band-Pass Filtering')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Additional analysis: Confidence intervals
print("\n" + "="*80)
print("SCENARIO 7: Confidence Intervals for Spectral Estimates")
print("="*80)

# Degrees of freedom for Welch
n_segments = int(np.ceil(len(x) / (256 / 2)))  # 50% overlap
df_welch = 2 * n_segments

# Chi-squared confidence interval
alpha = 0.05
lower_factor = df_welch / chi2.ppf(1 - alpha/2, df_welch)
upper_factor = df_welch / chi2.ppf(alpha/2, df_welch)

print(f"\nWelch's Method Confidence Intervals:")
print(f"  Effective degrees of freedom: {df_welch}")
print(f"  95% CI width factor: [{lower_factor:.2f}, {upper_factor:.2f}]")
print(f"  Example at 15 Hz peak:")

# Find 15 Hz peak
idx_15 = np.argmin(np.abs(freqs_welch - 15))
psd_at_15 = psd_welch[idx_15]
ci_lower = psd_at_15 * lower_factor
ci_upper = psd_at_15 * upper_factor

print(f"    Estimate: {psd_at_15:.4f}")
print(f"    95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# Multitaper (more precise)
df_mt = 2 * k_tapers
lower_factor_mt = df_mt / chi2.ppf(1 - alpha/2, df_mt)
upper_factor_mt = df_mt / chi2.ppf(alpha/2, df_mt)

idx_15_mt = np.argmin(np.abs(freqs_mt - 15))
psd_at_15_mt = psd_mt[idx_15_mt]
ci_lower_mt = psd_at_15_mt * lower_factor_mt
ci_upper_mt = psd_at_15_mt * upper_factor_mt

print(f"\nMultitaper Method:")
print(f"  Degrees of freedom: {df_mt}")
print(f"  95% CI width factor: [{lower_factor_mt:.2f}, {upper_factor_mt:.2f}]")
print(f"    Estimate: {psd_at_15_mt:.4f}")
print(f"    95% CI: [{ci_lower_mt:.4f}, {ci_upper_mt:.4f}]")
print(f"  → {(ci_upper - ci_lower) / (ci_upper_mt - ci_lower_mt):.1f}× wider CI for Welch")

# Final plot: CI visualization
fig2, ax = plt.subplots(1, 1, figsize=(12, 6))

# Welch with CI
ax.plot(freqs_welch, psd_welch, 'b-', linewidth=2, label='Welch estimate')
ax.fill_between(freqs_welch, psd_welch * lower_factor, psd_welch * upper_factor,
                alpha=0.3, color='blue', label='95% CI (Welch)')

# Multitaper with CI
ax.plot(freqs_mt, psd_mt, 'r-', linewidth=2, label='Multitaper estimate')
ax.fill_between(freqs_mt, psd_mt * lower_factor_mt, psd_mt * upper_factor_mt,
                alpha=0.3, color='red', label='95% CI (Multitaper)')

# True frequencies
for freq in [f1, f2, f3]:
    ax.axvline(freq, color='g', linestyle='--', alpha=0.5)

ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD')
ax.set_title('Spectral Estimates with 95% Confidence Intervals')
ax.set_xlim(0, 50)
ax.set_ylim(bottom=1e-4)
ax.set_yscale('log')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
