# Frequency Domain and Spectral Analysis in Time Series

## 1. Concept Skeleton
**Definition:** Decompose time series into frequency components via Fourier transform; analyze power spectral density to identify dominant cycles and periodicities  
**Purpose:** Detect hidden periodicities; separate signal from noise; filter specific frequencies; understand cyclical behavior; diagnose time series structure  
**Prerequisites:** Complex numbers, Fourier analysis, convolution, linear algebra, stationary processes, autocovariance functions

## 2. Comparative Framing
| Method | Periodogram | Welch's Method | Multitaper | Blackman-Tukey | Parametric (AR) |
|--------|------------|----------------|------------|----------------|-----------------|
| **Estimator** | Raw FFT | Averaged segments | Multiple tapers | Smoothed ACF | AR model fit |
| **Variance** | High (inconsistent) | Reduced | Low (optimal) | Reduced | Model-dependent |
| **Bias** | Low | Moderate | Low | Moderate | High (wrong order) |
| **Resolution** | High | Lower | High | Controllable | High |
| **Use Case** | Quick check | General purpose | High precision | Classical | Short series |

| Transform | Fourier Transform | Wavelet Transform | Hilbert Transform | Z-Transform | Laplace Transform |
|-----------|------------------|-------------------|-------------------|-------------|-------------------|
| **Domain** | Time → Frequency | Time-Frequency | Instantaneous freq | Discrete | Continuous (s-domain) |
| **Localization** | Global | Local (time-freq) | Time-varying | NA | NA |
| **Linearity** | Linear | Linear | Linear | Linear | Linear |
| **Invertible** | Yes | Yes | No (phase only) | Yes | Yes |
| **Best For** | Stationary signals | Non-stationary | Amplitude/phase | DSP, control | Continuous systems |

## 3. Examples + Counterexamples

**Simple Example:**  
Sunspot activity (11-year cycle): Periodogram shows sharp peak at frequency ω=2π/11. Power spectral density confirms regular oscillation dominates variance.

**Perfect Fit:**  
Ocean tides: Multiple periodicities (daily, monthly lunar). Spectral analysis identifies all harmonic components—diurnal (24h), semidiurnal (12h), fortnightly (14-day spring/neap). Precise frequency separation.

**Business Cycle:**  
Quarterly GDP: Spectral peak at 2-8 years (8-32 quarters). Band-pass filter extracts cyclical component. Fed uses for output gap estimation.

**Heartbeat (ECG):**  
Dominant frequency ~1 Hz (60 bpm). Spectral analysis detects arrhythmias (irregular peaks), diagnoses heart conditions. Time-frequency (wavelet) captures transient events.

**Poor Fit:**  
Stock returns: No significant spectral peaks—power decays smoothly (1/f noise). Randomness dominates, little cyclical structure. Spectral methods reveal lack of predictable cycles.

**Non-Stationary Failure:**  
Climate change data: Increasing variance over time. Fourier assumes stationarity—violates assumption. Wavelet transform needed (time-varying frequencies). Global spectrum misleading.

## 4. Layer Breakdown
```
Frequency Domain & Spectral Analysis Framework:

├─ Fourier Transform Theory:
│  ├─ Continuous Fourier Transform (CFT):
│  │   ├─ Forward Transform:
│  │   │   X(ω) = ∫_{-∞}^{∞} x(t) e^{-iωt} dt
│  │   │   Decomposes signal into sinusoids
│  │   │   ω: Angular frequency (radians/sec)
│  │   ├─ Inverse Transform:
│  │   │   x(t) = (1/2π) ∫_{-∞}^{∞} X(ω) e^{iωt} dω
│  │   │   Reconstructs time signal from frequencies
│  │   ├─ Properties:
│  │   │   Linearity: F{ax + by} = aF{x} + bF{y}
│  │   │   Time shift: F{x(t-τ)} = e^{-iωτ} X(ω)
│  │   │   Frequency shift: F{e^{iω₀t}x(t)} = X(ω-ω₀)
│  │   │   Convolution: F{x*y} = X(ω)Y(ω)
│  │   │   Parseval: ∫|x(t)|² dt = (1/2π)∫|X(ω)|² dω
│  │   └─ Interpretation:
│  │       X(ω): Amplitude & phase at frequency ω
│  │       |X(ω)|: Amplitude spectrum
│  │       arg(X(ω)): Phase spectrum
│  ├─ Discrete Fourier Transform (DFT):
│  │   ├─ Forward DFT:
│  │   │   X_k = Σ_{n=0}^{N-1} x_n e^{-i2πkn/N}
│  │   │   k = 0, 1, ..., N-1 (frequency bins)
│  │   ├─ Inverse DFT:
│  │   │   x_n = (1/N) Σ_{k=0}^{N-1} X_k e^{i2πkn/N}
│  │   ├─ Frequency Resolution:
│  │   │   Δω = 2π/N (radians per sample)
│  │   │   Δf = 1/(NΔt) Hz (if sampling interval Δt)
│  │   ├─ Nyquist Frequency:
│  │   │   ω_N = π (half sampling rate)
│  │   │   f_N = 1/(2Δt) Hz
│  │   │   Maximum detectable frequency
│  │   │   Aliasing: Frequencies above fold back
│  │   ├─ Symmetry:
│  │   │   Real signal: X_{N-k} = X_k* (conjugate symmetry)
│  │   │   Only need k=0 to N/2 (positive frequencies)
│  │   └─ Matrix Form:
│  │       X = W x, where W_kn = e^{-i2πkn/N}
│  │       W: DFT matrix (orthogonal)
│  └─ Fast Fourier Transform (FFT):
│      ├─ Cooley-Tukey Algorithm:
│      │   Divide-and-conquer recursion
│      │   Complexity: O(N log N) vs O(N²)
│      │   Requires N = 2^m (power of 2)
│      ├─ Radix-2 Decimation:
│      │   Split into even/odd indices
│      │   DFT_N = DFT_{N/2}(even) + W^k DFT_{N/2}(odd)
│      │   Recursive until N=1
│      ├─ Zero-Padding:
│      │   Pad to next power of 2 for efficiency
│      │   Interpolates spectrum (no new info)
│      └─ Implementation:
│          NumPy: np.fft.fft(), scipy.fft.fft()
│          FFTW library (fastest)
├─ Power Spectral Density (PSD):
│  ├─ Definition:
│  │   Distribution of signal power across frequencies
│  │   S(ω): Power per unit frequency
│  │   Measures contribution of each frequency to total variance
│  ├─ Relationship to Autocovariance:
│  │   ├─ Wiener-Khinchin Theorem:
│  │   │   S(ω) = Σ_{h=-∞}^{∞} γ(h) e^{-iωh}
│  │   │   PSD is Fourier transform of ACF
│  │   │   γ(h): Autocovariance at lag h
│  │   ├─ Inverse:
│  │   │   γ(h) = (1/2π) ∫_{-π}^{π} S(ω) e^{iωh} dω
│  │   │   ACF is inverse FT of PSD
│  │   └─ Total Variance:
│  │       Var(X) = γ(0) = (1/2π) ∫_{-π}^{π} S(ω) dω
│  │       Integrate PSD over all frequencies
│  ├─ Interpretation:
│  │   ├─ Peaks: Dominant periodicities (cycles)
│  │   ├─ Flat spectrum: White noise (all frequencies equal)
│  │   ├─ Low-frequency power: Slow trends, long memory
│  │   ├─ High-frequency power: Rapid fluctuations, noise
│  │   └─ Shape indicates process type:
│  │       AR(1): S(ω) ∝ 1/|1 - φe^{-iω}|²
│  │       MA(1): S(ω) ∝ |1 + θe^{-iω}|²
│  ├─ Units:
│  │   Power per frequency (e.g., W/Hz)
│  │   Depends on signal units squared
│  └─ Normalized Spectrum:
│      f(ω) = S(ω) / ∫S(ω)dω
│      Probability density over frequencies
│      Sums to 1
├─ Periodogram (PSD Estimator):
│  ├─ Definition:
│  │   I(ω_k) = (1/N) |X_k|²
│  │   Sample spectrum from finite data
│  │   Raw, unsmoothed estimate
│  ├─ Computation:
│  │   1. Compute DFT: X_k = FFT(x)
│  │   2. Square magnitude: |X_k|²
│  │   3. Normalize: Divide by N
│  ├─ Properties:
│  │   ├─ Asymptotic Bias:
│  │   │   E[I(ω)] → S(ω) as N → ∞
│  │   │   Consistent estimator (unbiased limit)
│  │   ├─ Variance Problem:
│  │   │   Var[I(ω)] ≈ S(ω)² (does NOT decrease with N)
│  │   │   Inconsistent! (variance doesn't vanish)
│  │   │   Periodogram "jumpy" even for large N
│  │   ├─ Chi-Squared Distribution:
│  │   │   2I(ω)/S(ω) ~ χ²(2) approximately
│  │   │   At each frequency independently
│  │   └─ Bias from Spectral Leakage:
│  │       Finite sample → rectangular window
│  │       Convolution with sinc function in frequency
│  │       Smears sharp peaks
│  ├─ Ordinate Values:
│  │   Scaled periodogram: (2πI(ω))/N
│  │   Matches theoretical spectrum normalization
│  └─ Frequencies:
│      ω_k = 2πk/N for k = 0, 1, ..., N/2
│      Fundamental frequency: 2π/N
├─ Spectral Estimation Methods:
│  ├─ Smoothed Periodogram (Daniell Kernel):
│  │   ├─ Idea: Average neighboring frequencies
│  │   │   Ŝ(ω) = Σ w_j I(ω + ω_j)
│  │   │   w_j: Smoothing weights (kernel)
│  │   ├─ Daniell Window:
│  │   │   Equal weights over bandwidth 2m+1
│  │   │   w_j = 1/(2m+1) for |j| ≤ m
│  │   ├─ Bias-Variance Tradeoff:
│  │   │   Larger m → lower variance, higher bias
│  │   │   Smooths over m frequencies
│  │   │   Reduces resolution
│  │   └─ Effective Degrees of Freedom:
│  │       df ≈ 2N/(2m+1)
│  │       More smoothing → more df → lower variance
│  ├─ Welch's Method:
│  │   ├─ Algorithm:
│  │   │   1. Divide series into L overlapping segments
│  │   │   2. Apply window (Hamming, Hanning) to each
│  │   │   3. Compute periodogram for each segment
│  │   │   4. Average across segments
│  │   ├─ Overlap:
│  │   │   Typically 50% overlap
│  │   │   Increases effective segments without more data
│  │   ├─ Windowing:
│  │   │   Hamming: w(n) = 0.54 - 0.46cos(2πn/(N-1))
│  │   │   Hanning: w(n) = 0.5(1 - cos(2πn/(N-1)))
│  │   │   Reduces spectral leakage
│  │   │   Tapers endpoints to zero
│  │   ├─ Variance Reduction:
│  │   │   Var[Ŝ_Welch] ≈ S(ω)² / L
│  │   │   L segments → L-fold variance reduction
│  │   ├─ Resolution Loss:
│  │   │   Segment length M < N
│  │   │   Coarser frequency grid: 2π/M vs 2π/N
│  │   └─ Parameters:
│  │       Segment length M = N/L
│  │       Trade resolution for variance
│  ├─ Multitaper Method (Thomson 1982):
│  │   ├─ Motivation:
│  │   │   Single taper (window) = one realization
│  │   │   Multiple orthogonal tapers → independent estimates
│  │   │   Average for variance reduction
│  │   ├─ Slepian Sequences (DPSS):
│  │   │   Discrete Prolate Spheroidal Sequences
│  │   │   Optimal concentration in frequency band [−W, W]
│  │   │   Orthogonal set of K tapers
│  │   │   Maximize energy in bandwidth 2W
│  │   ├─ Algorithm:
│  │   │   1. Choose bandwidth W and K tapers
│  │   │   2. Compute K eigenspectra: Ŝ_k(ω) = |Σ v_k(n)x_n e^{-iωn}|²
│  │   │   3. Average: Ŝ_MT(ω) = (1/K) Σ Ŝ_k(ω)
│  │   │   v_k(n): k-th Slepian taper
│  │   ├─ Advantages:
│  │   │   Low bias and variance
│  │   │   No ad-hoc windowing
│  │   │   Theoretically optimal
│  │   │   Adaptive weighting possible
│  │   ├─ Parameters:
│  │   │   Time-bandwidth product: NW
│  │   │   Number of tapers: K ≈ 2NW - 1
│  │   │   Typical: NW = 4, K = 7
│  │   └─ Degrees of Freedom:
│  │       df ≈ 2K
│  │       More tapers → lower variance
│  ├─ Blackman-Tukey Method:
│  │   ├─ Indirect Approach:
│  │   │   1. Estimate ACF: r̂(h) = (1/N)Σ x_t x_{t+h}
│  │   │   2. Truncate at lag M < N
│  │   │   3. Apply lag window w(h)
│  │   │   4. Fourier transform: Ŝ(ω) = Σ w(h)r̂(h)e^{-iωh}
│  │   ├─ Lag Window:
│  │   │   Bartlett: w(h) = 1 - |h|/M
│  │   │   Parzen: Quadratic taper
│  │   │   Tukey-Hanning: Raised cosine
│  │   ├─ Rationale:
│  │   │   ACF estimates poor at large lags (few pairs)
│  │   │   Truncation reduces variance
│  │   │   Window smooths further
│  │   └─ Equivalent to Smoothed Periodogram:
│  │       Different computational path, similar result
│  │       Lag window ↔ Frequency smoothing kernel
│  └─ Parametric Methods:
│      ├─ AR Spectral Estimation:
│      │   ├─ Model: x_t = Σ φ_i x_{t-i} + ε_t
│      │   ├─ Spectrum:
│      │   │   S(ω) = σ²_ε / |1 - Σ φ_i e^{-iωi}|²
│      │   │   Smooth, continuous
│      │   │   Sharp peaks possible (high-order AR)
│      │   ├─ Estimation:
│      │   │   Fit AR(p) via Yule-Walker, Burg, MLE
│      │   │   Choose order p (AIC, BIC)
│      │   │   Compute S(ω) from coefficients
│      │   ├─ Advantages:
│      │   │   High resolution (sharp peaks)
│      │   │   Good for short series
│      │   │   Smooth spectrum (no erratic jumps)
│      │   └─ Disadvantages:
│      │       Model misspecification → bias
│      │       Order selection critical
│      │       Can create spurious peaks
│      ├─ Maximum Entropy Method (MEM):
│      │   Equivalent to AR spectrum
│      │   Maximizes entropy subject to ACF constraints
│      └─ ARMA Spectrum:
│          S(ω) = σ²|1 + Σ θ_j e^{-iωj}|² / |1 - Σ φ_i e^{-iωi}|²
│          More flexible than pure AR
├─ Spectral Analysis in Practice:
│  ├─ Preprocessing:
│  │   ├─ Detrending:
│  │   │   Remove mean: x_t - x̄
│  │   │   Remove linear trend: Regression residuals
│  │   │   Necessary for non-stationary series
│  │   ├─ Differencing:
│  │   │   ∇x_t = x_t - x_{t-1}
│  │   │   Removes unit root, stochastic trend
│  │   │   Alters spectrum: S_∇(ω) = |1-e^{-iω}|² S(ω)
│  │   ├─ Tapering:
│  │   │   Apply window to entire series (cosine taper)
│  │   │   Reduces spectral leakage from endpoints
│  │   │   10% taper common (first/last 5%)
│  │   └─ Zero-Padding:
│  │       Extend with zeros to increase FFT length
│  │       Interpolates spectrum (cosmetic, no new info)
│  ├─ Peak Detection:
│  │   ├─ Identify Local Maxima:
│  │   │   Find ω where S(ω) > S(ω±Δω)
│  │   │   Significant peaks above threshold
│  │   ├─ Fisher's Test:
│  │   │   g = max I(ω) / Σ I(ω)
│  │   │   Test if largest peak is significant
│  │   │   Null: White noise (no peak)
│  │   ├─ Periodogram Ordinate Test:
│  │   │   Compare peak height to χ² threshold
│  │   │   I(ω) / σ² ~ χ²(2)
│  │   └─ Harmonic Analysis:
│  │       Multiple peaks at integer multiples
│  │       f, 2f, 3f, ... (fundamental + harmonics)
│  ├─ Bandwidth Selection:
│  │   ├─ Smoothing Bandwidth:
│  │   │   Wider → lower variance, higher bias
│  │   │   Narrower → higher variance, lower bias
│  │   ├─ Rule of Thumb:
│  │   │   m ≈ √N (Daniell smoothing span)
│  │   │   Balances bias-variance
│  │   └─ Cross-Validation:
│  │       Split data, optimize bandwidth
│  │       Minimize prediction error
│  ├─ Confidence Intervals:
│  │   ├─ Chi-Squared Approximation:
│  │   │   (df × Ŝ(ω)) / χ²_{α/2}(df) ≤ S(ω) ≤ (df × Ŝ(ω)) / χ²_{1-α/2}(df)
│  │   │   df: Equivalent degrees of freedom
│  │   │   Asymmetric intervals
│  │   ├─ Log Scale:
│  │   │   More symmetric on log(S(ω))
│  │   │   Easier to visualize
│  │   └─ Bootstrap:
│  │       Resample residuals, recompute spectrum
│  │       Empirical confidence bands
│  └─ Coherence and Cross-Spectrum:
│      ├─ Cross-Spectral Density:
│      │   S_xy(ω) = Cov(X(ω), Y(ω))
│      │   Measures linear association at frequency ω
│      │   Complex-valued: Amplitude + phase
│      ├─ Coherence:
│      │   C_xy(ω) = |S_xy(ω)|² / (S_xx(ω) S_yy(ω))
│      │   Analogous to R² at each frequency
│      │   Range: [0, 1]
│      │   High coherence → strong linear relationship
│      ├─ Phase Spectrum:
│      │   φ_xy(ω) = arg(S_xy(ω))
│      │   Lead/lag relationship at frequency ω
│      │   Positive → X leads Y
│      └─ Applications:
│          Input-output systems
│          Bivariate co-movement (business cycles)
│          Signal processing (filtering)
├─ Wavelet Analysis (Time-Frequency):
│  ├─ Motivation:
│  │   Fourier: Global frequency, no time localization
│  │   Non-stationary signals need time-varying spectrum
│  │   Transient events, changing cycles
│  ├─ Continuous Wavelet Transform (CWT):
│  │   ├─ Definition:
│  │   │   W(a, b) = (1/√a) ∫ x(t) ψ*((t-b)/a) dt
│  │   │   a: Scale (~ 1/frequency)
│  │   │   b: Time shift (location)
│  │   │   ψ(t): Mother wavelet
│  │   ├─ Mother Wavelets:
│  │   │   Morlet: ψ(t) = e^{iω₀t} e^{-t²/2} (complex, oscillatory)
│  │   │   Mexican Hat: ψ(t) = (1 - t²)e^{-t²/2} (real, symmetric)
│  │   │   Haar: Piecewise constant (simplest)
│  │   ├─ Scalogram:
│  │   │   |W(a, b)|²: Wavelet power spectrum
│  │   │   Time-frequency heatmap
│  │   │   High values → strong signal at scale a, time b
│  │   └─ Interpretation:
│  │       Small a: High frequency, fine detail
│  │       Large a: Low frequency, coarse structure
│  ├─ Discrete Wavelet Transform (DWT):
│  │   ├─ Dyadic Grid:
│  │   │   a = 2^j, b = k×2^j (powers of 2)
│  │   │   Efficient computation
│  │   ├─ Filter Bank:
│  │   │   Decompose into approximation + detail
│  │   │   Recursive: Multi-resolution analysis
│  │   │   Fast algorithm (O(N))
│  │   ├─ Wavelet Families:
│  │   │   Daubechies (db4, db10): Compact support
│  │   │   Symlets: Nearly symmetric
│  │   │   Coiflets: Vanishing moments
│  │   └─ Applications:
│  │       Signal denoising (threshold detail coefficients)
│  │       Compression (keep large coefficients)
│  │       Feature extraction
│  ├─ Wavelet Coherence:
│  │   Time-varying coherence between two signals
│  │   Identifies co-movement at specific times/frequencies
│  │   Popular in economics (business cycle synchronization)
│  └─ Edge Effects (Cone of Influence):
│      Boundary distortion from finite data
│      Results unreliable near edges
│      Cone of influence: Valid region
├─ Short-Time Fourier Transform (STFT):
│  ├─ Definition:
│  │   STFT(t, ω) = ∫ x(τ) w(τ - t) e^{-iωτ} dτ
│  │   Windowed Fourier transform
│  │   Slide window along time, compute DFT
│  ├─ Spectrogram:
│  │   |STFT(t, ω)|²: Power at time t, frequency ω
│  │   Time-frequency representation
│  ├─ Heisenberg Uncertainty:
│  │   Δt × Δω ≥ constant
│  │   Cannot have perfect time AND frequency resolution
│  │   Fixed window → fixed resolution
│  ├─ Comparison to Wavelets:
│  │   STFT: Fixed resolution (window size)
│  │   Wavelet: Adaptive (narrow at high freq, wide at low)
│  │   Wavelet generally preferred for non-stationary
│  └─ Applications:
│      Speech recognition (phonemes change over time)
│      Music analysis (notes, chords)
│      Radar, sonar (Doppler shift)
├─ Practical Considerations:
│  ├─ Choice of Method:
│  │   ├─ Quick exploratory: Raw periodogram
│  │   ├─ General purpose: Welch's method
│  │   ├─ High precision: Multitaper
│  │   ├─ Short series: AR spectrum
│  │   ├─ Non-stationary: Wavelet, STFT
│  │   └─ Two series: Cross-spectrum, coherence
│  ├─ Validation:
│  │   ├─ Surrogate Data Tests:
│  │   │   Phase randomization (preserve spectrum)
│  │   │   Bootstrap resampling
│  │   │   Test if peaks are significant vs noise
│  │   ├─ Residual Checks:
│  │   │   After filtering, check residuals white
│  │   │   Flat spectrum → no remaining structure
│  │   └─ Out-of-Sample:
│  │       Use estimated spectrum to predict
│  │       Validate on holdout data
│  ├─ Common Pitfalls:
│  │   ├─ Aliasing:
│  │   │   Under-sampling → high frequencies fold back
│  │   │   Must sample at >2× max frequency (Nyquist)
│  │   ├─ Spectral Leakage:
│  │   │   Finite data → smearing of peaks
│  │   │   Use tapers/windows
│  │   ├─ Spurious Peaks:
│  │   │   High-variance periodogram creates false peaks
│  │   │   Always smooth or use multiple tapers
│  │   ├─ Over-smoothing:
│  │   │   Lose resolution, miss real peaks
│  │   │   Balance with bandwidth choice
│  │   └─ Non-Stationarity:
│  │       Global spectrum misleading if frequencies change
│  │       Use time-frequency methods
│  └─ Software:
│      R: spectrum(), spec.pgram(), multitaper::spec.mtm()
│      Python: scipy.signal (periodogram, welch, spectrogram)
│              PyWavelets (wavelet analysis)
│      MATLAB: pwelch(), pmtm(), spectrogram()
└─ Applications:
   ├─ Economics/Finance:
   │   Business cycle identification (2-8 year band)
   │   Market microstructure (high-frequency patterns)
   │   Volatility cycles (GARCH persistence)
   ├─ Climate Science:
   │   El Niño cycles (3-7 years)
   │   Solar activity (11-year sunspot cycle)
   │   Ice age periodicities (Milankovitch cycles)
   ├─ Engineering:
   │   Vibration analysis (machinery diagnostics)
   │   Signal processing (noise reduction)
   │   Control systems (frequency response)
   ├─ Medicine:
   │   EEG (brain waves: delta, theta, alpha, beta)
   │   ECG (heart rate variability)
   │   Circadian rhythms (24-hour cycles)
   └─ Astronomy:
      Variable stars (pulsation periods)
      Exoplanet detection (orbital periods)
      Pulsar timing (millisecond periodicities)
```

**Interaction:** Observe time series → Preprocess (detrend, taper) → Compute FFT → Estimate PSD (periodogram, Welch, multitaper) → Identify peaks (dominant cycles) → Test significance → Interpret frequency components → Filter or forecast.

## 5. Mini-Project
Implement spectral estimation methods, compare performance, detect periodicities:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import chi2
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("FREQUENCY DOMAIN AND SPECTRAL ANALYSIS")
print("="*80)

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
        from statsmodels.tsa.ar_model import AutoReg
        
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
        import pywt
        
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
```

## 6. Challenge Round
1. **Aliasing Demonstration:** Undersample sine wave at frequency f. Sample at rate < 2f (violate Nyquist). Show aliased frequency appears at f_alias = |f_sample - f|. Plot periodogram—does it detect true or aliased frequency?

2. **Spectral Leakage:** Compare rectangular vs Hann window on pure sine wave not exactly at FFT bin frequency. Quantify leakage (power spread to adjacent bins). Which window better?

3. **Time-Varying Spectrum:** Simulate signal where dominant frequency changes linearly over time. Compute: (1) global FFT (misleading), (2) STFT spectrogram (shows transition), (3) wavelet scalogram. Which best captures evolution?

4. **Business Cycle Extraction:** Generate GDP-like series with trend, 8-year cycle, annual seasonal, noise. Use band-pass filter [6-10 years]. Does extracted cycle correlate with true cycle? Compare HP filter vs Butterworth.

5. **Multitaper Sensitivity:** Vary NW parameter (2, 4, 8) for multitaper. Fixed signal. How does K (number of tapers) affect: (1) variance of estimate, (2) frequency resolution? Plot bias-variance tradeoff.

## 7. Key References
- [Percival & Walden, "Spectral Analysis for Physical Applications" (1993)](https://www.cambridge.org/core/books/spectral-analysis-for-physical-applications/4A6999CEAA5B0F5EB5C1C66B11F2D7E1) - comprehensive spectral estimation textbook
- [Thomson, "Spectrum Estimation and Harmonic Analysis" (1982)](https://ieeexplore.ieee.org/document/1456701) - original multitaper method paper
- [Welch, "The Use of Fast Fourier Transform for Estimation of Power Spectra" (1967)](https://ieeexplore.ieee.org/document/1161901) - Welch's method foundation

---
**Status:** Core time series tool | **Complements:** Fourier Analysis, Signal Processing, ARIMA Modeling, Filtering, Wavelet Analysis, Cyclical Analysis
