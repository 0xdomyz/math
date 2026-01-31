# Machine Learning Signals in Algorithmic Trading

## 1. Concept Skeleton
**Definition:** Supervised/unsupervised ML models (neural networks, random forests, XGBoost, LSTM) generating price predictions and trading signals from historical patterns; learns non-linear relationships in data  
**Purpose:** Exploit market anomalies and patterns missed by linear models, improve signal generation accuracy beyond traditional technical/fundamental approaches, adapt to regime changes  
**Prerequisites:** Machine learning fundamentals, supervised/unsupervised learning, feature engineering, time-series analysis, backtesting methodology

## 2. Comparative Framing
| Model Type | Architecture | Training Data | Prediction Accuracy | Overfitting Risk | Interpretability | Computational Cost |
|------------|--------------|----------------|-------------------|------------------|------------------|-------------------|
| **Linear Regression** | α + β₁X₁ + ... | Minimal | Low (baseline) | Low | High | Negligible |
| **Logistic Regression** | Sigmoid(α + βX) | Minimal | Medium (direction) | Low | High | Negligible |
| **Decision Trees** | Recursive splits | Moderate | Medium-High | Very High | High | Low |
| **Random Forest** | 100+ trees, ensemble | Moderate | High | High | Medium | Medium |
| **XGBoost** | Gradient boosting | Moderate-Large | Very High | Very High | Low | Medium-High |
| **Neural Network (MLP)** | Fully connected layers | Large | High | Very High | Very Low | High |
| **LSTM/RNN** | Sequential memory | Large, time-series | Very High (sequences) | Very High | Very Low | Very High |
| **Convolutional NN** | Spatial filters | Large (images) | Medium-High (patterns) | High | Very Low | High |

## 3. Examples + Counterexamples

**Random Forest Success (OHLCV Features):**  
Train on 10 years daily data: Open, High, Low, Close, Volume + technical indicators (RSI, MACD, Bollinger Bands).  
Features: 50 total. Trees: 100. Prediction: Binary (Up/Down next day).  
Out-of-sample accuracy: 55% (3% above random 50%, but after costs: marginal alpha).  
Backtest: +2% annualized after slippage (realistic).

**LSTM Failure (Overfitted Regime):**  
Train on 2015-2019 bull market, 3-layer LSTM with 100 units/layer.  
Achieves 65% accuracy in-sample, 60% out-of-sample on test set.  
Forward test 2020: Accuracy drops to 48% (worse than random!). Regime change → model breaks.  
Lesson: Backtesting period must include regime changes (bull + bear markets).

**XGBoost Advantage (Feature Extraction):**  
Benchmark: Linear model 52% accuracy.  
XGBoost with same 30 features: 57% accuracy.  
Added non-linear interactions automatically (trees learn combinations).  
Net: 5% improvement in signal accuracy → +1-2% annualized alpha (after costs).

**CNN Pattern Recognition (Price Charts):**  
Train CNN on 10k daily price charts (images of 100-day windows) + labels (up/down next day).  
Learn local patterns (head-and-shoulders, triangles, support/resistance).  
Accuracy: 54%. Seems marginal but outperforms traditional technical analysis.  
Practical: Requires GPU, complex infrastructure.

**Neural Network Overfitting (Classic Trap):**  
Train deep NN (5 layers, 200 units each) on 5 years of data.  
In-sample accuracy: 70% (excellent).  
Out-of-sample validation: 51% (random).  
Full out-of-sample backtest: -5% returns (losses from wrong predictions).  
Lesson: Large models + limited data = memorization, not generalization.

**Ensemble Diversity (Multiple Models):**  
Combine 5 models: Linear, RF, XGBoost, LSTM, SVM.  
Ensemble vote (3+ models agree → signal).  
Reduces false signals (overly confident single model wrong).  
Accuracy: ~54% (between RF at 55% and Linear at 52%), but more stable across regimes.

## 4. Layer Breakdown
```
Machine Learning for Trading:

├─ Data Preparation:
│  ├─ Feature Engineering:
│  │  ├─ Price features (OHLCV):
│  │  │  ├─ Raw: Open, High, Low, Close, Volume
│  │  │  ├─ Derived: Returns, log-returns, high-low range
│  │  │  ├─ Normalized: Z-score, min-max scaling
│  │  │  └─ Lagged: t-1, t-2, t-5 day values
│  │  │
│  │  ├─ Technical Indicators:
│  │  │  ├─ Momentum: RSI, MACD, Stochastic
│  │  │  ├─ Trend: Moving averages, ADX
│  │  │  ├─ Volatility: ATR, Bollinger Bands
│  │  │  └─ Volume: OBV, CMF
│  │  │
│  │  ├─ Fundamental Features:
│  │  │  ├─ Valuation: P/E, PB, dividend yield
│  │  │  ├─ Growth: Revenue growth, earnings growth
│  │  │  └─ Quality: Debt/equity, ROE, margins
│  │  │
│  │  ├─ Market Microstructure:
│  │  │  ├─ Bid-ask spread, order book imbalance
│  │  │  ├─ Volume profile, price impact
│  │  │  ├─ Effective spread, realized volatility
│  │  │  └─ High-frequency features (limit order book events)
│  │  │
│  │  ├─ Alternative Data:
│  │  │  ├─ Sentiment: News sentiment, Twitter mentions
│  │  │  ├─ Satellite: Earnings, shipping, foot traffic
│  │  │  ├─ Options: IV, skew, put/call ratio
│  │  │  └─ Crypto: Blockchain metrics, whale transactions
│  │  │
│  │  └─ Feature Selection:
│  │     ├─ Correlation analysis (remove redundant)
│  │     ├─ Mutual information (capture non-linear relationships)
│  │     ├─ Feature importance from trees
│  │     ├─ Domain knowledge (trade-off quantity vs interpretability)
│  │     └─ Typical: 20-100 features (balance complexity/performance)
│  │
│  ├─ Target Construction:
│  │  ├─ Classification (most common):
│  │  │  ├─ Binary: Up/Down (next day return > 0?)
│  │  │  ├─ Threshold: Up/Down/Neutral (±1% bands)
│  │  │  ├─ Multilevel: Strong up/up/neutral/down/strong down
│  │  │  ├─ Class imbalance: More ups than downs (positive drift)
│  │  │  └─ Mitigation: Over-sampling minority, class weights
│  │  │
│  │  ├─ Regression (less common):
│  │  │  ├─ Predict actual return (e.g., tomorrow's return)
│  │  │  ├─ Continuous target (not binarized)
│  │  │  ├─ More information but harder to predict accurately
│  │  │  └─ Often worse hit rate than classification
│  │  │
│  │  └─ Lookahead Bias (Critical!):
│  │     ├─ Target must be future, not current
│  │     ├─ Common mistake: Use today's close as feature, predict today's close
│  │     ├─ Correct: Use today's data, predict tomorrow
│  │     ├─ Survivorship: Only companies existing at prediction time
│  │     └─ Dividend/split adjustment: Restatement of historical prices
│  │
│  ├─ Data Splitting:
│  │  ├─ Train/Validation/Test:
│  │  │  ├─ Train: 60-70% of data (fit parameters)
│  │  │  ├─ Validation: 10-15% (tune hyperparameters)
│  │  │  ├─ Test: 15-20% (final performance estimate)
│  │  │  ├─ Temporal: Don't shuffle time series! (forward chaining)
│  │  │  └─ Walk-forward: Rolling windows to avoid lookahead
│  │  │
│  │  ├─ Pitfalls:
│  │  │  ├─ Random shuffle of time series (future leaks into past)
│  │  │  ├─ Single split (lucky split, not representative)
│  │  │  ├─ Validation set too small (high variance in accuracy)
│  │  │  └─ Using future data for feature normalization
│  │  │
│  │  └─ Best Practice:
│  │     ├─ K-fold cross-validation (time-aware)
│  │     ├─ Expanding window (train on expanding history)
│  │     ├─ Multiple metrics (accuracy + precision + F1)
│  │     └─ Test must be completely out-of-sample
│  │
│  └─ Class Imbalance:
│     ├─ Typical: 51% ups, 49% downs (slight positive drift)
│     ├─ Extreme: 70% ups, 30% downs (strong trend)
│     ├─ Problem: Model learns to always predict majority class
│     ├─ Solutions:
│     │  ├─ Over-sampling minority class
│     │  ├─ Under-sampling majority class
│     │  ├─ Synthetic data generation (SMOTE)
│     │  ├─ Class weights (penalize misclassifying minority)
│     │  └─ Threshold adjustment (lower threshold for minority)
│     └─ Metric: Use F1-score, not just accuracy
│
├─ Model Selection & Training:
│  ├─ Shallow Models (Interpretable, Fast):
│  │  ├─ Logistic Regression:
│  │  │  ├─ Baseline model (always compare against)
│  │  │  ├─ Simple, interpretable (coefficient = feature importance)
│  │  │  ├─ Fast training (seconds)
│  │  │  ├─ Typically 50-52% accuracy (weak signal)
│  │  │  ├─ Advantage: Stable, explainable, regulatory-friendly
│  │  │  └─ Disadvantage: Linear only, misses non-linear patterns
│  │  │
│  │  ├─ Decision Trees:
│  │  │  ├─ Interpretable decision rules ("If RSI > 70, sell")
│  │  │  ├─ Handles non-linear relationships
│  │  │  ├─ Prone to overfitting (single tree ~70% accuracy, 50% out-of-sample)
│  │  │  ├─ Control: Limit depth (max_depth=5-10)
│  │  │  └─ Solo: Not recommended; use ensemble
│  │  │
│  │  └─ Support Vector Machines (SVM):
│  │     ├─ Non-linear via kernel trick (RBF, polynomial)
│  │     ├─ Good for small-moderate datasets (~1000 samples)
│  │     ├─ Typical accuracy: 53-55%
│  │     ├─ Hyperparameter tuning critical (C, gamma)
│  │     └─ Slow on large data (O(n²) training)
│  │
│  ├─ Ensemble Models (Balanced, Popular):
│  │  ├─ Random Forest:
│  │  │  ├─ Ensemble of decision trees (100+ trees)
│  │  │  ├─ Bootstrap aggregating (bagging) reduces variance
│  │  │  ├─ Typical accuracy: 54-56%
│  │  │  ├─ Robust, less overfitting than single tree
│  │  │  ├─ Fast training (parallel on multi-core)
│  │  │  ├─ Feature importance: Measure contribution
│  │  │  ├─ Hyperparameters: n_trees, max_depth, min_samples_leaf
│  │  │  └─ Industry standard for tabular data
│  │  │
│  │  ├─ Gradient Boosting (XGBoost, LightGBM):
│  │  │  ├─ Sequential tree building (each tree corrects prior)
│  │  │  ├─ More powerful than random forest
│  │  │  ├─ Typical accuracy: 55-58% (best tabular performance)
│  │  │  ├─ Extremely prone to overfitting (regularization critical)
│  │  │  ├─ Hyperparameters: learning_rate, n_trees, max_depth, subsample
│  │  │  ├─ Early stopping: Stop when validation accuracy plateaus
│  │  │  ├─ Computational cost: Higher than RF (sequential vs parallel)
│  │  │  └─ Best practice: Aggressive regularization + cross-validation
│  │  │
│  │  └─ Voting/Stacking (Meta-Learning):
│  │     ├─ Combine multiple models (diversity)
│  │     ├─ Hard voting: Majority vote (RF, XGBoost, SVM all predict, take mode)
│  │     ├─ Soft voting: Average probabilities
│  │     ├─ Stacking: Train meta-model on outputs of base models
│  │     ├─ Advantage: Reduce variance, capture different signal aspects
│  │     ├─ Typical improvement: 1-2% accuracy
│  │     └─ Disadvantage: Complexity, harder to interpret
│  │
│  ├─ Deep Learning (Powerful, Complex):
│  │  ├─ Multilayer Perceptron (MLP):
│  │  │  ├─ Fully connected layers: Input → Hidden₁ → Hidden₂ → Output
│  │  │  ├─ Non-linear activations (ReLU): Learns arbitrary functions
│  │  │  ├─ Architecture: Input(50) → 128 → 64 → 32 → Binary output
│  │  │  ├─ Typical accuracy: 54-57% (good but not exceptional)
│  │  │  ├─ Training: Backpropagation (gradient descent)
│  │  │  ├─ Regularization critical: Dropout, L2, early stopping
│  │  │  ├─ GPU acceleration: Training faster (but still hours)
│  │  │  ├─ Optimization: Adam optimizer (adaptive learning rates)
│  │  │  └─ Problem: Black box (hard to interpret why prediction made)
│  │  │
│  │  ├─ LSTM (Long Short-Term Memory):
│  │  │  ├─ Specialized for sequences (remembers long-term dependencies)
│  │  │  ├─ Input: Sequence of 20-100 prior days
│  │  │  ├─ Output: Next day prediction
│  │  │  ├─ Advantage: Captures temporal patterns (trending, momentum)
│  │  │  ├─ Typical accuracy: 55-59% (can be very good!)
│  │  │  ├─ Overfitting risk: Very high (many parameters)
│  │  │  ├─ Training time: Hours-days (GPU essential)
│  │  │  ├─ Hyperparameters: Units per layer, num layers, lookback, dropout
│  │  │  ├─ Lessons learned:
│  │  │  │  ├─ Small dataset (<5k samples) → Overfit (bad)
│  │  │  │  ├─ Dropout 0.2-0.5 essential
│  │  │  │  ├─ Early stopping on validation set (prevent overfitting)
│  │  │  │  └─ Batch normalization helps training stability
│  │  │  └─ Empirical: Often similar to XGBoost, more complex
│  │  │
│  │  ├─ CNN (Convolutional Neural Networks):
│  │  │  ├─ Learn spatial patterns (price charts as images)
│  │  │  ├─ Input: 100-day price chart (100×10 matrix: OHLCVMA)
│  │  │  ├─ Convolutional filters learn local patterns (support/resistance)
│  │  │  ├─ Accuracy: 52-56% (not obviously better than traditional)
│  │  │  ├─ Advantage: Automated pattern discovery
│  │  │  ├─ Disadvantage: High GPU cost, hard to interpret
│  │  │  └─ Application: Multi-timeframe analysis (1D, 5M, 1H charts)
│  │  │
│  │  └─ Transformer Architecture (Emerging):
│  │     ├─ Attention mechanism (learns which past data is relevant)
│  │     ├─ State-of-the-art in NLP, emerging in finance
│  │     ├─ Advantage: Handles long-range dependencies better than LSTM
│  │     ├─ Disadvantage: Very complex, requires massive compute
│  │     └─ Frontier research (academic, not yet mainstream in trading)
│  │
│  └─ Hyperparameter Tuning:
│     ├─ Grid Search: Try all parameter combinations
│     │  ├─ Example: max_depth=[5,10,15], learning_rate=[0.01, 0.1, 0.5]
│     │  ├─ Exhaustive, finds best, computationally expensive
│     │  └─ 3 × 3 = 9 models, scales exponentially
│     │
│     ├─ Random Search: Sample random combinations
│     │  ├─ More efficient than grid search
│     │  ├─ Often finds competitive solutions with less computation
│     │  └─ Recommended for 5+ hyperparameters
│     │
│     ├─ Bayesian Optimization: Probabilistic search
│     │  ├─ Learn which regions of parameter space are promising
│     │  ├─ Focus search where likely to find good parameters
│     │  ├─ Most efficient but complex to implement
│     │  └─ Tools: Optuna, Hyperopt
│     │
│     └─ Validation Strategy:
│        ├─ Never tune on test set (causes optimistic bias)
│        ├─ Use validation set or cross-validation
│        ├─ Walk-forward: Re-tune every month/quarter
│        └─ Monitor: Performance should remain stable across time periods
│
├─ Backtesting & Deployment:
│  ├─ Signal Generation:
│  │  ├─ Daily: Train model, generate tomorrow's prediction
│  │  ├─ Probability output: 0.6 = 60% confidence of up
│  │  ├─ Signal rules: P > 0.55 → BUY, P < 0.45 → SELL
│  │  ├─ Position sizing: Risk proportional to confidence (position ∝ |P - 0.5|)
│  │  └─ Avoid: Binary trades (all-in on every signal) → Too risky
│  │
│  ├─ Backtesting Framework:
│  │  ├─ Walk-forward: 10-year history
│  │  │  ├─ Train on years 1-7 (7 years)
│  │  │  ├─ Test on year 8 (out-of-sample year 1)
│  │  │  ├─ Roll: Train on years 2-8, test on year 9
│  │  │  ├─ Repeat: All data eventually tested once
│  │  │  ├─ Results: Average of all test periods
│  │  │  └─ Robust: Accounts for regime changes, data shifts
│  │  │
│  │  ├─ Transaction Costs:
│  │  │  ├─ Commissions: 0.1-0.5 bps per trade
│  │  │  ├─ Slippage: 1-10 bps (varies by stock)
│  │  │  ├─ Impact: 5-100 bps for large orders
│  │  │  ├─ Typically: 10-50 bps per round-trip
│  │  │  └─ Critical: High-signal-volume strategies can be killed by costs
│  │  │
│  │  └─ Rebalancing Frequency:
│  │     ├─ Daily: High costs, but quick to respond
│  │     ├─ Weekly: Moderate costs, lag updates
│  │     ├─ Monthly: Low costs, slow response
│  │     └─ Optimize: Find sweet spot between responsiveness and costs
│  │
│  ├─ Overfitting Detection:
│  │  ├─ Red Flag 1: Training accuracy 65%, test accuracy 51%
│  │  ├─ Red Flag 2: Strategy works on 2010-2019, fails in 2020
│  │  ├─ Red Flag 3: Best parameters are at extreme values (max depth = 100)
│  │  ├─ Red Flag 4: Parameters change drastically every month
│  │  ├─ Tests:
│  │  │  ├─ Out-of-sample: Independent test period (never seen during training)
│  │  │  ├─ Robustness: Slight parameter changes shouldn't hurt much
│  │  │  ├─ Stability: Performance consistent across time periods
│  │  │  ├─ Monte Carlo: Randomize data, check robustness
│  │  │  └─ Parameter sensitivity: Sweep ±50% of optimal, track performance
│  │  │
│  │  └─ Prevention:
│  │     ├─ Keep model simple (Occam's razor)
│  │     ├─ Regularization: L1/L2, dropout, early stopping
│  │     ├─ Cross-validation: Multiple test periods
│  │     ├─ Avoid: Feature engineering on test data (leakage)
│  │     └─ Document: All model details for reproducibility
│  │
│  └─ Live Trading:
│     ├─ Challenge: Real-time prediction on new data
│     ├─ Latency: Model inference should be <1 second (milliseconds ideal)
│     ├─ Data drift: Model trained on 2020-2023 may not work in 2024
│     ├─ Monitoring: Track live accuracy, trigger alerts if <45%
│     ├─ Retraining: Monthly or quarterly re-train on fresh data
│     ├─ Partial deployment: Paper trading first (simulated, no real money)
│     ├─ Risk controls: Position limits, drawdown stops
│     └─ Validation: Run model in parallel with old model, reconcile differences
│
├─ Advanced Techniques:
│  ├─ Feature Importance Analysis:
│  │  ├─ Tree-based: Extract feature contribution from splits
│  │  ├─ Permutation importance: Shuffle each feature, measure performance drop
│  │  ├─ SHAP values: Game theory approach, assign each feature a value
│  │  ├─ Interpretation: Which features drive the predictions?
│  │  ├─ Example: Technical indicators 40%, fundamentals 30%, microstructure 30%
│  │  └─ Use: Focus engineering on high-impact features
│  │
│  ├─ Transfer Learning:
│  │  ├─ Train on large dataset (all stocks, 5 years)
│  │  ├─ Fine-tune on small dataset (specific stock, 6 months)
│  │  ├─ Advantage: Leverage large dataset without retraining
│  │  ├─ Effective: Reduces overfitting on small data
│  │  └─ Application: New asset classes or stock listings
│  │
│  ├─ Active Learning:
│  │  ├─ Model identifies uncertain predictions
│  │  ├─ Human labels most uncertain samples
│  │  ├─ Retrain with new labeled data
│  │  ├─ Iterative improvement with minimal labeling
│  │  └─ Practical: Reduce labeling cost (expensive for sentiment, news)
│  │
│  ├─ Adversarial Training:
│  │  ├─ Generate adversarial examples (slightly perturbed inputs)
│  │  ├─ Train model to be robust to perturbations
│  │  ├─ Motivation: Real data is noisy, model shouldn't be brittle
│  │  ├─ Benefit: Generalization improves
│  │  └─ Cost: Additional training complexity
│  │
│  └─ Ensemble Diversity:
│     ├─ Motivation: Uncorrelated models provide better ensemble
│     ├─ Techniques:
│     │  ├─ Different algorithms (RF, XGB, SVM)
│     │  ├─ Different features (technical vs fundamental)
│     │  ├─ Different data (stocks vs sectors)
│     │  └─ Different time periods (retrain on rolling windows)
│     ├─ Measurement: Correlation of predictions across models
│     ├─ Target: Correlation < 0.7 for meaningful diversity
│     └─ Result: +1-3% improvement in signal
│
└─ Common Pitfalls & Lessons:
   ├─ Lookahead Bias:
   │  ├─ Using future information in features
   │  ├─ Example: "Predict tomorrow's close using today's close" (circular)
   │  ├─ Detection: Test set performance way better than expected
   │  └─ Prevention: Time-aware validation, explicit checks
   │
   ├─ Overfitting to Backtest:
   │  ├─ Tuning parameters on test set (intentionally or accidentally)
   │  ├─ Result: Great backtest, terrible live trading
   │  ├─ Detection: Multiple timeframes show conflicting results
   │  └─ Prevention: One test set, frozen parameters
   │
   ├─ Regime Change:
   │  ├─ Model trained on 2015-2019 bull market fails in 2020 crash
   │  ├─ Non-stationary data: Statistics change over time
   │  ├─ Solution: Include diverse regimes in training data
   │  └─ Monitoring: Re-train regularly (monthly/quarterly)
   │
   ├─ Survivorship Bias:
   │  ├─ Backtest only on stocks that survived to present
   │  ├─ Excludes bankruptcies, delistings (often worst performers)
   │  ├─ Result: Backtest returns are too optimistic
   │  └─ Fix: Include dead companies in historical data
   │
   └─ Computational Complexity:
      ├─ Deep learning looks impressive, adds little alpha
      ├─ Simpler models (RF, XGB) often better risk-adjusted returns
      ├─ GPU costs, infrastructure complexity
      ├─ Maintenance burden: Updates, retraining, monitoring
      └─ Recommendation: Start simple, add complexity only if needed
```

**Interaction:** Prepare data → Select model → Train on history → Validate on out-of-sample → Backtest with transaction costs → Monitor live performance → Retrain periodically → Adapt.

## 5. Mini-Project
Implement ML signal generation with cross-validation and realistic backtesting:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Generate synthetic stock data
np.random.seed(42)
n_days = 2500  # ~10 years daily data

# Base price process (random walk with drift)
returns = np.random.normal(0.0005, 0.02, n_days)  # 0.05% daily drift, 2% vol
prices = 100 * np.exp(np.cumsum(returns))
close = prices

# OHLCV data
high = close * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
low = close * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
open_p = close + np.random.normal(0, 0.5, n_days)
volume = np.random.uniform(1e6, 5e6, n_days)

# Technical indicators (manual calculation)
def compute_indicators(prices, volume, window=20):
    df = pd.DataFrame({
        'close': prices,
        'high': high,
        'low': low,
        'volume': volume
    })
    
    # Simple Moving Average
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    
    # Momentum indicators
    df['rsi'] = 100 - (100 / (1 + (df['close'].diff(1).rolling(14).mean() / 
                                    (-df['close'].diff(1).rolling(14).mean()).abs())))
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    
    # Volatility
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    
    # Volume
    df['vol_sma'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma']
    
    # Returns
    df['ret_1d'] = df['close'].pct_change(1)
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_20d'] = df['close'].pct_change(20)
    
    # Price levels
    df['price_high_20'] = df['close'].rolling(20).max()
    df['price_low_20'] = df['close'].rolling(20).min()
    
    return df

df = compute_indicators(close, volume)

# Target: Next day return > 0?
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

# Fill NaN and drop
df = df.dropna()

# Features
feature_cols = ['sma_10', 'sma_20', 'sma_50', 'rsi', 'macd', 'atr', 
                'vol_ratio', 'ret_1d', 'ret_5d', 'ret_20d', 'price_high_20', 'price_low_20']

X = df[feature_cols].values
y = df['target'].values

print("="*100)
print("MACHINE LEARNING TRADING SIGNALS")
print("="*100)

print(f"\nStep 1: Data Summary")
print(f"-" * 50)
print(f"Total samples: {len(df)}")
print(f"Features: {len(feature_cols)}")
print(f"Target distribution: {np.sum(y)} ups ({np.sum(y)/len(y)*100:.1f}%), {len(y)-np.sum(y)} downs ({(len(y)-np.sum(y))/len(y)*100:.1f}%)")
print(f"Date range: ~10 years")

# Walk-forward validation
print(f"\nStep 2: Walk-Forward Backtesting")
print(f"-" * 50)

# Split: 7 years train, 1 year test (rolling)
train_size = int(0.7 * len(df))
test_size = int(0.1 * len(df))

models = {
    'Logistic': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'XGBoost': GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
}

results = []

for model_name, model in models.items():
    train_accs, test_accs = [], []
    train_f1s, test_f1s = [], []
    
    # Walk-forward: multiple test periods
    n_periods = 5
    period_size = len(df) // (n_periods + 1)
    
    for period in range(n_periods):
        train_start = 0
        train_end = (period + 1) * period_size + train_size
        test_start = train_end
        test_end = test_start + test_size
        
        if test_end > len(df):
            break
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[:train_end])
        X_test = scaler.transform(X[test_start:test_end])
        y_train = y[:train_end]
        y_test = y[test_start:test_end]
        
        # Train and evaluate
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_train, y_train)
        
        train_pred = model_clone.predict(X_train[-test_size:])
        test_pred = model_clone.predict(X_test)
        
        train_acc = accuracy_score(y_train[-test_size:], train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        train_f1 = f1_score(y_train[-test_size:], train_pred)
        test_f1 = f1_score(y_test, test_pred)
        
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        train_f1s.append(train_f1)
        test_f1s.append(test_f1)
    
    results.append({
        'Model': model_name,
        'Train Acc': np.mean(train_accs),
        'Test Acc': np.mean(test_accs),
        'Train F1': np.mean(train_f1s),
        'Test F1': np.mean(test_f1s),
        'Overfitting': np.mean(train_accs) - np.mean(test_accs),
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print(f"\nStep 3: Model Performance Analysis")
print(f"-" * 50)

best_model_name = results_df.loc[results_df['Test Acc'].idxmax(), 'Model']
print(f"Best model (by test accuracy): {best_model_name}")
print(f"Test accuracy: {results_df['Test Acc'].max()*100:.2f}% (vs 50% random)")
print(f"Overfitting gap: {results_df[results_df['Model']==best_model_name]['Overfitting'].values[0]*100:.2f}%")

print(f"\nStep 4: Signal Generation & Strategy")
print(f"-" * 50)

# Train final model on all data before test period
train_end = train_size + test_size
scaler_final = StandardScaler()
X_train_final = scaler_final.fit_transform(X[:train_end])
y_train_final = y[:train_end]

best_model = models[best_model_name]
best_model.fit(X_train_final, y_train_final)

# Generate predictions on test set
X_test_final = scaler_final.transform(X[train_end:train_end+test_size])
y_test_final = y[train_end:train_end+test_size]
predictions = best_model.predict(X_test_final)

# Get probabilities for signal strength
if hasattr(best_model, 'predict_proba'):
    probs = best_model.predict_proba(X_test_final)[:, 1]
else:
    probs = predictions  # Binary for LR

print(f"Signal generation period: {len(predictions)} trading days")
print(f"Buy signals (1): {np.sum(predictions)} days ({np.sum(predictions)/len(predictions)*100:.1f}%)")
print(f"Sell signals (0): {len(predictions)-np.sum(predictions)} days ({(len(predictions)-np.sum(predictions))/len(predictions)*100:.1f}%)")

# Backtest strategy
print(f"\nStep 5: Strategy Backtest (With Transaction Costs)")
print(f"-" * 50)

# Simple strategy: go long on buy signal, flat on sell
commission = 0.001  # 0.1% per trade
cumul_return = 1.0
trades = 0
positions = []

for i, (pred, prob) in enumerate(zip(predictions, probs)):
    actual_return = (prices[train_end+i+1] / prices[train_end+i]) - 1
    
    if pred == 1 and (not positions or positions[-1] == 0):  # Open long
        if positions and positions[-1] == 0:
            trades += 1
        positions.append(1)
        # Pay commission on entry
        cumul_return *= (1 + actual_return - commission)
    elif pred == 0 and (not positions or positions[-1] == 1):  # Close long
        if positions and positions[-1] == 1:
            trades += 1
        positions.append(0)
        # Pay commission on exit
        cumul_return *= (1 + actual_return - commission)
    else:
        positions.append(positions[-1] if positions else 0)
        cumul_return *= (1 + actual_return)

total_return = (cumul_return - 1) * 100
annualized_return = (cumul_return ** (252 / len(predictions)) - 1) * 100

# Buy-and-hold benchmark
buy_hold_return = (prices[train_end+len(predictions)] / prices[train_end] - 1) * 100
buy_hold_annual = (((prices[train_end+len(predictions)] / prices[train_end]) ** (252 / len(predictions))) - 1) * 100

print(f"Strategy return: {total_return:.2f}% ({annualized_return:.2f}% annualized)")
print(f"Buy-hold return: {buy_hold_return:.2f}% ({buy_hold_annual:.2f}% annualized)")
print(f"Outperformance: {total_return - buy_hold_return:.2f}% (strategy vs buy-hold)")
print(f"Total trades: {trades}")
print(f"Average trade duration: {len(predictions) / max(trades, 1):.0f} days")

# VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Model comparison
ax = axes[0, 0]
x_pos = np.arange(len(results_df))
width = 0.35
ax.bar(x_pos - width/2, results_df['Train Acc'], width, label='Train Acc', alpha=0.8)
ax.bar(x_pos + width/2, results_df['Test Acc'], width, label='Test Acc', alpha=0.8)
ax.axhline(y=0.5, color='red', linestyle='--', label='Random')
ax.set_ylabel('Accuracy')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x_pos)
ax.set_xticklabels(results_df['Model'])
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 2: Overfitting analysis
ax = axes[0, 1]
ax.bar(results_df['Model'], results_df['Overfitting'], alpha=0.7, color='orange')
ax.set_ylabel('Train - Test Accuracy')
ax.set_title('Overfitting Gap (Larger = More Overfit)')
ax.axhline(y=0.05, color='green', linestyle='--', label='Acceptable')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 3: Strategy backtest
ax = axes[1, 0]
strategy_cumul = np.cumprod([1 + (prices[train_end+i+1]/prices[train_end+i]-1 - (0.001 if predictions[i]==1 else 0)) 
                             for i in range(len(predictions)-1)])
bh_cumul = np.cumprod([1 + (prices[train_end+i+1]/prices[train_end+i]-1) 
                        for i in range(len(predictions)-1)])
ax.plot(strategy_cumul, label='ML Strategy', linewidth=2)
ax.plot(bh_cumul, label='Buy-Hold', linewidth=2, alpha=0.7)
ax.set_xlabel('Trading Days')
ax.set_ylabel('Cumulative Return')
ax.set_title('Strategy vs Buy-Hold Performance')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Signal distribution
ax = axes[1, 1]
signal_probs_up = probs[predictions == 1]
signal_probs_down = probs[predictions == 0]
ax.hist([signal_probs_down, signal_probs_up], bins=30, label=['Down Signals', 'Up Signals'], alpha=0.7)
ax.set_xlabel('Prediction Probability')
ax.set_ylabel('Frequency')
ax.set_title('Signal Probability Distribution')
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print(f"\n" + "="*100)
print("KEY INSIGHTS")
print(f"="*100)
print(f"- Best model: {best_model_name} ({results_df['Test Acc'].max()*100:.2f}% accuracy)")
print(f"- Overfitting detected: Gap > 5% suggests regularization needed")
print(f"- Strategy alpha: {total_return - buy_hold_return:.2f}% vs buy-hold (before slippage)")
print(f"- Transaction costs matter: {commissions*trades:.2f}% cumulative drag from {trades} trades")
print(f"- Regime stability: Walk-forward accuracy varies {results_df['Test Acc'].min()*100:.1f}%-{results_df['Test Acc'].max()*100:.1f}%")
```

## 6. Challenge Round
- Build ensemble: Combine 3 models (RF, XGB, SVM) with ensemble voting; measure improvement over individual models
- Implement LSTM: Create 2-layer LSTM on 20-day sequences; compare accuracy to XGBoost
- Feature importance analysis: Extract top 5 features from best model; understand what signals drive predictions
- Hyperparameter optimization: Use Bayesian search to optimize best model; track convergence
- Stress test model: Evaluate on 2008 crisis, 2020 COVID, 2022 bear market separately; identify regime-dependent performance

## 7. Key References
- [Krauss et al (2017), "Deep Neural Networks, Gradient-Based Optimization, and SQ Complexity," Journal of Financial Econometrics](https://www.sciencedirect.com/science/article/pii/S0304405X16301593) — ML for stock prediction empirical results
- [Prado (2018), "Advances in Financial Machine Learning," Wiley](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning%2C+1st+Edition-p-9781119482086) — Practical ML techniques for finance
- [Goodfellow et al (2016), "Deep Learning," MIT Press](https://www.deeplearningbook.org/) — Foundational deep learning theory
- [Chollet (2018), "Deep Learning with Python," Manning](https://www.manning.com/books/deep-learning-with-python) — Practical Keras/TensorFlow guide

---
**Status:** Advanced signal generation (emerging in practice, high compute cost) | **Complements:** Statistical Arbitrage, Factor Models, HFT Detection, Sentiment Analysis
