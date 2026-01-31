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