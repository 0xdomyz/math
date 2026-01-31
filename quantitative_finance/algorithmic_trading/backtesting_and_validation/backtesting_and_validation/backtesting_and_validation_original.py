import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*70)
print("BACKTESTING AND VALIDATION FRAMEWORK")
print("="*70)

class BacktestEngine:
    """Event-driven backtesting engine"""
    
    def __init__(self, initial_capital=100000, commission=0.001, slippage=0.0005):
        self.initial_capital = initial_capital
        self.commission = commission  # Round-trip as fraction
        self.slippage = slippage  # Each way as fraction
        
        # State
        self.cash = initial_capital
        self.positions = {}  # {symbol: shares}
        self.portfolio_values = []
        self.trades = []
        self.dates = []
        
    def execute_trade(self, symbol, shares, price, date):
        """Execute a trade with transaction costs"""
        if shares == 0:
            return
        
        # Transaction costs
        notional = abs(shares * price)
        commission_cost = notional * self.commission
        slippage_cost = notional * self.slippage
        
        total_cost = commission_cost + slippage_cost
        
        # Update cash (negative for buys, positive for sells)
        self.cash -= shares * price + (total_cost if shares > 0 else -total_cost)
        
        # Update positions
        if symbol not in self.positions:
            self.positions[symbol] = 0
        self.positions[symbol] += shares
        
        # Record trade
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'commission': commission_cost,
            'slippage': slippage_cost,
            'total_cost': total_cost
        })
    
    def get_portfolio_value(self, prices):
        """Calculate total portfolio value"""
        holdings_value = sum(
            self.positions.get(symbol, 0) * prices.get(symbol, 0)
            for symbol in self.positions
        )
        return self.cash + holdings_value
    
    def record_state(self, date, prices):
        """Record portfolio state for this date"""
        portfolio_value = self.get_portfolio_value(prices)
        self.portfolio_values.append(portfolio_value)
        self.dates.append(date)
    
    def run_backtest(self, signals, prices):
        """
        Run backtest on signal dataframe
        signals: DataFrame with columns = symbols, values = target position (-1, 0, 1)
        prices: DataFrame with columns = symbols, values = prices
        """
        # Ensure aligned
        dates = signals.index.intersection(prices.index)
        
        for date in dates:
            current_prices = prices.loc[date].to_dict()
            target_positions = signals.loc[date].to_dict()
            
            # Calculate required trades
            for symbol in target_positions:
                target_shares = target_positions[symbol]
                current_shares = self.positions.get(symbol, 0)
                
                # Scale by capital allocation (simplified: equal weight)
                if target_shares != 0:
                    position_size = self.initial_capital * 0.1  # 10% per position
                    target_shares = int(position_size / current_prices[symbol]) * target_shares
                
                # Execute trade if needed
                trade_shares = target_shares - current_shares
                if trade_shares != 0 and symbol in current_prices:
                    self.execute_trade(symbol, trade_shares, current_prices[symbol], date)
            
            # Record state
            self.record_state(date, current_prices)
        
        return self.get_results()
    
    def get_results(self):
        """Calculate performance metrics"""
        portfolio_series = pd.Series(self.portfolio_values, index=self.dates)
        returns = portfolio_series.pct_change().dropna()
        
        # Total return
        total_return = (portfolio_series.iloc[-1] / self.initial_capital) - 1
        
        # CAGR
        n_years = len(returns) / 252
        cagr = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe
        sharpe = (returns.mean() * 252) / volatility if volatility > 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Drawdown duration
        in_drawdown = drawdown < 0
        drawdown_periods = []
        current_dd_length = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_dd_length += 1
            else:
                if current_dd_length > 0:
                    drawdown_periods.append(current_dd_length)
                current_dd_length = 0
        
        max_dd_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Trade stats
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'max_dd_duration': max_dd_duration,
            'num_trades': len(self.trades),
            'portfolio_series': portfolio_series,
            'returns': returns,
            'trades_df': trades_df
        }

class WalkForwardAnalysis:
    """Walk-forward optimization and validation"""
    
    def __init__(self, train_period=504, test_period=126, step=126):
        """
        train_period: Training window in days (504 = 2 years)
        test_period: Test window in days (126 = 6 months)
        step: Step size for rolling (126 = 6 months)
        """
        self.train_period = train_period
        self.test_period = test_period
        self.step = step
    
    def optimize_strategy(self, data, param_grid):
        """
        Find best parameters on training data
        Returns best params and their performance
        """
        best_sharpe = -np.inf
        best_params = None
        
        for params in param_grid:
            # Generate signals with these parameters
            signals = self.generate_signals(data, params)
            
            # Quick performance calc
            returns = signals.shift(1) * np.log(data / data.shift(1))
            sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params
        
        return best_params, best_sharpe
    
    def generate_signals(self, prices, params):
        """Generate trading signals based on parameters"""
        # Example: MA crossover with parameters
        fast_period = params.get('fast', 50)
        slow_period = params.get('slow', 200)
        
        ma_fast = prices.rolling(window=fast_period).mean()
        ma_slow = prices.rolling(window=slow_period).mean()
        
        signals = pd.Series(0, index=prices.index)
        signals[ma_fast > ma_slow] = 1
        signals[ma_fast < ma_slow] = -1
        
        return signals
    
    def run_walk_forward(self, prices):
        """Execute walk-forward analysis"""
        results = []
        n = len(prices)
        
        start = 0
        while start + self.train_period + self.test_period <= n:
            # Define windows
            train_end = start + self.train_period
            test_end = train_end + self.test_period
            
            train_data = prices.iloc[start:train_end]
            test_data = prices.iloc[train_end:test_end]
            
            # Parameter grid (simplified)
            param_grid = [
                {'fast': 20, 'slow': 50},
                {'fast': 50, 'slow': 200},
                {'fast': 10, 'slow': 30}
            ]
            
            # Optimize on training data
            best_params, train_sharpe = self.optimize_strategy(train_data, param_grid)
            
            # Test on out-of-sample data
            test_signals = self.generate_signals(test_data, best_params)
            test_returns = test_signals.shift(1) * np.log(test_data / test_data.shift(1))
            test_sharpe = (test_returns.mean() * 252) / (test_returns.std() * np.sqrt(252))
            
            results.append({
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'best_params': best_params,
                'train_sharpe': train_sharpe,
                'test_sharpe': test_sharpe,
                'test_returns': test_returns
            })
            
            # Roll forward
            start += self.step
        
        return results

class MonteCarloValidator:
    """Monte Carlo simulation for strategy validation"""
    
    def __init__(self, returns, n_simulations=10000):
        self.returns = returns
        self.n_simulations = n_simulations
    
    def bootstrap_trades(self):
        """Bootstrap individual trades"""
        # Assuming we have trade returns (not daily)
        # For daily returns, we'll bootstrap blocks
        results = []
        
        for _ in range(self.n_simulations):
            # Resample with replacement
            simulated_returns = np.random.choice(
                self.returns, size=len(self.returns), replace=True
            )
            
            # Calculate metrics
            cumulative = (1 + simulated_returns).prod() - 1
            sharpe = simulated_returns.mean() / simulated_returns.std() * np.sqrt(252)
            
            # Max drawdown
            cum_series = (1 + pd.Series(simulated_returns)).cumprod()
            running_max = cum_series.expanding().max()
            drawdown = (cum_series - running_max) / running_max
            max_dd = drawdown.min()
            
            results.append({
                'return': cumulative,
                'sharpe': sharpe,
                'max_dd': max_dd
            })
        
        return pd.DataFrame(results)
    
    def get_confidence_intervals(self, confidence=0.95):
        """Calculate confidence intervals"""
        results = self.bootstrap_trades()
        
        alpha = 1 - confidence
        lower = alpha / 2
        upper = 1 - lower
        
        ci = {
            'return': (results['return'].quantile(lower), results['return'].quantile(upper)),
            'sharpe': (results['sharpe'].quantile(lower), results['sharpe'].quantile(upper)),
            'max_dd': (results['max_dd'].quantile(lower), results['max_dd'].quantile(upper))
        }
        
        return ci, results

def detect_overfitting(in_sample_sharpe, out_sample_sharpe):
    """Simple overfitting detection"""
    ratio = in_sample_sharpe / out_sample_sharpe if out_sample_sharpe != 0 else np.inf
    
    if ratio < 1.0:
        return "No overfitting (surprising)"
    elif ratio < 1.5:
        return "Healthy (minimal degradation)"
    elif ratio < 2.0:
        return "Moderate overfitting"
    else:
        return "Severe overfitting"

# Generate synthetic data
def generate_market_data(n_days=1500, n_assets=3, seed=42):
    """Generate synthetic price data"""
    np.random.seed(seed)
    
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    data = {}
    for i in range(n_assets):
        trend = 0.0003 + np.random.uniform(-0.0002, 0.0002)
        vol = 0.015 + np.random.uniform(0, 0.01)
        
        returns = np.random.normal(trend, vol, n_days)
        prices = 100 * np.exp(returns.cumsum())
        
        data[f'Asset_{i+1}'] = prices
    
    return pd.DataFrame(data, index=dates)

# Scenario 1: Basic Backtest
print("\n" + "="*70)
print("SCENARIO 1: Basic Backtest with Transaction Costs")
print("="*70)

# Generate data
prices_df = generate_market_data(n_days=1000, n_assets=2)

# Simple MA crossover strategy
def generate_ma_signals(prices_df):
    signals = pd.DataFrame(0, index=prices_df.index, columns=prices_df.columns)
    
    for col in prices_df.columns:
        ma_fast = prices_df[col].rolling(window=50).mean()
        ma_slow = prices_df[col].rolling(window=200).mean()
        
        signals[col][ma_fast > ma_slow] = 1
        signals[col][ma_fast < ma_slow] = -1
    
    return signals

signals = generate_ma_signals(prices_df)

# Backtest
engine = BacktestEngine(initial_capital=100000, commission=0.001, slippage=0.0005)
results = engine.run_backtest(signals, prices_df)

print(f"\nStrategy Performance:")
print(f"  Total Return: {results['total_return']:.2%}")
print(f"  CAGR: {results['cagr']:.2%}")
print(f"  Volatility: {results['volatility']:.2%}")
print(f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}")
print(f"  Max Drawdown: {results['max_drawdown']:.2%}")
print(f"  Max DD Duration: {results['max_dd_duration']:.0f} days")
print(f"  Number of Trades: {results['num_trades']}")

# Buy and hold comparison
bh_value = (prices_df.mean(axis=1) / prices_df.mean(axis=1).iloc[0]) * 100000
bh_return = (bh_value.iloc[-1] / 100000) - 1

print(f"\nBuy & Hold (equal weight):")
print(f"  Total Return: {bh_return:.2%}")

# Scenario 2: Transaction Cost Sensitivity
print("\n" + "="*70)
print("SCENARIO 2: Transaction Cost Sensitivity Analysis")
print("="*70)

cost_levels = [0.0, 0.0005, 0.001, 0.002, 0.005]

print(f"\n{'Cost (bps)':<15} {'Sharpe':<12} {'Total Return':<15}")
print("-" * 42)

for cost in cost_levels:
    engine_cost = BacktestEngine(initial_capital=100000, commission=cost, slippage=cost/2)
    results_cost = engine_cost.run_backtest(signals, prices_df)
    
    print(f"{cost*10000:<15.1f} {results_cost['sharpe_ratio']:<12.3f} {results_cost['total_return']:<15.2%}")

print(f"\nTransaction costs significantly impact high-frequency strategies")

# Scenario 3: In-Sample vs Out-of-Sample
print("\n" + "="*70)
print("SCENARIO 3: In-Sample vs Out-of-Sample Validation")
print("="*70)

# Split data
split_point = int(len(prices_df) * 0.7)

prices_train = prices_df.iloc[:split_point]
prices_test = prices_df.iloc[split_point:]

# Train (in-sample)
signals_train = generate_ma_signals(prices_train)
engine_train = BacktestEngine()
results_train = engine_train.run_backtest(signals_train, prices_train)

# Test (out-of-sample)
signals_test = generate_ma_signals(prices_test)
engine_test = BacktestEngine()
results_test = engine_test.run_backtest(signals_test, prices_test)

print(f"\nIn-Sample (Training, 70% of data):")
print(f"  Sharpe Ratio: {results_train['sharpe_ratio']:.3f}")
print(f"  Total Return: {results_train['total_return']:.2%}")

print(f"\nOut-of-Sample (Testing, 30% of data):")
print(f"  Sharpe Ratio: {results_test['sharpe_ratio']:.3f}")
print(f"  Total Return: {results_test['total_return']:.2%}")

overfitting_assessment = detect_overfitting(
    results_train['sharpe_ratio'], 
    results_test['sharpe_ratio']
)

print(f"\nOverfitting Assessment: {overfitting_assessment}")
print(f"  Ratio (In/Out): {results_train['sharpe_ratio']/results_test['sharpe_ratio'] if results_test['sharpe_ratio'] != 0 else 'inf':.2f}")

# Scenario 4: Walk-Forward Analysis
print("\n" + "="*70)
print("SCENARIO 4: Walk-Forward Analysis")
print("="*70)

# Use single asset for simplicity
prices_single = prices_df['Asset_1']

wfa = WalkForwardAnalysis(train_period=504, test_period=126, step=126)
wf_results = wfa.run_walk_forward(prices_single)

print(f"\nWalk-Forward Windows: {len(wf_results)}")
print(f"\n{'Window':<10} {'Train Sharpe':<15} {'Test Sharpe':<15} {'Best Params':<20}")
print("-" * 60)

for i, result in enumerate(wf_results[:5]):  # Show first 5
    params_str = f"({result['best_params']['fast']}/{result['best_params']['slow']})"
    print(f"{i+1:<10} {result['train_sharpe']:<15.3f} {result['test_sharpe']:<15.3f} {params_str:<20}")

# Aggregate out-of-sample performance
all_test_returns = pd.concat([r['test_returns'] for r in wf_results])
aggregate_sharpe = (all_test_returns.mean() * 252) / (all_test_returns.std() * np.sqrt(252))

print(f"\nAggregate Out-of-Sample Sharpe: {aggregate_sharpe:.3f}")

# Scenario 5: Monte Carlo Validation
print("\n" + "="*70)
print("SCENARIO 5: Monte Carlo Bootstrap Validation")
print("="*70)

# Use strategy returns
strategy_returns = results['returns']

mc_validator = MonteCarloValidator(strategy_returns, n_simulations=1000)
confidence_intervals, mc_results = mc_validator.get_confidence_intervals(confidence=0.95)

print(f"\n95% Confidence Intervals (10,000 bootstraps):")
print(f"  Sharpe Ratio: [{confidence_intervals['sharpe'][0]:.3f}, {confidence_intervals['sharpe'][1]:.3f}]")
print(f"  Total Return: [{confidence_intervals['return'][0]:.2%}, {confidence_intervals['return'][1]:.2%}]")
print(f"  Max Drawdown: [{confidence_intervals['max_dd'][0]:.2%}, {confidence_intervals['max_dd'][1]:.2%}]")

# Check if Sharpe CI includes zero
if confidence_intervals['sharpe'][0] > 0:
    print(f"\n✓ Strategy statistically significant (lower bound > 0)")
else:
    print(f"\n✗ Strategy NOT statistically significant (includes zero)")

# Scenario 6: Look-Ahead Bias Detection
print("\n" + "="*70)
print("SCENARIO 6: Look-Ahead Bias Detection")
print("="*70)

# Correct implementation (no look-ahead)
prices_single = prices_df['Asset_1']
ma_50 = prices_single.rolling(window=50).mean()
ma_200 = prices_single.rolling(window=200).mean()

signals_correct = pd.Series(0, index=prices_single.index)
signals_correct[ma_50 > ma_200] = 1
signals_correct[ma_50 < ma_200] = -1

returns_correct = signals_correct.shift(1) * prices_single.pct_change()
sharpe_correct = (returns_correct.mean() * 252) / (returns_correct.std() * np.sqrt(252))

# Incorrect implementation (look-ahead bias)
signals_wrong = pd.Series(0, index=prices_single.index)
signals_wrong[ma_50 > ma_200] = 1
signals_wrong[ma_50 < ma_200] = -1

# Execute same day (uses current close for entry)
returns_wrong = signals_wrong * prices_single.pct_change()
sharpe_wrong = (returns_wrong.mean() * 252) / (returns_wrong.std() * np.sqrt(252))

print(f"\nCorrect (execute next day):")
print(f"  Sharpe Ratio: {sharpe_correct:.3f}")

print(f"\nWith Look-Ahead Bias (execute same day):")
print(f"  Sharpe Ratio: {sharpe_wrong:.3f}")
print(f"  Artificial boost: {(sharpe_wrong/sharpe_correct - 1)*100:.1f}%")

print(f"\nLook-ahead bias inflates backtested performance!")

# Visualizations
fig, axes = plt.subplots(3, 2, figsize=(16, 14))

# Plot 1: Equity curve with drawdowns
ax = axes[0, 0]
portfolio_series = results['portfolio_series']
cumulative = portfolio_series / portfolio_series.iloc[0]

ax.plot(cumulative.index, cumulative.values, 'b-', linewidth=2, label='Strategy')

# Drawdown shading
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max

ax.fill_between(cumulative.index, cumulative.values, running_max.values, 
                where=(cumulative < running_max), alpha=0.3, color='red', label='Drawdown')

ax.set_xlabel('Date')
ax.set_ylabel('Portfolio Value (Normalized)')
ax.set_title('Equity Curve with Drawdowns')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Underwater (drawdown) plot
ax = axes[0, 1]
ax.fill_between(drawdown.index, 0, drawdown.values*100, color='red', alpha=0.5)
ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
ax.set_xlabel('Date')
ax.set_ylabel('Drawdown (%)')
ax.set_title(f'Underwater Plot (Max DD: {results["max_drawdown"]:.2%})')
ax.grid(alpha=0.3)

# Plot 3: Return distribution
ax = axes[1, 0]
returns_clean = results['returns'].dropna()

ax.hist(returns_clean * 100, bins=50, density=True, alpha=0.7, edgecolor='black', color='skyblue')

# Overlay normal distribution
mu, sigma = returns_clean.mean(), returns_clean.std()
x = np.linspace(returns_clean.min(), returns_clean.max(), 100)
ax.plot(x * 100, stats.norm.pdf(x, mu, sigma) / 100, 'r-', linewidth=2, label='Normal')

ax.set_xlabel('Daily Return (%)')
ax.set_ylabel('Density')
ax.set_title('Return Distribution')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 4: Rolling Sharpe ratio
ax = axes[1, 1]
rolling_window = 252  # 1 year
rolling_sharpe = returns_clean.rolling(window=rolling_window).apply(
    lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252))
)

ax.plot(rolling_sharpe.index, rolling_sharpe.values, 'g-', linewidth=2)
ax.axhline(0, color='r', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(results['sharpe_ratio'], color='b', linestyle='--', linewidth=1, 
           alpha=0.5, label=f'Overall: {results["sharpe_ratio"]:.2f}')
ax.set_xlabel('Date')
ax.set_ylabel('Rolling 252-day Sharpe')
ax.set_title('Rolling Sharpe Ratio (Strategy Stability)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Monte Carlo distribution
ax = axes[2, 0]
ax.hist(mc_results['sharpe'], bins=50, density=True, alpha=0.7, edgecolor='black', color='orange')
ax.axvline(results['sharpe_ratio'], color='r', linestyle='--', linewidth=2, label='Actual')
ax.axvline(confidence_intervals['sharpe'][0], color='g', linestyle=':', linewidth=1.5, 
           label=f"95% CI")
ax.axvline(confidence_intervals['sharpe'][1], color='g', linestyle=':', linewidth=1.5)
ax.set_xlabel('Sharpe Ratio')
ax.set_ylabel('Density')
ax.set_title('Monte Carlo Bootstrap: Sharpe Distribution')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 6: Transaction cost impact
ax = axes[2, 1]
costs_plot = [0, 5, 10, 20, 50]
sharpes_plot = []

for cost_bps in costs_plot:
    cost_frac = cost_bps / 10000
    eng = BacktestEngine(initial_capital=100000, commission=cost_frac, slippage=cost_frac/2)
    res = eng.run_backtest(signals, prices_df)
    sharpes_plot.append(res['sharpe_ratio'])

ax.plot(costs_plot, sharpes_plot, 'ro-', linewidth=2, markersize=8)
ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.3)
ax.set_xlabel('Transaction Cost (bps round-trip)')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('Transaction Cost Sensitivity')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()