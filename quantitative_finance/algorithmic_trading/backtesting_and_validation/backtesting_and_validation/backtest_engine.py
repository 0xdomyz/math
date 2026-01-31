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
