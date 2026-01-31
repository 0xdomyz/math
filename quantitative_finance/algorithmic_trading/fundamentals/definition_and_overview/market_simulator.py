import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

np.random.seed(42)

# ============================================================================
# MARKET SIMULATOR
# ============================================================================

@dataclass

class MarketSimulator:
    """Realistic market with spread, impact, volume patterns."""
    
    def __init__(self, initial_price=100.0, n_periods=100):
        self.initial_price = initial_price
        self.n_periods = n_periods
        self.current_period = 0
        
        # Generate price path (GBM)
        returns = np.random.normal(0, 0.001, n_periods)
        self.prices = initial_price * np.exp(np.cumsum(returns))
        
        # Generate spread (mean-reverting, widens with volatility)
        self.spreads = 0.05 + 0.02 * np.abs(np.random.normal(0, 1, n_periods))
        
        # Generate volume (U-shaped intraday pattern)
        time_factor = np.linspace(0, 2*np.pi, n_periods)
        base_volume = 1000 * (1 + 0.5*np.cos(time_factor - np.pi))
        self.volumes = base_volume * (1 + 0.3*np.random.randn(n_periods))
        self.volumes = np.maximum(self.volumes, 100)
    
    def get_market_state(self) -> MarketState:
        """Return current market snapshot."""
        if self.current_period >= self.n_periods:
            return None
        
        mid = self.prices[self.current_period]
        spread = self.spreads[self.current_period]
        
        return MarketState(
            time=self.current_period,
            mid_price=mid,
            bid_price=mid - spread/2,
            ask_price=mid + spread/2,
            bid_size=int(self.volumes[self.current_period] * 0.1),
            ask_size=int(self.volumes[self.current_period] * 0.1),
            volume=int(self.volumes[self.current_period])
        )
    
    def execute_order(self, quantity: int, side: str) -> Tuple[float, float]:
        """
        Execute market order, return (avg_price, market_impact_bps).
        side: 'buy' or 'sell'
        """
        state = self.get_market_state()
        
        # Base price (cross spread)
        if side == 'buy':
            base_price = state.ask_price
        else:
            base_price = state.bid_price
        
        # Market impact: proportional to (quantity / volume)^0.5
        participation = quantity / state.volume if state.volume > 0 else 0
        impact_bps = 50 * np.sqrt(participation)  # Square root impact
        
        if side == 'buy':
            execution_price = base_price * (1 + impact_bps / 10000)
        else:
            execution_price = base_price * (1 - impact_bps / 10000)
        
        return execution_price, impact_bps
    
    def advance(self):
        """Move to next time period."""
        self.current_period += 1

# ============================================================================
# EXECUTION STRATEGIES
# ============================================================================
