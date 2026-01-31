import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

np.random.seed(42)

# ============================================================================
# MARKET SIMULATOR
# ============================================================================

@dataclass

class VWAPExecution(ExecutionStrategy):
    """Volume-Weighted Average Price: trade proportional to volume."""
    
    def __init__(self, duration=10):
        self.duration = duration
    
    def execute(self, market: MarketSimulator, total_qty: int, side: str) -> dict:
        arrival_state = market.get_market_state()
        arrival_price = arrival_state.mid_price
        
        # Forecast volume for scheduling
        volume_forecast = []
        start_period = market.current_period
        for i in range(self.duration):
            idx = start_period + i
            if idx < market.n_periods:
                volume_forecast.append(market.volumes[idx])
            else:
                volume_forecast.append(market.volumes[-1])
        
        total_forecast = sum(volume_forecast)
        
        # Execute proportional to volume
        execution_prices = []
        quantities = []
        
        for vol_forecast in volume_forecast:
            if market.get_market_state() is None:
                break
            
            slice_qty = int(total_qty * (vol_forecast / total_forecast))
            if slice_qty > 0:
                exec_price, _ = market.execute_order(slice_qty, side)
                execution_prices.append(exec_price)
                quantities.append(slice_qty)
            
            market.advance()
        
        avg_price = np.average(execution_prices, weights=quantities)
        slippage_bps = (avg_price - arrival_price) / arrival_price * 10000
        if side == 'sell':
            slippage_bps = -slippage_bps
        
        return {
            'strategy': 'VWAP',
            'avg_price': avg_price,
            'arrival_price': arrival_price,
            'slippage_bps': slippage_bps,
            'num_periods': len(execution_prices),
            'completion': sum(quantities) / total_qty
        }
