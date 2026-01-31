import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

np.random.seed(42)

# ============================================================================
# MARKET SIMULATOR
# ============================================================================

@dataclass

class TWAPExecution(ExecutionStrategy):
    """Time-Weighted Average Price: uniform slicing."""
    
    def __init__(self, n_slices=10):
        self.n_slices = n_slices
    
    def execute(self, market: MarketSimulator, total_qty: int, side: str) -> dict:
        arrival_state = market.get_market_state()
        arrival_price = arrival_state.mid_price
        
        slice_size = total_qty / self.n_slices
        execution_prices = []
        
        for _ in range(self.n_slices):
            if market.get_market_state() is None:
                break
            
            exec_price, _ = market.execute_order(int(slice_size), side)
            execution_prices.append(exec_price)
            market.advance()
        
        avg_price = np.mean(execution_prices)
        slippage_bps = (avg_price - arrival_price) / arrival_price * 10000
        if side == 'sell':
            slippage_bps = -slippage_bps
        
        return {
            'strategy': f'TWAP ({self.n_slices} slices)',
            'avg_price': avg_price,
            'arrival_price': arrival_price,
            'slippage_bps': slippage_bps,
            'num_periods': self.n_slices,
            'completion': len(execution_prices) / self.n_slices
        }
