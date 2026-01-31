import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

np.random.seed(42)

# ============================================================================
# MARKET SIMULATOR
# ============================================================================

@dataclass

class AggressiveExecution(ExecutionStrategy):
    """Execute entire order immediately (market order)."""
    
    def execute(self, market: MarketSimulator, total_qty: int, side: str) -> dict:
        arrival_state = market.get_market_state()
        arrival_price = arrival_state.mid_price
        
        exec_price, impact = market.execute_order(total_qty, side)
        market.advance()
        
        slippage_bps = (exec_price - arrival_price) / arrival_price * 10000
        if side == 'sell':
            slippage_bps = -slippage_bps
        
        return {
            'strategy': 'Aggressive',
            'avg_price': exec_price,
            'arrival_price': arrival_price,
            'slippage_bps': slippage_bps,
            'num_periods': 1,
            'completion': 1.0
        }
