import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

np.random.seed(42)

# ============================================================================
# MARKET SIMULATOR
# ============================================================================

@dataclass

class MarketState:
    """Snapshot of market at specific time."""
    time: int
    mid_price: float
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    volume: int  # Recent period volume
